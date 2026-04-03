#!/usr/bin/env python3
# Big Movers Viewer Server
# Usage: python Big_movers_server.py
# Then open http://localhost:5051/ in browser

import csv
import os
import json
import threading
import uuid
from bisect import bisect_right
from flask import Flask, jsonify, send_from_directory, request, Response
import pandas as pd
import yfinance as yf

import update_prices_yfinance as yahoo_updater

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=SCRIPT_DIR, static_url_path="")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Path configuration
RESULTS_CSV = os.path.join(SCRIPT_DIR, "big_movers_result.csv")
STOCKS_DIRS = [
    os.path.join(SCRIPT_DIR, "collected_stocks"),
]
COLLECTED_STOCKS_DIR = STOCKS_DIRS[0]

# SPY benchmark data source (UI "VS" overlay)
SPY_HIST_CSV = os.path.join(SCRIPT_DIR, "SPY Historical Data.csv")
_SPY_BARS_CACHE = None
_UNIVERSE_CLOSES_CACHE = None
_RS_SERIES_CACHE = {}
RESULT_FIELDS = ["year", "symbol", "gain_pct", "low_date", "high_date", "low_price", "high_price", "avg_vol_b"]
_REFRESH_JOBS = {}
_REFRESH_JOB_LOCK = threading.Lock()

def _normalize_date_maybe(raw):
    s = str(raw or "").strip()
    if not s:
        return ""
    # Already ISO-like
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    # MM/DD/YYYY -> YYYY-MM-DD
    if len(s) >= 10 and s[2] == "/" and s[5] == "/":
        mm = s[0:2]
        dd = s[3:5]
        yyyy = s[6:10]
        if mm.isdigit() and dd.isdigit() and yyyy.isdigit():
            return f"{yyyy}-{mm}-{dd}"
    return s

def _parse_float_maybe(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        # tolerate thousands separators
        s = s.replace(",", "")
        v = float(s)
        return v
    except (ValueError, TypeError):
        return None

def _parse_volume_maybe(x):
    """
    Parse volume like:
    - 52.00M
    - 1.23B
    - 450K
    - plain numeric
    """
    try:
        if x is None:
            return 0.0
        s = str(x).strip()
        if not s:
            return 0.0
        s = s.replace(",", "")
        if s.lower() == "nan":
            return 0.0

        mult = 1.0
        last = s[-1].upper()
        if last == "M":
            mult = 1e6
            s = s[:-1]
        elif last == "B":
            mult = 1e9
            s = s[:-1]
        elif last == "K":
            mult = 1e3
            s = s[:-1]

        v = float(s)
        return v * mult
    except (ValueError, TypeError):
        return 0.0

def _load_spy_bars():
    global _SPY_BARS_CACHE
    if _SPY_BARS_CACHE is not None:
        return _SPY_BARS_CACHE

    if not os.path.exists(SPY_HIST_CSV):
        _SPY_BARS_CACHE = []
        return _SPY_BARS_CACHE

    bars = []
    try:
        with open(SPY_HIST_CSV, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Support both:
                # - Date, Price, Open, High, Low, Vol.
                # - DateTime, Open, High, Low, Close, Volume
                raw_date = row.get("Date") or row.get("DateTime")
                date_str = _normalize_date_maybe(raw_date)
                if not date_str:
                    continue
                o = _parse_float_maybe(row.get("Open"))
                h = _parse_float_maybe(row.get("High"))
                l = _parse_float_maybe(row.get("Low"))
                c = _parse_float_maybe(row.get("Close"))
                if c is None:
                    c = _parse_float_maybe(row.get("Price"))
                v = _parse_volume_maybe(row.get("Volume"))
                if not v:
                    v = _parse_volume_maybe(row.get("Vol."))
                if c is None or c <= 0:
                    continue
                if o is None or h is None or l is None:
                    continue
                bars.append({
                    "time": date_str,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                })
    except Exception:
        bars = []

    bars.sort(key=lambda x: x.get("time") or "")
    _SPY_BARS_CACHE = bars
    return _SPY_BARS_CACHE


def _load_symbol_bars(symbol):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return []

    if symbol == "SPY":
        return _load_spy_bars()

    path = None
    for d in STOCKS_DIRS:
        for fname in [f"{symbol}.csv", f"{symbol.lower()}.csv"]:
            c = os.path.join(d, fname)
            if os.path.exists(c):
                path = c
                break
        if path:
            break

    if not path:
        return []

    bars = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return []
        if len(header) >= 2 and "date" in (header[1] or "").lower():
            fmt = "new"
        elif len(header) >= 1 and "date" in (header[0] or "").lower():
            fmt = "noindex"
        else:
            fmt = "old"
        for row in reader:
            try:
                if fmt == "new":
                    if len(row) < 7:
                        continue
                    t = row[1].strip()
                    o = float(row[2])
                    h = float(row[3])
                    l = float(row[4])
                    c = float(row[5])
                    v = float(row[6])
                elif fmt == "noindex":
                    if len(row) < 6:
                        continue
                    raw_t = row[0].strip()
                    if len(raw_t) == 10 and raw_t[2] == "/":
                        parts = raw_t.split("/")
                        t = f"{parts[2]}-{parts[0]:>02}-{parts[1]:>02}"
                    else:
                        t = raw_t
                    o = float(row[1])
                    h = float(row[2])
                    l = float(row[3])
                    c = float(row[4])
                    v = float(row[5])
                else:
                    if len(row) < 6:
                        continue
                    t = row[0].strip()
                    c = float(row[1])
                    o = float(row[2])
                    h = float(row[3])
                    l = float(row[4])
                    v = float(row[5])
                if c <= 0:
                    continue
                bars.append({
                    "time": t,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                })
            except (ValueError, IndexError):
                continue

    bars.sort(key=lambda x: x["time"])
    return bars


def _load_universe_close_cache():
    global _UNIVERSE_CLOSES_CACHE
    if _UNIVERSE_CLOSES_CACHE is not None:
        return _UNIVERSE_CLOSES_CACHE

    universe = {}
    stocks_dir = STOCKS_DIRS[0] if STOCKS_DIRS else None
    if not stocks_dir or not os.path.isdir(stocks_dir):
        _UNIVERSE_CLOSES_CACHE = universe
        return universe

    for fname in sorted(os.listdir(stocks_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        symbol = os.path.splitext(fname)[0].upper()
        bars = _load_symbol_bars(symbol)
        dates = []
        closes = []
        for bar in bars:
            t = bar.get("time")
            c = bar.get("close")
            if not t or c is None or c <= 0:
                continue
            dates.append(t)
            closes.append(float(c))
        if len(dates) >= 253:
            universe[symbol] = {"dates": dates, "closes": closes}

    _UNIVERSE_CLOSES_CACHE = universe
    return universe


def _reset_symbol_caches(symbol=None):
    global _UNIVERSE_CLOSES_CACHE
    _UNIVERSE_CLOSES_CACHE = None
    if symbol:
        _RS_SERIES_CACHE.pop(str(symbol).upper(), None)
    else:
        _RS_SERIES_CACHE.clear()


def _read_results_rows():
    if not os.path.exists(RESULTS_CSV):
        return []
    rows = []
    with open(RESULTS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({field: str(row.get(field, "") or "") for field in RESULT_FIELDS})
    return rows


def _write_results_rows(rows):
    with open(RESULTS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_download_frame(df):
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    frame = df.reset_index()
    if "Date" not in frame.columns and "Datetime" in frame.columns:
        frame = frame.rename(columns={"Datetime": "Date"})
    keep = [col for col in ("Date", "Open", "High", "Low", "Close", "Volume") if col in frame.columns]
    if "Date" not in keep:
        return pd.DataFrame()
    return frame[keep].copy()


def _download_symbol_history(symbol):
    df = yf.download(
        symbol,
        period="max",
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    frame = _flatten_download_frame(df)
    if frame.empty:
        return []
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.strftime("%Y-%m-%d")
    bars = []
    for rec in frame.to_dict("records"):
        t = _normalize_date_maybe(rec.get("Date"))
        o = _parse_float_maybe(rec.get("Open"))
        h = _parse_float_maybe(rec.get("High"))
        l = _parse_float_maybe(rec.get("Low"))
        c = _parse_float_maybe(rec.get("Close"))
        v = _parse_volume_maybe(rec.get("Volume"))
        if not t or None in (o, h, l, c):
            continue
        if c <= 0 or h <= 0 or l <= 0:
            continue
        bars.append({
            "time": t,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
    bars.sort(key=lambda x: x["time"])
    return bars


def _write_symbol_bars_csv(symbol, bars):
    os.makedirs(COLLECTED_STOCKS_DIR, exist_ok=True)
    path = os.path.join(COLLECTED_STOCKS_DIR, f"{symbol.upper()}.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        for bar in bars:
            writer.writerow([
                bar["time"],
                f"{bar['open']:.6f}".rstrip("0").rstrip("."),
                f"{bar['high']:.6f}".rstrip("0").rstrip("."),
                f"{bar['low']:.6f}".rstrip("0").rstrip("."),
                f"{bar['close']:.6f}".rstrip("0").rstrip("."),
                str(int(round(bar.get("volume", 0) or 0))),
            ])
    return path


def _compute_yearly_big_moves(symbol, bars):
    rows = []
    by_year = {}
    for idx, bar in enumerate(bars):
        year = str(bar.get("time", ""))[:4]
        if len(year) != 4 or not year.isdigit():
            continue
        by_year.setdefault(year, []).append((idx, bar))

    for year, items in sorted(by_year.items()):
        if len(items) < 2:
            continue
        min_low = None
        min_idx = None
        min_date = ""
        best = None
        for idx, bar in items:
            low = _parse_float_maybe(bar.get("low"))
            high = _parse_float_maybe(bar.get("high"))
            if low is not None and (min_low is None or low < min_low):
                min_low = low
                min_idx = idx
                min_date = bar["time"]
            if min_low is None or min_low <= 0 or high is None or high <= 0 or min_idx is None:
                continue
            gain_pct = ((high / min_low) - 1.0) * 100.0
            if best is None or gain_pct > best["gain_pct"]:
                period = bars[min_idx:idx + 1]
                daily_dollar = [
                    (_parse_float_maybe(day.get("close")) or 0.0) * (_parse_volume_maybe(day.get("volume")) or 0.0)
                    for day in period
                ]
                avg_monthly_dollar_b = (sum(daily_dollar) / len(daily_dollar) * 21.0 / 1e9) if daily_dollar else 0.0
                best = {
                    "year": year,
                    "symbol": symbol,
                    "gain_pct": gain_pct,
                    "low_date": min_date,
                    "high_date": bar["time"],
                    "low_price": min_low,
                    "high_price": high,
                    "avg_vol_b": avg_monthly_dollar_b,
                }
        if not best:
            continue
        rows.append({
            "year": best["year"],
            "symbol": best["symbol"],
            "gain_pct": f"{best['gain_pct']:.2f}",
            "low_date": best["low_date"],
            "high_date": best["high_date"],
            "low_price": f"{best['low_price']:.6f}".rstrip("0").rstrip("."),
            "high_price": f"{best['high_price']:.6f}".rstrip("0").rstrip("."),
            "avg_vol_b": f"{best['avg_vol_b']:.2f}",
        })
    return rows


def _import_symbol_and_scan(symbol):
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        raise ValueError("symbol required")
    bars = _download_symbol_history(symbol)
    if not bars:
        raise ValueError(f"No Yahoo history returned for {symbol}")
    csv_path = _write_symbol_bars_csv(symbol, bars)
    new_rows = _compute_yearly_big_moves(symbol, bars)
    if not new_rows:
        raise ValueError(f"No yearly big moves found for {symbol}")
    existing = [row for row in _read_results_rows() if str(row.get("symbol", "")).upper() != symbol]
    merged = existing + new_rows
    merged.sort(key=lambda row: (
        int(str(row.get("year", "0") or 0)),
        str(row.get("symbol", "")),
        str(row.get("low_date", "")),
        str(row.get("high_date", "")),
    ))
    _write_results_rows(merged)
    _reset_symbol_caches(symbol)
    return {
        "symbol": symbol,
        "csv_path": csv_path,
        "rows": new_rows,
        "bars": len(bars),
    }


def _refresh_chart_data_to_current():
    os.makedirs(COLLECTED_STOCKS_DIR, exist_ok=True)
    symbols = yahoo_updater.list_symbols_in_dirs([COLLECTED_STOCKS_DIR])
    if not symbols:
        symbols = yahoo_updater.read_symbols_from_result(RESULTS_CSV)
    targets = []
    missing = 0
    details = []
    for symbol in symbols:
        target = yahoo_updater.find_csv_target(symbol, [COLLECTED_STOCKS_DIR])
        if not target:
            missing += 1
            details.append({"symbol": symbol, "status": "missing"})
            continue
        targets.append(target)
    summary = yahoo_updater.batch_update_targets(targets, dry_run=False)
    _reset_symbol_caches()
    return {
        "updated": summary["updated"],
        "unchanged": summary["unchanged"],
        "missing": missing,
        "failed": summary["failed"],
        "details": details + summary["details"],
        "latest_market_date": summary.get("latest_market_date"),
    }


def _get_refresh_job_symbols():
    os.makedirs(COLLECTED_STOCKS_DIR, exist_ok=True)
    symbols = yahoo_updater.list_symbols_in_dirs([COLLECTED_STOCKS_DIR])
    if not symbols:
        symbols = yahoo_updater.read_symbols_from_result(RESULTS_CSV)
    return symbols


def _serialize_refresh_job(job):
    if not job:
        return None
    details = list(job.get("details", []))
    return {
        "id": job["id"],
        "state": job["state"],
        "total": job["total"],
        "processed": job["processed"],
        "updated": job["updated"],
        "unchanged": job["unchanged"],
        "missing": job["missing"],
        "failed": job["failed"],
        "current_symbol": job.get("current_symbol"),
        "current_index": job.get("current_index"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "latest_market_date": job.get("latest_market_date"),
        "error": job.get("error"),
        "last_detail": details[-1] if details else None,
    }


def _get_active_refresh_job():
    with _REFRESH_JOB_LOCK:
        for job in _REFRESH_JOBS.values():
            if job.get("state") == "running":
                return _serialize_refresh_job(job)
    return None


def _get_refresh_job(job_id):
    with _REFRESH_JOB_LOCK:
        job = _REFRESH_JOBS.get(job_id)
        return _serialize_refresh_job(job) if job else None


def _update_refresh_job(job_id, **patch):
    with _REFRESH_JOB_LOCK:
        job = _REFRESH_JOBS.get(job_id)
        if not job:
            return None
        job.update(patch)
        return _serialize_refresh_job(job)


def _run_refresh_job(job_id):
    with _REFRESH_JOB_LOCK:
        job = _REFRESH_JOBS.get(job_id)
        symbols = list(job.get("symbols", [])) if job else []
    try:
        targets = []
        latest_market_date = yahoo_updater.get_latest_market_date()
        _update_refresh_job(job_id, latest_market_date=latest_market_date)
        for symbol in symbols:
            target = yahoo_updater.find_csv_target(symbol, [COLLECTED_STOCKS_DIR])
            if not target:
                with _REFRESH_JOB_LOCK:
                    job = _REFRESH_JOBS.get(job_id)
                    if not job:
                        return
                    job["missing"] += 1
                    job["processed"] += 1
                    job["details"].append({"symbol": symbol, "status": "missing"})
                continue
            targets.append(target)

        total_targets = len(targets)
        symbol_index = {target.symbol: idx for idx, target in enumerate(targets, start=1)}

        def on_progress(symbol, detail):
            with _REFRESH_JOB_LOCK:
                job = _REFRESH_JOBS.get(job_id)
                if not job:
                    return
                job["current_symbol"] = symbol
                job["current_index"] = symbol_index.get(symbol, job["processed"] + 1)
                status = detail.get("status")
                if status == "updated":
                    job["updated"] += 1
                elif status == "current":
                    job["unchanged"] += 1
                else:
                    job["failed"] += 1
                job["processed"] += 1
                job["details"].append(detail)

        if total_targets:
            yahoo_updater.batch_update_targets(
                targets,
                dry_run=False,
                latest_market_date=latest_market_date,
                progress_callback=on_progress,
            )
        _reset_symbol_caches()
        _update_refresh_job(job_id, state="done", current_symbol=None, current_index=None, finished_at=pd.Timestamp.utcnow().isoformat())
    except Exception as exc:
        _update_refresh_job(job_id, state="error", current_symbol=None, current_index=None, finished_at=pd.Timestamp.utcnow().isoformat(), error=str(exc))


def _start_refresh_job():
    active = _get_active_refresh_job()
    if active:
        return active, False
    symbols = _get_refresh_job_symbols()
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "state": "running",
        "symbols": symbols,
        "total": len(symbols),
        "processed": 0,
        "updated": 0,
        "unchanged": 0,
        "missing": 0,
        "failed": 0,
        "current_symbol": None,
        "current_index": None,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "finished_at": None,
        "error": None,
        "details": [],
    }
    with _REFRESH_JOB_LOCK:
        _REFRESH_JOBS[job_id] = job
    thread = threading.Thread(target=_run_refresh_job, args=(job_id,), daemon=True)
    thread.start()
    return _serialize_refresh_job(job), True


def _weighted_ibd_score(closes, idx):
    lookbacks = (63, 126, 189, 252)
    if idx is None or idx < lookbacks[-1]:
        return None
    current = closes[idx]
    if current <= 0:
        return None
    weights = (0.4, 0.2, 0.2, 0.2)
    score = 0.0
    for weight, lb in zip(weights, lookbacks):
        base = closes[idx - lb]
        if base <= 0:
            return None
        score += weight * ((current / base) - 1.0)
    return score


def _compute_ibd_style_rs_series(symbol):
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return []
    if symbol in _RS_SERIES_CACHE:
        return _RS_SERIES_CACHE[symbol]

    universe = _load_universe_close_cache()
    target = universe.get(symbol)
    if not target:
        _RS_SERIES_CACHE[symbol] = []
        return []

    target_dates = target["dates"]
    target_closes = target["closes"]
    out = []

    for idx in range(252, len(target_dates)):
        date_str = target_dates[idx]
        target_score = _weighted_ibd_score(target_closes, idx)
        if target_score is None:
            continue

        peer_scores = []
        for peer in universe.values():
            peer_dates = peer["dates"]
            peer_idx = bisect_right(peer_dates, date_str) - 1
            if peer_idx < 252:
                continue
            score = _weighted_ibd_score(peer["closes"], peer_idx)
            if score is not None:
                peer_scores.append(score)

        if len(peer_scores) < 20:
            continue

        peer_scores.sort()
        rank = bisect_right(peer_scores, target_score)
        pct = int(round((rank / len(peer_scores)) * 99))
        pct = max(1, min(99, pct))
        out.append({"time": date_str, "rating": pct})

    _RS_SERIES_CACHE[symbol] = out
    return out


def _resolve_index_html_path():
    # Repo uses Big_movers.html; tolerate lowercase for clones on case-sensitive FS
    for name in ("Big_movers.html", "big_movers.html"):
        p = os.path.join(SCRIPT_DIR, name)
        if os.path.exists(p):
            return p
    return None


@app.route("/")
def index():
    html_path = _resolve_index_html_path()
    if not html_path:
        return f"Big_movers.html not found in {SCRIPT_DIR}", 404
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, mimetype="text/html")


@app.route("/api/results")
def api_results():
    if not os.path.exists(RESULTS_CSV):
        return jsonify({"error": "big_movers_result.csv not found"}), 404
    rows = []
    with open(RESULTS_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return jsonify(rows)


@app.route("/api/ohlcv")
def api_ohlcv():
    symbol = (request.args.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    try:
        bars = _load_symbol_bars(symbol)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not bars:
        return jsonify({"error": f"{symbol}.csv not found in any configured directory"}), 404
    return jsonify(bars)


@app.route("/api/rs_rating")
def api_rs_rating():
    symbol = (request.args.get("symbol") or "").strip().upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    if symbol == "SPY":
        return jsonify([])
    try:
        data = _compute_ibd_style_rs_series(symbol)
        return jsonify({
            "symbol": symbol,
            "method": "ibd_style_weighted_3_6_9_12m",
            "series": data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


DRAWINGS_FILE = os.path.join(DATA_DIR, "drawings.json")
LEGACY_DRAWINGS_FILE = os.path.join(SCRIPT_DIR, "drawings.json")
FAVORITES_FILE = os.path.join(DATA_DIR, "favorites.json")
SETUPS_FILE = os.path.join(DATA_DIR, "setups.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _read_json_file(path, fallback):
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback


def _write_json_file(path, data):
    _ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_state_with_legacy(primary_path, legacy_path, fallback):
    data = _read_json_file(primary_path, None)
    if data is not None:
        return data
    legacy = _read_json_file(legacy_path, None) if legacy_path else None
    if legacy is not None:
        try:
            _write_json_file(primary_path, legacy)
        except Exception:
            pass
        return legacy
    return fallback


@app.route("/api/drawings", methods=["GET", "POST"])
def api_drawings():
    if request.method == "GET":
        return jsonify(_read_state_with_legacy(DRAWINGS_FILE, LEGACY_DRAWINGS_FILE, {}))
    else:
        try:
            data = request.get_json(silent=True) or {}
            _write_json_file(DRAWINGS_FILE, data)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/api/favorites", methods=["GET", "POST"])
def api_favorites():
    if request.method == "GET":
        return jsonify(_read_state_with_legacy(FAVORITES_FILE, None, []))
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, list):
            data = []
        cleaned = [str(x) for x in data if str(x).strip()]
        _write_json_file(FAVORITES_FILE, cleaned)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/setups", methods=["GET", "POST"])
def api_setups():
    if request.method == "GET":
        return jsonify(_read_state_with_legacy(SETUPS_FILE, None, []))
    try:
        data = request.get_json(silent=True)
        if not isinstance(data, list):
            data = []
        cleaned = [item for item in data if isinstance(item, dict)]
        _write_json_file(SETUPS_FILE, cleaned)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/import_symbol", methods=["POST"])
def api_import_symbol():
    try:
        data = request.get_json(silent=True) or {}
        symbol = (data.get("symbol") or "").strip().upper()
        if not symbol:
            return jsonify({"error": "symbol required"}), 400
        result = _import_symbol_and_scan(symbol)
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/refresh_prices", methods=["GET", "POST"])
def api_refresh_prices():
    try:
        if request.method == "GET":
            job_id = (request.args.get("job_id") or "").strip()
            if job_id:
                job = _get_refresh_job(job_id)
                if not job:
                    return jsonify({"error": "refresh job not found"}), 404
                return jsonify({"ok": True, "job": job})
            active = _get_active_refresh_job()
            return jsonify({"ok": True, "job": active})
        job, started = _start_refresh_job()
        return jsonify({"ok": True, "job": job, "started": started})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Big Movers Viewer: http://localhost:5051/")
    app.run(host="127.0.0.1", port=5051, debug=False)
