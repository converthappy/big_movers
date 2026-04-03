#!/usr/bin/env python3
"""
Append the latest daily OHLCV rows from Yahoo Finance to local ticker CSVs.

Default behavior:
- reads symbols from big_movers_result.csv
- updates CSVs in the configured source folders if they exist
- otherwise falls back to collected_stocks/

Examples:
  python3 update_prices_yfinance.py
  python3 update_prices_yfinance.py --dir collected_stocks
  python3 update_prices_yfinance.py --symbol AAPL --symbol MSFT --dry-run
  python3 update_prices_yfinance.py --all-in-dir --dir collected_stocks
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Callable, Iterable

import pandas as pd
import yfinance as yf


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULT = os.path.join(SCRIPT_DIR, "big_movers_result.csv")
DEFAULT_COLLECTED = os.path.join(SCRIPT_DIR, "collected_stocks")

if platform.system() == "Windows":
    DEFAULT_SOURCE_DIRS = [
        r"D:\US_stocks_daily_data\delisted stocks from 2000",
        r"D:\US_stocks_daily_data\listed stocks from 2000",
    ]
else:
    _home = os.path.expanduser("~")
    DEFAULT_SOURCE_DIRS = [
        os.path.join(_home, "US_stocks_daily_data", "delisted stocks from 2000"),
        os.path.join(_home, "US_stocks_daily_data", "listed stocks from 2000"),
    ]


@dataclass
class CsvTarget:
    symbol: str
    path: str


@dataclass
class TargetState:
    target: CsvTarget
    header: list[str]
    body_count: int
    last_date: str
    fetch_from: date


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Update local daily OHLCV CSVs from Yahoo Finance.")
    p.add_argument("--result", default=DEFAULT_RESULT, help="Path to big_movers_result.csv")
    p.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        default=[],
        help="Directory containing ticker CSV files. Can be passed multiple times.",
    )
    p.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="Ticker to update. Can be passed multiple times.",
    )
    p.add_argument(
        "--all-in-dir",
        action="store_true",
        help="Update every CSV found in the target directories instead of only symbols from big_movers_result.csv.",
    )
    p.add_argument("--dry-run", action="store_true", help="Show planned updates without writing files.")
    p.add_argument(
        "--pause",
        type=float,
        default=0.2,
        help="Seconds to sleep between Yahoo requests. Default: 0.2",
    )
    return p.parse_args()


def normalize_date(raw: str) -> str:
    s = str(raw or "").strip()
    if not s:
        return ""
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    if len(s) >= 10 and s[2] == "/" and s[5] == "/":
        return f"{s[6:10]}-{s[0:2]}-{s[3:5]}"
    return s


def discover_target_dirs(explicit_dirs: list[str]) -> list[str]:
    if explicit_dirs:
        return [os.path.abspath(os.path.expanduser(d)) for d in explicit_dirs]
    existing = [d for d in DEFAULT_SOURCE_DIRS if os.path.isdir(d)]
    if existing:
        return existing
    return [DEFAULT_COLLECTED]


def read_symbols_from_result(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Result file not found: {path}")
    symbols: set[str] = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = (row.get("symbol") or "").strip().upper()
            if sym:
                symbols.add(sym)
    return sorted(symbols)


def list_symbols_in_dirs(dirs: Iterable[str]) -> list[str]:
    found: set[str] = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fname in os.listdir(d):
            if fname.lower().endswith(".csv"):
                found.add(os.path.splitext(fname)[0].upper())
    return sorted(found)


def find_csv_target(symbol: str, dirs: Iterable[str]) -> CsvTarget | None:
    for d in dirs:
        for fname in (f"{symbol}.csv", f"{symbol.lower()}.csv"):
            path = os.path.join(d, fname)
            if os.path.exists(path):
                return CsvTarget(symbol=symbol, path=path)
    return None


def read_existing_rows(path: str) -> tuple[list[str], list[list[str]], str]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    header = rows[0]
    body = rows[1:]
    date_idx = None
    for candidate in ("DateTime", "Date"):
        if candidate in header:
            date_idx = header.index(candidate)
            break
    if date_idx is None:
        raise ValueError(f"CSV missing Date/DateTime column: {path}")
    last_date = ""
    for row in reversed(body):
        if date_idx < len(row):
            normalized = normalize_date(row[date_idx])
            if normalized:
                last_date = normalized
                break
    if not last_date:
        raise ValueError(f"Could not determine latest date in {path}")
    return header, body, last_date


def choose_index_column_value(header: list[str], row_count: int) -> str:
    if not header:
        return ""
    first = (header[0] or "").strip().lower()
    if first in {"", "unnamed: 0"}:
        return ""
    return str(row_count)


def fetch_new_rows(symbol: str, start_date: str) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    fetch_from = start + timedelta(days=1)
    today = date.today()
    if fetch_from > today:
        return pd.DataFrame()
    df = yf.download(
        symbol,
        start=fetch_from.isoformat(),
        end=(today + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index()
    if "Date" not in df.columns:
        raise ValueError(f"Yahoo response for {symbol} did not include Date column")
    keep = [col for col in ("Date", "Open", "High", "Low", "Close", "Volume") if col in df.columns]
    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    return df


def get_latest_market_date(reference_symbol: str = "SPY") -> str:
    today = date.today()
    start = today - timedelta(days=14)
    df = yf.download(
        reference_symbol,
        start=start.isoformat(),
        end=(today + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        threads=False,
    )
    if df is None or df.empty:
        return today.isoformat()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df = df.reset_index()
    if "Date" not in df.columns:
        return today.isoformat()
    dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if dates.empty:
        return today.isoformat()
    return dates.max().strftime("%Y-%m-%d")


def format_number(value: object, is_volume: bool = False) -> str:
    if pd.isna(value):
        return ""
    if is_volume:
        return str(int(round(float(value))))
    text = ("%f" % float(value)).rstrip("0").rstrip(".")
    return text if text else "0"


def build_csv_rows(header: list[str], existing_count: int, df: pd.DataFrame) -> list[list[str]]:
    date_col = "DateTime" if "DateTime" in header else "Date"
    idx_value = choose_index_column_value(header, existing_count)
    rows: list[list[str]] = []
    for offset, rec in enumerate(df.to_dict("records")):
        row_map = {
            header[0]: idx_value if offset == 0 else choose_index_column_value(header, existing_count + offset),
            date_col: rec["Date"],
            "Open": format_number(rec.get("Open")),
            "High": format_number(rec.get("High")),
            "Low": format_number(rec.get("Low")),
            "Close": format_number(rec.get("Close")),
            "Volume": format_number(rec.get("Volume"), is_volume=True),
        }
        rows.append([row_map.get(col, "") for col in header])
    return rows


def read_target_state(target: CsvTarget) -> TargetState:
    header, body, last_date = read_existing_rows(target.path)
    start = datetime.strptime(last_date, "%Y-%m-%d").date()
    return TargetState(
        target=target,
        header=header,
        body_count=len(body),
        last_date=last_date,
        fetch_from=start + timedelta(days=1),
    )


def append_rows_to_target(state: TargetState, df: pd.DataFrame, dry_run: bool = False) -> tuple[int, str]:
    if df is None or df.empty:
        return 0, state.last_date
    rows = build_csv_rows(state.header, state.body_count, df)
    if not dry_run:
        with open(state.target.path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    latest = str(df["Date"].iloc[-1])
    return len(rows), latest


def _extract_symbol_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not isinstance(df.columns, pd.MultiIndex):
        out = df.reset_index()
        if "Date" not in out.columns:
            return pd.DataFrame()
        return out
    levels = [set(map(str, df.columns.get_level_values(i))) for i in range(df.columns.nlevels)]
    target = str(symbol)
    symbol_level = next((idx for idx, values in enumerate(levels) if target in values), None)
    if symbol_level is None:
        return pd.DataFrame()
    try:
        out = df.xs(target, axis=1, level=symbol_level)
    except Exception:
        return pd.DataFrame()
    if out is None or out.empty:
        return pd.DataFrame()
    out = out.reset_index()
    if "Date" not in out.columns:
        date_col = next((col for col in out.columns if str(col).lower().startswith("date")), None)
        if date_col is None:
            return pd.DataFrame()
        out = out.rename(columns={date_col: "Date"})
    return out


def fetch_new_rows_batch(states: list[TargetState], end_date: str | None = None) -> dict[str, pd.DataFrame]:
    if not states:
        return {}
    start = min(state.fetch_from for state in states)
    end = end_date or (date.today() + timedelta(days=1)).isoformat()
    tickers = " ".join(state.target.symbol for state in states)
    df = yf.download(
        tickers,
        start=start.isoformat(),
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        actions=False,
        group_by="ticker",
        threads=False,
    )
    if df is None or df.empty:
        return {state.target.symbol: pd.DataFrame() for state in states}
    out: dict[str, pd.DataFrame] = {}
    for state in states:
        frame = _extract_symbol_frame(df, state.target.symbol)
        if frame.empty:
            out[state.target.symbol] = pd.DataFrame()
            continue
        keep = [col for col in ("Date", "Open", "High", "Low", "Close", "Volume") if col in frame.columns]
        frame = frame[keep].copy()
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.dropna(subset=["Date"])
        frame["Date"] = frame["Date"].dt.strftime("%Y-%m-%d")
        frame = frame[frame["Date"] > state.last_date].reset_index(drop=True)
        out[state.target.symbol] = frame
    return out


def batch_update_targets(
    targets: list[CsvTarget],
    dry_run: bool = False,
    batch_size: int = 40,
    latest_market_date: str | None = None,
    progress_callback: Callable[[str, dict], None] | None = None,
) -> dict:
    latest_market_date = latest_market_date or get_latest_market_date()
    updated = 0
    unchanged = 0
    missing = 0
    failed = 0
    details: list[dict] = []
    batch: list[TargetState] = []

    def emit(symbol: str, detail: dict):
        details.append(detail)
        if progress_callback:
            progress_callback(symbol, detail)

    def flush_batch():
        nonlocal updated, unchanged, failed, batch
        if not batch:
            return
        try:
            data_by_symbol = fetch_new_rows_batch(batch, end_date=(date.today() + timedelta(days=1)).isoformat())
            for state in batch:
                try:
                    frame = data_by_symbol.get(state.target.symbol, pd.DataFrame())
                    added, latest = append_rows_to_target(state, frame, dry_run=dry_run)
                    if added:
                        updated += 1
                        emit(state.target.symbol, {"symbol": state.target.symbol, "status": "updated", "rows_added": added, "latest": latest})
                    else:
                        unchanged += 1
                        emit(state.target.symbol, {"symbol": state.target.symbol, "status": "current"})
                except Exception as exc:
                    failed += 1
                    emit(state.target.symbol, {"symbol": state.target.symbol, "status": "error", "error": str(exc)})
        except Exception as exc:
            err = str(exc)
            for state in batch:
                failed += 1
                emit(state.target.symbol, {"symbol": state.target.symbol, "status": "error", "error": err})
        finally:
            batch = []

    for target in targets:
        if not target:
            continue
        try:
            state = read_target_state(target)
        except Exception as exc:
            failed += 1
            emit(target.symbol, {"symbol": target.symbol, "status": "error", "error": str(exc)})
            continue
        if state.last_date >= latest_market_date:
            unchanged += 1
            emit(state.target.symbol, {"symbol": state.target.symbol, "status": "current"})
            continue
        batch.append(state)
        if len(batch) >= max(1, int(batch_size)):
            flush_batch()
    flush_batch()
    return {
        "updated": updated,
        "unchanged": unchanged,
        "missing": missing,
        "failed": failed,
        "details": details,
        "latest_market_date": latest_market_date,
    }


def update_one(target: CsvTarget, dry_run: bool = False) -> tuple[int, str]:
    header, body, last_date = read_existing_rows(target.path)
    df = fetch_new_rows(target.symbol, last_date)
    if df.empty:
        return 0, last_date
    rows = build_csv_rows(header, len(body), df)
    if not dry_run:
        with open(target.path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    latest = str(df["Date"].iloc[-1])
    return len(rows), latest


def main() -> int:
    args = parse_args()
    target_dirs = discover_target_dirs(args.dirs)

    requested_symbols = sorted({s.strip().upper() for s in args.symbol if s.strip()})
    if requested_symbols:
        symbols = requested_symbols
    elif args.all_in_dir:
        symbols = list_symbols_in_dirs(target_dirs)
    else:
        symbols = read_symbols_from_result(args.result)

    if not symbols:
        print("No symbols to update.")
        return 1

    print("Target directories:")
    for d in target_dirs:
        status = "OK" if os.path.isdir(d) else "MISSING"
        print(f"  [{status}] {d}")

    updated = 0
    unchanged = 0
    missing = 0
    failed = 0
    targets: list[CsvTarget] = []
    symbol_order: list[str] = []
    for idx, symbol in enumerate(symbols, start=1):
        target = find_csv_target(symbol, target_dirs)
        if not target:
            missing += 1
            print(f"[{idx}/{len(symbols)}] {symbol}: missing local CSV")
            continue
        targets.append(target)
        symbol_order.append(symbol)

    latest_market_date = get_latest_market_date()
    print(f"Latest market date: {latest_market_date}")
    index_by_symbol = {symbol: idx for idx, symbol in enumerate(symbol_order, start=1)}

    def on_progress(symbol: str, detail: dict):
        idx = index_by_symbol.get(symbol, "?")
        status = detail.get("status")
        if status == "updated":
            mode = "would append" if args.dry_run else "appended"
            print(f"[{idx}/{len(symbols)}] {symbol}: {mode} {detail.get('rows_added', 0)} row(s) -> {detail.get('latest', '')}")
        elif status == "current":
            print(f"[{idx}/{len(symbols)}] {symbol}: already current")
        else:
            print(f"[{idx}/{len(symbols)}] {symbol}: ERROR {detail.get('error', 'unknown error')}")

    summary = batch_update_targets(
        targets,
        dry_run=args.dry_run,
        latest_market_date=latest_market_date,
        progress_callback=on_progress,
    )
    updated += summary["updated"]
    unchanged += summary["unchanged"]
    failed += summary["failed"]

    print("\nSummary")
    print(f"  updated:   {updated}")
    print(f"  unchanged: {unchanged}")
    print(f"  missing:   {missing}")
    print(f"  failed:    {failed}")
    if args.dry_run:
        print("  mode:      dry-run")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
