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
from typing import Iterable

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

    for idx, symbol in enumerate(symbols, start=1):
        target = find_csv_target(symbol, target_dirs)
        if not target:
            missing += 1
            print(f"[{idx}/{len(symbols)}] {symbol}: missing local CSV")
            continue
        try:
            added, latest = update_one(target, dry_run=args.dry_run)
            if added:
                updated += 1
                mode = "would append" if args.dry_run else "appended"
                print(f"[{idx}/{len(symbols)}] {symbol}: {mode} {added} row(s) -> {latest}")
            else:
                unchanged += 1
                print(f"[{idx}/{len(symbols)}] {symbol}: already current")
        except Exception as exc:
            failed += 1
            print(f"[{idx}/{len(symbols)}] {symbol}: ERROR {exc}")
        if args.pause > 0 and idx < len(symbols):
            import time
            time.sleep(args.pause)

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
