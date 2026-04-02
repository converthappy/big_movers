# Big Movers — A Stock Research & Charting Platform

**Built on the Qullamaggie Breakout Methodology** — [notes on X (Twitter)](https://x.com/traderwillhu/status/2033669135628955960)

---

## What Is Big Movers?

Big Movers is a self-hosted stock research tool that identifies and visualizes the most powerful price moves in US equity history. It processes raw daily OHLCV data for over 12,000 US tickers — including delisted stocks — and surfaces the top performers each year for in-depth chart study.

The project is inspired by the trading methodology of Kristjan Kullamägi (Qullamaggie), which focuses on learning from stocks that made exceptional gains in a short period of time. By studying these historical moves in detail — their price structure, volume behavior, and timing — traders can develop a sharper eye for recognizing similar setups as they form in real time.

---

## The Charting Interface

Results from your CSV are shown in a browser-based charting platform served locally via a lightweight Python server.

**Stock List**
The left panel displays all selected stocks sorted by year and gain percentage. You can filter by minimum gain, symbol name, or year. Use the ↑↓ arrow keys to navigate between stocks instantly.

**Price Chart**
Clicking any stock loads a full daily candlestick chart with volume histogram. The chart automatically zooms to the move window, with markers showing the exact low and high dates. You can switch between **Daily**, **Weekly**, and **Monthly** timeframes at any time — all drawing tools and annotations carry over across timeframes.

**Moving Averages**
Configurable EMA and SMA overlays with custom periods and colors. Defaults are EMA 20, EMA 50, and MA 200. All settings are saved automatically and restored on next visit.

**Log / Linear Scale**
Toggle between logarithmic and linear price scale with a single click. Logarithmic scale is especially useful for studying moves of 100%+ where the structure is distorted on a linear chart.

---

## Drawing Tools

Big Movers includes a full set of annotation tools designed for technical analysis:

| Tool | Shortcut | Description |
|------|----------|-------------|
| Pan | V | Navigate and select drawings |
| H-Line | H | Horizontal price level line |
| Line | A | Extended line (both directions) |
| Ray | R | One-directional ray from anchor point |
| Segment | S | Fixed-length line between two points |
| Text | T | Free-form text label anywhere on chart |
| Measure | M | Rectangle tool showing gain % and bar count |

**Drawing styles** — each tool has independent color, line width, and style (solid, dashed, dotted) settings, all saved automatically.

**Editing** — select any drawing with a single click (highlights in yellow), then drag to move it. Text labels can be double-clicked to edit. Press Delete to remove a selected drawing. Ctrl+Z to undo.

**Persistence** — all drawings are saved per symbol to a local JSON file on the server. They reload automatically whenever you revisit a stock, and remain intact across timeframe changes.

---

## Updating Daily Prices

The viewer itself still serves local CSVs, but you can refresh those CSVs from Yahoo Finance with:

```bash
cd /Users/isaacwardle/Documents/big_movers
python3 update_prices_yfinance.py
```

By default this:
- reads symbols from `big_movers_result.csv`
- updates matching ticker CSVs in your configured source folders if they exist
- otherwise falls back to `collected_stocks/`

Useful options:

```bash
python3 update_prices_yfinance.py --dir collected_stocks
python3 update_prices_yfinance.py --symbol AAPL --symbol MSFT
python3 update_prices_yfinance.py --all-in-dir --dir collected_stocks
python3 update_prices_yfinance.py --dry-run
```

---

## Technical Stack

- **Server** — Python / Flask, runs locally on port 5051
- **Charts** — Lightweight Charts v3.8 (TradingView open-source library)
- **Data** — Local daily CSV files, no external API required
- **Storage** — Settings saved to browser localStorage; drawings saved to local JSON via the Flask server

Everything runs offline on your own machine. No cloud, no subscription, no data leaving your computer.

---

## Who Is This For?

Big Movers is built for traders and researchers who want to study price history systematically rather than relying on screeners that only show current market data. By reviewing hundreds of historical setups that share the same structure — strong fundamentals, tight consolidation, explosive breakout — you build the visual pattern recognition that is the foundation of discretionary trading.

The tool does not give buy or sell signals. It gives you the raw material to study, annotate, and learn from the market's best historical moves at your own pace.

---

## Disclaimer

Stock data and any statistics or charts derived from it are **not** guaranteed to be 100% accurate or complete. You are responsible for independently verifying the accuracy and suitability of the data before using it for any purpose.

---

## License

This project is licensed under the [MIT License](LICENSE).



## ☕ Support My Work

If you find this project helpful, please consider buying me a coffee. Building and maintaining this tool involves significant development time and AI token costs.

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support%20my%20work-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/willhu)
