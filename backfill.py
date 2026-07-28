"""
backfill.py
One-time historical data loader for market-pulse-pipeline.

What this does:
1. Loops through a fixed list of ~80 NSE tickers
2. For each ticker: fetches company metadata (name, sector) and 5 years of daily OHLCV
3. Validates each price row against basic data quality rules before inserting
4. Inserts/updates the `stocks` table with metadata
5. Inserts valid price rows into `stock_prices`; drops and logs invalid ones
6. COMMITS AFTER EACH TICKER (not one giant transaction) so progress survives
   interruptions - killing the script only ever loses the ticker in progress,
   never everything before it
7. Retries once on transient API failure before giving up on a ticker
8. Logs the overall run to `ingestion_log`
9. Logs any per-ticker failures AND per-row data quality rejections to `ingestion_failures`

Run once: python backfill.py
Safe to re-run: already-inserted rows are skipped via ON CONFLICT DO NOTHING.
"""

import os
import time
from datetime import datetime, timezone

import yfinance as yf
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS", "SBIN.NS",
    "TCS.NS", "BAJFINANCE.NS", "LT.NS", "HINDUNILVR.NS", "SUNPHARMA.NS",
    "ADANIENT.NS", "INFY.NS", "MARUTI.NS", "ADANIPORTS.NS", "AXISBANK.NS",
    "TITAN.NS", "KOTAKBANK.NS", "M&M.NS", "ITC.NS", "ULTRACEMCO.NS",
    "NTPC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "WIPRO.NS",
    "HCLTECH.NS", "ASIANPAINT.NS", "NESTLEIND.NS", "BAJAJFINSV.NS", "ONGC.NS",
    "COALINDIA.NS", "JSWSTEEL.NS", "GRASIM.NS", "TECHM.NS", "DRREDDY.NS",
    "CIPLA.NS", "EICHERMOT.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "BPCL.NS", "SHREECEM.NS", "UPL.NS",
    "TATACONSUM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "INDUSINDBK.NS", "BAJAJ-AUTO.NS",
    "VEDL.NS", "PIDILITIND.NS", "DABUR.NS", "GODREJCP.NS", "SIEMENS.NS",
    "AMBUJACEM.NS", "ACC.NS", "BANKBARODA.NS", "PNB.NS", "CANBK.NS",
    "IDFCFIRSTB.NS", "GAIL.NS", "IOC.NS", "HAL.NS", "BEL.NS",
    "ZOMATO.NS", "DMART.NS", "TRENT.NS", "PAGEIND.NS", "MOTHERSON.NS",
    "TVSMOTOR.NS", "BOSCHLTD.NS", "MRF.NS", "COLPAL.NS", "MARICO.NS",
    "BERGEPAINT.NS", "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARMA.NS", "ABB.NS",
]

MAX_RETRIES = 2  # 1 initial attempt + 1 retry
RETRY_DELAY_SECONDS = 3
HISTORY_PERIOD = "5y"


def fetch_stock_data(ticker: str):
    """
    Fetches company metadata + 5 years of daily OHLCV for one ticker.
    Retries once on failure. Returns (info_dict, price_dataframe) or raises
    the last exception if all attempts fail.
    """
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            history_df = yf_ticker.history(period=HISTORY_PERIOD)

            if history_df.empty:
                raise ValueError(f"No price history returned for {ticker}")

            return info, history_df
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt} failed for {ticker}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    raise last_error


def validate_row(row) -> str | None:
    """
    Data quality checks on ONE price row, considered in isolation.
    Returns None if the row is valid, or a string reason if it should be rejected.

    Checks (deliberately NOT anomaly detection - no history, no comparison,
    just internal logical consistency of this single row):
      1. OHLC values must not be missing/NULL
      2. low <= open <= high, low <= close <= high  (structural invariant)
      3. volume must not be negative
      4. volume == 0 combined with real intra-day price movement is a contradiction
    """
    o, h, l, c, v = row.get("Open"), row.get("High"), row.get("Low"), row.get("Close"), row.get("Volume")

    if pd.isna(o) or pd.isna(h) or pd.isna(l) or pd.isna(c) or pd.isna(v):
        return "missing OHLCV value"

    o, h, l, c, v = float(o), float(h), float(l), float(c), float(v)

    if not (l <= o <= h):
        return f"open ({o}) outside [low={l}, high={h}]"
    if not (l <= c <= h):
        return f"close ({c}) outside [low={l}, high={h}]"
    if v < 0:
        return f"negative volume ({v})"
    if v == 0 and not (o == h == l == c):
        return "zero volume but price moved intraday (contradiction)"

    return None


def upsert_stock(conn, ticker: str, info: dict) -> int:
    """
    Inserts (or updates) a row in `stocks` using metadata from yfinance.
    Returns the stock_id.
    """
    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector") or "Unknown"
    exchange = "NSE" if ticker.endswith(".NS") else "BSE"

    result = conn.execute(
        text("""
            INSERT INTO stocks (ticker, company_name, exchange, sector)
            VALUES (:ticker, :company_name, :exchange, :sector)
            ON CONFLICT (ticker) DO UPDATE
                SET company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector
            RETURNING stock_id
        """),
        {
            "ticker": ticker,
            "company_name": company_name,
            "exchange": exchange,
            "sector": sector,
        },
    )
    return result.scalar()


def insert_prices(conn, stock_id: int, ticker: str, history_df: pd.DataFrame):
    """
    Validates and inserts daily OHLCV rows into stock_prices for one stock.
    Invalid rows are dropped (not inserted) and returned separately so the
    caller can log them to ingestion_failures.
    Uses ON CONFLICT DO NOTHING since (stock_id, date) is the primary key
    and this script may be re-run.

    Returns (rows_inserted, rejected_rows) where rejected_rows is a list of
    (date, reason) tuples.
    """
    rows_inserted = 0
    rejected_rows = []

    for date_idx, row in history_df.iterrows():
        reason = validate_row(row)
        if reason is not None:
            rejected_rows.append((date_idx.date(), reason))
            continue

        conn.execute(
            text("""
                INSERT INTO stock_prices
                    (stock_id, date, open, high, low, close, adj_close, volume)
                VALUES
                    (:stock_id, :date, :open, :high, :low, :close, :adj_close, :volume)
                ON CONFLICT (stock_id, date) DO NOTHING
            """),
            {
                "stock_id": stock_id,
                "date": date_idx.date(),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "adj_close": round(float(row["Close"]), 2),  # yfinance auto-adjusts Close by default
                "volume": int(row["Volume"]),
            },
        )
        rows_inserted += 1

    return rows_inserted, rejected_rows


def log_ingestion_run(conn, source: str, result: str, total_expected: int, total_failed: int) -> int:
    """Inserts one row into ingestion_log. Returns the run_id."""
    run_result = conn.execute(
        text("""
            INSERT INTO ingestion_log (run_timestamp, source, result, total_expected, total_failed)
            VALUES (:run_timestamp, :source, :result, :total_expected, :total_failed)
            RETURNING run_id
        """),
        {
            "run_timestamp": datetime.now(timezone.utc),
            "source": source,
            "result": result,
            "total_expected": total_expected,
            "total_failed": total_failed,
        },
    )
    return run_result.scalar()


def log_failure(conn, run_id: int, stock_id, ticker: str, error_message: str):
    """Inserts one row into ingestion_failures."""
    conn.execute(
        text("""
            INSERT INTO ingestion_failures (run_id, stock_id, ticker, error_message)
            VALUES (:run_id, :stock_id, :ticker, :error_message)
        """),
        {"run_id": run_id, "stock_id": stock_id, "ticker": ticker, "error_message": error_message},
    )


def main():
    total_expected = len(TICKERS)
    failures = []          # (ticker, stock_id_or_None, error_message) - ticker-level failures
    total_price_rows = 0
    total_rejected_rows = 0

    print(f"Starting backfill for {total_expected} tickers ({HISTORY_PERIOD} of history each)...\n")

    # NOTE: one transaction PER TICKER, not one for the whole run.
    # If the script is interrupted, only the ticker currently in progress
    # is lost - everything committed before it is safely saved.
    for i, ticker in enumerate(TICKERS, start=1):
        print(f"[{i}/{total_expected}] {ticker}")
        try:
            info, history_df = fetch_stock_data(ticker)

            with engine.begin() as conn:
                stock_id = upsert_stock(conn, ticker, info)
                rows, rejected = insert_prices(conn, stock_id, ticker, history_df)

                total_price_rows += rows
                total_rejected_rows += len(rejected)

                if rejected:
                    for bad_date, reason in rejected:
                        log_failure(conn, run_id=None, stock_id=stock_id, ticker=ticker,
                                    error_message=f"row rejected for {bad_date}: {reason}")

            print(f"  -> inserted {rows} price rows, rejected {len(rejected)} (stock_id={stock_id})")

        except Exception as e:
            print(f"  -> FAILED: {e}")
            failures.append((ticker, None, str(e)))

    # Log the overall run summary AFTER the loop, in its own short transaction
    total_failed = len(failures)
    if total_failed == 0 and total_rejected_rows == 0:
        result = "success"
    elif total_failed == total_expected:
        result = "failed"
    else:
        result = "partial_failure"

    with engine.begin() as conn:
        run_id = log_ingestion_run(conn, source="stocks", result=result,
                                    total_expected=total_expected, total_failed=total_failed)
        for ticker, stock_id, error_message in failures:
            log_failure(conn, run_id, stock_id, ticker, error_message)

    print("\n--- Backfill summary ---")
    print(f"Total tickers attempted   : {total_expected}")
    print(f"Total price rows inserted : {total_price_rows}")
    print(f"Total rows rejected (DQ)  : {total_rejected_rows}")
    print(f"Total ticker-level failures: {total_failed}")
    print(f"Run logged as run_id={run_id}, result='{result}'")


if __name__ == "__main__":
    main()