"""
backfill_single_tickers.py
One-off loader for specific tickers - use this when a ticker in the main
TICKERS list needs fixing (renamed, delisted, replaced) instead of
re-running the entire 80-ticker backfill.

Usage: edit TICKERS_TO_FIX below, then run: python backfill_single_tickers.py
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

# Edit this list with just the ticker(s) you need to (re)fetch
TICKERS_TO_FIX = [
    "ETERNAL.NS",   # replaces ZOMATO.NS (company renamed)
]

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3
HISTORY_PERIOD = "5y"


def fetch_stock_data(ticker: str):
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
        {"ticker": ticker, "company_name": company_name, "exchange": exchange, "sector": sector},
    )
    return result.scalar()


def insert_prices(conn, stock_id: int, history_df: pd.DataFrame):
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
                "adj_close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            },
        )
        rows_inserted += 1
    return rows_inserted, rejected_rows


def create_ingestion_run(conn, source: str) -> int:
    result = conn.execute(
        text("""
            INSERT INTO ingestion_log (run_timestamp, source, result, total_expected, total_failed)
            VALUES (:run_timestamp, :source, 'in_progress', 0, 0)
            RETURNING run_id
        """),
        {"run_timestamp": datetime.now(timezone.utc), "source": source},
    )
    return result.scalar()


def update_ingestion_run(conn, run_id: int, result: str, total_expected: int, total_failed: int):
    conn.execute(
        text("""
            UPDATE ingestion_log
            SET result = :result, total_expected = :total_expected, total_failed = :total_failed
            WHERE run_id = :run_id
        """),
        {"run_id": run_id, "result": result, "total_expected": total_expected, "total_failed": total_failed},
    )


def log_failure(conn, run_id: int, stock_id, ticker: str, error_message: str):
    conn.execute(
        text("""
            INSERT INTO ingestion_failures (run_id, stock_id, ticker, error_message)
            VALUES (:run_id, :stock_id, :ticker, :error_message)
        """),
        {"run_id": run_id, "stock_id": stock_id, "ticker": ticker, "error_message": error_message},
    )


def main():
    total_expected = len(TICKERS_TO_FIX)
    ticker_failures = 0
    total_price_rows = 0
    total_rejected_rows = 0

    print(f"Backfilling {total_expected} corrected ticker(s): {TICKERS_TO_FIX}\n")

    with engine.begin() as conn:
        run_id = create_ingestion_run(conn, source="stocks_correction")
    print(f"Ingestion run created: run_id={run_id}\n")

    for i, ticker in enumerate(TICKERS_TO_FIX, start=1):
        print(f"[{i}/{total_expected}] {ticker}")
        try:
            info, history_df = fetch_stock_data(ticker)
            with engine.begin() as conn:
                stock_id = upsert_stock(conn, ticker, info)
                rows, rejected = insert_prices(conn, stock_id, history_df)
                total_price_rows += rows
                total_rejected_rows += len(rejected)
                for bad_date, reason in rejected:
                    log_failure(conn, run_id, stock_id, ticker, f"row rejected for {bad_date}: {reason}")
            print(f"  -> inserted {rows} price rows, rejected {len(rejected)} (stock_id={stock_id})")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            ticker_failures += 1
            with engine.begin() as conn:
                log_failure(conn, run_id, None, ticker, str(e))

    if ticker_failures == 0 and total_rejected_rows == 0:
        result = "success"
    elif ticker_failures == total_expected:
        result = "failed"
    else:
        result = "partial_failure"

    total_failed = ticker_failures + total_rejected_rows
    with engine.begin() as conn:
        update_ingestion_run(conn, run_id, result, total_expected, total_failed)

    print(f"\nDone. run_id={run_id}, result='{result}', "
          f"{total_price_rows} rows inserted, {ticker_failures} ticker failures")


if __name__ == "__main__":
    main()