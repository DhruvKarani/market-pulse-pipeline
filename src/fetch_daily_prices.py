"""
fetch_daily_prices.py
Daily incremental price ingestion for market-pulse-pipeline.

Separate from backfill.py by design (see Phase 3 notes): backfill is a
one-time historical load; this is what runs forever, once a day, adding
just the newest trading day per stock. Reuses the same validation and
upsert logic/patterns as backfill.py for consistency.

Uses period="5d" (not "1d") and takes only the most recent row - "1d" can
sometimes return empty depending on time-of-day/market settlement, so a
small buffer is safer while still being tiny compared to backfill's 5y pull.

Run daily (via GitHub Actions going forward): python fetch_daily_prices.py
Safe to re-run: ON CONFLICT DO NOTHING on (stock_id, date).
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

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 3


def get_all_stocks(conn) -> list[dict]:
    result = conn.execute(text("SELECT stock_id, ticker FROM stocks ORDER BY stock_id"))
    return [{"stock_id": row[0], "ticker": row[1]} for row in result]


def fetch_latest_row(ticker: str):
    """Fetches the most recent trading day's OHLCV for one ticker. Retries once."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            return data.iloc[[-1]]  # most recent row only, as a 1-row DataFrame
        except Exception as e:
            last_error = e
            print(f"  Attempt {attempt} failed for {ticker}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    raise last_error


def validate_row(row) -> str | None:
    """Same data quality checks as backfill.py - see Phase 4 notes."""
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


def insert_price_row(conn, stock_id: int, date_idx, row) -> bool:
    """Inserts one validated price row. Returns True if inserted, False if
    skipped (either failed validation or already existed)."""
    reason = validate_row(row)
    if reason is not None:
        return False, reason

    result = conn.execute(
        text("""
            INSERT INTO stock_prices
                (stock_id, date, open, high, low, close, adj_close, volume)
            VALUES
                (:stock_id, :date, :open, :high, :low, :close, :adj_close, :volume)
            ON CONFLICT (stock_id, date) DO NOTHING
            RETURNING stock_id
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
    return result.fetchone() is not None, None


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
    with engine.begin() as conn:
        run_id = create_ingestion_run(conn, source="stocks_daily")
        stocks = get_all_stocks(conn)

    total_expected = len(stocks)
    total_inserted = 0
    total_skipped_dup = 0
    total_failed = 0

    print(f"Fetching latest trading day for {total_expected} stocks...\n")

    for stock in stocks:
        ticker, stock_id = stock["ticker"], stock["stock_id"]
        try:
            latest = fetch_latest_row(ticker)
            date_idx = latest.index[0]
            row = latest.iloc[0]

            with engine.begin() as conn:
                inserted, reason = insert_price_row(conn, stock_id, date_idx, row)

                if reason is not None:
                    total_failed += 1
                    log_failure(conn, run_id, stock_id, ticker, f"row rejected: {reason}")
                    print(f"  {ticker}: REJECTED - {reason}")
                elif inserted:
                    total_inserted += 1
                    print(f"  {ticker}: inserted {date_idx.date()}")
                else:
                    total_skipped_dup += 1
                    print(f"  {ticker}: already had {date_idx.date()}, skipped")

        except Exception as e:
            total_failed += 1
            print(f"  {ticker}: FAILED - {e}")
            with engine.begin() as conn:
                log_failure(conn, run_id, stock_id, ticker, str(e))

        time.sleep(0.5)  # small delay between requests - reduces risk of Yahoo
                          # rate-limiting/returning incomplete data when hitting
                          # many tickers in quick succession

    if total_failed == 0:
        result = "success"
    elif total_failed == total_expected:
        result = "failed"
    else:
        result = "partial_failure"

    with engine.begin() as conn:
        update_ingestion_run(conn, run_id, result, total_expected, total_failed)

    print(f"\n--- Daily price ingestion summary ---")
    print(f"New rows inserted     : {total_inserted}")
    print(f"Already existed (skip): {total_skipped_dup}")
    print(f"Failed/rejected       : {total_failed}")
    print(f"Run: run_id={run_id}, result='{result}'")


if __name__ == "__main__":
    main()