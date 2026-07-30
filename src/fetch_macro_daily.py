"""
fetch_macro_daily.py
Daily macro indicators ingestion for market-pulse-pipeline.

What this does:
1. Fetches USD/INR, Crude Oil, and S&P 500 from yfinance (tradeable instruments)
2. Fetches the Fed Funds Rate from FRED (a policy value, not a tradeable instrument)
3. Writes all of today's values into macro_indicators (long format: date, indicator_name, value)
4. Logs the run to ingestion_log; logs any per-indicator failures to ingestion_failures

Two different sources, one script, one daily ingestion run - see project notes
for why: both feed the same "today's macro snapshot" and should succeed/fail
together as one coherent daily task, not two separately scheduled jobs.

Run daily: python fetch_macro_daily.py
Safe to re-run: ON CONFLICT DO NOTHING on (date, indicator_name).
"""

import os
from datetime import datetime, timezone, date

import yfinance as yf
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
FRED_API_KEY = os.getenv("FRED_API_KEY")
engine = create_engine(DATABASE_URL)

# yfinance tickers for tradeable macro instruments
YFINANCE_INDICATORS = {
    "USD_INR": "INR=X",
    "CRUDE_OIL": "CL=F",
    "SP500": "^GSPC",
}

# FRED series IDs for policy/economic values
FRED_INDICATORS = {
    "FED_RATE": "FEDFUNDS",
}


def fetch_yfinance_macro():
    """
    Fetches the latest available value for each yfinance-based indicator.
    Returns a list of (indicator_name, value) tuples for successes, and
    a separate list of (indicator_name, error_message) for failures.
    """
    results = []
    errors = []

    for indicator_name, ticker in YFINANCE_INDICATORS.items():
        try:
            data = yf.Ticker(ticker).history(period="5d")
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            latest_close = float(data["Close"].iloc[-1])
            results.append((indicator_name, round(latest_close, 4)))
        except Exception as e:
            errors.append((indicator_name, str(e)))

    return results, errors


def fetch_fred_data():
    """
    Fetches the latest value for each FRED-based indicator using FRED's
    public REST API. Returns (results, errors) in the same shape as
    fetch_yfinance_macro().
    """
    results = []
    errors = []

    for indicator_name, series_id in FRED_INDICATORS.items():
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": FRED_API_KEY,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 1,  # only need the most recent observation
            }
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            observations = response.json().get("observations", [])

            if not observations:
                raise ValueError(f"No observations returned for series {series_id}")

            latest_value = observations[0]["value"]
            if latest_value == ".":  # FRED uses "." to represent missing data
                raise ValueError(f"FRED returned missing value ('.') for {series_id}")

            results.append((indicator_name, round(float(latest_value), 4)))
        except Exception as e:
            errors.append((indicator_name, str(e)))

    return results, errors


def insert_macro_value(conn, indicator_date: date, indicator_name: str, value: float):
    """Inserts one row into macro_indicators. Skips silently if already present."""
    conn.execute(
        text("""
            INSERT INTO macro_indicators (date, indicator_name, value)
            VALUES (:date, :indicator_name, :value)
            ON CONFLICT (date, indicator_name) DO NOTHING
        """),
        {"date": indicator_date, "indicator_name": indicator_name, "value": value},
    )


def create_ingestion_run(conn, source: str) -> int:
    """Creates the ingestion_log row up front so failures can reference a real run_id."""
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


def log_failure(conn, run_id: int, indicator_name: str, error_message: str):
    """
    Logs a failed indicator fetch. Note: ingestion_failures.stock_id stays NULL here
    since macro indicators aren't tied to any stock - the ticker column holds the
    indicator name instead so the failure is still queryable/identifiable.
    """
    conn.execute(
        text("""
            INSERT INTO ingestion_failures (run_id, stock_id, ticker, error_message)
            VALUES (:run_id, NULL, :indicator_name, :error_message)
        """),
        {"run_id": run_id, "indicator_name": indicator_name, "error_message": error_message},
    )


def main():
    today = datetime.now(timezone.utc).date()
    total_expected = len(YFINANCE_INDICATORS) + len(FRED_INDICATORS)

    print(f"Fetching macro indicators for {today}...\n")

    with engine.begin() as conn:
        run_id = create_ingestion_run(conn, source="macro")

    yf_results, yf_errors = fetch_yfinance_macro()
    fred_results, fred_errors = fetch_fred_data()

    all_results = yf_results + fred_results
    all_errors = yf_errors + fred_errors

    with engine.begin() as conn:
        for indicator_name, value in all_results:
            insert_macro_value(conn, today, indicator_name, value)
            print(f"  {indicator_name}: {value}")

        for indicator_name, error_message in all_errors:
            log_failure(conn, run_id, indicator_name, error_message)
            print(f"  {indicator_name}: FAILED - {error_message}")

        total_failed = len(all_errors)
        if total_failed == 0:
            result = "success"
        elif total_failed == total_expected:
            result = "failed"
        else:
            result = "partial_failure"

        update_ingestion_run(conn, run_id, result, total_expected, total_failed)

    print(f"\nRun complete: run_id={run_id}, result='{result}', "
          f"{len(all_results)} succeeded, {total_failed} failed")


if __name__ == "__main__":
    main()