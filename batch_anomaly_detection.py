"""
batch_anomaly_detection.py
One-time (or on-demand) pass that runs volume-price divergence detection
over ALL existing historical data in stock_prices, one stock at a time,
and writes any detected anomalies into the anomalies table.

Run whenever you want to (re)scan full history: python batch_anomaly_detection.py
Safe to re-run: existing anomaly rows for the same (stock_id, date, anomaly_type)
are not duplicated (see note on uniqueness below).
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# This script lives at repo root, but anomaly_detection.py (the module it
# depends on) lives in src/ - there is no root-level copy. Running this
# exactly as the README instructs (`python batch_anomaly_detection.py` from
# repo root) previously failed immediately with
# `ModuleNotFoundError: No module named 'anomaly_detection'`, since Python
# only searches the script's own directory (repo root) and site-packages,
# not src/, for a bare top-level import. Adding src/ to sys.path before the
# import - rather than moving the file or converting the repo to a package
# layout, either of which would be a bigger restructuring than this bug
# warrants - fixes the crash with a minimal, localized change.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from anomaly_detection import detect_anomalies_for_stock

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


def get_all_stock_ids(conn) -> list[int]:
    result = conn.execute(text("SELECT stock_id FROM stocks ORDER BY stock_id"))
    return [row[0] for row in result]


def get_price_history(conn, stock_id: int) -> pd.DataFrame:
    """Fetches full price history for one stock as a DataFrame.
    close/volume are cast to float here because Postgres NUMERIC columns
    come back as Python Decimal via psycopg2, and Decimal can't be mixed
    with float in pandas arithmetic (rolling means, pct_change, etc.).
    NUMERIC precision matters for storage; float is fine for statistics."""
    result = conn.execute(
        text("""
            SELECT stock_id, date, close, volume
            FROM stock_prices
            WHERE stock_id = :stock_id
            ORDER BY date
        """),
        {"stock_id": stock_id},
    )
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "date", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df


def insert_anomaly(conn, anomaly: dict):
    """
    Inserts one anomaly row. Relies on the UNIQUE(stock_id, date, anomaly_type)
    constraint + ON CONFLICT DO NOTHING to keep this script safely re-runnable -
    enforced at the database level rather than checked manually in application code.
    Returns True if a new row was actually inserted, False if it was a no-op (duplicate).
    """
    result = conn.execute(
        text("""
            INSERT INTO anomalies (stock_id, date, anomaly_type, anomaly_category, severity_score)
            VALUES (:stock_id, :date, :anomaly_type, :anomaly_category, :severity_score)
            ON CONFLICT (stock_id, date, anomaly_type) DO NOTHING
            RETURNING anomaly_id
        """),
        anomaly,
    )
    return result.fetchone() is not None


def main():
    with engine.begin() as conn:
        stock_ids = get_all_stock_ids(conn)

    print(f"Running anomaly detection across {len(stock_ids)} stocks...\n")

    total_anomalies_found = 0
    total_market_events = 0
    total_dq_issues = 0

    for i, stock_id in enumerate(stock_ids, start=1):
        with engine.begin() as conn:
            price_df = get_price_history(conn, stock_id)

            if price_df.empty:
                continue

            anomalies = detect_anomalies_for_stock(price_df)

            inserted_count = 0
            for anomaly in anomalies:
                if insert_anomaly(conn, anomaly):
                    inserted_count += 1
                    if anomaly["anomaly_category"] == "market_event":
                        total_market_events += 1
                    else:
                        total_dq_issues += 1

            total_anomalies_found += inserted_count

        if inserted_count > 0:
            print(f"[{i}/{len(stock_ids)}] stock_id={stock_id}: {inserted_count} anomalies found")

    print("\n--- Batch anomaly detection summary ---")
    print(f"Total stocks scanned      : {len(stock_ids)}")
    print(f"Total anomalies inserted  : {total_anomalies_found}")
    print(f"  market_event            : {total_market_events}")
    print(f"  data_quality_issue      : {total_dq_issues}")


if __name__ == "__main__":
    main()