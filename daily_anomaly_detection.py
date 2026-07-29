"""
daily_anomaly_detection.py
Meant to run daily, right after the daily price ingestion step, as part of
the automated pipeline. Unlike batch_anomaly_detection.py (which scans full
history), this only classifies TODAY's newest row per stock - but still
needs the preceding ~20 days of history to compute that stock's rolling
baseline, since a single row alone has nothing to compare against.

Reuses the exact same detection logic as the batch script (anomaly_detection.py)
so both stay consistent - only the amount of history pulled and which row's
classification actually gets acted on differs.

Run daily: python daily_anomaly_detection.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from anomaly_detection import compute_baselines, classify_row, ROLLING_WINDOW

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# Need the rolling window's worth of prior history PLUS today itself
DAYS_TO_FETCH = ROLLING_WINDOW + 1


def get_all_stock_ids(conn) -> list[int]:
    result = conn.execute(text("SELECT stock_id FROM stocks ORDER BY stock_id"))
    return [row[0] for row in result]


def get_recent_price_history(conn, stock_id: int) -> pd.DataFrame:
    """
    Fetches just the last (ROLLING_WINDOW + 1) days of price history for one
    stock - enough to compute a valid rolling baseline for the most recent day.
    """
    result = conn.execute(
        text("""
            SELECT stock_id, date, close, volume
            FROM stock_prices
            WHERE stock_id = :stock_id
            ORDER BY date DESC
            LIMIT :limit
        """),
        {"stock_id": stock_id, "limit": DAYS_TO_FETCH},
    )
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "date", "close", "volume"])
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.sort_values("date").reset_index(drop=True)  # back to ascending order


def insert_anomaly(conn, anomaly: dict) -> bool:
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

    print(f"Running daily anomaly check for {len(stock_ids)} stocks...\n")

    total_checked = 0
    total_anomalies = 0

    for stock_id in stock_ids:
        with engine.begin() as conn:
            price_df = get_recent_price_history(conn, stock_id)

            # Not enough history yet for this stock to have a valid baseline
            if len(price_df) < DAYS_TO_FETCH:
                continue

            df_with_baselines = compute_baselines(price_df)
            latest_row = df_with_baselines.iloc[-1]  # only classify TODAY's row

            total_checked += 1
            result = classify_row(latest_row)

            if result is not None:
                anomaly = {
                    "stock_id": int(latest_row["stock_id"]),
                    "date": latest_row["date"],
                    **result,
                }
                if insert_anomaly(conn, anomaly):
                    total_anomalies += 1
                    print(f"  stock_id={stock_id}: {result['anomaly_type']} "
                          f"({result['anomaly_category']}, severity={result['severity_score']})")

    print(f"\nDone. {total_checked} stocks checked, {total_anomalies} new anomalies flagged today.")


if __name__ == "__main__":
    main()