"""
sentiment_correlation.py
Phase 8a: correlation study between headline sentiment and next-day price
movement. Deliberately NOT a predictive model - just checks whether the two
signals tend to move together at all, using Pearson correlation. Modeling
(Phase 8b) comes later, after the full pipeline is built.

Approach:
1. For each (stock_id, date) that has at least one headline, compute that
   day's AVERAGE sentiment (a stock can have multiple headlines same day).
2. For each such stock/date, find the NEXT TRADING DAY that actually has a
   row in stock_prices (not just date+1, since weekends/holidays have no
   row at all) using the LEAD() window function, and compute next-day return:
   (next_close - today_close) / today_close.
3. Compute Pearson correlation between avg_sentiment and next_day_return
   across all these (stock, date) pairs.

Run: python sentiment_correlation.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# This single query does most of the real work:
#  - CTE `daily_sentiment`: average sentiment per (stock, date) from headlines
#  - CTE `price_with_next`: uses LEAD() over each stock's own price history
#    (ordered by date) to grab the NEXT ROW's close - which is automatically
#    the next actual trading day for that stock, regardless of weekend/
#    holiday gaps, since LEAD() operates on existing rows, not calendar dates.
#  - Final SELECT joins sentiment to that next-day return.
QUERY = """
WITH daily_sentiment AS (
    SELECT
        hs.stock_id,
        nh.published_date AS date,
        AVG(nh.sentiment_score) AS avg_sentiment,
        COUNT(*) AS headline_count
    FROM news_headlines nh
    JOIN headline_stocks hs ON nh.headline_id = hs.headline_id
    GROUP BY hs.stock_id, nh.published_date
),
price_with_next AS (
    SELECT
        stock_id,
        date,
        close,
        LEAD(close) OVER (PARTITION BY stock_id ORDER BY date) AS next_close
    FROM stock_prices
)
SELECT
    ds.stock_id,
    s.ticker,
    ds.date,
    ds.avg_sentiment,
    ds.headline_count,
    pwn.close AS today_close,
    pwn.next_close,
    (pwn.next_close - pwn.close) / pwn.close AS next_day_return
FROM daily_sentiment ds
JOIN price_with_next pwn ON ds.stock_id = pwn.stock_id AND ds.date = pwn.date
JOIN stocks s ON ds.stock_id = s.stock_id
WHERE pwn.next_close IS NOT NULL
ORDER BY ds.date, s.ticker;
"""


def main():
    with engine.begin() as conn:
        result = conn.execute(text(QUERY))
        rows = result.fetchall()
        columns = result.keys()

    df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        print("No overlapping (stock, date) pairs with both sentiment AND a "
              "next trading day yet - need more accumulated headline data "
              "with dates that have price history the day after.")
        return

    df["avg_sentiment"] = df["avg_sentiment"].astype(float)
    df["next_day_return"] = df["next_day_return"].astype(float)

    print(f"Analyzing {len(df)} (stock, date) pairs with both sentiment and next-day price data...\n")
    print(df[["ticker", "date", "avg_sentiment", "headline_count", "next_day_return"]].to_string(index=False))

    correlation = df["avg_sentiment"].corr(df["next_day_return"])

    print(f"\n--- Correlation result ---")
    print(f"Pearson correlation (sentiment vs next-day return): {correlation:.4f}")
    print(f"Sample size: {len(df)} observations")

    if len(df) < 30:
        print("\nNote: sample size is small - this correlation is not yet "
              "statistically meaningful. Needs more days of accumulated "
              "headline data before drawing real conclusions.")


if __name__ == "__main__":
    main()