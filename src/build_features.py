"""
build_features.py
Phase 8b: builds the feature set used for next-day return prediction.

Reuses the exact same rolling-baseline logic as anomaly_detection.py
(daily_return, volume_ratio, price_zscore) for consistency - one shared
definition of these signals across the whole project, not two.

Features (all scale-independent - no raw price, see project notes):
  daily_return, volume_ratio, price_zscore  (per stock, from stock_prices)
  usd_inr, crude_oil, sp500, fed_rate       (shared across all stocks, from macro_indicators)
  avg_sentiment                             (optional - only for Model B, from news_headlines)

Target: next_day_return = (next_close - close) / close

Chronological split (NOT random) - see project notes on lookahead bias.
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from anomaly_detection import compute_baselines

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)


def get_all_stock_ids(conn) -> list[int]:
    result = conn.execute(text("SELECT stock_id FROM stocks ORDER BY stock_id"))
    return [row[0] for row in result]


def get_price_history(conn, stock_id: int) -> pd.DataFrame:
    """Full price history for one stock, cast to float (see Phase 6 notes
    on NUMERIC/Decimal vs float in pandas arithmetic)."""
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


def get_macro_wide(conn) -> pd.DataFrame:
    """
    macro_indicators is stored long-format (date, indicator_name, value).
    Pivots it to wide format (one column per indicator) since each training
    row needs all macro values as separate features, not separate rows.
    """
    result = conn.execute(text("SELECT date, indicator_name, value FROM macro_indicators"))
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=["date", "indicator_name", "value"])
    df["value"] = df["value"].astype(float)
    wide = df.pivot_table(index="date", columns="indicator_name", values="value").reset_index()
    wide.columns.name = None
    return wide


def get_daily_sentiment(conn) -> pd.DataFrame:
    """Average sentiment per (stock_id, date) - only used for Model B."""
    result = conn.execute(
        text("""
            SELECT hs.stock_id, nh.published_date AS date, AVG(nh.sentiment_score) AS avg_sentiment
            FROM news_headlines nh
            JOIN headline_stocks hs ON nh.headline_id = hs.headline_id
            GROUP BY hs.stock_id, nh.published_date
        """)
    )
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "date", "avg_sentiment"])
    df["avg_sentiment"] = df["avg_sentiment"].astype(float)
    return df


def build_stock_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes daily_return, volume_ratio, price_zscore (via the SAME
    compute_baselines used by anomaly detection), plus next_day_return
    as the target - using LEAD-equivalent shift(-1), which automatically
    uses the next EXISTING row (skips weekends/holidays correctly, same
    reasoning as the LEAD() window function in sentiment_correlation.py).
    """
    df = compute_baselines(price_df)
    df["next_close"] = df["close"].shift(-1)
    df["next_day_return"] = (df["next_close"] - df["close"]) / df["close"]
    return df


def build_full_feature_set(include_sentiment: bool = False) -> pd.DataFrame:
    """
    Builds the complete training-ready feature set across all stocks.
    include_sentiment=True builds Model B's feature set (smaller window,
    only dates with sentiment data survive the join).
    """
    with engine.begin() as conn:
        stock_ids = get_all_stock_ids(conn)
        macro_wide = get_macro_wide(conn)
        sentiment_df = get_daily_sentiment(conn) if include_sentiment else None

        all_stocks_features = []
        for stock_id in stock_ids:
            price_df = get_price_history(conn, stock_id)
            if price_df.empty:
                continue
            features = build_stock_features(price_df)
            all_stocks_features.append(features)

    full_df = pd.concat(all_stocks_features, ignore_index=True)

    # Merge macro features (shared across all stocks, joined on date only)
    full_df = full_df.merge(macro_wide, on="date", how="left")

    if include_sentiment:
        full_df = full_df.merge(sentiment_df, on=["stock_id", "date"], how="inner")
        # inner join deliberately shrinks to only (stock, date) pairs that
        # HAVE sentiment - this is what makes Model B's dataset small

    # Drop rows with no valid baseline yet (first 20 days per stock) or no
    # next_day_return (the very last day per stock, which has no "tomorrow")
    feature_cols = ["daily_return", "volume_ratio", "price_zscore", "next_day_return"]
    full_df = full_df.dropna(subset=feature_cols)

    return full_df


if __name__ == "__main__":
    # Quick manual check when run directly
    df_a = build_full_feature_set(include_sentiment=False)
    print(f"Model A feature set: {len(df_a)} rows")
    print(df_a[["stock_id", "date", "daily_return", "volume_ratio", "price_zscore", "next_day_return"]].head())

    df_b = build_full_feature_set(include_sentiment=True)
    print(f"\nModel B feature set: {len(df_b)} rows")
    print(df_b[["stock_id", "date", "avg_sentiment", "next_day_return"]].head())