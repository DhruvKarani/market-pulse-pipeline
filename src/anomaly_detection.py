"""
anomaly_detection.py
Core volume-price divergence detection logic, shared by both the batch
(historical) and daily anomaly detection scripts.

Design recap (see project notes for full reasoning):
- Baselines are computed PER STOCK, over a rolling 20-trading-day window -
  "unusual" only means something relative to that stock's own normal behavior.
- Decision grid:
    significant price move + significant volume  -> market_event
    significant price move + normal/low volume    -> data_quality_issue
      (a real price move requires real trades; if volume doesn't back up
       the move, the price change itself is suspect, not confirmed)
    normal price move (any volume)                -> no anomaly (out of scope for v1)
- severity_score (0-1) blends BOTH price-move magnitude and volume-move
  magnitude, since a big move on huge volume is more severe than the same
  move on merely-elevated volume.
"""

import pandas as pd
import numpy as np

ROLLING_WINDOW = 20

# How many standard deviations a day's return must be from the rolling mean
# to count as a "significant" price move.
# Note: 2.0 was tested first and flagged ~7.5% of days per stock - higher than
# the ~5% theoretical rate for a normal distribution, consistent with real
# stock returns having "fat tails" (extreme moves happen more often than a
# clean normal distribution predicts). Raised to 2.5 to flag rarer, more
# standout days while still surfacing enough anomalies to be useful.
PRICE_ZSCORE_THRESHOLD = 2.5

# How many multiples of the rolling average volume counts as "significant" volume.
# Note: 2.0 produced a 46% market_event / 54% data_quality_issue split across
# 78 blue-chip stocks with cleanly validated data (0 rows failed the hard DQ
# checks in backfill) - implausible for over half of significant price moves
# to be genuinely bad data on liquid, heavily-traded stocks. That pointed to
# the volume bar being too strict rather than the data being wrong. Lowered
# to 1.5 so moderately elevated (but real) volume still counts as confirmation.
VOLUME_RATIO_THRESHOLD = 1.5


def compute_baselines(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a DataFrame of ONE stock's price history, sorted by date ascending,
    with columns: date, close, volume.

    Adds columns:
      daily_return       - % change in close vs previous day
      return_mean_20d    - rolling 20-day mean of daily_return (excludes today)
      return_std_20d     - rolling 20-day stddev of daily_return (excludes today)
      price_zscore       - how many std devs today's return is from the rolling mean
      volume_mean_20d    - rolling 20-day mean of volume (excludes today)
      volume_ratio       - today's volume / volume_mean_20d
    """
    df = price_df.sort_values("date").reset_index(drop=True).copy()

    df["daily_return"] = df["close"].pct_change()

    # shift(1) before rolling so "today" is never included in its own baseline -
    # otherwise an extreme day would inflate the very average it's being compared to
    df["return_mean_20d"] = df["daily_return"].shift(1).rolling(ROLLING_WINDOW).mean()
    df["return_std_20d"] = df["daily_return"].shift(1).rolling(ROLLING_WINDOW).std()
    df["volume_mean_20d"] = df["volume"].shift(1).rolling(ROLLING_WINDOW).mean()

    df["price_zscore"] = (df["daily_return"] - df["return_mean_20d"]) / df["return_std_20d"]
    df["volume_ratio"] = df["volume"] / df["volume_mean_20d"]

    return df


def classify_row(row) -> dict | None:
    """
    Applies the decision grid to one row (which must already have
    price_zscore and volume_ratio computed). Returns None if no anomaly,
    or a dict with anomaly_type, anomaly_category, severity_score.
    """
    price_z = row["price_zscore"]
    volume_ratio = row["volume_ratio"]

    # Not enough history yet to have a valid baseline (first 20 days)
    if pd.isna(price_z) or pd.isna(volume_ratio):
        return None

    price_significant = abs(price_z) >= PRICE_ZSCORE_THRESHOLD
    volume_significant = volume_ratio >= VOLUME_RATIO_THRESHOLD

    if not price_significant:
        return None  # normal day, or volume-only spike (explicitly out of scope for v1)

    # Blend both signals into one 0-1 severity score.
    # Normalize each roughly onto a 0-1 scale, then average them.
    price_component = min(abs(price_z) / 5.0, 1.0)       # cap at z=5 -> 1.0
    volume_component = min(volume_ratio / 10.0, 1.0)      # cap at 10x avg volume -> 1.0
    severity_score = round((price_component + volume_component) / 2, 2)

    if volume_significant:
        return {
            "anomaly_type": "volume_price_divergence_confirmed",
            "anomaly_category": "market_event",
            "severity_score": severity_score,
        }
    else:
        return {
            "anomaly_type": "volume_price_divergence_unconfirmed",
            "anomaly_category": "data_quality_issue",
            "severity_score": severity_score,
        }


def detect_anomalies_for_stock(price_df: pd.DataFrame) -> list[dict]:
    """
    Full pipeline for one stock: compute baselines, classify every row,
    return a list of anomaly dicts (each including date + stock_id) for
    rows that triggered an anomaly.
    """
    df = compute_baselines(price_df)
    anomalies = []

    for _, row in df.iterrows():
        result = classify_row(row)
        if result is not None:
            anomalies.append({
                "stock_id": row["stock_id"],
                "date": row["date"],
                **result,
            })

    return anomalies