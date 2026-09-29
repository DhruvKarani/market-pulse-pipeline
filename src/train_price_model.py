"""
train_price_model.py
Phase 8b: trains and evaluates next-day return prediction models.

Design recap (see project notes for full reasoning):
- Target: next_day_return (regression) - NOT raw price, to avoid the
  "predict today = tomorrow" naive-baseline trap that raw price prediction
  is vulnerable to.
- Chronological train/test split - NOT random - to avoid lookahead bias
  (training on data from after the test period, which no real deployment
  would ever have access to).
- Three comparisons: naive baseline (predict 0), Linear Regression on
  Model A's feature set (price/volume/macro, full 5yr), Linear Regression
  on Model B's feature set (+ sentiment, ~30-day overlap window only).
- RMSE and MAE both reported; RMSE used as the deciding metric.

Run: python train_price_model.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from build_features import build_full_feature_set

TEST_SET_FRACTION = 0.2  # most recent 20% of dates held out for testing


def chronological_split(df: pd.DataFrame, feature_cols: list[str], target_col: str = "next_day_return"):
    """
    Splits by DATE, not randomly - the earliest (1 - TEST_SET_FRACTION)
    portion of dates becomes training data, the most recent portion becomes
    the test set. This means the model is only ever evaluated on dates it
    could not possibly have seen during training - the real-world condition
    any deployed model would face.
    """
    df = df.sort_values("date")
    unique_dates = df["date"].unique()
    split_idx = int(len(unique_dates) * (1 - TEST_SET_FRACTION))
    split_date = unique_dates[split_idx]

    train_df = df[df["date"] < split_date]
    test_df = df[df["date"] >= split_date]

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    return X_train, X_test, y_train, y_test, split_date


def evaluate(y_true, y_pred, label: str):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"  {label:35s} RMSE={rmse:.5f}  MAE={mae:.5f}")
    return rmse, mae


def run_model_a():
    print("=" * 60)
    print("MODEL A: price/volume features, full 5-year history")
    print("=" * 60)
    print("(Note: macro indicators excluded for now - macro_indicators only")
    print(" has a handful of recent dates so far, since daily macro ingestion")
    print(" just started; left-joining it against 5yr of price history and")
    print(" dropping NaN rows collapsed ~95k rows down to ~150. Revisit once")
    print(" enough daily macro history has accumulated, or macro is backfilled.)\n")

    df = build_full_feature_set(include_sentiment=False)
    feature_cols = ["daily_return", "volume_ratio", "price_zscore"]

    df = df.dropna(subset=feature_cols)
    print(f"Training on features: {feature_cols}")
    print(f"Total usable rows: {len(df)}\n")

    X_train, X_test, y_train, y_test, split_date = chronological_split(df, feature_cols)
    print(f"Train: {len(X_train)} rows (before {split_date})")
    print(f"Test:  {len(X_test)} rows (from {split_date} onward)\n")

    print("Results:")
    naive_preds = np.zeros(len(y_test))  # naive baseline: always predict 0
    evaluate(y_test, naive_preds, "Naive baseline (predict 0)")

    model = LinearRegression()
    model.fit(X_train, y_train)
    lr_preds = model.predict(X_test)
    evaluate(y_test, lr_preds, "Linear Regression (Model A)")

    print(f"\n  Feature coefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"    {name:15s} {coef:+.6f}")

    return model


def run_model_b():
    print("\n" + "=" * 60)
    print("MODEL B: price/volume/macro + sentiment, ~30-day overlap window")
    print("=" * 60)

    df = build_full_feature_set(include_sentiment=True)
    feature_cols = ["daily_return", "volume_ratio", "price_zscore", "avg_sentiment"]
    # Macro excluded here too - see Model A's note on macro_indicators coverage

    df = df.dropna(subset=feature_cols)
    print(f"Training on features: {feature_cols}")
    print(f"Total usable rows: {len(df)}\n")

    if len(df) < 30:
        print("  WARNING: sample size is very small - results below are exploratory "
              "only, not statistically reliable. This is expected and documented; "
              "NewsAPI's free tier only provides ~30 days of historical headlines.\n")

    if len(df) < 10:
        print("  Not enough data yet to even attempt a train/test split. "
              "Run this again after more days of accumulated headline data.")
        return None

    X_train, X_test, y_train, y_test, split_date = chronological_split(df, feature_cols)
    print(f"Train: {len(X_train)} rows (before {split_date})")
    print(f"Test:  {len(X_test)} rows (from {split_date} onward)\n")

    print("Results:")
    naive_preds = np.zeros(len(y_test))
    evaluate(y_test, naive_preds, "Naive baseline (predict 0)")

    model = LinearRegression()
    model.fit(X_train, y_train)
    lr_preds = model.predict(X_test)
    evaluate(y_test, lr_preds, "Linear Regression (Model B)")

    print(f"\n  Feature coefficients:")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"    {name:15s} {coef:+.6f}")

    return model


if __name__ == "__main__":
    run_model_a()
    run_model_b()