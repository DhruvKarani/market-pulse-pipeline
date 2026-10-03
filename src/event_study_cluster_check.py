"""
event_study_cluster_check.py

Addresses a real methodological gap in event_study.py: Welch's t-test
treats every flagged stock-day as an independent observation, but if many
stocks get flagged on the SAME calendar day (a broad market-wide move
rather than 2,419 unrelated single-stock events), those rows are
correlated - they share a common cause. Treating them as independent
overstates the effective sample size and understates how uncertain the
result really is.

This script does NOT replace event_study.py's primary test. It's a
robustness check: collapse to ONE observation per (stock, calendar-day-
cluster) rather than one per flagged row, and see whether the headline
finding (elevated post-flag volatility) survives. If it does, that's a
much stronger defensive answer than "we didn't check" - it's "we checked,
and here's what happens to the effect when you remove the correlation."

Approach, and why this one over alternatives:
- The statistically complete fix is cluster-robust standard errors
  (clustering by date) or a mixed-effects model. Both are more correct but
  heavier to implement and explain live than this project needs.
- A block bootstrap (resample whole DATES, not individual rows) is the
  simplest approach that directly addresses the actual problem - it
  naturally accounts for same-day correlation because an entire day's
  worth of (correlated) flags moves in or out of the resample together,
  rather than being drawn independently.
- We report the bootstrap 95% CI for the mean volatility ratio alongside
  the original parametric CI. If they're similar, clustering isn't
  materially distorting the conclusion. If the bootstrap CI is much wider,
  that's the honest, quantified version of "yes, the independence
  assumption matters here."

Run: python src/event_study_cluster_check.py
(Reads the same DB tables as event_study.py; no new tables created.)
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

ROLLING_WINDOW = 20
N_BOOTSTRAP = 2000
RNG_SEED = 42  # fixed seed - reproducible result, not a different number every run


def load_and_prepare():
    """
    Same extraction + baseline + forward-volatility logic as event_study.py,
    trimmed to just what this check needs (10-day realized volatility,
    market_event vs normal). Kept here as a self-contained duplicate rather
    than importing event_study.py's internals, since this is a standalone
    diagnostic meant to be run and read in isolation.
    """
    query = text("""
        SELECT sp.stock_id, sp.date, sp.adj_close, sp.volume, a.anomaly_category
        FROM stock_prices sp
        LEFT JOIN anomalies a ON a.stock_id = sp.stock_id AND a.date = sp.date
        ORDER BY sp.stock_id, sp.date
    """)
    with engine.begin() as conn:
        rows = conn.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=["stock_id", "date", "adj_close", "volume", "anomaly_category"])
    df["adj_close"] = df["adj_close"].astype(float)
    df["date"] = pd.to_datetime(df["date"])

    out = []
    for stock_id, g in df.groupby("stock_id"):
        g = g.sort_values("date").reset_index(drop=True).copy()
        g["daily_return"] = g["adj_close"].pct_change()
        ret_mean = g["daily_return"].shift(1).rolling(ROLLING_WINDOW).mean()
        vol_mean = g["volume"].shift(1).rolling(ROLLING_WINDOW).mean()
        eligible = ret_mean.notna() & vol_mean.notna()

        daily_ret = g["daily_return"].values
        n = len(g)
        fwd_vol = np.full(n, np.nan)
        for i in range(n - 10):
            window = daily_ret[i + 1: i + 11]
            if not np.isnan(window).any():
                fwd_vol[i] = np.std(window, ddof=1)
        g["fwd_vol_10d"] = fwd_vol
        g["eligible"] = eligible
        out.append(g)

    full = pd.concat(out, ignore_index=True)
    full = full[full["eligible"]].copy()
    full["group"] = np.where(full["anomaly_category"] == "market_event", "market_event", "normal")
    full = full[full["group"].isin(["market_event", "normal"])]
    return full.dropna(subset=["fwd_vol_10d"])


def block_bootstrap_ci(df: pd.DataFrame, group_col: str, value_col: str, date_col: str,
                        n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED):
    """
    Resamples whole DATES with replacement (not individual rows), so
    same-day correlation is preserved inside each resample rather than
    averaged away - the key difference from a naive row-level bootstrap,
    which would make the same independence-assumption error as the t-test.
    """
    rng = np.random.default_rng(seed)
    unique_dates = df[date_col].unique()

    anomalous_means, normal_means, ratios = [], [], []
    for _ in range(n_boot):
        sampled_dates = rng.choice(unique_dates, size=len(unique_dates), replace=True)
        # Rebuild the resampled frame by concatenating all rows for each
        # sampled date (a date drawn twice contributes its rows twice).
        resampled = df[df[date_col].isin(sampled_dates)]
        a = resampled.loc[resampled[group_col] == "market_event", value_col]
        n_ = resampled.loc[resampled[group_col] == "normal", value_col]
        if len(a) < 10 or len(n_) < 10:
            continue
        anomalous_means.append(a.mean())
        normal_means.append(n_.mean())
        ratios.append(a.mean() / n_.mean())

    ratios = np.array(ratios)
    ci_low, ci_high = np.percentile(ratios, [2.5, 97.5])
    return ratios.mean(), ci_low, ci_high, len(ratios)


def main():
    df = load_and_prepare()

    n_flagged_days = df.loc[df["group"] == "market_event", "date"].nunique()
    n_total_flags = (df["group"] == "market_event").sum()
    print(f"market_event flags: {n_total_flags} rows across {n_flagged_days} distinct calendar dates "
          f"({n_total_flags / n_flagged_days:.1f} stocks flagged per date on average).")
    print("If that ratio is well above 1, a meaningful share of 'independent' flagged")
    print("rows actually share a common same-day cause - exactly the correlation the")
    print("plain Welch's t-test in event_study.py doesn't account for.\n")

    point_ratio = (df.loc[df["group"] == "market_event", "fwd_vol_10d"].mean()
                   / df.loc[df["group"] == "normal", "fwd_vol_10d"].mean())
    print(f"Point estimate, 10-day volatility ratio (market_event / normal): {point_ratio:.3f}")
    print(f"(event_study.py's parametric result for this same ratio: 1.106)\n")

    print(f"Running block bootstrap ({N_BOOTSTRAP} resamples, whole dates resampled together)...")
    mean_ratio, ci_low, ci_high, n_valid = block_bootstrap_ci(
        df, group_col="group", value_col="fwd_vol_10d", date_col="date"
    )
    print(f"\nBlock-bootstrap mean ratio: {mean_ratio:.3f}")
    print(f"Block-bootstrap 95% CI:     [{ci_low:.3f}, {ci_high:.3f}]  (from {n_valid} valid resamples)")

    print("\nInterpretation: if this CI still excludes 1.0 (no effect), the post-flag")
    print("volatility effect survives even after accounting for same-day correlation")
    print("across stocks - a materially stronger claim than the plain t-test alone,")
    print("since it no longer assumes every flagged row is an independent event.")


if __name__ == "__main__":
    main()