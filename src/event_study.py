"""
event_study.py
Statistical event study: do stocks flagged as anomalous (volume-price
divergence) show meaningfully different forward returns / volatility than
normal days?

Run: python src/event_study.py
Requires: pip install scipy   (t-test / CI / Levene's; not otherwise used by the pipeline)
Requires the `event_study_results` table to exist - run create_tables.py
(or src/create_tables.py) first if you haven't already, since this script
now writes its results back to Postgres, not just to a local CSV.

--------------------------------------------------------------------------
DESIGN DECISIONS (confirmed with the user before implementation)
--------------------------------------------------------------------------
1. Forward horizons tested: 1, 3, 5, 10 trading days.
2. "Anomalous" (primary) = anomaly_category == 'market_event'.
   'data_quality_issue' analyzed separately/exploratorily.
3. Returns use `adj_close`, checked for population before trusting it.
4. "Normal" = same 78 stocks, past 20-day warm-up, not flagged with any
   anomaly.

ROUND 2: added Levene's test + direct realized-volatility test, since a
null MEAN-return result doesn't rule out a VOLATILITY effect.

ROUND 3: exploratory + segmentation results now also captured into output,
not just the primary market_event tests - surfaced a real finding (see
README methods section): data_quality_issue shows a suspiciously clean
mean-return effect, likely a split/adjustment-lag artifact rather than a
real market effect (flagged, not yet confirmed against raw data) - while
the market_event volatility effect shows a genuine dose-response pattern
across high/low volume-spike segments, which is the more trustworthy
finding of the two.

ROUND 4 (this version): PIPELINE INTEGRATION.
  - Results are now written to a new `event_study_results` Postgres table
    (see create_tables.py), append-only with a run_timestamp, rather than
    only ever living in a local CSV that gets overwritten each run and
    never leaves your laptop. This means results are queryable, persist
    across runs, and let you track later whether findings (e.g. the
    volatility effect) hold up as more data accumulates - something a
    single CSV snapshot can't do.
  - The CSV is still written too (kept as a fast local artifact / input to
    the GitHub Actions job summary), but the DB table is now the source of
    truth.
  - This script is wired into a SEPARATE, non-daily GitHub Actions workflow
    (.github/workflows/event_study.yml) - deliberately NOT added to the
    existing daily_pipeline.yml. Reasoning: the daily pipeline's job is
    same-day ingestion + same-day anomaly flagging, which needs to run
    every trading day. This event study is a slow-moving statistical
    re-validation of the detector itself - re-running it daily would (a)
    waste compute re-testing on ~1 new day of data each time, (b) spam the
    results table with near-duplicate rows, and (c) blur the distinction
    between "operational pipeline" and "periodic model validation" in the
    Actions history. A weekly cadence is enough to notice drift without
    the noise.
--------------------------------------------------------------------------
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from scipy import stats

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

ROLLING_WINDOW = 20
HORIZONS = [1, 3, 5, 10]
MIN_GROUP_SIZE = 30
MIN_VOL_HORIZON = 3

# Columns written to event_study_results, in DB column order. Kept as an
# explicit list (rather than relying on dict key order from whatever test
# function produced the row) so the INSERT is robust to which test type
# produced a given result dict - missing keys are written as NULL.
RESULT_COLUMNS = [
    "test", "label", "anomaly_group", "segment", "skipped",
    "n_anomalous", "n_normal", "mean_anomalous", "mean_normal",
    "t_stat", "p_value", "ci_low", "ci_high", "mannwhitney_p",
    "var_anomalous", "var_normal", "levene_stat", "levene_p",
    "mean_vol_anomalous", "mean_vol_normal", "vol_ratio",
]


def check_adj_close_populated(conn) -> bool:
    result = conn.execute(text("""
        SELECT
            COUNT(*) AS total,
            COUNT(*) FILTER (WHERE adj_close IS NULL) AS null_count,
            COUNT(*) FILTER (WHERE adj_close = close) AS equal_count
        FROM stock_prices
    """)).fetchone()
    total, null_count, equal_count = result
    if total == 0:
        print("stock_prices is empty - nothing to analyze.")
        return False
    null_pct = null_count / total * 100
    equal_pct = equal_count / total * 100
    print(f"adj_close check: {total} rows, {null_pct:.1f}% NULL, "
          f"{equal_pct:.1f}% exactly equal to close.")
    if null_pct > 5:
        print("WARNING: adj_close has meaningful NULLs. Refusing to proceed "
              "silently - inspect data before rerunning.")
        return False
    return True


def check_results_table_exists(conn) -> bool:
    """
    Fail with a clear message rather than a raw SQL error if create_tables
    hasn't been run yet with the event_study_results table added.
    """
    exists = conn.execute(text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = 'event_study_results'
        )
    """)).scalar()
    if not exists:
        print("ERROR: 'event_study_results' table does not exist. "
              "Run `python create_tables.py` (updated DDL) before running "
              "this script.")
    return exists


def load_prices_and_anomalies(conn) -> pd.DataFrame:
    query = text("""
        SELECT
            sp.stock_id, s.ticker, sp.date, sp.close, sp.adj_close, sp.volume,
            a.anomaly_category, a.severity_score
        FROM stock_prices sp
        JOIN stocks s ON s.stock_id = sp.stock_id
        LEFT JOIN anomalies a
            ON a.stock_id = sp.stock_id AND a.date = sp.date
        ORDER BY sp.stock_id, sp.date
    """)
    rows = conn.execute(query).fetchall()
    df = pd.DataFrame(rows, columns=[
        "stock_id", "ticker", "date", "close", "adj_close",
        "volume", "anomaly_category", "severity_score"
    ])
    df["close"] = df["close"].astype(float)
    df["adj_close"] = df["adj_close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_baseline_eligibility(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("date").reset_index(drop=True).copy()
    g["daily_return"] = g["adj_close"].pct_change()
    ret_mean = g["daily_return"].shift(1).rolling(ROLLING_WINDOW).mean()
    ret_std = g["daily_return"].shift(1).rolling(ROLLING_WINDOW).std()
    vol_mean = g["volume"].shift(1).rolling(ROLLING_WINDOW).mean()
    g["price_zscore"] = (g["daily_return"] - ret_mean) / ret_std
    g["volume_ratio"] = g["volume"] / vol_mean
    g["baseline_eligible"] = ret_mean.notna() & vol_mean.notna()
    return g


def add_forward_returns_and_volatility(group: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    g = group.sort_values("date").reset_index(drop=True).copy()
    price = g["adj_close"].values
    daily_ret = g["daily_return"].values
    n = len(g)

    for h in horizons:
        fwd_return = np.full(n, np.nan)
        fwd_vol = np.full(n, np.nan)

        for i in range(n - h):
            start_price = price[i]
            end_price = price[i + h]
            if start_price and not np.isnan(start_price) and not np.isnan(end_price):
                fwd_return[i] = (end_price - start_price) / start_price

            if h >= MIN_VOL_HORIZON:
                window = daily_ret[i + 1: i + h + 1]
                if not np.isnan(window).any() and len(window) == h:
                    fwd_vol[i] = np.std(window, ddof=1)

        g[f"fwd_return_{h}d"] = fwd_return
        if h >= MIN_VOL_HORIZON:
            g[f"fwd_vol_{h}d"] = fwd_vol

    return g


def build_dataset() -> pd.DataFrame:
    with engine.begin() as conn:
        if not check_results_table_exists(conn):
            sys.exit(1)
        if not check_adj_close_populated(conn):
            sys.exit(1)
        raw = load_prices_and_anomalies(conn)

    if raw.empty:
        print("No data found in stock_prices.")
        sys.exit(1)

    processed = []
    for stock_id, group in raw.groupby("stock_id"):
        g = add_baseline_eligibility(group)
        g = add_forward_returns_and_volatility(g, HORIZONS)
        processed.append(g)
    df = pd.concat(processed, ignore_index=True)
    df = df[df["baseline_eligible"]].copy()

    df["group"] = np.where(
        df["anomaly_category"] == "market_event", "market_event",
        np.where(df["anomaly_category"] == "data_quality_issue",
                 "data_quality_issue", "normal")
    )
    return df


def run_ttest_and_ci(anomalous, normal, label, group="market_event", segment="primary"):
    a = anomalous.dropna()
    n_ = normal.dropna()

    if len(a) < MIN_GROUP_SIZE or len(n_) < MIN_GROUP_SIZE:
        print(f"  [{label}] SKIPPED - insufficient n "
              f"(anomalous={len(a)}, normal={len(n_)}, need >= {MIN_GROUP_SIZE}).")
        return {"test": "mean_return", "label": label, "anomaly_group": group, "segment": segment,
                "n_anomalous": len(a), "n_normal": len(n_), "skipped": True}

    t_stat, p_value = stats.ttest_ind(a, n_, equal_var=False)
    mean_a = a.mean()
    sem_a = a.sem()
    dof = len(a) - 1
    ci_low, ci_high = stats.t.interval(0.95, dof, loc=mean_a, scale=sem_a)
    u_stat, u_p = stats.mannwhitneyu(a, n_, alternative="two-sided")

    print(f"\n  --- {label} [{group}/{segment}] ---")
    print(f"  n(anomalous)={len(a)}, n(normal)={len(n_)}")
    print(f"  mean return | anomalous = {mean_a:.4%}   normal = {n_.mean():.4%}")
    print(f"  Welch t-stat = {t_stat:.3f}, p-value = {p_value:.4f}")
    print(f"  95% CI: [{ci_low:.4%}, {ci_high:.4%}]   Mann-Whitney p = {u_p:.4f}")

    return {
        "test": "mean_return", "label": label, "anomaly_group": group, "segment": segment,
        "skipped": False, "n_anomalous": len(a), "n_normal": len(n_),
        "mean_anomalous": mean_a, "mean_normal": n_.mean(),
        "t_stat": t_stat, "p_value": p_value,
        "ci_low": ci_low, "ci_high": ci_high, "mannwhitney_p": u_p,
    }


def run_levene_on_returns(anomalous, normal, label, group="market_event", segment="primary"):
    a = anomalous.dropna()
    n_ = normal.dropna()
    if len(a) < MIN_GROUP_SIZE or len(n_) < MIN_GROUP_SIZE:
        print(f"  [{label} | Levene] SKIPPED - insufficient n.")
        return {"test": "levene_returns", "label": label, "anomaly_group": group, "segment": segment,
                "n_anomalous": len(a), "n_normal": len(n_), "skipped": True}

    stat, p = stats.levene(a, n_)
    print(f"  [{label} | Levene spread, {group}/{segment}] stat={stat:.3f}, p={p:.4f} "
          f"(var: anomalous={a.var():.6f}, normal={n_.var():.6f})")
    return {"test": "levene_returns", "label": label, "anomaly_group": group, "segment": segment,
            "skipped": False, "n_anomalous": len(a), "n_normal": len(n_),
            "var_anomalous": a.var(), "var_normal": n_.var(), "levene_stat": stat, "levene_p": p}


def run_ttest_on_volatility(anomalous_vol, normal_vol, label, group="market_event", segment="primary"):
    a = anomalous_vol.dropna()
    n_ = normal_vol.dropna()
    if len(a) < MIN_GROUP_SIZE or len(n_) < MIN_GROUP_SIZE:
        print(f"  [{label} | realized vol] SKIPPED - insufficient n "
              f"(anomalous={len(a)}, normal={len(n_)}).")
        return {"test": "volatility", "label": label, "anomaly_group": group, "segment": segment,
                "n_anomalous": len(a), "n_normal": len(n_), "skipped": True}

    t_stat, p_value = stats.ttest_ind(a, n_, equal_var=False)
    u_stat, u_p = stats.mannwhitneyu(a, n_, alternative="two-sided")
    mean_a, mean_n = a.mean(), n_.mean()
    ratio = mean_a / mean_n if mean_n else np.nan

    print(f"\n  --- {label} REALIZED VOLATILITY [{group}/{segment}] ---")
    print(f"  mean stdev | anomalous = {mean_a:.4%}   normal = {mean_n:.4%}   (ratio={ratio:.2f}x)")
    print(f"  Welch t-stat = {t_stat:.3f}, p-value = {p_value:.4f}   Mann-Whitney p = {u_p:.4f}")

    return {"test": "volatility", "label": label, "anomaly_group": group, "segment": segment,
            "skipped": False, "n_anomalous": len(a), "n_normal": len(n_),
            "mean_vol_anomalous": mean_a, "mean_vol_normal": mean_n, "vol_ratio": ratio,
            "t_stat": t_stat, "p_value": p_value, "mannwhitney_p": u_p}


def write_results_to_db(results: list[dict]):
    """
    Append-only insert, one run_timestamp shared across the whole run (set
    once here, not per-row via DB default) so every row from this run can
    be queried together as a single "batch" later, e.g.:
        SELECT * FROM event_study_results
        WHERE run_timestamp = (SELECT MAX(run_timestamp) FROM event_study_results)
    """
    run_ts = datetime.now(timezone.utc)
    rows = []
    for r in results:
        row = {col: r.get(col) for col in RESULT_COLUMNS}
        row["run_timestamp"] = run_ts
        rows.append(row)

    df = pd.DataFrame(rows)
    # NaN -> None so Postgres gets real NULLs, not the string "NaN"
    df = df.where(pd.notnull(df), None)

    insert_cols = ["run_timestamp"] + RESULT_COLUMNS
    placeholders = ", ".join(f":{c}" for c in insert_cols)
    stmt = text(f"""
        INSERT INTO event_study_results ({", ".join(insert_cols)})
        VALUES ({placeholders})
    """)

    with engine.begin() as conn:
        conn.execute(stmt, df[insert_cols].to_dict(orient="records"))

    print(f"\nWrote {len(df)} rows to event_study_results (run_timestamp={run_ts.isoformat()}).")


def main():
    df = build_dataset()

    print(f"\nDataset built: {len(df)} baseline-eligible stock-days across "
          f"{df['stock_id'].nunique()} stocks.")
    print(df["group"].value_counts().to_string())

    results = []

    print("\n===== PRIMARY: mean forward return, market_event vs normal =====")
    for h in HORIZONS:
        col = f"fwd_return_{h}d"
        results.append(run_ttest_and_ci(
            df.loc[df["group"] == "market_event", col],
            df.loc[df["group"] == "normal", col],
            f"{h}-day forward return", group="market_event", segment="primary"))

    print("\n===== Levene's test on return SPREAD, market_event vs normal =====")
    for h in HORIZONS:
        col = f"fwd_return_{h}d"
        results.append(run_levene_on_returns(
            df.loc[df["group"] == "market_event", col],
            df.loc[df["group"] == "normal", col],
            f"{h}-day forward return", group="market_event", segment="primary"))

    print("\n===== REALIZED VOLATILITY, market_event vs normal =====")
    for h in HORIZONS:
        if h < MIN_VOL_HORIZON:
            continue
        col = f"fwd_vol_{h}d"
        results.append(run_ttest_on_volatility(
            df.loc[df["group"] == "market_event", col],
            df.loc[df["group"] == "normal", col],
            f"{h}-day", group="market_event", segment="primary"))

    print("\n===== EXPLORATORY: data_quality_issue vs normal (mean return) =====")
    for h in HORIZONS:
        col = f"fwd_return_{h}d"
        results.append(run_ttest_and_ci(
            df.loc[df["group"] == "data_quality_issue", col],
            df.loc[df["group"] == "normal", col],
            f"{h}-day forward return", group="data_quality_issue", segment="exploratory"))

    print("\n===== EXPLORATORY: data_quality_issue vs normal (realized volatility) =====")
    for h in HORIZONS:
        if h < MIN_VOL_HORIZON:
            continue
        col = f"fwd_vol_{h}d"
        results.append(run_ttest_on_volatility(
            df.loc[df["group"] == "data_quality_issue", col],
            df.loc[df["group"] == "normal", col],
            f"{h}-day", group="data_quality_issue", segment="exploratory"))

    print("\n===== SEGMENTATION: high vs low volume-spike, within market_event =====")
    me = df[df["group"] == "market_event"].copy()
    if len(me) >= MIN_GROUP_SIZE * 2:
        median_ratio = me["volume_ratio"].median()
        high_spike = me[me["volume_ratio"] >= median_ratio]
        low_spike = me[me["volume_ratio"] < median_ratio]
        normal_all = df.loc[df["group"] == "normal"]
        print(f"Median volume_ratio split = {median_ratio:.2f}x "
              f"(high n={len(high_spike)}, low n={len(low_spike)})")
        for h in HORIZONS:
            col = f"fwd_return_{h}d"
            results.append(run_ttest_and_ci(high_spike[col], normal_all[col], f"{h}-day forward return",
                                             group="market_event", segment="high_volume_spike"))
            results.append(run_ttest_and_ci(low_spike[col], normal_all[col], f"{h}-day forward return",
                                             group="market_event", segment="low_volume_spike"))
        for h in HORIZONS:
            if h < MIN_VOL_HORIZON:
                continue
            col = f"fwd_vol_{h}d"
            results.append(run_ttest_on_volatility(high_spike[col], normal_all[col], f"{h}-day",
                                                     group="market_event", segment="high_volume_spike"))
            results.append(run_ttest_on_volatility(low_spike[col], normal_all[col], f"{h}-day",
                                                     group="market_event", segment="low_volume_spike"))
    else:
        print(f"SKIPPED - only {len(me)} market_event rows, need >= {MIN_GROUP_SIZE * 2}.")

    results = [r for r in results if r is not None]

    # Local CSV: fast to inspect without a DB round-trip, and useful as a
    # GitHub Actions artifact / job summary input. The DB table is the
    # durable, queryable source of truth across runs - see module docstring.
    out = pd.DataFrame(results)
    out.to_csv("event_study_results.csv", index=False)
    print(f"\nWrote {len(out)} rows to event_study_results.csv (local).")

    write_results_to_db(results)


if __name__ == "__main__":
    main()