"""
verify_volatility_consistency.py

Standalone, no-DB-required script that reproduces the single most
important sanity check behind the Phase 9 event study's volatility
finding: that two INDEPENDENTLY computed measures of dispersion agree
with each other mathematically.

Why this check exists:
The event study has two different tests for "does volatility change after
an anomaly flag":
  1. Levene's test on the SPREAD of endpoint forward-returns (fwd_return_Nd)
     - gives a variance for the anomalous group and a variance for the
       normal group.
  2. A direct REALIZED VOLATILITY measure - the standard deviation of
     daily returns over the forward window - compared between groups via
     a separate Welch's t-test.

These two measures are computed from different underlying data (endpoint
returns vs. the full path of daily returns) using different statistical
tests. If dispersion is actually elevated after a flag, both measures
should point the same direction and - because variance is just
(standard deviation)^2 - the RATIO of realized volatilities should be
approximately the SQUARE ROOT of the ratio of variances. If this
consistency check were to fail (the two measures disagreeing, or one
showing an effect the other doesn't), that would be a strong signal
something upstream was wrong - a bug, or two tests accidentally measuring
different things. It held up closely at every horizon, which is part of
why the volatility finding is reported as credible rather than
coincidental.

Run: python src/verify_volatility_consistency.py
No DB connection or dependencies beyond the Python standard library -
intentionally reproducible on the spot (e.g. live in an interview) from
numbers already produced by a real event_study.py run. The numbers below
are hardcoded from the actual event_study_results.csv output on the real
dataset (97,603 baseline-eligible stock-days, 78 NSE stocks, ~5 years) -
see README.md Phase 9 for the full run.
"""

import math

# --- Real measured values, market_event vs normal, from event_study.py ---
# (var_anomalous / var_normal from the levene_returns rows;
#  mean_vol_anomalous / mean_vol_normal from the volatility rows;
#  vol_ratio as printed directly by the script)
RESULTS = [
    {
        "horizon": "3-day",
        "var_anomalous": 0.001400192363490992,
        "var_normal":    0.0009073034271597864,
        "measured_vol_ratio": 1.2524691204180793,
    },
    {
        "horizon": "5-day",
        "var_anomalous": 0.002209894830031807,
        "var_normal":    0.0015194799780322316,
        "measured_vol_ratio": 1.1780042554376524,
    },
    {
        "horizon": "10-day",
        "var_anomalous": 0.003701101444160547,
        "var_normal":    0.003000196039646895,
        "measured_vol_ratio": 1.1064652547244611,
    },
]


def main():
    print("Consistency check: sqrt(variance ratio from Levene's test on")
    print("endpoint returns) should approximately equal the independently")
    print("measured realized-volatility ratio, if the volatility effect is")
    print("real rather than an artifact of one specific test.\n")

    header = f"{'Horizon':<8} {'Var ratio':>10} {'sqrt(var ratio)':>16} {'Measured vol ratio':>20} {'Diff':>8}"
    print(header)
    print("-" * len(header))

    for r in RESULTS:
        variance_ratio = r["var_anomalous"] / r["var_normal"]
        predicted_vol_ratio = math.sqrt(variance_ratio)
        measured = r["measured_vol_ratio"]
        pct_diff = abs(predicted_vol_ratio - measured) / measured * 100

        print(f"{r['horizon']:<8} {variance_ratio:>10.3f} {predicted_vol_ratio:>16.3f} "
              f"{measured:>20.3f} {pct_diff:>7.1f}%")

    print("\nInterpretation: the predicted ratio (derived purely from the Levene")
    print("variance test on ENDPOINT returns) and the measured ratio (from the")
    print("independently computed REALIZED VOLATILITY test on the full daily-return")
    print("path) agree within a few percent at every horizon, despite coming from")
    print("different data transformations and different statistical tests. That's")
    print("the basis for treating the volatility finding as a real, internally")
    print("consistent effect rather than an artifact of one specific test choice.")


if __name__ == "__main__":
    main()