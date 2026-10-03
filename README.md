# Market Pulse Pipeline

A daily data engineering pipeline for Indian stock market and global financial data — ingestion, data quality validation, volume-price divergence anomaly detection, and news sentiment analysis, built on PostgreSQL (Supabase) with automated scheduling via GitHub Actions.

## What this does

- Ingests daily OHLCV price data for 78 NSE-listed stocks (Nifty 50 + Sensex constituents) via yfinance
- Ingests global macro indicators (USD/INR, crude oil, S&P 500, US Fed funds rate) from yfinance and FRED
- Validates every row against data quality rules before it enters the warehouse — bad data is rejected and logged, never silently inserted
- Detects volume-price divergence anomalies: separates genuine market events (price move confirmed by volume) from likely data quality issues (price move with no volume behind it)
- Pulls financial news headlines, scores sentiment (VADER), links headlines to specific stocks, and studies correlation between sentiment and next-day price movement
- Sends a daily email summary (via Resend) covering pipeline health, anomalies detected, and the day's most notable headline sentiment — since GitHub Actions only notifies on job failure, not on what the pipeline actually found
- Runs automatically every trading day via GitHub Actions

## Architecture

```
yfinance ──┐
           ├──► stock_prices ──► anomaly detection ──► anomalies
FRED    ───┘         │
                     └──► (joined with sentiment for correlation study)

NewsAPI ──► news_headlines ──► headline_stocks (many-to-many) ──► stocks
                │
                └──► VADER sentiment scoring

Every ingestion run ──► ingestion_log (summary) + ingestion_failures (per-item detail)

End of each daily run ──► daily_summary_email.py ──► Resend API ──► inbox
```

8 tables: `stocks`, `stock_prices`, `macro_indicators`, `news_headlines`, `headline_stocks`, `ingestion_log`, `ingestion_failures`, `anomalies`.

## Key design decisions

**PostgreSQL over SQLite/flat files** — GitHub Actions runners are ephemeral (a fresh VM every run, nothing persists between runs), so a file-based database would lose all historical data daily. Postgres runs as a separate, persistent server on Supabase; the runner is just a client that connects, writes, and disconnects.

**Surrogate keys + unique natural keys** — `stocks.stock_id` (SERIAL) is the primary key referenced everywhere, while `ticker` stays `UNIQUE` but not primary. This protects against ticker changes needing to propagate through every foreign key, and keeps joins on a small integer rather than a string. At 78 stocks this is a stylistic choice more than a performance necessity — the reasoning matters more than the scale here.

**NUMERIC over FLOAT for all prices** — avoids binary floating-point approximation errors that would compound across calculations. Learned in practice: NUMERIC values come back as Python `Decimal` via psycopg2, which doesn't mix with `float` in pandas arithmetic — had to explicitly cast to `float` at the analysis boundary.

**Long/narrow format for `macro_indicators`** — `(date, indicator_name, value)` instead of one column per indicator, so adding a new indicator never requires a schema change.

**Data quality checks vs anomaly detection are deliberately separate concepts.** Data quality checks validate one row in isolation — no history needed (OHLC consistency, non-negative volume, a zero-volume-with-price-movement contradiction, no missing values). Anomaly detection requires history — a 20-day rolling per-stock baseline for both price return z-score and volume ratio.

**Volume-price divergence logic:**
| Price move | Volume | Classification |
|---|---|---|
| Significant | Significant | `market_event` — real move, confirmed by real trading activity |
| Significant | Normal/low | `data_quality_issue` — a real price move requires real trades; if volume doesn't back it up, the price change itself is suspect |
| Normal | Significant | out of scope for v1 (would need its own anomaly type) |
| Normal | Normal | no anomaly |

Thresholds were tuned, not guessed: an initial volume ratio threshold of 2.0x produced a 54% "data_quality_issue" rate across 78 blue-chip, cleanly-validated stocks — implausible given 0 rows failed hard validation in backfill. Lowered to 1.5x, producing a more plausible 65% market_event / 35% data_quality_issue split.

**Per-ticker transaction commits, not one giant transaction** — discovered the hard way: an early version wrapped the entire 80-ticker backfill in one transaction, and interrupting the script mid-run rolled back everything, including tickers that had already succeeded. Fixed by committing after each ticker individually — a crash now only costs the ticker in progress.

**Broad NewsAPI queries + company-name matching, not one query per stock** — the free tier's 100 requests/day can't sustain 78 individual per-stock queries. Broad market queries are run instead, and headlines are matched to stocks via a cleaned "short name" substring check. Known limitation: this heuristic missed/mismatched cases in testing (e.g. "titanium" falsely matching "Titan" via plain substring search; "Tata" alone falsely matching every Tata-group headline after over-aggressive suffix stripping) — both were found and fixed (word-boundary regex matching; keeping fuller distinctive names instead of truncating to 1-2 words). A production version would use NER or fuzzy matching instead.

**GitHub Actions over Airflow** — GitHub Actions is a CI/CD scheduler (YAML-defined steps, cron trigger), not a true orchestrator. It has no task-dependency graph, per-task retry policies, or monitoring dashboard like Airflow does — sequencing here is just steps running top-to-bottom in one job. Appropriate for this project's scale (4 sequential daily scripts); would migrate to Airflow if this grew into dozens of interdependent, cross-team tasks.

**Daily pipeline scheduled for evening IST, not right after market close** — found real evidence that yfinance can return a row with genuine `NaN` OHLC values (but populated volume) if queried too soon after a trading day ends; the daily price script's data quality validation correctly caught and rejected these. The GitHub Actions schedule runs at 8:00 PM IST (2:30 PM UTC) to give the data time to settle.

**A separate daily email step, not relying on GitHub's built-in notifications** — GitHub Actions only emails on workflow *failure*, which tells you the job ran but nothing about what it actually found. Added a final pipeline step (`daily_summary_email.py`) that queries that day's `ingestion_log`, `anomalies`, and `news_headlines` and sends a formatted summary via Resend's API — separating "did the job run" (GitHub's job) from "what did the job find" (this project's job). Uses Resend's free test sender rather than a verified custom domain, since a personal project doesn't need production-grade email deliverability.

## Phase 8b — Predictive modeling (exploratory)

Extended the correlation study into an actual next-day return prediction model, scoped deliberately to keep the ML separate from the data engineering: Linear Regression predicting `next_day_return` (not raw price — predicting price directly is vulnerable to a naive "tomorrow = today" baseline looking deceptively accurate), trained with a **chronological** train/test split (not random, to avoid lookahead bias — a random split would let the model train on data from after the test period, an advantage no real deployment would have), evaluated against a naive baseline (predict 0% return) using RMSE/MAE.

Two models were built:
- **Model A** — price/volume features only (`daily_return`, `volume_ratio`, `price_zscore`), trained on the full 5-year backfilled dataset (~95,000 rows)
- **Model B** — same features + average daily sentiment, trained on the ~30-day window where sentiment data actually exists (NewsAPI's free tier only provides ~30 days of historical headlines)

**Result: Model A's RMSE (0.01680) was statistically indistinguishable from the naive baseline (0.01679)** — the model did not meaningfully outperform "just predict zero every day." Feature coefficients were near-zero across the board, consistent with the well-known finding that daily stock returns behave close to a random walk and simple technical features carry little standalone next-day predictive power. This is reported as an honest negative result, not hidden or massaged — a model that doesn't beat a naive baseline is itself a meaningful, defensible finding, and chasing a better number by adding features until something "wins" would be overfitting by search rather than real signal.

Model B (9 training rows) was too small to draw any real conclusion from and is documented as exploratory only.

**Known limitation surfaced during this phase:** an early version of Model A accidentally left-joined macro indicators (`macro_indicators`) against 5 years of price data and dropped rows with missing values — but since daily macro ingestion only recently started, this silently collapsed the training set from ~95,000 rows to ~150 before being caught via a row-count sanity check. Macro features were removed from both models for now; backfilling 5 years of macro history (yfinance + FRED both support this) is a documented future enhancement, not built in this phase.

## Phase 9 — Event study: does the anomaly detector predict anything? (statistical validation)

The anomaly detector (Phase 4) flags days where price and volume both move sharply together — but flagging an unusual day and that day having predictive value for what happens *next* are two different claims. This phase tests the second one directly against ~5 years of real data (97,603 baseline-eligible stock-days across 78 stocks; 2,419 `market_event` flags, 1,313 `data_quality_issue` flags), using two-sample statistical tests rather than eyeballing a chart.

**Design, and the bias it guards against:** forward returns for a flagged/normal day `t` are computed strictly from `t+1` onward, using `adj_close[t]` as the base price — never `t` itself, since the return that *produced* the flag already happened by that day's close. Measuring from `t` instead of `t+1` would partly re-test the same return that defined the anomaly, guaranteeing a circular, meaningless "finding." `market_event` (price move confirmed by volume) is tested as the primary hypothesis; `data_quality_issue` (price move *not* confirmed by volume — flagged by the detector's own logic as more likely a bad print) is kept as a separate, exploratory comparison rather than pooled in, since pooling would let noisy data quietly inflate the headline result.

**Result 1 — no significant directional return effect.** Welch's t-test comparing mean forward return after `market_event` days vs. normal days across 1/3/5/10-day horizons: p = 0.365, 0.782, 0.424, 0.017. Three of four horizons are nowhere near significance; the one nominal hit (10-day) doesn't survive a Bonferroni correction for testing four horizons (α/4 = 0.0125), and the underlying effect (0.90% vs 0.60% mean return) is smaller than typical transaction costs on these names. **Honest conclusion: the detector does not predict which direction a stock moves next**, and this is reported as a real finding, not a failed one — a rule-based anomaly detector isn't obligated to also be a return-prediction model, and testing that claim directly (rather than assuming it) is the point of doing statistics instead of taking the detector's usefulness on faith.

**Result 2 — a real, cross-validated volatility effect.** The null result above only tests *direction*; it says nothing about whether price action gets *choppier* after a flag. Realized volatility (stdev of daily returns over the forward window, same lookahead-bias guard) is 25% higher than normal 3 days after a `market_event` flag, decaying to 18% and 11% higher by day 5 and day 10 (Welch's t-test p < 10⁻¹⁴ at every horizon). This survives four independent checks: (a) Levene's test on the endpoint-return spread agrees directionally at every horizon, (b) the two measures are internally consistent — `√(variance ratio) ≈ volatility ratio` holds within ~2.4% at every horizon (reproducible on the spot via `src/verify_volatility_consistency.py`, no DB needed), which two unrelated calculations wouldn't do by coincidence, (c) segmenting `market_event` by volume-spike strength shows a clean dose-response relationship (3-day volatility ratio: 1.31x for high-volume-confirmed flags vs. 1.19x for low — stronger signal, stronger effect, the shape a real relationship should have, not noise), and (d) a block bootstrap (`src/event_study_cluster_check.py`) resampling whole calendar dates together — not individual rows — to account for same-day correlation across stocks (2,419 `market_event` flags span only 902 distinct dates, ~2.7 stocks/date) gives a 95% CI of [1.070, 1.137] for the 10-day ratio, matching the plain t-test's point estimate exactly and still clearly excluding "no effect." The plain Welch's t-test's independence assumption was a real approximation worth checking, not just a theoretical nitpick — and the effect holds up once that assumption is relaxed.

**A caveat surfaced by the exploratory comparison, not yet resolved:** `data_quality_issue` — the category the detector's own logic flags as likely-noisy — shows a *cleaner, stronger* mean-return effect than `market_event` does (e.g. 10-day: +0.53pp, p = 0.0006, no decay pattern). That's backwards from what you'd want if the effect were real market behavior. An earlier hypothesis here (a split/adjustment-lag artifact in `adj_close`) turned out not to fit the ingestion code: `backfill.py` and `fetch_daily_prices.py` both set `adj_close` directly from yfinance's already-adjusted `Close`, so there's no separate adjustment step to lag — see the `adj_close` limitation below for the real, still-unresolved hazard this raises instead. The cause of the `data_quality_issue` mean-return pattern itself remains unexplained and is flagged here rather than left out, since a real statistical validation reports the finding that complicates the story, not just the ones that support it. `data_quality_issue`'s volatility effect (4–7% higher, vs. `market_event`'s 11–25%) is much weaker, which is at least evidence the volatility finding isn't being driven by the same cause.

**What this phase demonstrates methodologically:** lookahead-bias-safe return construction, Welch's t-test (not Student's — no basis to assume equal variance between groups) with 95% confidence intervals, a non-parametric Mann-Whitney cross-check given fat-tailed return distributions, explicit multiple-comparison correction rather than cherry-picking the significant horizon, and reporting a null result on the originally-hypothesized effect (direction) alongside a genuine positive result on a different, related one (volatility) — rather than reframing the question after the fact to manufacture a win.

Runs weekly via `.github/workflows/event_study.yml` (deliberately not daily — this is slow statistical re-validation, not same-day ingestion) and writes every test result to the `event_study_results` table, append-only with a `run_timestamp`, so findings can be tracked for drift as more data accumulates rather than living only in a point-in-time snapshot.

## Known limitations / future work

- 3 of 80 original tickers failed backfill due to real corporate actions (Tata Motors demerger, Zomato→Eternal rename) — one fixed (`ETERNAL.NS`), others documented as known gaps
- Company-name-to-headline matching is a substring heuristic, not NER — will miss nicknames, abbreviations, and ticker-only mentions
- Sentiment scoring via VADER doesn't understand financial-domain nuance or sarcasm
- Sentiment-price correlation study needs more accumulated days of headline data before results are statistically meaningful
- Macro indicators (`macro_indicators`) only have daily-forward coverage, not 5 years of backfilled history - excluded from Phase 8b modeling as a result; backfilling this is a natural next step
- Model A's feature set (single-day return, volume ratio, z-score) showed no predictive signal beyond a naive baseline - multi-day momentum features, sector-relative signals, or a wider macro feature set (once backfilled) are the more promising next directions, rather than more complex models on the same weak features
- `adj_close` is currently 100% identical to `close` across all 99,241 price rows — confirmed as expected, not a bug: both `backfill.py` and `fetch_daily_prices.py` populate `adj_close` directly from yfinance's `Close`, which yfinance auto-adjusts by default. The real, still-open hazard is forward-looking rather than a problem with current data: `backfill.py`'s one-time pull adjusts the full 5-year history consistently as of backfill date, but `fetch_daily_prices.py`'s daily pull only adjusts that single day's row. If any of the 78 tickers splits or issues a bonus after backfill, old rows stay on the pre-split basis while new daily rows land post-split, producing a fake jump exactly at the join with nothing in the pipeline to catch it. No safeguard implemented yet.
- Phase 9's event study doesn't control for sector, market-cap, or macro regime (e.g. whether flagged days cluster during broad high-volatility market periods generally, independent of what the detector itself is catching) — `src/event_study_cluster_check.py` addresses the *same-day-across-stocks* correlation specifically (block bootstrap, 95% CI [1.070, 1.137], confirms the volatility effect survives), but does not address the separate, still-open question of whether the effect is partly just mechanical volatility clustering (big moves tend to precede more big moves regardless of cause) rather than something specific to what the detector flags. The weaker volatility effect in `data_quality_issue` (4–7%) vs. `market_event` (11–25%) is suggestive evidence against a purely mechanical explanation, but not a clean regime control.
- The daily anomaly-detection job (`src/daily_anomaly_detection.py`) had an off-by-one bug (`DAYS_TO_FETCH = ROLLING_WINDOW + 1` left one baseline return short, so `classify_row` always returned `None`) — fixed to `+ 2` and verified with a synthetic-spike test, but this means the live daily job likely flagged nothing in production until the fix was deployed; all `anomalies` rows prior to that are from `batch_anomaly_detection.py` backfill runs, not the daily job
- `batch_anomaly_detection.py` (repo root) imported `anomaly_detection` from `src/` with no path set up, so running it exactly as documented raised `ModuleNotFoundError` — fixed with an explicit `sys.path` insert
- News `published_date` previously took the first 10 characters of NewsAPI's UTC `publishedAt` directly; headlines published late UTC evening could land on the wrong IST trading day. Fixed in `fetch_news_sentiment.py` with explicit UTC→IST conversion

## Results so far

- **95,394** price rows backfilled across **77 of 80** target tickers (5 years of history each); 3 failures traced to real corporate actions (demerger, company rename), not pipeline bugs
- **0** rows failed hard data quality validation during backfill — OHLC consistency, non-negative volume, and missing-value checks all passed cleanly on real yfinance data
- **3,732** anomalies detected across 78 stocks after threshold tuning: **65%** classified as `market_event` (price move confirmed by volume), **35%** as `data_quality_issue` (price move unconfirmed by volume)
- **2** real matching bugs found and fixed in the news-to-stock linking logic during manual verification of live output (substring false-positive, over-aggressive name truncation)
- 4 macro indicators (USD/INR, crude oil, S&P 500, Fed funds rate) ingested successfully from two independent sources in a single daily run
- Event study (Phase 9) tested 97,603 baseline-eligible stock-days: no significant post-flag directional return effect after multiple-comparison correction, but a statistically robust and cross-validated post-flag volatility effect (up to 1.31x normal daily volatility for volume-confirmed flags, decaying over ~10 trading days)

## Tech stack

Python · PostgreSQL (Supabase) · SQLAlchemy · yfinance · FRED API · NewsAPI · VADER · SciPy · Resend · pandas · GitHub Actions

## Setup

1. Clone the repo, `pip install -r requirements.txt`
2. Create a `.env` file with `DATABASE_URL`, `FRED_API_KEY`, `NEWSAPI_KEY`, `RESEND_API_KEY`
3. Run `python src/create_tables.py` to create the schema
4. Run `python backfill.py` for historical data (one-time) — this file lives at repo root, not `src/`
5. Run `python batch_anomaly_detection.py` to scan historical anomalies — also repo root
6. Going forward, `src/fetch_daily_prices.py`, `src/fetch_macro_daily.py`, `src/fetch_news_sentiment.py`, `src/daily_anomaly_detection.py`, and `src/daily_summary_email.py` run automatically via `.github/workflows/daily_pipeline.yml`
7. `python src/event_study.py` re-runs the statistical validation (Phase 9) against current data and writes results to `event_study_results`; runs weekly via `.github/workflows/event_study.yml`, or trigger manually via `workflow_dispatch`
8. `python src/verify_volatility_consistency.py` reproduces the Phase 9 cross-check math standalone (no DB connection needed); `python src/event_study_cluster_check.py` re-runs the same-day-clustering robustness check against current data
