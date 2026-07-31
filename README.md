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

## Known limitations / future work

- 3 of 80 original tickers failed backfill due to real corporate actions (Tata Motors demerger, Zomato→Eternal rename) — one fixed (`ETERNAL.NS`), others documented as known gaps
- Company-name-to-headline matching is a substring heuristic, not NER — will miss nicknames, abbreviations, and ticker-only mentions
- Sentiment scoring via VADER doesn't understand financial-domain nuance or sarcasm
- Sentiment-price correlation study needs more accumulated days of headline data before results are statistically meaningful

## Results so far

- **95,394** price rows backfilled across **77 of 80** target tickers (5 years of history each); 3 failures traced to real corporate actions (demerger, company rename), not pipeline bugs
- **0** rows failed hard data quality validation during backfill — OHLC consistency, non-negative volume, and missing-value checks all passed cleanly on real yfinance data
- **3,732** anomalies detected across 78 stocks after threshold tuning: **65%** classified as `market_event` (price move confirmed by volume), **35%** as `data_quality_issue` (price move unconfirmed by volume)
- **2** real matching bugs found and fixed in the news-to-stock linking logic during manual verification of live output (substring false-positive, over-aggressive name truncation)
- 4 macro indicators (USD/INR, crude oil, S&P 500, Fed funds rate) ingested successfully from two independent sources in a single daily run

## Tech stack

Python · PostgreSQL (Supabase) · SQLAlchemy · yfinance · FRED API · NewsAPI · VADER · Resend · pandas · GitHub Actions

## Setup

1. Clone the repo, `pip install -r requirements.txt`
2. Create a `.env` file with `DATABASE_URL`, `FRED_API_KEY`, `NEWSAPI_KEY`, `RESEND_API_KEY`
3. Run `python src/create_tables.py` to create the schema
4. Run `python src/backfill.py` for historical data (one-time)
5. Run `python src/batch_anomaly_detection.py` to scan historical anomalies
6. Going forward, `src/fetch_daily_prices.py`, `src/fetch_macro_daily.py`, `src/fetch_news_sentiment.py`, `src/daily_anomaly_detection.py`, and `src/daily_summary_email.py` run automatically via `.github/workflows/daily_pipeline.yml`
