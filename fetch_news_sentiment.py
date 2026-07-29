"""
fetch_news_sentiment.py
Daily news + sentiment ingestion for market-pulse-pipeline.

Approach (see project notes for full reasoning):
- NewsAPI free tier has a 100 req/day limit, and we have 78 stocks - one
  query per stock isn't sustainable. Instead we run a small set of BROAD
  market queries, then match company names within the returned headlines.
- Matching uses a simplified "short name" per company (common corporate
  suffixes like Limited/Ltd/Industries stripped) and a substring check.
  This is a known-imperfect heuristic - won't catch nicknames, abbreviations,
  or ticker-only mentions. A production version would use NER or fuzzy
  matching; this is a reasonable scalable first pass for a learning project.
- Sentiment via VADER: lightweight, rule-based, no API/training needed,
  outputs naturally in -1 to 1 range. Doesn't understand financial nuance
  specifically (sarcasm, domain jargon) - a known, acceptable limitation.

Run daily: python fetch_news_sentiment.py
"""

import os
import re
from datetime import datetime, timezone, date

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

engine = create_engine(DATABASE_URL)
newsapi = NewsApiClient(api_key=NEWSAPI_KEY)
analyzer = SentimentIntensityAnalyzer()

# Broad queries covering Indian market news generally, instead of one query
# per stock - keeps us well within NewsAPI's free tier request budget.
BROAD_QUERIES = [
    "Indian stock market",
    "NSE Sensex Nifty",
    "India economy stocks",
]

ARTICLES_PER_QUERY = 20  # keep this modest - free tier budget

# Only strip PURE corporate-structure words here, not descriptive words like
# "Consultancy Services" or "Industries" - those are often exactly what makes
# a name distinctive within a business group (e.g. stripping down to bare
# "Tata" falsely matched every Tata-group headline, not just TCS specifically).
SUFFIXES_TO_STRIP = [
    "limited", "ltd", "ltd.", "corporation", "corp", "inc", "inc.",
    "& co", "and co", "co.",
]


def get_short_name(company_name: str) -> str:
    """
    Strips only pure corporate-structure suffixes (Limited/Ltd/Corp/Inc) and
    returns the full remaining name as the match key - NOT truncated to 1-2
    words, since truncating threw away the distinctive part of group-company
    names (e.g. "Tata Consultancy Services" -> bare "tata", which then
    matched any Tata-group headline). Matching is still substring-based and
    imperfect by design - see module docstring.
    """
    name = company_name.lower()
    for suffix in SUFFIXES_TO_STRIP:
        name = re.sub(rf"\b{re.escape(suffix)}\b", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def get_all_stocks(conn) -> list[dict]:
    result = conn.execute(text("SELECT stock_id, company_name FROM stocks"))
    return [{"stock_id": row[0], "company_name": row[1], "short_name": get_short_name(row[1])}
            for row in result]


def fetch_headlines() -> list[dict]:
    """Runs the broad queries against NewsAPI, returns a deduplicated list
    of raw article dicts (title, source, published date)."""
    seen_titles = set()
    articles = []

    for query in BROAD_QUERIES:
        try:
            response = newsapi.get_everything(
                q=query,
                language="en",
                sort_by="publishedAt",
                page_size=ARTICLES_PER_QUERY,
            )
            for article in response.get("articles", []):
                title = article.get("title")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                articles.append({
                    "title": title,
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt"),
                })
        except Exception as e:
            print(f"  Query '{query}' failed: {e}")

    return articles


def match_stocks_to_headline(headline_text: str, stocks: list[dict]) -> list[int]:
    """
    Returns stock_ids whose short_name appears in the headline text as a
    whole word/phrase - NOT plain substring matching, which would match
    "titan" inside "titanium". Uses \\b word boundaries around the short_name.
    """
    text_lower = headline_text.lower()
    matched = []
    for s in stocks:
        if not s["short_name"]:
            continue
        pattern = r"\b" + re.escape(s["short_name"]) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(s["stock_id"])
    return matched


def score_sentiment(headline_text: str) -> float:
    scores = analyzer.polarity_scores(headline_text)
    return round(scores["compound"], 2)  # VADER's compound score is already -1 to 1


def insert_headline(conn, headline_text: str, published_date, source: str, sentiment_score: float) -> int:
    result = conn.execute(
        text("""
            INSERT INTO news_headlines (headline_text, published_date, source, sentiment_score)
            VALUES (:headline_text, :published_date, :source, :sentiment_score)
            RETURNING headline_id
        """),
        {
            "headline_text": headline_text,
            "published_date": published_date,
            "source": source,
            "sentiment_score": sentiment_score,
        },
    )
    return result.scalar()


def link_headline_to_stock(conn, headline_id: int, stock_id: int):
    conn.execute(
        text("""
            INSERT INTO headline_stocks (headline_id, stock_id)
            VALUES (:headline_id, :stock_id)
            ON CONFLICT (headline_id, stock_id) DO NOTHING
        """),
        {"headline_id": headline_id, "stock_id": stock_id},
    )


def create_ingestion_run(conn, source: str) -> int:
    result = conn.execute(
        text("""
            INSERT INTO ingestion_log (run_timestamp, source, result, total_expected, total_failed)
            VALUES (:run_timestamp, :source, 'in_progress', 0, 0)
            RETURNING run_id
        """),
        {"run_timestamp": datetime.now(timezone.utc), "source": source},
    )
    return result.scalar()


def update_ingestion_run(conn, run_id: int, result: str, total_expected: int, total_failed: int):
    conn.execute(
        text("""
            UPDATE ingestion_log
            SET result = :result, total_expected = :total_expected, total_failed = :total_failed
            WHERE run_id = :run_id
        """),
        {"run_id": run_id, "result": result, "total_expected": total_expected, "total_failed": total_failed},
    )


def main():
    with engine.begin() as conn:
        run_id = create_ingestion_run(conn, source="news")
        stocks = get_all_stocks(conn)

    print(f"Fetching headlines across {len(BROAD_QUERIES)} broad queries...\n")
    articles = fetch_headlines()
    print(f"Fetched {len(articles)} unique headlines.\n")

    total_inserted = 0
    total_linked = 0
    total_unmatched = 0  # headlines that matched no stock at all - still stored, just unlinked

    with engine.begin() as conn:
        for article in articles:
            try:
                published_date = article["published_at"][:10] if article["published_at"] else date.today().isoformat()
                sentiment = score_sentiment(article["title"])

                headline_id = insert_headline(
                    conn, article["title"], published_date, article["source"], sentiment
                )
                total_inserted += 1

                matched_stock_ids = match_stocks_to_headline(article["title"], stocks)
                if not matched_stock_ids:
                    total_unmatched += 1
                for stock_id in matched_stock_ids:
                    link_headline_to_stock(conn, headline_id, stock_id)
                    total_linked += 1

            except Exception as e:
                print(f"  Failed to process headline '{article['title'][:50]}...': {e}")

        update_ingestion_run(conn, run_id, "success", len(articles), 0)

    print("--- News + sentiment summary ---")
    print(f"Headlines inserted        : {total_inserted}")
    print(f"Headline-stock links made : {total_linked}")
    print(f"Headlines matched to no stock: {total_unmatched}")


if __name__ == "__main__":
    main()