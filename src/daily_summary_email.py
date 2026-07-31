"""
daily_summary_email.py
Phase 11: sends a daily email summary of what the pipeline found today -
anomalies detected, notable headlines/sentiment, and overall ingestion
health - since GitHub Actions only notifies on job failure, not on content.

Uses Resend's REST API directly via `requests` (no extra SDK needed,
consistent with how FRED is called in fetch_macro_daily.py).

Run as the LAST step in the daily pipeline, after all other scripts.
"""

import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RECIPIENT_EMAIL = "dhrubkarani22@gmail.com"

engine = create_engine(DATABASE_URL)


def get_todays_anomalies(conn):
    result = conn.execute(
        text("""
            SELECT s.ticker, a.anomaly_type, a.anomaly_category, a.severity_score
            FROM anomalies a
            JOIN stocks s ON a.stock_id = s.stock_id
            WHERE a.date = CURRENT_DATE
            ORDER BY a.severity_score DESC
        """)
    )
    return result.fetchall()


def get_todays_top_headlines(conn, limit=5):
    result = conn.execute(
        text("""
            SELECT nh.headline_text, nh.sentiment_score, s.ticker
            FROM news_headlines nh
            JOIN headline_stocks hs ON nh.headline_id = hs.headline_id
            JOIN stocks s ON hs.stock_id = s.stock_id
            WHERE nh.published_date = CURRENT_DATE
            ORDER BY ABS(nh.sentiment_score) DESC
            LIMIT :limit
        """),
        {"limit": limit},
    )
    return result.fetchall()


def get_todays_ingestion_health(conn):
    result = conn.execute(
        text("""
            SELECT source, result, total_expected, total_failed
            FROM ingestion_log
            WHERE run_timestamp::date = CURRENT_DATE
            ORDER BY run_timestamp
        """)
    )
    return result.fetchall()


def build_email_html(anomalies, headlines, ingestion_runs) -> str:
    today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")

    html = f"<h2>Market Pulse Daily Summary — {today_str}</h2>"

    # Ingestion health first - if something failed, that's the most
    # important thing to see immediately
    html += "<h3>Pipeline Health</h3><ul>"
    if not ingestion_runs:
        html += "<li>No ingestion runs recorded today.</li>"
    for source, result, total_expected, total_failed in ingestion_runs:
        icon = "✅" if result == "success" else ("⚠️" if result == "partial_failure" else "❌")
        html += f"<li>{icon} <b>{source}</b>: {result} ({total_failed}/{total_expected} failed)</li>"
    html += "</ul>"

    html += f"<h3>Anomalies Detected Today ({len(anomalies)})</h3>"
    if anomalies:
        html += "<ul>"
        for ticker, anomaly_type, category, severity in anomalies[:15]:
            html += f"<li><b>{ticker}</b>: {anomaly_type} ({category}) — severity {severity}</li>"
        html += "</ul>"
        if len(anomalies) > 15:
            html += f"<p><i>...and {len(anomalies) - 15} more.</i></p>"
    else:
        html += "<p>No anomalies flagged today.</p>"

    html += f"<h3>Most Notable Headlines Today</h3>"
    if headlines:
        html += "<ul>"
        for headline_text, sentiment, ticker in headlines:
            tone = "🟢" if sentiment > 0.2 else ("🔴" if sentiment < -0.2 else "⚪")
            html += f"<li>{tone} <b>{ticker}</b> ({sentiment}): {headline_text}</li>"
        html += "</ul>"
    else:
        html += "<p>No headlines matched to stocks today.</p>"

    return html


def send_email(html_content: str):
    response = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
        json={
            "from": "Market Pulse Pipeline <onboarding@resend.dev>",
            "to": [RECIPIENT_EMAIL],
            "subject": f"Market Pulse Daily Summary — {datetime.now(timezone.utc).strftime('%b %d, %Y')}",
            "html": html_content,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def main():
    with engine.begin() as conn:
        anomalies = get_todays_anomalies(conn)
        headlines = get_todays_top_headlines(conn)
        ingestion_runs = get_todays_ingestion_health(conn)

    html = build_email_html(anomalies, headlines, ingestion_runs)

    try:
        result = send_email(html)
        print(f"Email sent successfully. Resend id: {result.get('id')}")
    except Exception as e:
        print(f"Failed to send email: {e}")
        raise


if __name__ == "__main__":
    main()