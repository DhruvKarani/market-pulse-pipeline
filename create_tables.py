import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS stocks (
        stock_id       SERIAL PRIMARY KEY,
        ticker         VARCHAR(20) UNIQUE NOT NULL,
        company_name   VARCHAR(255),
        exchange       VARCHAR(10),
        sector         VARCHAR(100) NOT NULL DEFAULT 'Unknown'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS stock_prices (
        stock_id    INTEGER REFERENCES stocks(stock_id),
        date        DATE,
        open        NUMERIC(10,2),
        high        NUMERIC(10,2),
        low         NUMERIC(10,2),
        close       NUMERIC(10,2),
        adj_close   NUMERIC(10,2),
        volume      BIGINT,
        PRIMARY KEY (stock_id, date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS macro_indicators (
        date            DATE,
        indicator_name  VARCHAR(50),
        value           NUMERIC(12,4),
        PRIMARY KEY (date, indicator_name)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS news_headlines (
        headline_id     SERIAL PRIMARY KEY,
        headline_text   TEXT,
        published_date  DATE,
        source          TEXT,
        sentiment_score NUMERIC(3,2)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS headline_stocks (
        headline_id  INTEGER REFERENCES news_headlines(headline_id),
        stock_id     INTEGER REFERENCES stocks(stock_id),
        PRIMARY KEY (headline_id, stock_id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_log (
        run_id          SERIAL PRIMARY KEY,
        run_timestamp   TIMESTAMPTZ,
        source          TEXT,
        result          TEXT,
        total_expected  INT,
        total_failed    INT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS ingestion_failures (
        failure_id      SERIAL PRIMARY KEY,
        run_id          INTEGER REFERENCES ingestion_log(run_id),
        stock_id        INTEGER REFERENCES stocks(stock_id),
        error_message   TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS anomalies (
        stock_id          INTEGER,
        date              DATE,
        anomaly_id        SERIAL PRIMARY KEY,
        anomaly_type      TEXT,
        anomaly_category  TEXT,
        severity_score    NUMERIC(3,2),
        FOREIGN KEY (stock_id, date) REFERENCES stock_prices(stock_id, date)
    );
    """,
]

def create_all_tables():
    with engine.begin() as conn:
        for statement in DDL_STATEMENTS:
            conn.execute(text(statement))
    print("All 8 tables created (or already existed).")

if __name__ == "__main__":
    create_all_tables()