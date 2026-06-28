"""Optional PostgreSQL logging for predictions.

This module provides simple helpers to initialize a predictions table and
append prediction records to a PostgreSQL database when `DATABASE_URL` is
configured in `app.config.Config`.
"""
import os
import json
from datetime import datetime
from typing import Optional
import psycopg2
from psycopg2.extras import RealDictCursor


def init_db(database_url: str):
    """Create predictions table if it doesn't exist."""
    if not database_url:
        return
    try:
        with psycopg2.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        fraud_probability DOUBLE PRECISION NOT NULL,
                        actual_label INTEGER,
                        meta JSONB
                    )
                    """
                )
                conn.commit()
    except Exception:
        # Do not raise — DB logging is optional
        return


def log_prediction(database_url: str, fraud_prob: float, actual_label: Optional[int] = None, meta: Optional[dict] = None):
    """Insert a prediction record into the DB."""
    if not database_url:
        return
    try:
        with psycopg2.connect(database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO predictions (timestamp, fraud_probability, actual_label, meta)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (datetime.utcnow(), float(fraud_prob), actual_label, json.dumps(meta) if meta is not None else None),
                )
                conn.commit()
    except Exception:
        # Silently ignore DB failures to avoid impacting API
        return
