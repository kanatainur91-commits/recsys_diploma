import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


DATABASE_URL = os.getenv("DATABASE_URL")

engine = None
SessionLocal = None


if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True
    )

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )


def create_tables():
    """Создание таблиц PostgreSQL."""

    if engine is None:
        return

    with engine.begin() as connection:

        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                movie_id BIGINT NOT NULL,
                reward SMALLINT NOT NULL CHECK (reward IN (0, 1)),
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """))

        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS bandit_stats (
                movie_id BIGINT PRIMARY KEY,
                alpha DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                beta DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """))

        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS impressions (
                id BIGSERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL,
                movie_id BIGINT NOT NULL,
                reward SMALLINT,
                timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            )
        """))