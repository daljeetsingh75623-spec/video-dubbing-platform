"""URL helpers for SQLAlchemy engine construction.

Kept in their own module (no engine creation, no settings import) so both
``db/session.py`` and Alembic's ``env.py`` can use them without pulling in
module-level side effects.
"""


def sync_database_url(url: str) -> str:
    """Map an async driver URL to its sync equivalent for Celery/Alembic.

    Handles the two async drivers this project uses: Postgres asyncpg becomes
    psycopg2; SQLite aiosqlite becomes plain sqlite (the default dialect has no
    explicit driver token). Anything else is returned unchanged.
    """
    if "+aiosqlite" in url:
        return url.replace("+aiosqlite", "")
    if "+asyncpg" in url:
        return url.replace("+asyncpg", "+psycopg2")
    return url
