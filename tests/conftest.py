"""
Shared pytest fixtures. Integration tests use an in-memory SQLite DB (async)
instead of the real Postgres, and monkeypatch Celery's .delay() to a no-op
so API tests don't require a live Redis/worker.
"""
from __future__ import annotations

import os
import tempfile

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from db.models import Base

# Don't initialize OpenTelemetry tracing during tests (settings are read once
# and cached on first import of api.main, so this must happen before it).
os.environ.setdefault("OTEL_ENABLED", "false")
# Keep the per-route rate limiter from tripping during the whole test suite
# (the limiter key is the remote address, so every request in the run shares
# one bucket).
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")
# Give the storage backend a real scratch dir (the stub TTS now uploads its
# generated WAV chunks through storage, exactly like a real provider).
os.environ.setdefault("STORAGE_LOCAL_PATH", tempfile.mkdtemp(prefix="vdp_test_storage_"))
# Point the Celery pipeline's DB (sync engine) at a file-backed SQLite DB.
_test_db = os.path.join(tempfile.mkdtemp(prefix="vdp_test_db_"), "test.db")
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_test_db.replace(os.sep, '/')}")


@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session_factory(test_engine):
    return async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def client(test_session_factory, monkeypatch):
    from db.session import get_db
    import api.main as main_module

    async def _override_get_db():
        async with test_session_factory() as session:
            yield session

    main_module.app.dependency_overrides[get_db] = _override_get_db

    class _FakeDelay:
        def delay(self, *args, **kwargs):
            return None

    monkeypatch.setattr("workers.tasks.run_pipeline", _FakeDelay())

    transport = ASGITransport(app=main_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    main_module.app.dependency_overrides.clear()