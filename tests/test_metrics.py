"""Prometheus /metrics endpoint: renders self-contained metrics and the app
counters/gauges react to real events (job creation, 429 rejections, HTTP)."""
from __future__ import annotations

import asyncio
import io

import pytest

from core import metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    metrics.reset()
    yield
    metrics.reset()


def _count(text: str, name: str) -> int:
    return sum(
        int(line.rsplit(" ", 1)[1])
        for line in text.splitlines()
        if line.startswith(name + " ") or line.startswith(name + "{")
    )


def _upload():
    return {"file": ("clip.mp4", io.BytesIO(b"not a real video"), "video/mp4")}, {"target_language": "es"}


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_families(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "# HELP vdp_jobs_created_total" in resp.text
    assert "# TYPE vdp_jobs_created_total counter" in resp.text
    assert "# TYPE vdp_uploads_active gauge" in resp.text
    assert "# TYPE vdp_http_requests_total counter" in resp.text
    assert "# TYPE vdp_upload_rejections_total counter" in resp.text


@pytest.mark.asyncio
async def test_upload_increments_created_counter_and_queue_depth(client, monkeypatch):
    import api.main as main_module
    from config.settings import get_settings

    main_module.app.state.upload_semaphore = asyncio.Semaphore(get_settings().max_concurrent_uploads)

    async def _short_duration(path):
        return 1000

    monkeypatch.setattr("api.validation.get_video_duration_ms", _short_duration)

    before = _count((await client.get("/metrics")).text, "vdp_jobs_created_total")

    resp = await client.post("/videos", files=_upload()[0], data=_upload()[1])
    assert resp.status_code == 201

    text = (await client.get("/metrics")).text
    assert _count(text, "vdp_jobs_created_total") == before + 1
    assert "# TYPE vdp_queue_depth gauge" in text


@pytest.mark.asyncio
async def test_queue_full_rejection_increments_labeled_counter(client, test_session_factory, monkeypatch):
    import uuid
    from datetime import datetime, timezone

    from db.models import Job

    async with test_session_factory() as s:
        for _ in range(2):
            s.add(
                Job(
                    id=uuid.uuid4(),
                    status="queued",
                    source_video_key="uploads/x/source.mp4",
                    target_language="es",
                    stt_provider="stub",
                    translation_provider="stub",
                    tts_provider="stub",
                    diarization_provider="stub",
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc),
                )
            )
        await s.commit()

    from config.settings import Settings

    s = Settings()
    s.queue_max_length = 2
    monkeypatch.setattr("api.routers.videos.get_settings", lambda: s)

    resp = await client.post("/videos", files=_upload()[0], data=_upload()[1])
    assert resp.status_code == 429

    text = (await client.get("/metrics")).text
    assert 'vdp_upload_rejections_total{reason="queue_full"} 1' in text
    assert "vdp_queue_depth 2" in text


@pytest.mark.asyncio
async def test_http_counter_uses_route_template(client):
    await client.get("/health")

    text = (await client.get("/metrics")).text
    assert 'vdp_http_requests_total{method="GET", route="/health", status="200"} 1' in text


@pytest.mark.asyncio
async def test_metrics_stays_up_when_db_down(client, monkeypatch):
    import api.main as main_module
    from db.session import get_db
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    # Simulate an unreachable database: the override session fails on query.
    bad_engine = create_async_engine("sqlite+aiosqlite:////nonexistent/dir/x.db")
    bad_factory = async_sessionmaker(bad_engine, expire_on_commit=False)

    async def _bad_get_db():
        async with bad_factory() as session:
            yield session

    main_module.app.dependency_overrides[get_db] = _bad_get_db
    try:
        resp = await client.get("/metrics")
    finally:
        main_module.app.dependency_overrides.clear()
        await bad_engine.dispose()

    assert resp.status_code == 200
    assert "vdp_queue_depth -1" in resp.text
    assert "vdp_http_requests_total" in resp.text
