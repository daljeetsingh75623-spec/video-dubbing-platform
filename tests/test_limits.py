"""Enforcement of the configurable operational limits: queue capacity and
upload concurrency must reject with 429 + Retry-After when saturated."""
from __future__ import annotations

import asyncio
import io
import uuid
from datetime import datetime, timezone

import pytest

from config.settings import Settings
from db.models import Job


async def _seed_queued_jobs(session_factory, n: int) -> None:
    async with session_factory() as s:
        for _ in range(n):
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


def _upload(files=None):
    files = files or {"file": ("clip.mp4", io.BytesIO(b"not a real video"), "video/mp4")}
    return {"file": files["file"]}, {"target_language": "es"}


@pytest.mark.asyncio
async def test_upload_rejected_when_queue_full(client, test_session_factory, monkeypatch):
    await _seed_queued_jobs(test_session_factory, 2)

    s = Settings()
    s.queue_max_length = 2
    monkeypatch.setattr("api.routers.videos.get_settings", lambda: s)

    resp = await client.post("/videos", files=_upload()[0], data=_upload()[1])
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "60"
    assert "queue" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_rejected_when_concurrency_saturated(client, monkeypatch):
    import api.main as main_module

    main_module.app.state.upload_semaphore = asyncio.Semaphore(0)

    resp = await client.post("/videos", files=_upload()[0], data=_upload()[1])
    assert resp.status_code == 429
    assert resp.headers.get("retry-after") == "5"
    assert "concurrent" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_succeeds_when_limits_open(client, monkeypatch):
    import api.main as main_module
    from config.settings import get_settings

    # Other limit tests replace the app-wide semaphore; restore a healthy one
    # so this test exercises the real open path.
    main_module.app.state.upload_semaphore = asyncio.Semaphore(
        get_settings().max_concurrent_uploads
    )

    async def _short_duration(path):
        return 1000

    monkeypatch.setattr("api.validation.get_video_duration_ms", _short_duration)

    resp = await client.post("/videos", files=_upload()[0], data=_upload()[1])
    assert resp.status_code == 201
    assert resp.json()["status"] == "queued"
