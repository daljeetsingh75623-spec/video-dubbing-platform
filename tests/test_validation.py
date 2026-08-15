from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.validation import validate_duration


@pytest.mark.asyncio
async def test_validate_duration_accepts_within_limit(monkeypatch):
    async def _fake(path):
        return 599_000

    monkeypatch.setattr("api.validation.get_video_duration_ms", _fake)
    assert await validate_duration("/tmp/clip.mp4", 600) is None


@pytest.mark.asyncio
async def test_validate_duration_rejects_over_limit(monkeypatch):
    async def _fake(path):
        return 601_000

    monkeypatch.setattr("api.validation.get_video_duration_ms", _fake)
    with pytest.raises(HTTPException) as excinfo:
        await validate_duration("/tmp/clip.mp4", 600)
    assert excinfo.value.status_code == 422


@pytest.mark.asyncio
async def test_validate_duration_rejects_corrupt_file(monkeypatch):
    async def _fake(path):
        raise RuntimeError("ffprobe failed")

    monkeypatch.setattr("api.validation.get_video_duration_ms", _fake)
    with pytest.raises(HTTPException) as excinfo:
        await validate_duration("/tmp/clip.mp4", 600)
    assert excinfo.value.status_code == 400
