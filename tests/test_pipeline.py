"""
Pipeline tests that run the real Celery stage tasks in-process (no broker/worker
needed) against SQLite + local storage.

The stub TTS generates and uploads real (silent) WAV chunks, so the full
diarize -> transcribe -> translate -> synthesize -> sync_av -> package chain is
exercised end-to-end. ffmpeg itself is monkeypatched so the suite runs without
it, but the AV-sync filter-building and muxing code paths run for real.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from db.models import Base, Job, JobEvent, TranscriptRecord
from db.session import get_sync_db, sync_database_url
from storage.factory import get_storage_backend

import core.av_sync as av
import workers.tasks as tasks


@pytest.fixture
def seeded_tables():
    """Ensure the file-backed SQLite DB (from conftest's DATABASE_URL) has tables."""
    engine = create_engine(sync_database_url(tasks._settings.database_url))
    Base.metadata.create_all(engine)
    yield engine


def _make_job(status="queued", **overrides):
    job = Job(
        id=uuid.uuid4(), status=status,
        source_video_key=f"uploads/{uuid.uuid4()}/source.mp4",
        target_language="es",
        stt_provider="stub", translation_provider="stub",
        tts_provider="stub", diarization_provider="stub",
        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc),
    )
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


def _seed_source_video(job, storage):
    """Put a dummy source video in storage so sync_av can download it."""
    path = Path(tasks.WORK_DIR) / f"{job.id}_src.mp4"
    path.write_bytes(b"fake-video")
    asyncio.run(storage.upload(str(path), job.source_video_key))


def test_sync_database_url_converts_async_drivers():
    assert sync_database_url("postgresql+asyncpg://x/y") == "postgresql+psycopg2://x/y"
    assert sync_database_url("sqlite+aiosqlite:///./x.db") == "sqlite:///./x.db"
    assert sync_database_url("postgresql://x/y") == "postgresql://x/y"


def test_full_pipeline_completes_with_stub_providers(seeded_tables, monkeypatch):
    storage = get_storage_backend()

    async def _fake_ffmpeg(cmd):
        from pathlib import Path
        Path(cmd[-1]).write_bytes(b"fake-media")
        return None

    async def _no_rubberband():
        return False

    async def _fake_video_duration(_path):
        return 10_000

    async def _fake_audio_duration(_path):
        return 3_000

    monkeypatch.setattr(av, "_run_ffmpeg", _fake_ffmpeg)
    monkeypatch.setattr(av, "_check_rubberband", _no_rubberband)
    monkeypatch.setattr(av, "get_video_duration_ms", _fake_video_duration)
    monkeypatch.setattr(av, "_get_audio_duration_ms", _fake_audio_duration)

    with get_sync_db() as db:
        job = _make_job()
        db.add(job)
        db.commit()
        job_id = str(job.id)

    _seed_source_video(job, storage)

    tasks.diarize_stage(job_id)
    tasks.transcribe_stage(job_id)
    tasks.translate_stage(job_id)
    tasks.synthesize_stage(job_id)
    tasks.sync_av_stage(job_id)
    tasks.package_output_stage(job_id)

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job.status == "completed", job.error_message
        assert job.output_video_key
        assert job.output_srt_key
        assert job.output_transcript_key
        assert job.source_language == "en"
        records = (
            db.query(TranscriptRecord)
            .filter(TranscriptRecord.job_id == job.id)
            .order_by(TranscriptRecord.start_ms)
            .all()
        )
        assert len(records) == 10  # stub diarization: 10 alternating 3s turns
        assert all(r.speaker_label in ("SPEAKER_00", "SPEAKER_01") for r in records)
        assert all(r.translated_text and r.translated_text.startswith("[es]") for r in records)

    # Real outputs must have been written through the storage backend.
    for key in (job.output_video_key, job.output_srt_key, job.output_transcript_key):
        assert asyncio.run(storage.exists(key)), f"missing output: {key}"
    assert asyncio.run(storage.exists(job.source_video_key))


def test_sync_av_and_package_are_idempotent(seeded_tables, monkeypatch):
    """Re-running sync_av/package after completion must not regress the job."""
    storage = get_storage_backend()

    async def _fake_ffmpeg(cmd):
        from pathlib import Path
        Path(cmd[-1]).write_bytes(b"fake-media")
        return None

    async def _no_rubberband():
        return False

    async def _fake_video_duration(_path):
        return 10_000

    async def _fake_audio_duration(_path):
        return 3_000

    monkeypatch.setattr(av, "_run_ffmpeg", _fake_ffmpeg)
    monkeypatch.setattr(av, "_check_rubberband", _no_rubberband)
    monkeypatch.setattr(av, "get_video_duration_ms", _fake_video_duration)
    monkeypatch.setattr(av, "_get_audio_duration_ms", _fake_audio_duration)

    with get_sync_db() as db:
        job = _make_job()
        db.add(job)
        db.commit()
        job_id = str(job.id)

    _seed_source_video(job, storage)

    tasks.diarize_stage(job_id)
    tasks.transcribe_stage(job_id)
    tasks.translate_stage(job_id)
    tasks.synthesize_stage(job_id)
    tasks.sync_av_stage(job_id)
    tasks.package_output_stage(job_id)

    tasks.sync_av_stage(job_id)
    tasks.package_output_stage(job_id)

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job.status == "completed"
        assert job.error_message is None


def test_recover_stale_jobs_marks_inflight_failed(seeded_tables, monkeypatch):
    class _FakeSettings:
        stale_job_timeout_seconds = 1

    # Patch the call-time lookup rather than the module-level snapshot: other
    # tests (test_auth) clear the settings cache, so _settings can be stale.
    monkeypatch.setattr(tasks, "get_settings", lambda: _FakeSettings())

    with get_sync_db() as db:
        stuck = _make_job(status="syncing")
        stuck.updated_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        fresh = _make_job(status="syncing")
        fresh.updated_at = datetime.now(timezone.utc) - timedelta(seconds=0.1)
        queued = _make_job(status="queued")
        queued.updated_at = datetime.now(timezone.utc) - timedelta(seconds=60)
        db.add_all([stuck, fresh, queued])
        db.commit()
        stuck_id = str(stuck.id)
        fresh_id = str(fresh.id)
        queued_id = str(queued.id)

    recovered = tasks.recover_stale_jobs()
    assert recovered == 1

    with get_sync_db() as db:
        assert db.get(Job, uuid.UUID(stuck_id)).status == "failed"
        # In-flight but recent -> untouched; queued is deliberately left alone
        # (a long queue is backpressure, not a dead job).
        assert db.get(Job, uuid.UUID(fresh_id)).status == "syncing"
        assert db.get(Job, uuid.UUID(queued_id)).status == "queued"


def test_mark_failed_records_error(seeded_tables):
    with get_sync_db() as db:
        job = _make_job(status="syncing")
        db.add(job)
        db.commit()
        job_id = str(job.id)

    tasks.mark_failed(None, None, None, job_id)

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        assert job.status == "failed"
        assert job.error_message is not None
        event = (
            db.query(JobEvent)
            .filter(JobEvent.job_id == job.id, JobEvent.stage == "failed")
            .first()
        )
        assert event is not None
