from __future__ import annotations

import asyncio
import os
import tempfile
import uuid

import structlog
from celery import chain
from sqlalchemy import select

from config.settings import get_settings
from core.domain import Transcript, TranscriptSegment, TranslatedSegment, TranslatedTranscript
from db.models import Job, JobEvent, TranscriptRecord
from db.session import get_sync_db
from providers.errors import ProviderError, ProviderTimeoutError, ProviderUnavailableError
from providers.factory import ProviderFactory
from storage.factory import get_storage_backend
from workers.celery_app import celery_app

log = structlog.get_logger()


def _run_async(coro):
    return asyncio.run(coro)


def _set_status(job_id: str, status: str, message: str | None = None, error: str | None = None) -> None:
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if not job:
            return
        job.status = status
        if error is not None:
            job.error_message = error
        db.add(JobEvent(id=uuid.uuid4(), job_id=job.id, stage=status, message=message))
        db.commit()


def _is_cancelled(job_id: str) -> bool:
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        return bool(job and job.status == "cancelled")


_RETRYABLE = (ProviderTimeoutError, ProviderUnavailableError)
_TASK_KWARGS = dict(
    bind=True,
    autoretry_for=_RETRYABLE,
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)


@celery_app.task(**_TASK_KWARGS)
def diarize_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "diarizing")
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
    if not job:
        return job_id
    provider = ProviderFactory.get_diarization(job.diarization_provider)
    try:
        result = _run_async(provider.diarize(job.source_video_key))
    except ProviderError as e:
        log.error("diarization_failed", job_id=job_id, error=str(e))
        raise
    log.info("diarization_complete", job_id=job_id, num_speakers=result.num_speakers)
    return job_id


@celery_app.task(**_TASK_KWARGS)
def transcribe_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "transcribing")
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
    if not job:
        return job_id
    provider = ProviderFactory.get_stt(job.stt_provider)
    try:
        transcript = _run_async(provider.transcribe(job.source_video_key))
    except ProviderError as e:
        log.error("transcription_failed", job_id=job_id, error=str(e))
        raise
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if not job:
            return job_id
        job.source_language = transcript.source_language
        for seg in transcript.segments:
            db.add(TranscriptRecord(
                id=uuid.uuid4(), job_id=job.id, speaker_label=seg.speaker_id,
                start_ms=seg.start_ms, end_ms=seg.end_ms,
                source_text=seg.text, confidence=seg.confidence,
            ))
        db.commit()
    return job_id


def _load_transcript_segments(job_id: str) -> list[TranscriptRecord]:
    with get_sync_db() as db:
        return list(
            db.scalars(
                select(TranscriptRecord)
                .where(TranscriptRecord.job_id == uuid.UUID(job_id))
                .order_by(TranscriptRecord.start_ms)
            ).all()
        )


@celery_app.task(**_TASK_KWARGS)
def translate_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "translating")
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
    if not job:
        return job_id
    records = _load_transcript_segments(job_id)
    if not records:
        return job_id

    transcript = Transcript(
        source_language=job.source_language,
        segments=[
            TranscriptSegment(
                speaker_id=r.speaker_label,
                start_ms=r.start_ms,
                end_ms=r.end_ms,
                text=r.source_text,
                language=job.source_language,
                confidence=r.confidence,
            )
            for r in records
        ],
    )
    provider = ProviderFactory.get_translation(job.translation_provider)
    try:
        translated = _run_async(provider.translate(transcript, job.target_language))
    except ProviderError as e:
        log.error("translation_failed", job_id=job_id, error=str(e))
        raise

    with get_sync_db() as db:
        records = db.scalars(
            select(TranscriptRecord)
            .where(TranscriptRecord.job_id == uuid.UUID(job_id))
            .order_by(TranscriptRecord.start_ms)
        ).all()
        by_start = {r.start_ms: r for r in records}
        for seg in translated.segments:
            rec = by_start.get(seg.start_ms)
            if rec is not None:
                rec.translated_text = seg.translated_text
        db.commit()
    log.info("translation_complete", job_id=job_id, num_segments=len(translated.segments))
    return job_id


@celery_app.task(**_TASK_KWARGS)
def synthesize_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "synthesizing")
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
    if not job:
        return job_id
    records = _load_transcript_segments(job_id)
    if not records:
        return job_id

    translated = TranslatedTranscript(
        target_language=job.target_language,
        segments=[
            TranslatedSegment(
                speaker_id=r.speaker_label,
                start_ms=r.start_ms,
                end_ms=r.end_ms,
                source_text=r.source_text,
                translated_text=r.translated_text or r.source_text,
                target_language=job.target_language,
            )
            for r in records
        ],
    )
    provider = ProviderFactory.get_tts(job.tts_provider)
    try:
        result = _run_async(provider.synthesize(translated))
    except ProviderError as e:
        log.error("synthesis_failed", job_id=job_id, error=str(e))
        raise
    log.info("synthesis_complete", job_id=job_id, num_segments=len(result.segments))
    return job_id


@celery_app.task(**_TASK_KWARGS)
def sync_av_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "syncing")
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
    if not job:
        return job_id

    output_key = f"outputs/{job_id}/dubbed.mp4"
    storage = get_storage_backend()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp_path = tmp.name
        _run_async(storage.download(job.source_video_key, tmp_path))
        _run_async(storage.upload(tmp_path, output_key))
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if not job:
            return job_id
        job.output_video_key = output_key
        db.commit()
    log.info("av_sync_complete", job_id=job_id, output_video_key=output_key)
    return job_id


def _format_srt_timestamp(ms: int) -> str:
    ms = max(0, ms)
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, millis = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _build_srt(records: list[TranscriptRecord]) -> str:
    lines: list[str] = []
    for i, r in enumerate(records, start=1):
        text = (r.translated_text or r.source_text).replace("\r", "").replace("\n", " ")
        lines.append(str(i))
        lines.append(f"{_format_srt_timestamp(r.start_ms)} --> {_format_srt_timestamp(r.end_ms)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


@celery_app.task(**_TASK_KWARGS)
def package_output_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "packaging")
    records = _load_transcript_segments(job_id)

    srt_key = f"outputs/{job_id}/subtitles.srt"
    storage = get_storage_backend()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w", encoding="utf-8") as tmp:
            tmp.write(_build_srt(records))
            tmp_path = tmp.name
        _run_async(storage.upload(tmp_path, srt_key))
    finally:
        if tmp_path:
            os.unlink(tmp_path)

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if not job:
            return job_id
        job.status = "completed"
        job.output_srt_key = srt_key
        db.add(JobEvent(id=uuid.uuid4(), job_id=job.id, stage="completed", message="Pipeline finished"))
        db.commit()
    log.info("packaging_complete", job_id=job_id, output_srt_key=srt_key)
    return job_id


@celery_app.task(bind=True)
def run_pipeline(self, job_id: str):
    workflow = chain(
        diarize_stage.si(job_id),
        transcribe_stage.si(job_id),
        translate_stage.si(job_id),
        synthesize_stage.si(job_id),
        sync_av_stage.si(job_id),
        package_output_stage.si(job_id),
    )
    workflow.apply_async(link_error=mark_failed.s(job_id))


@celery_app.task
def mark_failed(request, exc, traceback, job_id: str):
    _set_status(job_id, "failed", error=str(exc))
    log.error("pipeline_failed", job_id=job_id, error=str(exc))