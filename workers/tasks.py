from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid

import structlog
from celery import chain, signature
from sqlalchemy import delete, select

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


_DONE_BY_STAGE = {
    "diarize": "diarized",
    "transcribe": "transcribed",
    "translate": "translated",
    "synthesize": "synthesized",
    "sync_av": "synced",
    "package": "completed",
}
_DONE_ORDER = ["diarized", "transcribed", "translated", "synthesized", "synced", "completed"]


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


def _stage_done(job_id: str, stage: str) -> bool:
    done = _DONE_BY_STAGE.get(stage)
    if done is None:
        return False
    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
    if not job:
        return True
    if job.status == "completed":
        return True
    if job.status in ("queued", "failed", "cancelled"):
        return False
    try:
        return _DONE_ORDER.index(job.status) >= _DONE_ORDER.index(done)
    except ValueError:
        return False


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
    if _stage_done(job_id, "diarize"):
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
    _set_status(job_id, "diarized", message=f"num_speakers={result.num_speakers}")
    log.info("diarization_complete", job_id=job_id, num_speakers=result.num_speakers)
    return job_id


@celery_app.task(**_TASK_KWARGS)
def transcribe_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    if _stage_done(job_id, "transcribe"):
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
        db.execute(delete(TranscriptRecord).where(TranscriptRecord.job_id == job.id))
        job.source_language = transcript.source_language
        for seg in transcript.segments:
            db.add(TranscriptRecord(
                id=uuid.uuid4(), job_id=job.id, speaker_label=seg.speaker_id,
                start_ms=seg.start_ms, end_ms=seg.end_ms,
                source_text=seg.text, confidence=seg.confidence,
            ))
        db.commit()
    _set_status(job_id, "transcribed", message=f"segments={len(transcript.segments)}")
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
    if _stage_done(job_id, "translate"):
        return job_id
    _set_status(job_id, "translating")

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if not job:
            return job_id
        records = list(
            db.scalars(
                select(TranscriptRecord)
                .where(TranscriptRecord.job_id == job.id)
                .order_by(TranscriptRecord.start_ms)
            ).all()
        )
        transcript = Transcript(
            segments=[
                TranscriptSegment(
                    speaker_id=r.speaker_label, start_ms=r.start_ms, end_ms=r.end_ms,
                    text=r.source_text, confidence=r.confidence,
                )
                for r in records
            ],
            source_language=job.source_language,
        )
        target_language = job.target_language
        provider_name = job.translation_provider

    provider = ProviderFactory.get_translation(provider_name)
    try:
        translated = _run_async(provider.translate(transcript, target_language))
    except ProviderError as e:
        log.error("translation_failed", job_id=job_id, error=str(e))
        raise

    with get_sync_db() as db:
        records = list(
            db.scalars(
                select(TranscriptRecord)
                .where(TranscriptRecord.job_id == uuid.UUID(job_id))
                .order_by(TranscriptRecord.start_ms)
            ).all()
        )
        for record, seg in zip(records, translated.segments):
            record.translated_text = seg.translated_text
        db.commit()

    _set_status(job_id, "translated", message=f"segments={len(translated.segments)}")
    log.info("translation_complete", job_id=job_id, num_segments=len(translated.segments))
    return job_id


@celery_app.task(**_TASK_KWARGS)
def synthesize_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    if _stage_done(job_id, "synthesize"):
        return job_id
    _set_status(job_id, "synthesizing")

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if not job:
            return job_id
        records = list(
            db.scalars(
                select(TranscriptRecord)
                .where(TranscriptRecord.job_id == job.id)
                .order_by(TranscriptRecord.start_ms)
            ).all()
        )
        translated = TranslatedTranscript(
            segments=[
                TranslatedSegment(
                    speaker_id=r.speaker_label, start_ms=r.start_ms, end_ms=r.end_ms,
                    source_text=r.source_text,
                    translated_text=r.translated_text or r.source_text,
                    target_language=job.target_language,
                )
                for r in records
            ],
            target_language=job.target_language,
        )
        provider_name = job.tts_provider

    provider = ProviderFactory.get_tts(provider_name)
    try:
        synthesis = _run_async(provider.synthesize(translated))
    except ProviderError as e:
        log.error("synthesis_failed", job_id=job_id, error=str(e))
        raise

    _set_status(
        job_id,
        "synthesized",
        message=json.dumps([
            {"speaker_id": s.speaker_id, "start_ms": s.start_ms, "end_ms": s.end_ms,
             "audio_path": s.audio_path}
            for s in synthesis.segments
        ]),
    )
    log.info("synthesis_complete", job_id=job_id, num_segments=len(synthesis.segments))
    return job_id

@celery_app.task(**_TASK_KWARGS)
def sync_av_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "syncing")

    import json
    import tempfile

    from core.av_sync import build_composite_audio_track, get_video_duration_ms, mux_audio_onto_video
    from db.models import JobEvent
    from storage.factory import get_storage_backend

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        source_video_key = job.source_video_key

        # Pull the synthesized-segment list written by synthesize_stage.
        event = (
            db.query(JobEvent)
            .filter(JobEvent.job_id == job.id, JobEvent.stage == "synthesized")
            .order_by(JobEvent.created_at.desc())
            .first()
        )
        synth_segments = json.loads(event.message) if event and event.message else []

    storage = get_storage_backend()

    async def _run():
        local_video = f"/tmp/{job_id}_source.mp4"
        await storage.download(source_video_key, local_video)
        duration_ms = await get_video_duration_ms(local_video)

        local_audio_paths: dict[str, str] = {}
        for seg in synth_segments:
            local_path = f"/tmp/{job_id}_{seg['audio_path'].replace('/', '_')}"
            await storage.download(seg["audio_path"], local_path)
            local_audio_paths[seg["audio_path"]] = local_path

        from core.domain import SynthesizedSegment
        seg_objs = [
            SynthesizedSegment(
                speaker_id=s["speaker_id"], start_ms=s["start_ms"], end_ms=s["end_ms"],
                audio_path=s["audio_path"], target_language="",
            )
            for s in synth_segments
        ]

        composite_audio_path = f"/tmp/{job_id}_composite.wav"
        await build_composite_audio_track(seg_objs, local_audio_paths, duration_ms, composite_audio_path)

        output_video_path = f"/tmp/{job_id}_dubbed.mp4"
        await mux_audio_onto_video(local_video, composite_audio_path, output_video_path)

        output_key = f"outputs/{job_id}/dubbed.mp4"
        await storage.upload(output_video_path, output_key)
        return output_key

    output_video_key = _run_async(_run())

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.output_video_key = output_video_key
        db.commit()

    return job_id


@celery_app.task(**_TASK_KWARGS)
def package_output_stage(self, job_id: str) -> str:
    if _is_cancelled(job_id):
        return job_id
    _set_status(job_id, "packaging")

    import tempfile

    from core.srt import build_srt
    from storage.factory import get_storage_backend

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        records = (
            db.query(TranscriptRecord)
            .filter(TranscriptRecord.job_id == job.id)
            .order_by(TranscriptRecord.start_ms)
            .all()
        )
        srt_segments = [
            {
                "start_ms": r.start_ms, "end_ms": r.end_ms,
                "speaker_label": r.speaker_label,
                "text": r.translated_text or r.source_text,
            }
            for r in records
        ]
        transcript_text = "\n".join(
            f"[{r.speaker_label or 'UNKNOWN'}] {r.source_text}"
            + (f" -> {r.translated_text}" if r.translated_text else "")
            for r in records
        )

    srt_content = build_srt(srt_segments)

    storage = get_storage_backend()

    async def _upload_outputs():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(srt_content)
            srt_path = f.name
        srt_key = f"outputs/{job_id}/subtitles.srt"
        await storage.upload(srt_path, srt_key)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(transcript_text)
            transcript_path = f.name
        transcript_key = f"outputs/{job_id}/transcript.txt"
        await storage.upload(transcript_path, transcript_key)

        return srt_key, transcript_key

    srt_key, transcript_key = _run_async(_upload_outputs())

    with get_sync_db() as db:
        job = db.get(Job, uuid.UUID(job_id))
        if job is None:
            raise ValueError(f"Job {job_id} not found")
        job.output_srt_key = srt_key
        job.output_transcript_key = transcript_key
        db.commit()

    _set_status(job_id, "completed", message="Pipeline finished")
    return job_id

@celery_app.task(bind=True)
def run_pipeline(self, job_id: str):
    workflow = chain(
        signature("workers.tasks.diarize_stage", args=[job_id], immutable=True),
        signature("workers.tasks.transcribe_stage", args=[job_id], immutable=True),
        signature("workers.tasks.translate_stage", args=[job_id], immutable=True),
        signature("workers.tasks.synthesize_stage", args=[job_id], immutable=True),
        signature("workers.tasks.sync_av_stage", args=[job_id], immutable=True),
        signature("workers.tasks.package_output_stage", args=[job_id], immutable=True),
    )
    workflow.apply_async(link_error=signature("workers.tasks.mark_failed", args=[job_id]))


@celery_app.task
def mark_failed(request, exc, traceback, job_id: str):
    _set_status(job_id, "failed", error=str(exc))
    log.error("pipeline_failed", job_id=job_id, error=str(exc))