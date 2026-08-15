import asyncio
import os
import tempfile
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask
from api.auth import require_api_key
from api.rate_limit import default_rate_limit, limiter

router = APIRouter(prefix="/videos", tags=["videos"], dependencies=[Depends(require_api_key)])

from api.schemas.jobs import (
    JobCreateResponse,
    JobStatusResponse,
    TranscriptResponse,
    TranscriptSegmentResponse,
)
from api.validation import read_and_validate_upload, validate_duration, validate_target_language
from config.settings import get_settings
from core import metrics
from db.models import AuditLog, Job, JobEvent, TranscriptRecord
from db.session import get_db
from storage.factory import get_storage_backend


async def _enqueue_or_fail(db: AsyncSession, job: Job, action: str) -> None:
    """Queue the pipeline; if enqueueing fails the job is marked failed
    instead of silently dangling in 'queued' (and 500ing the client)."""
    from workers.tasks import dispatch_pipeline

    try:
        dispatch_pipeline(str(job.id))
    except Exception as e:  # noqa: BLE001 - any broker/connectivity error
        job.status = "failed"
        job.error_message = f"Failed to enqueue pipeline: {e}"
        job.updated_at = datetime.now(timezone.utc)
        db.add(JobEvent(id=uuid.uuid4(), job_id=job.id, stage="failed", message=str(e)))
        db.add(AuditLog(id=uuid.uuid4(), job_id=job.id, action=action, actor="anonymous"))
        await db.commit()


@router.post("", response_model=JobCreateResponse, status_code=201)
@limiter.limit(default_rate_limit)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    target_language: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    settings = get_settings()
    target_language = validate_target_language(target_language)

    # Queue capacity gate: refuse new uploads once the queued backlog is at
    # max_queue_length, so the queue can't grow unboundedly.
    queued = (
        await db.execute(
            select(func.count()).select_from(Job).where(Job.status == "queued")
        )
    ).scalar_one()
    if queued >= settings.queue_max_length:
        metrics.inc(metrics.upload_rejections_total, "queue_full")
        raise HTTPException(
            status_code=429,
            detail="Server is at capacity — the queue is full. Try again later.",
            headers={"Retry-After": "60"},
        )

    # Upload concurrency gate: bound how many uploads are validated, probed and
    # stored at once. Returns 429 immediately when at capacity (retry later).
    upload_semaphore: asyncio.Semaphore = request.app.state.upload_semaphore
    if upload_semaphore.locked():
        metrics.inc(metrics.upload_rejections_total, "concurrency")
        raise HTTPException(
            status_code=429,
            detail="Too many concurrent uploads. Try again shortly.",
            headers={"Retry-After": "5"},
        )

    metrics.inc(metrics.uploads_active)
    try:
        async with upload_semaphore:
            content = await read_and_validate_upload(
                file, settings.allowed_video_formats, settings.max_upload_size_mb
            )

            job_id = uuid.uuid4()
            storage = get_storage_backend()
            filename = file.filename or ""
            ext = filename.rsplit(".", 1)[-1].lower()
            key = f"uploads/{job_id}/source.{ext}"

            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                await validate_duration(tmp_path, settings.max_video_duration_seconds)
                await storage.upload(tmp_path, key)
            finally:
                if tmp_path:
                    os.unlink(tmp_path)

            job = Job(
                id=job_id,
                status="queued",
                source_video_key=key,
                target_language=target_language,
                stt_provider=settings.stt_provider,
                translation_provider=settings.translation_provider,
                tts_provider=settings.tts_provider,
                diarization_provider=settings.diarization_provider,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            db.add(job)
            db.add(JobEvent(id=uuid.uuid4(), job_id=job_id, stage="queued", message="Job created"))
            db.add(AuditLog(id=uuid.uuid4(), job_id=job_id, action="upload", actor="anonymous",
                             detail={"filename": file.filename, "target_language": target_language}))
            await db.commit()
    finally:
        metrics.dec(metrics.uploads_active)

    metrics.inc(metrics.jobs_created_total)
    await _enqueue_or_fail(db, job, action="upload")

    return JobCreateResponse(job_id=job_id, status=job.status, target_language=target_language)

@router.get("")
@limiter.limit(default_rate_limit)
async def list_jobs(request: Request, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).order_by(Job.created_at.desc()).limit(10))
    jobs = result.scalars().all()
    return [
        {
            "job_id": str(j.id),
            "status": j.status,
            "source_language": j.source_language,
            "target_language": j.target_language,
            "error_message": j.error_message,
            "created_at": j.created_at,
        }
        for j in jobs
    ]


@router.get("/{job_id}/status", response_model=JobStatusResponse)
@limiter.limit(default_rate_limit)
async def get_status(request: Request, job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job.id, status=job.status, source_language=job.source_language,
        target_language=job.target_language, retry_count=job.retry_count,
        error_message=job.error_message, created_at=job.created_at, updated_at=job.updated_at,
    )


@router.get("/{job_id}/transcript", response_model=TranscriptResponse)
@limiter.limit(default_rate_limit)
async def get_transcript(request: Request, job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    result = await db.execute(
        select(TranscriptRecord).where(TranscriptRecord.job_id == job_id).order_by(TranscriptRecord.start_ms)
    )
    segments = result.scalars().all()
    return TranscriptResponse(
        job_id=job_id,
        segments=[
            TranscriptSegmentResponse(
                speaker_label=s.speaker_label, start_ms=s.start_ms, end_ms=s.end_ms,
                source_text=s.source_text, translated_text=s.translated_text,
            )
            for s in segments
        ],
    )


@router.get("/{job_id}/download")
@limiter.limit(default_rate_limit)
async def download_video(request: Request, job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.output_video_key:
        raise HTTPException(409, f"Job is not completed yet (status: {job.status})")
    storage = get_storage_backend()
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    await storage.download(job.output_video_key, tmp_path)
    return FileResponse(
        tmp_path,
        media_type="video/mp4",
        filename="dubbed.mp4",
        content_disposition_type="inline",
        background=BackgroundTask(os.unlink, tmp_path),
    )


@router.get("/{job_id}/subtitles")
@limiter.limit(default_rate_limit)
async def download_subtitles(request: Request, job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.output_srt_key:
        raise HTTPException(409, "Subtitles not available yet")
    storage = get_storage_backend()
    with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
        tmp_path = tmp.name
    await storage.download(job.output_srt_key, tmp_path)
    return FileResponse(
        tmp_path,
        media_type="application/x-subrip",
        filename="subtitles.srt",
        background=BackgroundTask(os.unlink, tmp_path),
    )


@router.post("/{job_id}/retry", response_model=JobStatusResponse)
@limiter.limit(default_rate_limit)
async def retry_job(request: Request, job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in ("failed", "cancelled"):
        raise HTTPException(409, f"Cannot retry a job in status '{job.status}'")

    settings = get_settings()
    if job.retry_count >= settings.retry_count:
        raise HTTPException(409, "Max retry count exceeded")

    job.status = "queued"
    job.error_message = None
    job.retry_count += 1
    job.updated_at = datetime.now(timezone.utc)
    db.add(JobEvent(id=uuid.uuid4(), job_id=job_id, stage="queued", message="Retry requested"))
    db.add(AuditLog(id=uuid.uuid4(), job_id=job_id, action="retry", actor="anonymous"))
    await db.commit()

    await _enqueue_or_fail(db, job, action="retry")

    return JobStatusResponse(
        job_id=job.id, status=job.status, source_language=job.source_language,
        target_language=job.target_language, retry_count=job.retry_count,
        error_message=job.error_message, created_at=job.created_at, updated_at=job.updated_at,
    )


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
@limiter.limit(default_rate_limit)
async def cancel_job(request: Request, job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in ("completed", "cancelled"):
        raise HTTPException(409, f"Cannot cancel a job in status '{job.status}'")

    job.status = "cancelled"
    job.updated_at = datetime.now(timezone.utc)
    db.add(JobEvent(id=uuid.uuid4(), job_id=job_id, stage="cancelled", message="Cancelled by user"))
    db.add(AuditLog(id=uuid.uuid4(), job_id=job_id, action="cancel", actor="anonymous"))
    await db.commit()

    return JobStatusResponse(
        job_id=job.id, status=job.status, source_language=job.source_language,
        target_language=job.target_language, retry_count=job.retry_count,
        error_message=job.error_message, created_at=job.created_at, updated_at=job.updated_at,
    )