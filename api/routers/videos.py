from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas.jobs import (
    JobCreateResponse,
    JobStatusResponse,
    TranscriptResponse,
    TranscriptSegmentResponse,
)
from api.validation import read_and_validate_upload
from config.settings import get_settings
from db.models import AuditLog, Job, JobEvent, TranscriptRecord
from db.session import get_db
from storage.factory import get_storage_backend

router = APIRouter(prefix="/videos", tags=["videos"])


@router.post("", response_model=JobCreateResponse, status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    target_language: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    settings = get_settings()
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

    from workers.tasks import run_pipeline
    run_pipeline.delay(str(job_id))

    return JobCreateResponse(job_id=job_id, status=job.status, target_language=target_language)


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_status(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job.id, status=job.status, source_language=job.source_language,
        target_language=job.target_language, retry_count=job.retry_count,
        error_message=job.error_message, created_at=job.created_at, updated_at=job.updated_at,
    )


@router.get("/{job_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
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
async def download_video(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "completed" or not job.output_video_key:
        raise HTTPException(409, f"Job is not completed yet (status: {job.status})")
    storage = get_storage_backend()
    url = await storage.get_url(job.output_video_key)
    return {"url": url}


@router.get("/{job_id}/subtitles")
async def download_subtitles(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.output_srt_key:
        raise HTTPException(409, "Subtitles not available yet")
    storage = get_storage_backend()
    url = await storage.get_url(job.output_srt_key)
    return {"url": url}


@router.post("/{job_id}/retry", response_model=JobStatusResponse)
async def retry_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
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

    from workers.tasks import run_pipeline
    run_pipeline.delay(str(job_id))

    return JobStatusResponse(
        job_id=job.id, status=job.status, source_language=job.source_language,
        target_language=job.target_language, retry_count=job.retry_count,
        error_message=job.error_message, created_at=job.created_at, updated_at=job.updated_at,
    )


@router.post("/{job_id}/cancel", response_model=JobStatusResponse)
async def cancel_job(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
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