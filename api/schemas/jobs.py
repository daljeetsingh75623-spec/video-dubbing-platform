from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class JobCreateResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    target_language: str


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    source_language: str | None
    target_language: str
    retry_count: int
    error_message: str | None
    created_at: datetime
    updated_at: datetime


class TranscriptSegmentResponse(BaseModel):
    speaker_label: str | None
    start_ms: int
    end_ms: int
    source_text: str
    translated_text: str | None


class TranscriptResponse(BaseModel):
    job_id: uuid.UUID
    segments: list[TranscriptSegmentResponse]


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["Unsupported video format: .wmv"])