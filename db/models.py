from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


def _uuid_col():
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_col()
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    source_video_key: Mapped[str] = mapped_column(String(512))
    source_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    target_language: Mapped[str] = mapped_column(String(16))

    stt_provider: Mapped[str] = mapped_column(String(64))
    translation_provider: Mapped[str] = mapped_column(String(64))
    tts_provider: Mapped[str] = mapped_column(String(64))
    diarization_provider: Mapped[str] = mapped_column(String(64))

    output_video_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    output_srt_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    output_transcript_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    speakers: Mapped[list["Speaker"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    transcripts: Mapped[list["TranscriptRecord"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    events: Mapped[list["JobEvent"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class Speaker(Base):
    __tablename__ = "speakers"

    id: Mapped[uuid.UUID] = _uuid_col()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    speaker_label: Mapped[str] = mapped_column(String(32))
    total_speaking_ms: Mapped[int] = mapped_column(Integer, default=0)
    reference_audio_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    voice_id: Mapped[str | None] = mapped_column(String(128), nullable=True)

    job: Mapped[Job] = relationship(back_populates="speakers")


class TranscriptRecord(Base):
    __tablename__ = "transcripts"

    id: Mapped[uuid.UUID] = _uuid_col()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    speaker_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    start_ms: Mapped[int] = mapped_column(Integer)
    end_ms: Mapped[int] = mapped_column(Integer)
    source_text: Mapped[str] = mapped_column(Text)
    translated_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    job: Mapped[Job] = relationship(back_populates="transcripts")


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[uuid.UUID] = _uuid_col()
    job_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job: Mapped[Job] = relationship(back_populates="events")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = _uuid_col()
    job_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    detail: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    job: Mapped["Job"] = relationship()