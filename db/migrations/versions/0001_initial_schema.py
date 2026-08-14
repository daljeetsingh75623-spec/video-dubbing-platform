"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-08-14
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("source_video_key", sa.String(512), nullable=False),
        sa.Column("source_language", sa.String(16), nullable=True),
        sa.Column("target_language", sa.String(16), nullable=False),
        sa.Column("stt_provider", sa.String(64), nullable=False),
        sa.Column("translation_provider", sa.String(64), nullable=False),
        sa.Column("tts_provider", sa.String(64), nullable=False),
        sa.Column("diarization_provider", sa.String(64), nullable=False),
        sa.Column("output_video_key", sa.String(512), nullable=True),
        sa.Column("output_srt_key", sa.String(512), nullable=True),
        sa.Column("output_transcript_key", sa.String(512), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_jobs_status", "jobs", ["status"])

    op.create_table(
        "speakers",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaker_label", sa.String(32), nullable=False),
        sa.Column("total_speaking_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("reference_audio_key", sa.String(512), nullable=True),
        sa.Column("voice_id", sa.String(128), nullable=True),
    )
    op.create_index("ix_speakers_job_id", "speakers", ["job_id"])

    op.create_table(
        "transcripts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("speaker_label", sa.String(32), nullable=True),
        sa.Column("start_ms", sa.Integer(), nullable=False),
        sa.Column("end_ms", sa.Integer(), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
    )
    op.create_index("ix_transcripts_job_id", "transcripts", ["job_id"])

    op.create_table(
        "job_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_job_events_job_id", "job_events", ["job_id"])

    op.create_table(
        "audit_logs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("actor", sa.String(128), nullable=True),
        sa.Column("detail", postgresql.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_index("ix_job_events_job_id", table_name="job_events")
    op.drop_table("job_events")
    op.drop_index("ix_transcripts_job_id", table_name="transcripts")
    op.drop_table("transcripts")
    op.drop_index("ix_speakers_job_id", table_name="speakers")
    op.drop_table("speakers")
    op.drop_index("ix_jobs_status", table_name="jobs")
    op.drop_table("jobs")