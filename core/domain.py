"""
Core domain models shared across all provider interfaces and pipeline stages.

Keeping these in one place (rather than duplicating per-provider) is what lets
providers/* stay swappable: every STTProvider implementation must return a
Transcript, every TranslationProvider must accept and return one, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    TRANSLATING = "translating"
    SYNTHESIZING = "synthesizing"
    SYNCING = "syncing"
    PACKAGING = "packaging"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SpeakerSegment:
    """A single contiguous span of time attributed to one speaker."""
    speaker_id: str          # stable id within the job, e.g. "SPEAKER_00"
    start_ms: int
    end_ms: int


@dataclass
class DiarizationResult:
    segments: list[SpeakerSegment] = field(default_factory=list)
    num_speakers: int = 0


@dataclass
class TranscriptSegment:
    speaker_id: str | None
    start_ms: int
    end_ms: int
    text: str
    language: str | None = None      # detected language code, e.g. "en"
    confidence: float | None = None


@dataclass
class Transcript:
    segments: list[TranscriptSegment] = field(default_factory=list)
    source_language: str | None = None

    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


@dataclass
class TranslatedSegment:
    speaker_id: str | None
    start_ms: int
    end_ms: int
    source_text: str
    translated_text: str
    target_language: str


@dataclass
class TranslatedTranscript:
    segments: list[TranslatedSegment] = field(default_factory=list)
    target_language: str = ""


@dataclass
class SynthesizedSegment:
    speaker_id: str | None
    start_ms: int
    end_ms: int
    audio_path: str          # path/key of the synthesized audio chunk in storage
    target_language: str


@dataclass
class SynthesisResult:
    segments: list[SynthesizedSegment] = field(default_factory=list)