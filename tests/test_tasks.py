from __future__ import annotations

import uuid

from workers.tasks import _build_srt, _format_srt_timestamp


def test_format_srt_timestamp():
    assert _format_srt_timestamp(0) == "00:00:00,000"
    assert _format_srt_timestamp(1_000) == "00:00:01,000"
    assert _format_srt_timestamp(61_234) == "00:01:01,234"
    assert _format_srt_timestamp(3_661_234) == "01:01:01,234"
    assert _format_srt_timestamp(-500) == "00:00:00,000"


def test_build_srt_uses_translated_text_when_available():
    record = _make_record(start_ms=0, end_ms=1500, source="hello", translated="hola")
    srt = _build_srt([record])

    assert "1" in srt.splitlines()[0]
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "hola" in srt
    assert "hello" not in srt


def test_build_srt_falls_back_to_source_text():
    record = _make_record(start_ms=500, end_ms=900, source="bonjour", translated=None)
    srt = _build_srt([record])

    assert "bonjour" in srt
    assert "00:00:00,500 --> 00:00:00,900" in srt


def _make_record(start_ms, end_ms, source, translated):
    from db.models import TranscriptRecord

    return TranscriptRecord(
        id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        speaker_label="SPEAKER_00",
        start_ms=start_ms,
        end_ms=end_ms,
        source_text=source,
        translated_text=translated,
        confidence=0.99,
    )
