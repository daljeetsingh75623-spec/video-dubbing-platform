from __future__ import annotations

import pytest

from core.av_sync import build_composite_audio_track
from core.domain import SynthesizedSegment


def _seg(speaker, start_ms, end_ms, audio_path):
    return SynthesizedSegment(
        speaker_id=speaker, start_ms=start_ms, end_ms=end_ms,
        audio_path=audio_path, target_language="es",
    )


def _build(monkeypatch, audio_ms, window_ms, has_rubberband=True, extra_overrun_ms=0):
    """Runs build_composite_audio_track with faked durations and returns the
    ffmpeg filter_complex string (without running real ffmpeg)."""
    captured = {}

    async def _fake_duration(path):
        return audio_ms + extra_overrun_ms

    async def _fake_rubberband_check():
        return has_rubberband

    async def _fake_run_ffmpeg(cmd):
        captured["cmd"] = cmd

    monkeypatch.setattr("core.av_sync._get_audio_duration_ms", _fake_duration)
    monkeypatch.setattr("core.av_sync._check_rubberband", _fake_rubberband_check)
    monkeypatch.setattr("core.av_sync._run_ffmpeg", _fake_run_ffmpeg)

    segment = _seg("SPEAKER_00", 1000, 1000 + window_ms, "chunk.mp3")
    coro = build_composite_audio_track(
        [segment], {"chunk.mp3": "/tmp/chunk.mp3"}, 10_000, "/tmp/out.wav",
    )
    return coro, captured


@pytest.mark.asyncio
async def test_short_chunk_not_stretched_or_slowed(monkeypatch):
    # audio shorter than window -> leave at natural pace, just delay it.
    coro, captured = _build(monkeypatch, audio_ms=1_500, window_ms=3_000)
    await coro
    chain = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
    assert "rubberband" not in chain
    assert "atrim" not in chain
    assert "adelay=1000|1000" in chain


@pytest.mark.asyncio
async def test_long_chunk_stretched_to_fit(monkeypatch):
    # audio longer than window -> stretch just enough (no hard trim needed).
    coro, captured = _build(monkeypatch, audio_ms=2_400, window_ms=2_000)
    await coro
    chain = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
    assert "rubberband=tempo=1.200" in chain
    assert "atrim" not in chain


@pytest.mark.asyncio
async def test_extreme_overrun_capped_then_trimmed(monkeypatch):
    # audio far longer than the window -> capped speedup + hard-trim tail.
    coro, captured = _build(monkeypatch, audio_ms=6_000, window_ms=2_000)
    await coro
    chain = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
    assert "rubberband=tempo=1.350" in chain
    assert "atrim=end=2.000" in chain


@pytest.mark.asyncio
async def test_no_rubberband_falls_back_to_trim(monkeypatch):
    # ffmpeg build without rubberband -> trim the overflow instead of erroring.
    coro, captured = _build(monkeypatch, audio_ms=3_000, window_ms=2_000, has_rubberband=False)
    await coro
    chain = captured["cmd"][captured["cmd"].index("-filter_complex") + 1]
    assert "rubberband" not in chain
    assert "atrim=end=2.000" in chain
