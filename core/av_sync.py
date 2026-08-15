"""
ffmpeg-based audio-video muxing: takes per-segment synthesized audio chunks
(each with its own start_ms/end_ms) and lays them onto a silent copy of the
original video's timeline, then muxes that composite audio track back onto
the original video stream.

Approach: build one continuous audio track exactly as long as the source
video, place each TTS chunk at its original segment's start_ms, pad gaps
with silence, then swap the source video's audio for this composite track.
Chunks that would overflow their window are pitch-preserving time-stretched
only as much as needed (never slowed down) and hard-trimmed if still too
long, so dubbed speech keeps a natural pace and segments never overlap.
Real studio dubbing bends this further with per-sentence re-timing; noted
as a known limitation in the architecture doc.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from core.domain import SynthesizedSegment

# Never stretch a chunk faster than this — higher values make speech sound
# rushed. If a chunk still overflows at this cap it is hard-trimmed to its
# window instead (see build_composite_audio_track), which guarantees no
# overlap at the cost of a tiny amount of chopped audio at the tail.
MAX_SPEEDUP = 1.35

_HAS_RUBBERBAND: bool | None = None


async def _check_rubberband() -> bool:
    """Lazily probe whether the local ffmpeg build has the `rubberband`
    (pitch-preserving) filter. Some minimal builds omit librubberband; callers
    then fall back to hard-trimming instead of crashing."""
    global _HAS_RUBBERBAND
    if _HAS_RUBBERBAND is not None:
        return _HAS_RUBBERBAND
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-filters",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        _HAS_RUBBERBAND = b"rubberband" in stdout
    except Exception:
        _HAS_RUBBERBAND = False
    return _HAS_RUBBERBAND


async def get_video_duration_ms(video_path: str) -> int:
    """ffprobe the source video's duration."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {stderr.decode()}")
    return int(float(stdout.decode().strip()) * 1000)


async def _get_audio_duration_ms(audio_path: str) -> int:
    """ffprobe a single audio chunk's duration."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {stderr.decode()}")
    return int(float(stdout.decode().strip()) * 1000)


async def build_composite_audio_track(
    segments: list[SynthesizedSegment],
    local_audio_paths: dict[str, str],   # audio_path key -> local file path
    total_duration_ms: int,
    output_path: str,
) -> str:
    """
    Builds one WAV file spanning total_duration_ms, with each segment's
    audio placed at its start_ms offset via ffmpeg's `adelay` + `amix`.

    Timing rules (chosen so dubbed speech stays natural):
    - A chunk that fits its window is placed as-is (natural pace, no
      slow-down — slowing speech makes it sound unnatural).
    - A chunk longer than its window is pitch-preserving time-stretched
      (rubberband) at the smallest rate needed to avoid bleeding into the
      next segment, capped at MAX_SPEEDUP.
    - If even the capped stretch still overruns, the tail is hard-trimmed
      (atrim) so chunks can never overlap.
    """
    if not segments:
        # No dubbed audio at all — emit silence for the full duration so
        # sync_av_stage still has something valid to mux.
        cmd = [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"anullsrc=r=44100:cl=stereo",
            "-t", str(total_duration_ms / 1000),
            output_path,
        ]
        await _run_ffmpeg(cmd)
        return output_path

    segments = sorted(segments, key=lambda s: s.start_ms)
    has_rubberband = await _check_rubberband()

    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, seg in enumerate(segments):
        local_path = local_audio_paths[seg.audio_path]
        inputs += ["-i", local_path]
        window_ms = max(seg.end_ms - seg.start_ms, 1)
        audio_ms = await _get_audio_duration_ms(local_path)

        chain = ""
        if audio_ms > window_ms:
            tempo = audio_ms / window_ms
            if has_rubberband and abs(tempo - 1.0) > 0.03:
                # Stretch just enough to fit, capped for naturalness.
                tempo = min(tempo, MAX_SPEEDUP)
                chain = f"rubberband=tempo={tempo:.3f},asetpts=PTS-STARTPTS,"
                if audio_ms / tempo > window_ms:
                    chain += f"atrim=end={window_ms / 1000:.3f},asetpts=PTS-STARTPTS,"
            else:
                # No rubberband available (or overrun is negligible):
                # hard-trim the chunk to its window so it can't overlap.
                chain = f"atrim=end={window_ms / 1000:.3f},asetpts=PTS-STARTPTS,"

        # adelay pads the start of this clip with silence up to start_ms
        filter_parts.append(f"[{i}:a]{chain}adelay={seg.start_ms}|{seg.start_ms}[a{i}]")

    mix_inputs = "".join(f"[a{i}]" for i in range(len(segments)))
    filter_complex = ";".join(filter_parts) + (
        f";{mix_inputs}amix=inputs={len(segments)}:duration=longest:normalize=0,apad[aout]"
    )

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-t", str(total_duration_ms / 1000),
        output_path,
    ]
    await _run_ffmpeg(cmd)
    return output_path


async def mux_audio_onto_video(video_path: str, audio_path: str, output_path: str) -> str:
    """Replaces the source video's audio track with the composite dub track."""
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",       # don't re-encode video — fast, lossless
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path,
    ]
    await _run_ffmpeg(cmd)
    return output_path


async def _run_ffmpeg(cmd: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({' '.join(cmd)}):\n{stderr.decode()[-2000:]}")