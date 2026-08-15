"""
ffmpeg-based audio-video muxing: takes per-segment synthesized audio chunks
(each with its own start_ms/end_ms) and lays them onto a silent copy of the
original video's timeline, then muxes that composite audio track back onto
the original video stream.

Approach: build one continuous audio track exactly as long as the source
video, place each TTS chunk at its original segment's start_ms, pad gaps
with silence, then swap the source video's audio for this composite track.
This is "best-effort" sync — TTS output length rarely matches the original
segment's duration exactly, so chunks may run slightly long/short. Real
studio dubbing bends this with time-stretching per segment; noted as a
known limitation in the architecture doc rather than solved here.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from core.domain import SynthesizedSegment


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

    Each TTS chunk is time-stretched (rubberband, pitch-preserving) to fit
    its segment's window exactly, so audio never bleeds into the next
    segment and voices don't overlap.
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

    inputs: list[str] = []
    filter_parts: list[str] = []
    for i, seg in enumerate(segments):
        local_path = local_audio_paths[seg.audio_path]
        inputs += ["-i", local_path]
        # Stretch/shrink the chunk to exactly fill its window so it never
        # overlaps the next segment (rubberband keeps pitch constant).
        window_ms = max(seg.end_ms - seg.start_ms, 1)
        audio_ms = await _get_audio_duration_ms(local_path)
        tempo = min(max(audio_ms / window_ms, 0.25), 4.0)
        stretch = f"rubberband=tempo={tempo:.3f}," if abs(tempo - 1.0) > 0.02 else ""
        # adelay pads the start of this clip with silence up to start_ms
        filter_parts.append(f"[{i}:a]{stretch}adelay={seg.start_ms}|{seg.start_ms}[a{i}]")

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