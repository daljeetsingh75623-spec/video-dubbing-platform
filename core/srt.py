"""SRT subtitle file generation from timestamped, translated transcript segments."""
from __future__ import annotations


def _ms_to_srt_timestamp(ms: int) -> str:
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def build_srt(segments: list[dict]) -> str:
    """
    Args:
        segments: list of dicts with start_ms, end_ms, text (use
            translated_text if available, else source_text — caller decides
            which language the SRT should be in).

    Returns:
        Full .srt file content as a string.
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start = _ms_to_srt_timestamp(seg["start_ms"])
        end = _ms_to_srt_timestamp(seg["end_ms"])
        speaker_prefix = f"[{seg['speaker_label']}] " if seg.get("speaker_label") else ""
        lines.append(f"{i}\n{start} --> {end}\n{speaker_prefix}{seg['text']}\n")
    return "\n".join(lines)