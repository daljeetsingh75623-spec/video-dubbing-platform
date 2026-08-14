from __future__ import annotations

from fastapi import HTTPException, UploadFile

try:
    import magic
except ImportError:  # libmagic is not available (e.g. on Windows)
    magic = None

_ALLOWED_MIME_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "application/octet-stream",
}


def validate_extension(filename: str, allowed_formats: list[str]) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '.{ext}'. Allowed: {', '.join(allowed_formats)}",
        )
    return ext


def validate_size(size_bytes: int, max_size_mb: int) -> None:
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_bytes / 1024 / 1024:.1f}MB). Max is {max_size_mb}MB.",
        )


def validate_mime_sniff(header_bytes: bytes) -> None:
    if magic is None:
        return
    detected = magic.from_buffer(header_bytes, mime=True)
    if detected not in _ALLOWED_MIME_TYPES and not detected.startswith("video/"):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not look like a video (detected: {detected}).",
        )


async def read_and_validate_upload(
    file: UploadFile,
    allowed_formats: list[str],
    max_size_mb: int,
) -> bytes:
    validate_extension(file.filename or "", allowed_formats)

    header = await file.read(4096)
    validate_mime_sniff(header)

    rest = await file.read()
    content = header + rest
    validate_size(len(content), max_size_mb)

    await file.seek(0)
    return content