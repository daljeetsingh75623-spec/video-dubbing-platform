from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from storage.local import LocalStorageBackend


@pytest.fixture
def backend(tmp_path):
    return LocalStorageBackend(base_path=str(tmp_path))


@pytest.mark.asyncio
async def test_upload_and_download_roundtrip(backend, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello world")

    key = await backend.upload(src, "uploads/test.txt")
    assert key == "uploads/test.txt"
    assert await backend.exists(key)

    dest = tmp_path / "dest.txt"
    await backend.download(key, dest)
    assert dest.read_text() == "hello world"


@pytest.mark.asyncio
async def test_delete_removes_file(backend, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("data")
    key = await backend.upload(src, "uploads/del.txt")
    assert await backend.exists(key)

    await backend.delete(key)
    assert not await backend.exists(key)


@pytest.mark.asyncio
async def test_exists_false_for_missing_key(backend):
    assert not await backend.exists("nope/missing.txt")


@pytest.mark.asyncio
async def test_path_traversal_rejected(backend):
    with pytest.raises(ValueError, match="path traversal"):
        backend._resolve("../../etc/passwd")


@pytest.mark.asyncio
async def test_get_url_returns_file_uri(backend, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("x")
    key = await backend.upload(src, "uploads/u.txt")
    url = await backend.get_url(key)
    assert url.startswith("file://")