from __future__ import annotations

import shutil
from pathlib import Path

from storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Filesystem-backed storage rooted at `base_path`. Dev/test default."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        p = (self.base_path / key).resolve()
        if self.base_path.resolve() not in p.parents and p != self.base_path.resolve():
            raise ValueError(f"Invalid storage key (path traversal): {key}")
        return p

    async def upload(self, local_path: str | Path, key: str) -> str:
        dest = self._resolve(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, dest)
        return key

    async def download(self, key: str, local_path: str | Path) -> str:
        src = self._resolve(key)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, local_path)
        return str(local_path)

    async def get_url(self, key: str, expires_seconds: int = 3600) -> str:
        return self._resolve(key).as_uri()

    async def delete(self, key: str) -> None:
        p = self._resolve(key)
        if p.exists():
            p.unlink()

    async def exists(self, key: str) -> bool:
        return self._resolve(key).exists()