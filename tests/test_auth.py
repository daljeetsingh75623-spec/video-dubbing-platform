from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.auth import require_api_key


@pytest.mark.asyncio
async def test_auth_disabled_allows_no_key(monkeypatch):
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("API_KEY_AUTH_ENABLED", "false")
    await require_api_key(x_api_key=None)  # should not raise
    settings_module.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_auth_enabled_rejects_missing_key(monkeypatch):
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("API_KEY_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "secret123")
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401
    settings_module.get_settings.cache_clear()


@pytest.mark.asyncio
async def test_auth_enabled_accepts_correct_key(monkeypatch):
    from config import settings as settings_module
    settings_module.get_settings.cache_clear()
    monkeypatch.setenv("API_KEY_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "secret123")
    await require_api_key(x_api_key="secret123")  # should not raise
    settings_module.get_settings.cache_clear()