from __future__ import annotations

from fastapi import Header, HTTPException

from config.settings import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """
    FastAPI dependency: enforces X-API-Key header when api_key_auth_enabled
    is set. No-op (open) when auth is disabled — keeps local dev/demo
    frictionless while making prod-mode auth a one-line config flip.
    """
    settings = get_settings()
    if not settings.api_key_auth_enabled:
        return
    if not settings.api_key:
        raise HTTPException(500, "API key auth is enabled but no API_KEY is configured")
    if x_api_key != settings.api_key:
        raise HTTPException(401, "Invalid or missing API key")