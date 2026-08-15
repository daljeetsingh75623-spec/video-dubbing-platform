"""Security enforcement: rate limiting actually rejects on breach, and
target_language is validated as input."""
from __future__ import annotations

import io

import pytest
from fastapi import HTTPException

from api.validation import validate_target_language


@pytest.mark.asyncio
async def test_rate_limit_enforced_and_config_driven(client, monkeypatch):
    import api.rate_limit as rate_limit_module
    from config.settings import Settings

    low = Settings()
    low.rate_limit_per_minute = 1
    monkeypatch.setattr(rate_limit_module, "get_settings", lambda: low)

    assert (await client.get("/health")).status_code == 200

    breaching = await client.get("/health")
    assert breaching.status_code == 429
    assert breaching.headers.get("retry-after")


@pytest.mark.asyncio
async def test_upload_rejects_bad_target_language(client):
    files = {"file": ("clip.mp4", io.BytesIO(b"not a real video"), "video/mp4")}
    resp = await client.post("/videos", files=files, data={"target_language": "not-a-language!"})
    assert resp.status_code == 400
    assert "language code" in resp.json()["detail"].lower()


def test_validate_target_language_accepts_iso_codes():
    assert validate_target_language("es") == "es"
    assert validate_target_language("ES") == "ES"
    assert validate_target_language("zh-CN") == "zh-CN"
    assert validate_target_language("pt-BR") == "pt-BR"


def test_validate_target_language_rejects_invalid():
    for bad in ("", "e", "english", "en_US_extra", "es!", "123"):
        with pytest.raises(HTTPException) as excinfo:
            validate_target_language(bad)
        assert excinfo.value.status_code == 400
