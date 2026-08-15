"""
Shared rate-limiting infrastructure (slowapi).

The limit string is resolved through a callable at request time so the limit
stays live-configurable through any config channel (env / file / secrets /
cloud) and is trivially overridable in tests.
"""
from __future__ import annotations

import re

from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import get_settings

_WINDOW_RE = re.compile(r"(\d+)\s+per\s+(second|minute|hour|day)")
_UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def default_rate_limit() -> str:
    return f"{get_settings().rate_limit_per_minute}/minute"


def retry_after_seconds(limit_detail: str) -> int:
    """Derive a Retry-After header from slowapi's limit detail
    (e.g. "10 per 1 minute" -> 60)."""
    match = _WINDOW_RE.search(limit_detail)
    if not match:
        return 60
    return _UNIT_SECONDS[match.group(2)]


limiter = Limiter(key_func=get_remote_address)
