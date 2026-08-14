"""
Common exception types raised by provider implementations.

Celery tasks catch these specifically (rather than bare Exception) to decide
retry vs. fail-fast behavior — see workers/tasks.py.
"""


class ProviderError(Exception):
    """Base class for all provider-level failures."""


class ProviderTimeoutError(ProviderError):
    """Raised when a provider call exceeds its configured timeout."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider signals rate limiting (HTTP 429 or equivalent)."""


class ProviderAuthError(ProviderError):
    """Raised when a provider rejects credentials — not retryable, fail fast."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is down / unreachable — retryable with backoff."""