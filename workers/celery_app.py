from __future__ import annotations

import os

from celery import Celery

from config.settings import get_settings

settings = get_settings()

# Celery does NOT read CELERY_* env vars into configuration on its own, so
# the operator's CELERY_TASK_ALWAYS_EAGER flag must be applied explicitly.
# Eager mode runs every task in-process (broker and worker bypassed) which is
# how local dev runs without Redis. It also means the result backend is never
# contacted, but we still switch it off redis so that chain handling in eager
# mode cannot accidentally hang on a connection retry.
is_eager = os.environ.get("CELERY_TASK_ALWAYS_EAGER", "").lower() in ("1", "true", "yes")
if is_eager and settings.celery_result_backend.startswith("redis"):
    _result_backend = "cache+memory://"
else:
    _result_backend = settings.celery_result_backend

celery_app = Celery(
    "video_dubbing",
    broker=settings.celery_broker_url,
    backend=_result_backend,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=settings.processing_timeout_seconds,
    task_soft_time_limit=max(settings.processing_timeout_seconds - 30, 30),
    task_always_eager=is_eager,
    task_eager_propagates=is_eager and os.environ.get("CELERY_TASK_EAGER_PROPAGATES", "").lower()
    in ("1", "true", "yes"),
    # Bound concurrent processing jobs via config instead of a hardcoded CLI
    # flag, so it is settable through env / file / secrets / cloud config.
    worker_concurrency=settings.max_concurrent_processing_jobs,
)

celery_app.conf.beat_schedule = {
    "recover-stale-jobs": {
        "task": "workers.tasks.recover_stale_jobs",
        "schedule": settings.stale_job_recovery_interval_seconds,
    },
}