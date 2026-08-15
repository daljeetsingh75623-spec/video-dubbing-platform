import asyncio

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, Response
from slowapi.errors import RateLimitExceeded
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from api.rate_limit import default_rate_limit, limiter, retry_after_seconds
from api.routers import videos
from config.settings import get_settings
from core import metrics
from core.telemetry import init_otel
from db.models import Job
from db.session import get_db

settings = get_settings()
log = structlog.get_logger()

app = FastAPI(
    title="AI Video Dubbing Platform",
    version="0.1.0",
    description="Upload a video, get it back dubbed into another language.",
)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc.detail}"},
        headers={"Retry-After": str(retry_after_seconds(exc.detail))},
    )

# Upload concurrency gate: bounds how many uploads run validation/storage at
# once. The value comes from config (env / file / secrets / cloud config).
app.state.upload_semaphore = asyncio.Semaphore(settings.max_concurrent_uploads)

init_otel(app)

app.include_router(videos.router)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info("request_start", method=request.method, path=request.url.path)
    response = await call_next(request)
    log.info("request_end", method=request.method, path=request.url.path, status=response.status_code)
    # Use the route template ("/videos/{job_id}/status") instead of the raw
    # URL so dynamic paths don't explode label cardinality.
    route = getattr(request.scope.get("route"), "path", request.url.path)
    metrics.inc(metrics.http_requests_total, request.method, route, str(response.status_code))
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
@limiter.limit(default_rate_limit)
async def health(request: Request):
    return {"status": "ok", "env": settings.app_env}


@app.get("/metrics", include_in_schema=False)
@limiter.limit(default_rate_limit)
async def metrics_endpoint(request: Request, db: AsyncSession = Depends(get_db)):
    # Keep the scrape alive during a DB outage: report queue depth as "unknown"
    # (-1) rather than failing the whole endpoint, so Prometheus can still
    # scrape HTTP/job counters and alert on the platform being degraded.
    try:
        queued = (
            await db.execute(select(func.count()).select_from(Job).where(Job.status == "queued"))
        ).scalar_one()
        metrics.set_gauge(metrics.queue_depth, queued)
    except Exception:  # noqa: BLE001 - observability must not take the API down
        log.warning("metrics_queue_depth_unavailable")
        metrics.set_gauge(metrics.queue_depth, -1)
    return Response(
        content=metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# Static frontend must be mounted last so catch-all "/" doesn't shadow
# API routes like /health and /videos (Starlette matches in registration order).
from fastapi.staticfiles import StaticFiles  # noqa: E402

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")