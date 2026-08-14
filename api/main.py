from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.routers import videos
from config.settings import get_settings

settings = get_settings()
log = structlog.get_logger()

limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])

app = FastAPI(
    title="AI Video Dubbing Platform",
    version="0.1.0",
    description="Upload a video, get it back dubbed into another language.",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

app.include_router(videos.router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    log.info("request_start", method=request.method, path=request.url.path)
    response = await call_next(request)
    log.info("request_end", method=request.method, path=request.url.path, status=response.status_code)
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}