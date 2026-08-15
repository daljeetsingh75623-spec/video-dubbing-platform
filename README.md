# AI-Powered Video Dubbing Platform

Translates and dubs videos with multiple speakers from a source language into
a target language, preserving speaker identity, conversation flow, audio-video
synchronization, and subtitle alignment.

Built as a production-minded FastAPI + Celery service: every AI stage sits
behind a swappable provider interface, all limits are configuration-driven,
and the worker tier scales horizontally off a Redis-backed job queue.

## Features

- **Video upload** with format / size / duration validation (MP4, MOV, AVI, MKV)
- **Speaker diarization** with consistent speaker IDs (pyannote.audio)
- **Speech recognition** with automatic language detection (faster-whisper)
- **Translation** that preserves segment structure, tone, and speaker mapping (Gemini)
- **AI voice generation** per speaker (Edge TTS / gTTS)
- **Audio-video synchronization** via ffmpeg (pitch-preserving time-stretch + mux)
- **Outputs**: dubbed video, SRT subtitles, transcript text, processing logs
- **Job API**: status, transcript, retry, cancel, download
- **Config-driven**: env vars > YAML > defaults, no code changes required
- **Provider abstraction** for STT / translation / TTS / diarization
- **Observability**: structured logging (structlog) + OpenTelemetry tracing
- **Docker Compose** deployment with healthchecks

## Architecture

```
Client ──▶ FastAPI ──▶ PostgreSQL (jobs, transcripts, events)
    │          │
    │          └── enqueue ──▶ Redis (Celery broker/backend)
    ▼                                  │
Celery workers ──▶ Provider factory ───┤ (pyannote, whisper, Gemini, edge-tts)
    │                                  │
    └──▶ ffmpeg (AV sync) ──▶ MinIO / S3 / local (video, audio, outputs)
```

- API and workers are separate stateless processes sharing the same DB and
  object storage — the worker tier scales horizontally by adding replicas.
- The pipeline is a Celery **task chain**: `diarize → transcribe → translate
  → synthesize → sync_av → package`. Each stage is independently retryable,
  writes its own status/event, and persists output before the next stage runs.
- Full details, scaling strategy (5 → 500 → 5,000 videos/day), failure
  recovery, and design trade-offs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick Start (Docker Compose)

```bash
cp .env.example .env
docker compose up --build
```

This starts API (:8000), Celery worker, Redis, PostgreSQL, and MinIO.

- OpenAPI docs: http://localhost:8000/docs
- Frontend UI: http://localhost:8000
- Health check: http://localhost:8000/health

### Run a dubbing job

```bash
curl -X POST http://localhost:8000/videos \
  -F "file=@sample.mp4" \
  -F "target_language=es"

curl http://localhost:8000/videos/<job_id>/status
curl http://localhost:8000/videos/<job_id>/transcript
curl http://localhost:8000/videos/<job_id>/download -o dubbed.mp4
curl http://localhost:8000/videos/<job_id>/subtitles -o subtitles.srt
```

## Local Development

Requires Python 3.11+ with ffmpeg/ffprobe on PATH.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (or `source .venv/bin/activate` on macOS/Linux)
pip install -r requirements.txt
uvicorn api.main:app --reload            # API
celery -A workers.celery_app worker --loglevel=info   # worker
```

Tests use an in-memory SQLite database and a no-op Celery `.delay()`, so they
run without any infrastructure:

```bash
pytest
```

## Configuration

Precedence: **environment variables > YAML (`CONFIG_FILE`, defaults to
`config/default.yaml`) > built-in defaults**.

Key settings (full list in `config/default.yaml` and `config/settings.py`):

| Setting | Default | Description |
|---|---|---|
| `MAX_UPLOAD_SIZE_MB` | 500 | Max upload size |
| `MAX_VIDEO_DURATION_SECONDS` | 600 | Max video length (enforced via ffprobe) |
| `ALLOWED_VIDEO_FORMATS` | mp4,mov,avi,mkv | Allowed extensions |
| `MAX_CONCURRENT_UPLOADS` | 10 | Upload concurrency cap |
| `MAX_CONCURRENT_PROCESSING_JOBS` | 5 | Worker concurrency |
| `PROCESSING_TIMEOUT_SECONDS` | 1800 | Per-task hard time limit |
| `RETRY_COUNT` | 3 | Task retries (exponential backoff + jitter) |
| `QUEUE_MAX_LENGTH` | 1000 | Queue depth cap |
| `TRANSLATION_BATCH_SIZE` | 25 | Max transcript segments per translation call (avoids truncated JSON) |
| `STT_PROVIDER` | stub | `stub`, `faster_whisper` |
| `TRANSLATION_PROVIDER` | stub | `stub`, `gemini` |
| `TTS_PROVIDER` | stub | `stub`, `edge_tts`, `gtts` |
| `DIARIZATION_PROVIDER` | stub | `stub`, `pyannote` |
| `DIARIZATION_MIN_SPEAKERS` | unset | Force a lower bound on detected speakers (e.g. `2` for a known conversation) |
| `DIARIZATION_MAX_SPEAKERS` | unset | Force an upper bound on detected speakers |
| `STORAGE_BACKEND` | local | `local` or `s3` (MinIO-compatible) |
| `API_KEY_AUTH_ENABLED` | false | Require `X-API-Key` header |
| `RATE_LIMIT_PER_MINUTE` | 60 | Global per-IP rate limit |
| `OTEL_ENABLED` | true | OpenTelemetry tracing (console if no endpoint) |

Swapping models is a config change, not a code change. For example, to use the
real local stack:

```bash
STT_PROVIDER=faster_whisper DIARIZATION_PROVIDER=pyannote \
TRANSLATION_PROVIDER=gemini GEMINI_API_KEY=... TTS_PROVIDER=edge_tts \
docker compose up --build
```

`pyannote` requires a HuggingFace token (`HUGGINGFACE_TOKEN`), and the gated
model must be accepted once on hf.co/pyannote/speaker-diarization-3.1.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/videos` | Upload video + `target_language` |
| GET | `/videos` | List recent jobs |
| GET | `/videos/{id}/status` | Job status |
| GET | `/videos/{id}/transcript` | Source/translated transcript |
| GET | `/videos/{id}/download` | Download dubbed video |
| GET | `/videos/{id}/subtitles` | Download SRT subtitles |
| POST | `/videos/{id}/retry` | Retry a failed/cancelled job |
| POST | `/videos/{id}/cancel` | Cancel a running job |
| GET | `/health` | Liveness probe |

Interactive docs at `/docs`.

## Project Structure

```
api/                 FastAPI app (routers, auth, validation, schemas)
config/              Settings + YAML defaults (env > yaml > defaults)
core/                Domain models, AV sync (ffmpeg), SRT, telemetry
db/                  SQLAlchemy models, session, Alembic migrations
providers/           Pluggable AI providers (stt, translation, tts, diarization)
storage/             Storage backends (local, s3/minio)
workers/             Celery app + pipeline task chain
tests/               Unit + integration tests
docs/                Architecture document
frontend/            Minimal upload/status UI
docker/              Dockerfile
```

## Observability

- All services log structured JSON-ish events via `structlog`.
- `core/telemetry.py` initializes OpenTelemetry: FastAPI request spans are
  exported to the OTLP endpoint in `OTEL_EXPORTER_ENDPOINT`, or to the console
  when unset. Point it at a Jaeger/OTel Collector / Grafana Tempo for tracing.
- `GET /metrics` exposes Prometheus-format job counters, queue depth, upload /
  processing gauges, and HTTP counters.

## Logs

**Docker Compose** — follow one or all services:

```bash
docker compose logs -f                       # all services
docker compose logs -f api                   # or: worker, beat, redis, postgres, minio
docker compose logs --tail=100 api           # last N lines
```

**Local dev** — run uvicorn / celery in the foreground; structured `structlog`
events print to the terminal, e.g. `request_end`, `tts_segment`,
`stale_job_recovered`, `pipeline_failed`.

**Traces** — with `OTEL_ENABLED=true` and no `OTEL_EXPORTER_ENDPOINT`, spans
are printed to the console instead of exported.

## Known Limitations

- **Speaker voices**: Edge TTS maps each speaker to a distinct voice from a
  per-language pool, so speaker identity is preserved up to the pool size
  (e.g. 8 voices for English). This is still not true voice *cloning* — the
  dubbed voice won't match the original speaker's timbre. Reference-audio
  cloning (ElevenLabs / Coqui) is the intended production upgrade; the
  `speaker_reference_audio` interface is defined but not yet wired to a
  cloning provider.
- **Single target language per job**: multi-target-language dubbing is not yet
  implemented.
- **Best-effort lip sync**: chunks are placed at their original segment
  offsets and only pitch-preserving time-stretched when they would overlap
  the next segment (never slowed down). No per-sentence re-timing against the
  original audio, so timing follows the source transcript boundaries.
- **CPU-only**: pyannote/whisper run on CPU in the containers; `device="cuda"`
  is supported in code but not exposed via config/compose.

## Troubleshooting

**"All speakers sound the same"**
1. Restart the stack so the `.env` config and latest code are actually loaded:
   `docker compose up -d --build` (env/`docker-compose.yml` changes need a
   restart; the worker also caches code at startup).
2. Confirm the speaker separation happened upstream — check the transcript
   for a job: `GET /videos/{id}/transcript`. If every segment shows the same
   `speaker_label`, diarization/attribution found only one speaker. If the
   video is a known 2+ person conversation, set
   `DIARIZATION_MIN_SPEAKERS=2` in `.env` (pyannote can merge similar
   voices into one speaker).
3. Verify which voice was used per segment in the worker logs — each segment
   logs `tts_segment` with its `speaker_id` and `voice`. The `synthesized`
   job event also stores the chosen `voice` per segment.

**Dubbed audio is fast/slow or overlapping**
- Ensure the source segments come from *real* diarization
  (`DIARIZATION_PROVIDER=pyannote`) rather than the stub, which fabricates
  fixed 3-second turns that don't match real speech.
- The AV-sync stage only stretches chunks that would overflow their window
  (capped at 1.35x, never slowed down) and hard-trims anything still too
  long, so overlap is prevented by design.

## License

For the original assignment: internal take-home submission.
