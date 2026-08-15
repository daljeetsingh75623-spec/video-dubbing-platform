# Architecture Document — AI Video Dubbing Platform

## 1. High-Level Architecture

┌──────────┐ ┌─────────────┐ ┌───────────────┐
│ Client │────▶│ FastAPI │────▶│ PostgreSQL │
│ (curl/ │ │ (api/) │ │ (jobs, specs, │
│ Postman)│◀────│ │◀────│ transcripts) │
└──────────┘ └──────┬──────┘ └───────────────┘
│
│ enqueue job
▼
┌─────────────┐
│ Redis │◀─── broker + result backend
│ (Celery MQ) │
└──────┬──────┘
│
▼
┌─────────────┐ ┌───────────────┐
│ Celery │────▶│ MinIO / S3 │
│ Worker │◀────│ (video, audio, │
│ (workers/) │ │ outputs) │
└──────┬──────┘ └───────────────┘
│
▼
┌─────────────────────┐
│ Provider Factory │
│ (providers/) │
│ ┌─────┬─────┬─────┐│
│ │diar.│ STT │trans││
│ ├─────┼─────┼─────┤│
│ │ TTS │ │ ││
│ └─────┴─────┴─────┘│
└─────────────────────┘


The API and worker are separate processes (separate containers) sharing the
same codebase, DB, and object storage — this is what makes the worker tier
horizontally scalable independent of the API tier.

## 2. Component Interactions

1. Client uploads video → `POST /videos` → API validates (format/size/mime-sniff) → uploads to storage → creates `Job` row (status=`queued`) → enqueues `run_pipeline` task → returns `job_id` immediately (non-blocking).
2. Celery worker picks up the job, runs a **task chain**: `diarize → transcribe → translate → synthesize → sync_av → package_output`. Each stage is its own retryable Celery task, writes its own status + `JobEvent` row on entry, and persists its output to Postgres/storage before the next stage starts.
3. Client polls `GET /videos/{id}/status` (or a future SSE/WebSocket endpoint fed by Redis pub/sub) to track progress.
4. On completion, `GET /videos/{id}/download` and `/subtitles` return pre-signed/local URLs to the final artifacts.

## 3. Processing Pipeline

| Stage | Input | Output | Provider(s) |
|---|---|---|---|
| Diarization | source video | speaker-labeled time segments | pyannote.audio (local) |
| Transcription | source audio | timestamped transcript, speaker-attributed | faster-whisper (local) |
| Translation | transcript | translated transcript, same segment structure | Gemini (API) |
| Synthesis | translated transcript | per-segment TTS audio chunks | edge-tts (free, local call) |
| AV Sync | audio chunks + original video | composite audio track muxed onto original video | ffmpeg |
| Packaging | transcript + video | SRT file, dubbed video, transcript text, all uploaded to storage | — |

Every stage above the ffmpeg layer sits behind a provider **abstract base
class** (`providers/*/base.py`) and is resolved at runtime by
`ProviderFactory` based on config — swapping `faster_whisper` for
`deepgram`, or `gemini` for `openai`/`claude`, is a one-line config change
with zero code touched in the API, worker, or pipeline logic.

## 4. AI Model Selection Strategy

All four provider categories chose **free-tier / local** options for this
submission, prioritizing zero-friction reproducibility for reviewers over
peak quality:

- **Diarization — pyannote.audio**: strong open-source diarization
  accuracy; requires a free HuggingFace token (gated model license, no
  payment). Trade-off: CPU inference is slow on longer clips — GPU
  recommended for production.
- **STT — faster-whisper**: local inference, no API key, no per-request
  cost or rate limit — ideal for repeated test runs during a take-home
  assignment. Trade-off: accuracy slightly below Whisper large models
  depending on chosen model size; `base` used for speed.
- **Translation — Gemini (gemini-2.5-flash)**: free-tier API key, strong
  translation quality, fast. Swappable for OpenAI/Claude/DeepL via the same
  interface — only `TranslationProvider.translate()` needs implementing.
- **TTS — edge-tts**: free, no API key, good voice quality via Microsoft
  Edge's neural voices. Speaker "identity" is approximated by alternating
  between two voices per language per speaker index — **not** true voice
  cloning. Production would swap to ElevenLabs or Coqui with per-speaker
  reference audio for genuine voice preservation.

## 5. Queue Architecture

Celery + Redis (broker + result backend). Chosen over RabbitMQ/Kafka/SQS for
simplicity — Redis is already a dependency for job-status pub/sub, so it
serves double duty without adding another infra component. `task_acks_late`
ensures a task is redelivered if a worker dies mid-processing rather than
being silently lost.

## 6. Scaling Strategy — 5 → 500 → 5,000 videos/day

**5 videos/day (current setup):** single API container, single worker
container with `--concurrency=4`, single Postgres/Redis/MinIO instance. No
changes needed.

**500 videos/day:**
- Scale `worker` horizontally: `docker compose up --scale worker=4` (or
  equivalent in an orchestrator). Workers are stateless — all state lives in
  Postgres/Redis/object storage — so this requires zero application code
  changes.
- Redis queue depth becomes the autoscaling signal: a Kubernetes
  `HorizontalPodAutoscaler` (or equivalent) watching Celery queue length via
  a custom metric can add/remove worker replicas automatically.
- Postgres connection pooling (PgBouncer) becomes worthwhile at this scale.

**5,000 videos/day:**
- Full container orchestration (Kubernetes/ECS/AKS/GKE) with worker pools
  separated **by stage** — diarization/STT are CPU/GPU-heavy and benefit
  from dedicated node pools with GPU scheduling, while translation/TTS are
  I/O-bound (waiting on external APIs) and can run on cheap CPU-only nodes
  with high concurrency.
- Chunked processing for long videos: split a video into segments before
  diarization so no single task holds a worker slot for the video's full
  duration.
- Sharded job distribution by `job_id` hash across multiple Redis/Celery
  queue instances to avoid a single broker becoming a bottleneck.
- Object storage (S3) already scales horizontally with no changes needed;
  Postgres would move to a managed, read-replica-backed instance.
- **Not implemented in this submission** (described only, per assignment
  allowance): Kubernetes manifests, multi-region deployment, full
  Prometheus/Grafana stack. Structured logs (`structlog`) + basic
  OpenTelemetry tracing demonstrate the observability pattern without
  standing up the full stack.

## 7. Failure Recovery

- **Provider-level failures** (timeout, rate limit, transient
  unavailability): caught as typed exceptions (`providers/errors.py`),
  retried via Celery's `autoretry_for` with exponential backoff + jitter, up
  to `retry_count` (configurable).
- **Auth failures** (`ProviderAuthError`): intentionally **not** retried —
  fail fast, since retrying a bad API key wastes time and quota.
- **Partial pipeline failure**: each stage commits its output to Postgres
  before the next stage begins, so a failure at stage N doesn't lose
  progress from stages 1..N-1. A `retry` on a failed job re-runs the full
  chain currently (idempotent by design — re-running diarization/STT is
  safe since they're deterministic given the same input), though a future
  optimization could resume from the last successful stage.
- **Corrupted/invalid uploads**: rejected synchronously at upload time via
  extension check, size check, and mime-sniffing (`python-magic`) before a
  job is even created — no wasted pipeline capacity on bad input.
- **Cooperative cancellation**: `POST /{id}/cancel` sets job status to
  `cancelled`; each pipeline stage checks this flag before starting its own
  work and exits early rather than being force-killed mid-task, avoiding
  partial/corrupted outputs.

## 8. Deployment Strategy

Docker Compose for this submission (api, worker, redis, postgres, minio),
each with healthchecks and explicit startup dependency ordering. Production
deployment would move to:
- Container images pushed to a registry, deployed via Kubernetes/ECS
- Managed Postgres (RDS/Cloud SQL) and managed Redis (ElastiCache) instead
  of self-hosted containers
- S3 (real, not MinIO) for object storage
- Secrets (API keys, DB credentials) injected via a secrets manager
  (AWS Secrets Manager / Azure Key Vault) rather than `.env` files — the
  config system already supports this transparently since env vars are the
  highest-precedence config source.

## 9. Design Trade-offs

| Decision | Trade-off accepted |
|---|---|
| Free-tier/local providers over paid APIs | Slightly lower ceiling on quality (esp. voice cloning) in exchange for zero-cost reproducibility |
| Best-effort AV sync (chunk placement, no time-stretching) | Simpler ffmpeg pipeline; occasional slight audio drift on segments where TTS output length differs from original speech duration |
| Full pipeline re-run on retry (not resume-from-stage) | Simpler retry semantics; acceptable since all stages are idempotent, at the cost of re-doing completed work |
| Synchronous full-file read on upload (not streaming) | Simpler validation code; would need to move to streaming + incremental size checks for very large files at higher scale |
| SQLite for integration tests, Postgres for real deployment | Fast, dependency-free test runs; accepts a small risk of SQLite/Postgres behavioral differences not being caught by tests |
