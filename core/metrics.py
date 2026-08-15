"""
Minimal self-contained Prometheus metrics registry.

Deliberately dependency-free (no prometheus_client / otel SDK wiring) so the
/health-adjacent scrape endpoint works in any deployment. Sample collection is
GIL-atomic (plain dict reads/writes of ints), which is fine for an app of this
size; swap for prometheus_client + OTel metric SDK if you need more rigor.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_ESCAPES = str.maketrans({"\\": "\\\\", '"': '\\"', "\n": "\\n"})


@dataclass
class _Family:
    name: str
    help: str
    mtype: str  # "counter" | "gauge"
    label_names: tuple[str, ...]
    samples: dict[tuple[str, ...], int] = field(default_factory=dict)


_REGISTRY: list[_Family] = []


def _family(name: str, help: str, mtype: str, label_names: tuple[str, ...] = ()) -> _Family:
    fam = _Family(name=name, help=help, mtype=mtype, label_names=label_names)
    _REGISTRY.append(fam)
    return fam


# --- metric definitions -------------------------------------------------------

jobs_created_total = _family("vdp_jobs_created_total", "Total jobs created (uploaded and queued).", "counter")
jobs_completed_total = _family("vdp_jobs_completed_total", "Total jobs that finished successfully.", "counter")
jobs_failed_total = _family("vdp_jobs_failed_total", "Total jobs that failed (incl. stale-job recovery).", "counter")
uploads_active = _family("vdp_uploads_active", "Uploads currently being validated and stored.", "gauge")
processing_active = _family("vdp_processing_active", "Pipeline runs currently executing in this process.", "gauge")
queue_depth = _family("vdp_queue_depth", "Jobs currently waiting in the queue.", "gauge")
upload_rejections_total = _family(
    "vdp_upload_rejections_total",
    "Uploads rejected with 429 by an operational limit.",
    "counter",
    ("reason",),
)
http_requests_total = _family(
    "vdp_http_requests_total",
    "HTTP requests handled.",
    "counter",
    ("method", "route", "status"),
)


# --- mutation helpers ---------------------------------------------------------

def inc(fam: _Family, *label_values: str) -> None:
    key = tuple(label_values)
    fam.samples[key] = fam.samples.get(key, 0) + 1


def dec(fam: _Family, *label_values: str) -> None:
    key = tuple(label_values)
    fam.samples[key] = fam.samples.get(key, 0) - 1


def set_gauge(fam: _Family, value: int) -> None:
    fam.samples[()] = value


def reset() -> None:
    """Clear all samples (test isolation only)."""
    for fam in _REGISTRY:
        fam.samples.clear()


# --- rendering -----------------------------------------------------------------

def render() -> str:
    lines: list[str] = []
    for fam in _REGISTRY:
        lines.append(f"# HELP {fam.name} {fam.help}")
        lines.append(f"# TYPE {fam.name} {fam.mtype}")
        samples = fam.samples or {(): 0}
        for label_values, value in sorted(samples.items()):
            if fam.label_names:
                pairs = sorted(zip(fam.label_names, label_values), key=lambda kv: kv[0])
                rendered = ", ".join(f'{k}="{_escape(str(v))}"' for k, v in pairs)
                lines.append(f"{fam.name}{{{rendered}}} {value}")
            else:
                lines.append(f"{fam.name} {value}")
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.translate(_ESCAPES)
