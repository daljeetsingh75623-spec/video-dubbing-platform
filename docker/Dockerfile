FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch/torchaudio first, from the CPU wheel index — much
# smaller download than the default CUDA-bundled wheels (~200MB vs ~2GB+),
# and this is a CPU-only container so CUDA wheels would be dead weight.
# Versions must match requirements.txt (pyannote.audio 4.x needs torch>=2.8).
# torchaudio 2.13.0+cpu isn't published on the CPU index yet — 2.11.0 is the
# newest pairing available; the local venv runs the same torch/torchaudio pair.
# torchcodec must also come from this CPU index: the PyPI wheel is built
# against CUDA torch and fails to import (libnvrtc.so.13) on the CPU build.
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 \
    torch==2.13.0 torchaudio==2.11.0 torchcodec==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]