FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only torch/torchaudio first, from the CPU wheel index — much
# smaller download than the default CUDA-bundled wheels (~200MB vs ~2GB+),
# and this is a CPU-only container so CUDA wheels would be dead weight.
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 \
    torch==2.4.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 --retries 10 -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]