# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base

# ---- Runtime hygiene -------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Europe/Zagreb

WORKDIR /app

# ---- Dependencies (cached layer) ------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application -----------------------------------------------------------
COPY main.py .

# Non-root user; /data is where we persist the JSONL log
RUN useradd --create-home --uid 1000 tracker \
    && mkdir -p /data \
    && chown tracker:tracker /data
USER tracker

VOLUME ["/data"]

# Default target = Zagreb Grič, poll every 3 min, log to /data/clusters.jsonl.
# Override any of these with `docker run ... python main.py --lat ...`
ENTRYPOINT ["python", "main.py"]
CMD ["--lat", "45.8146", "--lon", "15.9722", \
     "--poll", "180", \
     "--dump-jsonl", "/data/clusters.jsonl"]
