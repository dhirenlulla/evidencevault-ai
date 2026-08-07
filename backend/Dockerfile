FROM python:3.10-slim

# PYTHONDONTWRITEBYTECODE keeps .pyc files out of the image (they're
# useless in a container that's rebuilt from scratch every deploy).
# PYTHONUNBUFFERED makes stdout/stderr flush immediately instead of
# buffering, which matters for `docker compose logs -f` showing
# output in real time rather than in delayed chunks.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is only needed for the container HEALTHCHECK below; nothing
# in the application itself depends on system packages beyond what
# pip installs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch BEFORE the rest of requirements.txt.
# sentence-transformers depends on torch but doesn't pin it to a
# CPU build, so plain `pip install torch` on Linux resolves to the
# CUDA-enabled wheel and pulls several GB of NVIDIA libraries
# (cublas, cudnn, cusolver, the cuda-toolkit, ...) that this
# container has no GPU access to and will never use - both
# EMBEDDING_DEVICE and RERANKER_DEVICE are "cpu". Installing the
# CPU wheel first means it already satisfies sentence-transformers'
# torch requirement, so the requirements.txt install below never
# reaches for the CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install the rest of the dependencies BEFORE copying
# application source. Docker caches each layer by its inputs: as
# long as requirements.txt is unchanged, this install layer is
# reused from cache on every rebuild instead of reinstalling every
# dependency each time application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Running as root inside a container removes an isolation boundary
# for free with no upside: a vulnerability in any dependency becomes
# a root-level compromise of the container instead of a contained,
# unprivileged one.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/uploads \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# start_period is deliberately generous. Importing sentence-transformers
# (and therefore torch) at process startup is not instant even before
# any model weights are loaded, and Phase 10B will add eager model
# warm-up at startup - budgeting time for that now avoids revisiting
# this value later.
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]