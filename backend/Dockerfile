# ── IWT Detection API ── Dockerfile
# Build: docker build -t iwt-api .
# Run:   docker compose up  (from repo root, see docker-compose.yml)
#
# Data directories are mounted as volumes at runtime; they are NOT baked into
# the image so sensitive/large data files stay off the registry.

# ---------------------------------------------------------------------------
# Stage 1 – base Python image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# System packages needed for Pillow, scikit-learn, and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2 – install Python dependencies
# ---------------------------------------------------------------------------
FROM base AS deps

# Copy requirements first to maximise Docker layer cache
COPY requirements.txt requirements-dev.txt ./

# Install PyTorch CPU wheels first (smaller than GPU build, fine for offline demo)
RUN pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 3 – runtime image
# ---------------------------------------------------------------------------
FROM base AS runtime

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy backend source (excludes .venv, __pycache__, checkpoints via .dockerignore)
COPY . .

# Volumes for data and model checkpoints are mounted at runtime.
# Create mount-point stubs so Docker doesn't auto-create them as root-owned dirs.
RUN mkdir -p /data/lexicons /data/processed /app/models/checkpoints

# Non-root user for security best practice
RUN useradd -m iwt && chown -R iwt:iwt /app /data
USER iwt

# Health-check: poll GET /health every 30 s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Run with uvicorn; workers=1 to keep the lazy-loaded models in one process.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
