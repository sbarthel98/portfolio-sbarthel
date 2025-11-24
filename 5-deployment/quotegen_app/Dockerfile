# Optimized Dockerfile for smaller builds and faster caching
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Reduce pip noise and cache usage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal build deps only if needed for binary packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application source and only required artefacts
COPY src/quotegen /app/src/quotegen
COPY backend /app/backend
COPY artefacts /app/artefacts
COPY assets /app/assets
COPY quotegen.toml pyproject.toml uv.lock /app/

# Ensure Python can import local package code without installing a wheel
ENV PYTHONPATH=/app

WORKDIR /app/backend

EXPOSE 8000
CMD ["python", "app.py"]