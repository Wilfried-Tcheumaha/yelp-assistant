FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Superlinked defaults to Path.home() / ".cache" when MODEL_CACHE_DIR isn't set.
# In some containers the non-root user's home is unset (Path.home() becomes /nonexistent),
# causing PermissionError when it tries to create cache folders.
ENV HOME=/tmp
ENV MODEL_CACHE_DIR=/tmp/superlinked-model-cache
ENV TRANSFORMERS_CACHE=/tmp/transformers-cache
ENV HF_HOME=/tmp/hf-home

# Copy workspace root files for dependency resolution
COPY pyproject.toml uv.lock ./

# Copy package files and source
COPY api ./api

# Superlinked reads YAML_FILENAME ("config.yaml") from process cwd (/app/api/src)
COPY config.yaml api/src/config.yaml

ENV UV_COMPILE_BYTECODE=1

# Build deps (psutil may not have wheels for this image/arch, so it falls back to compiling)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential linux-libc-dev && \
    rm -rf /var/lib/apt/lists/*

# Ensure cache dirs are created with permissive access.
RUN mkdir -p "$MODEL_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_HOME"
RUN chmod -R 777 "$MODEL_CACHE_DIR" "$TRANSFORMERS_CACHE" "$HF_HOME"

# Install dependencies including workspace packages
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package api

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set PATH to use the virtual environment
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/api/src"

# Create non-root user and set permissions
RUN addgroup --system app && \
    adduser --system --ingroup app app && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose the FastAPI port
EXPOSE 8000

WORKDIR /app/api/src

# Command to run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]