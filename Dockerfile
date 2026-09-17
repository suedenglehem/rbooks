# library-rag application image.
#
# The app runs as a single coordinator process that spawns bounded worker
# subprocesses (per PRD §3). Qdrant and the optional llama.cpp answer server are
# separate compose services, not baked into this image.
#
# Build:  docker build -t library-rag .
# Run:    see compose.yaml (profiles: app, qdrant, llama)

FROM python:3.12-slim AS runtime

# uv for reproducible dependency installation from uv.lock.
RUN pip install --no-cache-dir uv==0.12.15

WORKDIR /app

# Install the exact locked dependencies into an isolated venv.
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --frozen --no-dev

# Non-root runtime user.
RUN useradd --uid 1000 app && mkdir -p /data && chown -R app:app /data /app
USER app

# /data is where the operator mounts the storage roots (archive/state/qdrant/...);
# the config inside the container must point its roots there.
ENV LIBRARY_RAG_CONFIG=/data/config.yaml
WORKDIR /app

EXPOSE 8000

# Default: run the API. The compose file overrides this for coordinator/worker roles.
CMD ["library-rag", "serve"]
