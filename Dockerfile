FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Avoid interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace/autoresearch

# Copy dependency files first (layer caching — only re-install when deps change)
COPY pyproject.toml uv.lock .python-version ./

# Install all dependencies (expensive — cached unless deps change)
RUN uv sync

# Copy project source
COPY prepare.py train.py ./

# Data is mounted at runtime, not baked into the image.
# Mount your data volume at /root/.cache/autoresearch/
