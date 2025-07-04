FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

# ---- Build stage ----
FROM base AS build

# Install build dependencies for C extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy only dependency files first for better cache utilization
COPY pyproject.toml uv.lock* ./

# 2. Install dependencies (this layer is cached unless dependencies change)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# 3. Now copy the rest of your application code
COPY . .

# 4. Install the project itself (fast, and only re-runs if your code changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# ---- Runtime stage ----
FROM base AS runtime

WORKDIR /app

# Copy everything from /app in build stage to /app in runtime stage
COPY --from=build /app /app

EXPOSE 8080

CMD ["uv", "run", "main.py"]
