FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency spec first (caching layer)
COPY pyproject.toml .

# Install Python deps (cannot use -e . yet since source is not copied)
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0" \
    "websockets>=12.0" \
    "httpx>=0.25.0"

# Copy source
COPY . .

# Install the package itself (now source is available)
RUN pip install --no-cache-dir -e . 2>/dev/null || true

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
