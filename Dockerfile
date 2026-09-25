# Dockerfile — packages Ledger's API into a container image.
FROM python:3.12-slim

# Don't buffer print output; put downloaded models inside the image folder.
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app
RUN pip install --no-cache-dir uv

# Install dependencies first. Docker caches this layer, so code changes
# don't trigger a slow reinstall of every library.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY ledger ./ledger
COPY evals ./evals
COPY start.sh ./start.sh
RUN chmod +x start.sh

# Download the two search models at build time, so the first request isn't slow.
RUN uv run --no-sync python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000
CMD ["./start.sh"]
