# Build from the repository root, so rag_core is in the context:
#   docker build -f apps/support_copilot/Dockerfile -t support-copilot .
#   docker run --rm -p 8000:8000 --env-file .env -v copilot-data:/app/data support-copilot
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DATA_DIR=/app/data

WORKDIR /app

# CPU-only torch first: the default PyPI wheel on Linux pulls several GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The RAG engine: only its package sources, never its .env, data or venv.
COPY packages/rag_core/pyproject.toml packages/rag_core/README.md /opt/rag_core/
COPY packages/rag_core/src /opt/rag_core/src
RUN pip install /opt/rag_core

COPY apps/support_copilot/pyproject.toml apps/support_copilot/README.md ./
COPY apps/support_copilot/src ./src
RUN pip install .

RUN useradd --create-home app && mkdir -p /app/data /app/.cache && chown -R app /app
USER app

# Bake the embedding model into the image so the container works offline (~470 MB).
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"

VOLUME /app/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "support_copilot.main:app", "--host", "0.0.0.0", "--port", "8000"]
