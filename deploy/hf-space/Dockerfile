# Hugging Face Space (Docker SDK, free CPU). This folder is the whole Space repo: the code
# is downloaded from github.com/erbolka25/rag-support-platform at a pinned commit.
# Secrets (Space settings): OPENAI_API_KEY, LLM_BASE_URL, LLM_MODEL.
FROM python:3.13-slim

ARG PLATFORM_REF=dd45013b001ee9a08079b01811416f818c78a39a

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    DATA_DIR=/app/data \
    COMPANY_NAME=TechNomad \
    LLM_PROVIDER=openai \
    LLM_MAX_TOKENS=1500 \
    RATE_LIMIT_PER_HOUR=20 \
    DEMO_KB_DIR=/opt/platform/apps/support_copilot/demo/kb \
    DEMO_TICKETS_FILE=/opt/platform/apps/support_copilot/demo/shop_tickets.json

WORKDIR /app

# CPU-only torch first: the default PyPI wheel on Linux pulls several GB of CUDA libraries.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# The monorepo at the pinned commit: the rag_core engine, then the copilot on top of it.
ADD https://github.com/erbolka25/rag-support-platform/archive/${PLATFORM_REF}.tar.gz /tmp/platform.tar.gz
RUN mkdir -p /opt/platform \
    && tar -xzf /tmp/platform.tar.gz -C /opt/platform --strip-components=1 \
    && rm /tmp/platform.tar.gz \
    && pip install /opt/platform/packages/rag_core \
    && pip install /opt/platform/apps/support_copilot

# Spaces run the container as uid 1000.
RUN useradd --create-home --uid 1000 user && mkdir -p /app/data /app/.cache && chown -R user /app
USER user

# Bake the embedding model into the image: no download on every cold start.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

EXPOSE 7860
# One worker: the per-IP limiter lives in process memory. --proxy-headers: the client IP
# comes from the Spaces proxy, which is the only way in.
CMD ["uvicorn", "support_copilot.main:app", "--host", "0.0.0.0", "--port", "7860", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
