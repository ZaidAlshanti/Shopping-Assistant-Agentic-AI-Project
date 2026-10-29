# ---------- builder ----------
FROM public.ecr.aws/docker/library/python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# CPU-only torch first, so the multi-GB CUDA build is never pulled
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# language model used by the PII validator
RUN python -m spacy download en_core_web_lg

# ---------- final ----------
FROM public.ecr.aws/docker/library/python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LOGLEVEL="INFO"
ENV GUARDRAILS_LOG_LEVEL="INFO"
ENV OTEL_SDK_DISABLED=true
ENV HF_HOME=/opt/hf_cache

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./config.py ./config.py

EXPOSE 8000

CMD uvicorn guardrails_api.app:create_app --workers 1 --host 0.0.0.0 --port 8001 --timeout-keep-alive 20 --timeout-graceful-shutdown 60