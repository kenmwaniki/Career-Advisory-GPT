# ----------------------------------------
# Career GPT RAG API - Hugging Face Space
# ----------------------------------------
FROM python:3.11-slim

# Core env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/data/.huggingface \
    RAG_DB_DIR=/data/chroma_db \
    RAG_PORT=7860

# System deps (add more if your loaders need them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project files
COPY . .

# Space port
EXPOSE 7860

# Entrypoint
CMD ["bash", "bootstrap.sh"]
