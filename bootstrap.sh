#!/bin/bash
set -e

echo "[bootstrap] Career GPT Space starting..."

# Ensure /data exists
mkdir -p /data /data/uploads /data/chroma_db

# Re-ingest control:
# - If DB is empty -> ingest
# - Or if RAG_REINGEST=1 -> force re-ingest (wipes DB if RAG_REBUILD=1)
NEED_INGEST=0
if [ ! -d "/data/chroma_db" ] || [ -z "$(ls -A /data/chroma_db 2>/dev/null)" ]; then
  NEED_INGEST=1
elif [ "${RAG_REINGEST:-0}" = "1" ]; then
  NEED_INGEST=1
fi

if [ "$NEED_INGEST" = "1" ]; then
  echo "[bootstrap] Ingesting documents into /data/chroma_db ..."
  if [ "${RAG_REBUILD:-0}" = "1" ]; then
    echo "[bootstrap] Rebuild requested; wiping existing DB first."
    rm -rf /data/chroma_db && mkdir -p /data/chroma_db
  fi

  if [ -f "ingest.py" ]; then
    # Choose embedding provider/model via env (defaults shown)
    # RAG_EMBED_PROVIDER: hf_local | hf_inference | ollama
    # RAG_EMBED_MODEL: e.g. BAAI/bge-small-en-v1.5
    python ingest.py \
      --docs /data/uploads \
      --db /data/chroma_db \
      --embed-provider "${RAG_EMBED_PROVIDER:-hf_local}" \
      --embed-model "${RAG_EMBED_MODEL:-BAAI/bge-small-en-v1.5}" \
      --device "${RAG_DEVICE:-cpu}" \
      --embed-batch "${RAG_EMBED_BATCH:-32}" \
      --chunk-size "${RAG_CHUNK_SIZE:-900}" \
      --chunk-overlap "${RAG_CHUNK_OVERLAP:-180}" \
      --min-chars "${RAG_MIN_CHARS:-200}" \
      $( [ "${RAG_BGE_PREFIX:-1}" = "0" ] && echo --no-bge-prefix )
  else
    echo "[bootstrap] WARNING: ingest.py not found; skipping ingestion."
  fi
else
  echo "[bootstrap] Existing Chroma DB detected — skipping ingestion."
fi

# Optional: watch mode (incremental indexing)
if [ "${RAG_WATCH:-0}" = "1" ]; then
  echo "[bootstrap] Watch mode enabled (background)."
  nohup python ingest.py \
    --docs /data/uploads \
    --db /data/chroma_db \
    --embed-provider "${RAG_EMBED_PROVIDER:-hf_local}" \
    --embed-model "${RAG_EMBED_MODEL:-BAAI/bge-small-en-v1.5}" \
    --device "${RAG_DEVICE:-cpu}" \
    --watch --interval "${RAG_WATCH_INTERVAL:-5}" \
    --chunk-size "${RAG_CHUNK_SIZE:-900}" \
    --chunk-overlap "${RAG_CHUNK_OVERLAP:-180}" \
    --min-chars "${RAG_MIN_CHARS:-200}" \
    $( [ "${RAG_BGE_PREFIX:-1}" = "0" ] && echo --no-bge-prefix ) \
  >/tmp/rag_watch.log 2>&1 &
fi

echo "[bootstrap] Launching FastAPI on port ${RAG_PORT:-7860} ..."
python app.py
