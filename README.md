# Career GPT RAG API (Hugging Face Space)

FastAPI RAG service over Chroma + embeddings, deployed as a Docker Space.
- **Persistent DB**: `/data/chroma_db`
- **Uploads folder**: `/data/uploads`
- **API port**: `7860`

## Endpoints
- `GET /healthz` — service health
- `POST /ask` — RAG answer (requires header `Authorization: Bearer <RAG_API_KEY>`)

## Environment Variables (set in Space → Settings → Variables & secrets)
- `RAG_API_KEY` — required for `/ask`
- `RAG_DB_DIR=/data/chroma_db`
- `HUGGINGFACEHUB_API_TOKEN` — if using HF Inference LLM
- `HF_LLM_REPO=mistralai/Mistral-7B-Instruct-v0.3` — default in app.py

### Ingestion options
- `RAG_REINGEST=1` — force re-ingest on boot
- `RAG_REBUILD=1` — wipe DB first (use with care)
- `RAG_WATCH=1` — watch `/data/uploads` and re-index incrementally

### Uploading documents
Upload `.pdf`, `.html`, `.docx`, `.txt`, `.md`, `.csv` to `/data/uploads` via the Space Files UI, then set `RAG_REINGEST=1` (and restart) or enable `RAG_WATCH=1`.

### Quick test

```bash
curl -X POST "https://<your-space>.hf.space/ask" \
  -H "Authorization: Bearer $RAG_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the International Scholars Program?"}'
