#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG API (FastAPI) over Chroma + Ollama
- /healthz GET: health check
- /ask POST: { "question": "...", "top_k": 4 } -> answer with citations

Env vars (all optional):
- RAG_DB_DIR=chroma_db
- RAG_EMBED_MODEL=mxbai-embed-large
- RAG_LLM_MODEL=llama3:instruct
- RAG_TOP_K=4
- RAG_FETCH_K=16
- RAG_LAMBDA=0.5
- RAG_NUM_CTX=8192
- RAG_NUM_PREDICT=512
- RAG_TEMPERATURE=0.2
- RAG_FALLBACK_MSG="I am Career GPT for International Scholars Program..."
- RAG_API_KEY=<string>   # if set, require header: Authorization: Bearer <key>
- RAG_HOST=0.0.0.0
- RAG_PORT=8000
- RAG_CORS_ORIGINS=*
"""

import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# RunnablePassthrough moved in newer LangChain; support both locations
try:
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    from langchain.schema.runnable import RunnablePassthrough

# -------------------- Config --------------------
DB_DIR = os.getenv("RAG_DB_DIR", "chroma_db")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3:instruct")
TOP_K_DEFAULT = int(os.getenv("RAG_TOP_K", "4"))
FETCH_K = int(os.getenv("RAG_FETCH_K", str(max(TOP_K_DEFAULT * 3, 12))))
LAMBDA_MMR = float(os.getenv("RAG_LAMBDA", "0.5"))

NUM_CTX = int(os.getenv("RAG_NUM_CTX", "8192"))
NUM_PREDICT = int(os.getenv("RAG_NUM_PREDICT", "512"))
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))

FALLBACK_MSG = os.getenv(
    "RAG_FALLBACK_MSG",
    "I am Career GPT for International Scholars Program and I’m still under training. "
    "I hope I’ll keep learning and improve my responses next time."
)

API_KEY = os.getenv("RAG_API_KEY")  # if set, require Bearer token
HOST = os.getenv("RAG_HOST", "0.0.0.0")
PORT = int(os.getenv("RAG_PORT", "8000"))
CORS_ORIGINS = os.getenv("RAG_CORS_ORIGINS", "*")

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("rag_api")

# -------------------- Vector DB / Retriever --------------------
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Use MMR for diverse retrieval
def build_retriever(k: int):
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MMR},
    )

# -------------------- LLM --------------------
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=TEMPERATURE,
    num_ctx=NUM_CTX,
    num_predict=NUM_PREDICT,
)

SYSTEM_RULES = (
    "You are a careful RAG assistant for the International Scholars Program.\n"
    "Use only the information inside <context> to answer.\n"
    "If the answer is not fully supported by the context, say exactly: \"I don’t know.\""
)

prompt = ChatPromptTemplate.from_template(
    f"{SYSTEM_RULES}\n\n"
    "<question>\n{question}\n</question>\n\n"
    "<context>\n{context}\n</context>\n\n"
    "Answer concisely and include source tags like [1], [2] where relevant."
)

output_parser = StrOutputParser()

# RAG chain graph:
# inputs -> {context, question} -> prompt -> llm -> parser -> string
chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | llm
    | output_parser
)

# -------------------- Helpers --------------------
def format_docs(docs) -> str:
    """
    Join retrieved docs into a single context string; include source + page when available.
    """
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f"[{i}] ({src}" + (f", p.{page}" if page is not None else "") + ")"
        parts.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(parts)


def custom_fallback() -> str:
    return FALLBACK_MSG


def normalize_unknown(answer: str) -> str:
    lowered = answer.strip().lower()
    phrases = [
        "i don't know", "i do not know", "not in the context",
        "cannot find", "unsure", "no context"
    ]
    if any(p in lowered for p in phrases):
        return custom_fallback()
    return answer


def answer_question(question: str, k: int = TOP_K_DEFAULT) -> dict:
    """
    Returns a dict: { answer: str, citations: [ {index:int, source:str, page:int|None} ], used_k:int }
    """
    retriever = build_retriever(k)
    docs = retriever.get_relevant_documents(question)

    if not docs:
        return {"answer": custom_fallback(), "citations": [], "used_k": k}

    context = format_docs(docs)
    try:
        raw = chain.invoke({"question": question, "context": context})
        answer = normalize_unknown(raw)
    except Exception as e:
        log.exception("LLM error")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    # Build citations summary (index + source + page)
    cits = []
    for i, d in enumerate(docs, 1):
        cits.append({
            "index": i,
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page"),
        })

    return {"answer": answer, "citations": cits, "used_k": k}


# -------------------- FastAPI --------------------
app = FastAPI(title="Career GPT RAG API", version="1.0.0")

# CORS (wide-open by default; tighten in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_api_key(authorization: Optional[str] = Header(None)):
    """Optional Bearer token auth, enabled if RAG_API_KEY is set."""
    if API_KEY:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ", 1)[1].strip()
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")
    return True


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override retrieval K")


class AskResponse(BaseModel):
    answer: str
    citations: List[dict]
    used_k: int


@app.get("/healthz")
def healthz():
    # verify DB can be opened and embeddings are present
    try:
        _ = vectordb._collection.count()  # quick sanity check
        return {"status": "ok", "db_dir": DB_DIR, "embed_model": EMBED_MODEL, "llm": LLM_MODEL}
    except Exception as e:
        log.exception("Health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _ok: bool = Depends(require_api_key)):
    try:
        k = req.top_k or TOP_K_DEFAULT
        result = answer_question(req.question, k=k)
        return AskResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled /ask error")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- CLI server --------------------
if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting RAG API on {HOST}:{PORT} (DB={DB_DIR}, EMBED={EMBED_MODEL}, LLM={LLM_MODEL})")
    uvicorn.run("rag_qa:app", host=HOST, port=PORT, reload=False)

