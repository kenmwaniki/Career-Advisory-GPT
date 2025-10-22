#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Career GPT RAG API — FastAPI over Chroma + HuggingFace embeddings + HuggingFace Inference LLM
Configured for Hugging Face Spaces deployment.

Changes from local version:
- DB_DIR -> /data/chroma_db  (persistent storage)
- PORT   -> 7860             (Spaces default)
- LLM uses Hugging Face Inference API instead of Ollama (Spaces cannot run Ollama)
"""

import os, sys, logging, warnings, time
from typing import List, Optional, Iterable

# -------------------- Quiet warnings --------------------
if not sys.warnoptions:
    warnings.simplefilter("ignore")
for cat in (DeprecationWarning, UserWarning, FutureWarning):
    warnings.filterwarnings("ignore", category=cat)
warnings.filterwarnings("ignore", message=".*LangChainDeprecationWarning.*")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
log = logging.getLogger("rag_api")
for _noisy in ["httpx", "chromadb", "uvicorn", "langchain", "asyncio"]:
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# -------------------- Imports --------------------
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

try:
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFaceEndpoint

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
try:
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    from langchain.schema.runnable import RunnablePassthrough

# -------------------- Config --------------------
ENV = os.getenv
DB_DIR = ENV("RAG_DB_DIR", "/data/chroma_db")       # <-- persistent location
EMBED_PROVIDER = ENV("RAG_EMBED_PROVIDER", "hf_local").lower()
EMBED_MODEL = ENV("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
DEVICE = ENV("RAG_DEVICE", "cpu")
HF_TOKEN = ENV("HUGGINGFACEHUB_API_TOKEN", "")
USE_PREFIX = ENV("RAG_BGE_PREFIX", "1") not in ("0", "false", "False")
EMBED_BATCH = int(ENV("RAG_EMBED_BATCH", "32"))
TOP_K_DEFAULT = int(ENV("RAG_TOP_K", "4"))
FETCH_K = int(ENV("RAG_FETCH_K", str(max(TOP_K_DEFAULT * 3, 12))))
LAMBDA_MMR = float(ENV("RAG_LAMBDA", "0.5"))
NUM_CTX = int(ENV("RAG_NUM_CTX", "8192"))
NUM_PREDICT = int(ENV("RAG_NUM_PREDICT", "512"))
TEMPERATURE = float(ENV("RAG_TEMPERATURE", "0.2"))
FALLBACK_MSG = ENV(
    "RAG_FALLBACK_MSG",
    "I am Career GPT for International Scholars Program and I’m still under training. "
    "I hope I’ll keep learning and improve my responses next time."
)
API_KEY = ENV("RAG_API_KEY")
HOST = ENV("RAG_HOST", "0.0.0.0")
PORT = int(ENV("RAG_PORT", "7860"))                 # <-- Spaces port
CORS_ORIGINS = ENV("RAG_CORS_ORIGINS", "*")

# -------------------- Embeddings --------------------
from langchain.embeddings.base import Embeddings
import numpy as np

def batched(iterable: Iterable, n: int):
    b = []
    for x in iterable:
        b.append(x)
        if len(b) >= n:
            yield b
            b = []
    if b:
        yield b

class BGEAdapter(Embeddings):
    """Add 'passage:' and 'query:' prefixes (BGE best practice)."""
    def __init__(self, base: Embeddings, use_prefixes: bool = True):
        self.base = base
        self.use_prefixes = use_prefixes
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.use_prefixes:
            texts = [f"passage: {t}" for t in texts]
        return self.base.embed_documents(texts)
    def embed_query(self, text: str) -> List[float]:
        if self.use_prefixes:
            text = f"query: {text}"
        return self.base.embed_query(text)

def build_embeddings(provider: str, model: str, device: str,
                     use_prefixes: bool, hf_token: str, batch_size: int) -> Embeddings:
    base = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info(f"Embedding provider: HF local on {device}")
    return BGEAdapter(base, use_prefixes=use_prefixes)

embeddings = build_embeddings(
    provider=EMBED_PROVIDER,
    model=EMBED_MODEL,
    device=DEVICE,
    use_prefixes=USE_PREFIX,
    hf_token=HF_TOKEN,
    batch_size=EMBED_BATCH,
)

# -------------------- Vector DB / Retriever --------------------
vectordb = Chroma(persist_directory=DB_DIR,
                  embedding_function=embeddings,
                  collection_metadata={"hnsw:space": "cosine"})
def build_retriever(k: int):
    return vectordb.as_retriever(search_type="mmr",
                                 search_kwargs={"k": k, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MMR})

# -------------------- LLM (Hugging Face Inference) --------------------
HF_LLM_REPO = ENV("HF_LLM_REPO", "mistralai/Mistral-7B-Instruct-v0.3")
llm = HuggingFaceEndpoint(
    repo_id=HF_LLM_REPO,
    task="text-generation",
    max_new_tokens=NUM_PREDICT,
    temperature=TEMPERATURE,
    timeout=120,
    huggingfacehub_api_token=HF_TOKEN,
)

SYSTEM_RULES = (
    "You are a careful RAG assistant for the International Scholars Program.\n"
    "Use only the information inside <context> to answer.\n"
    "If the answer is not fully supported by the context, say exactly: \"I don’t know.\""
)
prompt = ChatPromptTemplate.from_template(
    f"{SYSTEM_RULES}\n\n<question>\n{{question}}\n</question>\n\n<context>\n{{context}}\n</context>\n\n"
    "Answer concisely and include source tags like [1], [2] where relevant."
)
chain = ({"context": RunnablePassthrough(), "question": RunnablePassthrough()}
         | prompt | llm | StrOutputParser())

# -------------------- Helpers --------------------
def format_docs(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        tag = f"[{i}] ({src}" + (f", p.{page}" if page is not None else "") + ")"
        parts.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(parts)

def fallback_msg() -> str:
    return FALLBACK_MSG

def normalize_unknown(answer: str) -> str:
    lowered = answer.strip().lower()
    for p in ["i don't know", "i do not know", "not in the context", "cannot find", "unsure", "no context"]:
        if p in lowered:
            return fallback_msg()
    return answer

def answer_question(question: str, k: int = TOP_K_DEFAULT) -> dict:
    docs = build_retriever(k).invoke(question)
    if not docs:
        return {"answer": fallback_msg(), "citations": [], "used_k": k}
    context = format_docs(docs)
    try:
        raw = chain.invoke({"question": question, "context": context})
        answer = normalize_unknown(raw)
    except Exception as e:
        log.exception("LLM error")
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")
    cits = [{"index": i,
             "source": d.metadata.get("source", "unknown"),
             "page": d.metadata.get("page")}
            for i, d in enumerate(docs, 1)]
    return {"answer": answer, "citations": cits, "used_k": k}

# -------------------- FastAPI --------------------
app = FastAPI(title="Career GPT RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_api_key(authorization: Optional[str] = Header(None)):
    if API_KEY:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.split(" ", 1)[1].strip()
        if token != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")
    return True

class AskRequest(BaseModel):
    question: str = Field(..., min_length=2)
    top_k: Optional[int] = Field(None, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    citations: list
    used_k: int

@app.get("/healthz")
def healthz():
    try:
        meta = vectordb.get(include=["metadatas"], limit=1)
        return {"status": "ok",
                "db_dir": DB_DIR,
                "docs_indexed": len(meta.get("ids", [])),
                "embed_provider": EMBED_PROVIDER,
                "embed_model": EMBED_MODEL,
                "llm": HF_LLM_REPO}
    except Exception as e:
        log.exception("Health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _ok: bool = Depends(require_api_key)):
    k = req.top_k or TOP_K_DEFAULT
    try:
        return AskResponse(**answer_question(req.question, k=k))
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Unhandled /ask error")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Runner --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
