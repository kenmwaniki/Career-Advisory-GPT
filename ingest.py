#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust RAG ingestion for Chroma + Embeddings (HF Local / HF Inference API / Ollama)

Supported file types: PDF, HTML, DOCX, TXT/MD, CSV (+ watch mode)

Features
- Incremental ingest with stable chunk IDs (sha1(file|page|content)) -> no dup chunks
- Rich metadata for citations: source, source_path, page, mtime
- Clean/normalize text (page number + whitespace heuristics)
- Config via CLI flags and env vars
- Rebuild & dry-run modes, detailed logs
- Optional watch mode (polling) for auto-reindex on file changes
- Embeddings providers:
    * hf_local     -> sentence-transformers (device: cpu/cuda)
    * hf_inference -> Hugging Face Inference API (token required)
    * ollama       -> OllamaEmbeddings (existing behavior)

BGE best practices:
- L2 normalization (cosine space)
- Prefix: "passage: " for docs, "query: " for queries (toggle with --no-bge-prefix)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from unidecode import unidecode

# LangChain + Chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from langchain.embeddings.base import Embeddings
from chromadb.config import Settings as ChromaSettings

# Loaders
from langchain_community.document_loaders import (
    PyMuPDFLoader,  # PDF
    BSHTMLLoader,   # HTML (BeautifulSoup)
    Docx2txtLoader, # DOCX
    TextLoader,     # TXT/MD
    CSVLoader,      # CSV
)

# -------------------- Defaults (overridable via env) --------------------
ENV = os.getenv
DEFAULT_DOCS_DIR = ENV("RAG_DOCS_DIR", "docs")
DEFAULT_DB_DIR = ENV("RAG_DB_DIR", "chroma_db")

# Provider: "hf_local" | "hf_inference" | "ollama"
DEFAULT_EMBED_PROVIDER = ENV("RAG_EMBED_PROVIDER", "hf_local")
DEFAULT_EMBED_MODEL = ENV("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Device for hf_local
DEFAULT_DEVICE = ENV("RAG_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")

# HF token for hf_inference (or for gated/private models if needed)
DEFAULT_HF_TOKEN = ENV("HUGGINGFACEHUB_API_TOKEN", ENV("HF_TOKEN", ""))

DEFAULT_USE_PREFIX = ENV("RAG_BGE_PREFIX", "1") not in ("0", "false", "False")

DEFAULT_CHUNK_SIZE = int(ENV("RAG_CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(ENV("RAG_CHUNK_OVERLAP", "180"))
DEFAULT_MIN_CHARS = int(ENV("RAG_MIN_CHARS", "200"))  # drop tiny chunks

DEFAULT_WATCH_INTERVAL = int(ENV("RAG_WATCH_INTERVAL", "5"))  # seconds
DEFAULT_BATCH_SIZE = int(ENV("RAG_EMBED_BATCH", "32"))        # embedding batch size (hf_inference wrapper)

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("rag_ingest")

# -------------------- Helpers --------------------
def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def normalize_text(txt: str) -> str:
    """
    Normalize/clean text for better retrieval.
    - ascii transliteration to reduce unicode noise
    - strip trailing spaces
    - drop lines that are just numbers (page numbers)
    - collapse excessive blank lines/spaces
    """
    if not txt:
        return ""
    txt = unidecode(txt)
    lines = [re.sub(r"\s+$", "", line) for line in txt.splitlines()]
    lines = [l for l in lines if not re.fullmatch(r"\d{1,4}", l) and l.strip()]
    txt = "\n".join(lines)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"[ \t]{2,}", " ", txt)
    return txt.strip()

def assign_common_metadata(doc: Document, path: Path, page: Optional[int] = None) -> None:
    doc.metadata = dict(doc.metadata or {})
    doc.metadata["source"] = path.name
    doc.metadata["source_path"] = str(path.resolve())
    if page is not None and doc.metadata.get("page") is None:
        doc.metadata["page"] = page
    try:
        doc.metadata["mtime"] = int(path.stat().st_mtime)
    except OSError:
        doc.metadata["mtime"] = 0

def load_pdf(path: Path) -> List[Document]:
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    out: List[Document] = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, d.metadata.get("page"))
        if d.page_content:
            out.append(d)
    return out

def load_html(path: Path) -> List[Document]:
    loader = BSHTMLLoader(str(path))
    docs = loader.load()
    out: List[Document] = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, None)
        if d.page_content:
            out.append(d)
    return out

def load_docx(path: Path) -> List[Document]:
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    out: List[Document] = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, None)
        if d.page_content:
            out.append(d)
    return out

def load_text_like(path: Path) -> List[Document]:
    loader = TextLoader(str(path), autodetect_encoding=True)
    docs = loader.load()
    out: List[Document] = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, None)
        if d.page_content:
            out.append(d)
    return out

def load_csv(path: Path) -> List[Document]:
    """
    Loads CSV as one Document per row, including header mapping in content.
    """
    loader = CSVLoader(str(path))
    docs = loader.load()
    out: List[Document] = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, None)
        if d.page_content:
            out.append(d)
    return out

SUPPORTED_SUFFIXES = {
    ".pdf", ".html", ".htm", ".docx", ".txt", ".md", ".markdown", ".csv"
}

def discover_files(docs_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(p)
    return files

def chunk_documents(
    raw_docs: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    chunks = [c for c in chunks if len(c.page_content.strip()) >= min_chars]
    return chunks

def make_chunk_id(doc: Document) -> str:
    src = doc.metadata.get("source_path", doc.metadata.get("source", "unknown"))
    page = str(doc.metadata.get("page"))
    basis = f"{src}|{page}|{doc.page_content}"
    return sha1(basis)

def ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def batched(iterable: Iterable[Any], n: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

# -------------------- Embedding Adapters --------------------
class BGEAdapter(Embeddings):
    """
    Wraps any LangChain Embeddings and applies BGE prefixes:
      - "passage: " for embed_documents
      - "query: "   for embed_query
    """
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

class HFInferenceEmbeddings(Embeddings):
    """
    Minimal embeddings wrapper using Hugging Face Inference API feature-extraction.
    - Mean-pools token embeddings
    - L2-normalizes vectors
    - Works for any feature-extraction model (e.g., BAAI/bge-small-en-v1.5)
    """
    def __init__(
        self,
        model: str,
        token: str,
        timeout: float = 60.0,
        max_retries: int = 5,
        batch_size: int = 32,
    ):
        from huggingface_hub import InferenceClient  # lazy import
        if not token:
            raise ValueError("HF Inference API requires a token. Set HUGGINGFACEHUB_API_TOKEN or pass --hf-token.")
        self.client = InferenceClient(token=token, timeout=timeout)
        self.model = model
        self.max_retries = max_retries
        self.batch_size = max(1, batch_size)

    @staticmethod
    def _mean_pool(mat: List[List[float]]) -> List[float]:
        arr = np.asarray(mat, dtype=np.float32)
        v = arr.mean(axis=0)
        norm = np.linalg.norm(v) + 1e-12
        return (v / norm).tolist()

    def _fe(self, text: str) -> List[float]:
        for i in range(self.max_retries):
            try:
                # Returns [seq_len x dim]
                mat = self.client.feature_extraction(model=self.model, inputs=text)
                return self._mean_pool(mat)
            except Exception as e:
                # Respect rate limits/backoffs
                if i == self.max_retries - 1:
                    raise
                sleep_s = max(0.5, 2 ** i * 0.5)
                log.warning(f"HF Inference backoff ({i+1}/{self.max_retries}): {e}. Sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)
        return []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for batch in batched(texts, self.batch_size):
            for t in batch:
                out.append(self._fe(t))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._fe(text)

def build_embeddings(
    provider: str,
    model: str,
    device: str,
    use_prefixes: bool,
    hf_token: str,
    batch_size: int,
) -> Embeddings:
    provider = provider.lower()
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings  # lazy import
        base = OllamaEmbeddings(model=model)
        log.info("Embedding provider: Ollama")
        return BGEAdapter(base, use_prefixes=use_prefixes)

    if provider == "hf_inference":
        base = HFInferenceEmbeddings(model=model, token=hf_token, batch_size=batch_size)
        log.info("Embedding provider: HF Inference API")
        return BGEAdapter(base, use_prefixes=use_prefixes)

    # hf_local
    from langchain_community.embeddings import HuggingFaceEmbeddings  # lazy import
    base = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},  # cosine-friendly
    )
    log.info(f"Embedding provider: HF local (sentence-transformers) on {device}")
    return BGEAdapter(base, use_prefixes=use_prefixes)

# -------------------- Ingest Core --------------------
def _wipe_dir(path: Path) -> None:
    if not path.exists():
        return
    for p in sorted(path.glob("**/*"), reverse=True):
        try:
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except Exception as e:
            log.debug(f"Skipping removal for {p}: {e}")

def _build_vectordb(db_dir: Path, embeddings: Embeddings) -> Chroma:
    client_settings = ChromaSettings(
        is_persistent=True,
        persist_directory=str(db_dir),
        anonymized_telemetry=False,
    )
    # Note: collection_metadata space "cosine" matches normalized embeddings
    return Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        client_settings=client_settings,
    )

def _load_docs_for_paths(files: List[Path]) -> List[Document]:
    loaders = {
        ".pdf": load_pdf,
        ".html": load_html,
        ".htm": load_html,
        ".docx": load_docx,
        ".txt": load_text_like,
        ".md": load_text_like,
        ".markdown": load_text_like,
        ".csv": load_csv,
    }
    raw_docs: List[Document] = []
    for path in tqdm(files, desc="Loading files", unit="file"):
        try:
            fn = loaders.get(path.suffix.lower())
            if fn:
                raw_docs.extend(fn(path))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")
    return raw_docs

def ingest_once(
    docs_dir: Path,
    db_dir: Path,
    embed_provider: str,
    embed_model: str,
    device: str,
    use_prefixes: bool,
    hf_token: str,
    batch_size: int,
    rebuild: bool = False,
    dry_run: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Dict[str, Any]:

    ensure_dirs(docs_dir)
    ensure_dirs(db_dir)

    if rebuild and not dry_run:
        _wipe_dir(db_dir)
        ensure_dirs(db_dir)
        log.warning("Rebuild mode: existing DB wiped.")

    log.info(f"Using embeddings: model={embed_model} provider={embed_provider}")
    embeddings = build_embeddings(
        provider=embed_provider,
        model=embed_model,
        device=device,
        use_prefixes=use_prefixes,
        hf_token=hf_token,
        batch_size=batch_size,
    )

    vectordb = _build_vectordb(db_dir, embeddings)

    files = discover_files(docs_dir)
    if not files:
        log.warning(f"No supported files found in {docs_dir.resolve()}")
        return {"added": 0, "skipped": 0, "total_chunks": 0, "files": 0}

    raw_docs = _load_docs_for_paths(files)
    if not raw_docs:
        log.warning("No documents loaded after parsing.")
        return {"added": 0, "skipped": 0, "total_chunks": 0, "files": len(files)}

    chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap, min_chars)
    if not chunks:
        log.warning("No chunks produced (check chunking params / min_chars).")
        return {"added": 0, "skipped": 0, "total_chunks": 0, "files": len(files)}

    ids = [make_chunk_id(c) for c in chunks]

    # find existing ids (batched)
    existing: set[str] = set()
    for batch in batched(ids, 500):
        try:
            res = vectordb.get(ids=batch)
            if res and res.get("ids"):
                existing.update(res["ids"])
        except Exception:
            # .get may raise for non-existing ids; safe to ignore
            pass

    to_add_docs: List[Document] = []
    to_add_ids: List[str] = []
    skipped = 0

    for doc, _id in zip(chunks, ids):
        if _id in existing:
            skipped += 1
            continue
        to_add_docs.append(doc)
        to_add_ids.append(_id)

    log.info(f"Total chunks: {len(chunks)} | To add: {len(to_add_docs)} | Skipped (dups): {skipped}")

    if dry_run:
        log.info("Dry-run mode: not writing to DB.")
        return {
            "added": len(to_add_docs),
            "skipped": skipped,
            "total_chunks": len(chunks),
            "files": len(files),
            "dry_run": True,
        }

    added = 0
    for batch_docs, batch_ids in zip(batched(to_add_docs, 256), batched(to_add_ids, 256)):
        try:
            # explicit call to make intent clear
            Chroma.add_documents(vectordb, documents=batch_docs, ids=batch_ids)
            added += len(batch_docs)
        except Exception as e:
            log.error(f"Error adding batch ({len(batch_docs)} docs): {e}")

    try:
        vectordb.persist()
    except Exception as e:
        log.error(f"Persist error: {e}")

    return {
        "added": added,
        "skipped": skipped,
        "total_chunks": len(chunks),
        "files": len(files),
        "db_dir": str(db_dir.resolve()),
        "embed_model": embed_model,
        "embed_provider": embed_provider,
    }

# -------------------- Watch Mode (polling) --------------------
def build_mtime_index(docs_dir: Path) -> Dict[str, float]:
    idx: Dict[str, float] = {}
    for f in discover_files(docs_dir):
        try:
            idx[str(f.resolve())] = f.stat().st_mtime
        except Exception:
            pass
    return idx

def watch_and_ingest(
    docs_dir: Path,
    db_dir: Path,
    embed_provider: str,
    embed_model: str,
    device: str,
    use_prefixes: bool,
    hf_token: str,
    batch_size: int,
    interval: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
) -> None:
    log.info(f"Watching {docs_dir.resolve()} every {interval}s for changes...")
    baseline = build_mtime_index(docs_dir)
    while True:
        time.sleep(interval)
        curr = build_mtime_index(docs_dir)

        added_paths = [p for p in curr.keys() if p not in baseline]
        changed_paths = [p for p, mt in curr.items() if p in baseline and mt > baseline[p]]
        removed_paths = [p for p in baseline.keys() if p not in curr]

        if not (added_paths or changed_paths or removed_paths):
            continue

        if removed_paths:
            log.warning(f"{len(removed_paths)} files removed since last scan (not deleting existing vectors).")

        if added_paths or changed_paths:
            log.info(f"Detected {len(added_paths)} new and {len(changed_paths)} modified files. Re-ingesting incrementally...")
            summary = ingest_once(
                docs_dir=Path(docs_dir),
                db_dir=Path(db_dir),
                embed_provider=embed_provider,
                embed_model=embed_model,
                device=device,
                use_prefixes=use_prefixes,
                hf_token=hf_token,
                batch_size=batch_size,
                rebuild=False,
                dry_run=False,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chars=min_chars,
            )
            log.info(f"Watch ingest summary: {summary}")

        baseline = curr

# -------------------- CLI --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest documents (PDF/HTML/DOCX/TXT/MD/CSV) into Chroma for RAG.")
    # I/O
    p.add_argument("--docs", default=DEFAULT_DOCS_DIR, help=f"Docs directory (default: {DEFAULT_DOCS_DIR})")
    p.add_argument("--db", default=DEFAULT_DB_DIR, help=f"Chroma DB directory (default: {DEFAULT_DB_DIR})")

    # Embeddings
    p.add_argument("--embed-provider", default=DEFAULT_EMBED_PROVIDER,
                   choices=["hf_local", "hf_inference", "ollama"],
                   help=f"Embedding provider (default: {DEFAULT_EMBED_PROVIDER})")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                   help=f"Embedding model name (default: {DEFAULT_EMBED_MODEL})")
    p.add_argument("--device", default=DEFAULT_DEVICE, help=f"'cpu' or 'cuda' (hf_local only, default: {DEFAULT_DEVICE})")
    p.add_argument("--hf-token", default=DEFAULT_HF_TOKEN, help="HF token (hf_inference or gated/private models)")
    p.add_argument("--no-bge-prefix", action="store_true", help="Disable 'passage:/query:' prefixes for embeddings")
    p.add_argument("--embed-batch", type=int, default=DEFAULT_BATCH_SIZE, help=f"Embedding batch size (hf_inference): default {DEFAULT_BATCH_SIZE}")

    # Ingest
    p.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the DB")
    p.add_argument("--dry-run", action="store_true", help="Do everything except write to DB")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Chunk size (default: {DEFAULT_CHUNK_SIZE})")
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})")
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help=f"Drop chunks shorter than this (default: {DEFAULT_MIN_CHARS})")

    # Watch
    p.add_argument("--watch", action="store_true", help="Watch for file changes and ingest incrementally (polling)")
    p.add_argument("--interval", type=int, default=DEFAULT_WATCH_INTERVAL, help=f"Watch poll interval seconds (default: {DEFAULT_WATCH_INTERVAL})")
    return p.parse_args()

def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        names = {signal.SIGINT: "SIGINT", signal.SIGTERM: "SIGTERM"}
        log.warning(f"Received {names.get(signum, signum)}. Exiting gracefully...")
        sys.exit(130 if signum == signal.SIGINT else 143)

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        # Not all environments allow signal hooks (e.g., Windows threads)
        pass

def main() -> None:
    _install_signal_handlers()
    args = parse_args()
    docs_dir = Path(args.docs)
    db_dir = Path(args.db)

    use_prefixes = not args.no_bge_prefix

    log.info(f"Docs dir: {docs_dir.resolve()}")
    log.info(f"DB dir:   {db_dir.resolve()}")
    log.info(f"Embed:    provider={args.embed_provider} model={args.embed_model}")

    try:
        if args.watch:
            # Prime the DB once before watching
            summary = ingest_once(
                docs_dir=docs_dir,
                db_dir=db_dir,
                embed_provider=args.embed_provider,
                embed_model=args.embed_model,
                device=args.device,
                use_prefixes=use_prefixes,
                hf_token=args.hf_token,
                batch_size=args.embed_batch,
                rebuild=args.rebuild,
                dry_run=args.dry_run,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
            )
            log.info(f"Initial ingest summary: {summary}")
            if args.dry_run:
                log.info("Dry-run set; skipping watch loop.")
                return
            watch_and_ingest(
                docs_dir=docs_dir,
                db_dir=db_dir,
                embed_provider=args.embed_provider,
                embed_model=args.embed_model,
                device=args.device,
                use_prefixes=use_prefixes,
                hf_token=args.hf_token,
                batch_size=args.embed_batch,
                interval=args.interval,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
            )
        else:
            summary = ingest_once(
                docs_dir=docs_dir,
                db_dir=db_dir,
                embed_provider=args.embed_provider,
                embed_model=args.embed_model,
                device=args.device,
                use_prefixes=use_prefixes,
                hf_token=args.hf_token,
                batch_size=args.embed_batch,
                rebuild=args.rebuild,
                dry_run=args.dry_run,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
            )
            log.info(f"Ingest summary: {summary}")
    except KeyboardInterrupt:
        log.warning("Interrupted.")
        sys.exit(130)
    except Exception:
        log.exception("Ingest failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
