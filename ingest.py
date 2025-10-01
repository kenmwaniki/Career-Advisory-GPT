#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robust RAG ingestion for Chroma + Ollama embeddings.

Now supports: PDF, HTML, DOCX, TXT/MD, CSV (+ watch mode)

Features
- Incremental ingest with stable chunk IDs (sha1(file|page|content)) -> no dup chunks
- Rich metadata for citations: source, source_path, page, mtime, sha1
- Clean/normalize text (page number + whitespace heuristics)
- Config via CLI flags and env vars
- Rebuild & dry-run modes, detailed logs
- Optional watch mode (polling) for auto-reindex on file changes
"""

import os
import re
import sys
import time
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Iterable, Dict, Any, Optional

from tqdm import tqdm
from unidecode import unidecode

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

# Loaders
from langchain_community.document_loaders import (
    PyMuPDFLoader,         # PDF
    BSHTMLLoader,          # HTML (BeautifulSoup)
    Docx2txtLoader,        # DOCX
    TextLoader,            # TXT/MD
    CSVLoader,             # CSV
)

# -------------------- Defaults (overridable via env) --------------------
ENV = os.getenv
DEFAULT_DOCS_DIR = ENV("RAG_DOCS_DIR", "docs")
DEFAULT_DB_DIR = ENV("RAG_DB_DIR", "chroma_db")
DEFAULT_EMBED_MODEL = ENV("RAG_EMBED_MODEL", "mxbai-embed-large")

DEFAULT_CHUNK_SIZE = int(ENV("RAG_CHUNK_SIZE", "900"))
DEFAULT_CHUNK_OVERLAP = int(ENV("RAG_CHUNK_OVERLAP", "180"))
DEFAULT_MIN_CHARS = int(ENV("RAG_MIN_CHARS", "200"))  # drop tiny chunks

DEFAULT_WATCH_INTERVAL = int(ENV("RAG_WATCH_INTERVAL", "5"))  # seconds

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ingest")

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

def assign_common_metadata(doc: Document, path: Path, page: Optional[int] = None):
    doc.metadata = dict(doc.metadata or {})
    doc.metadata["source"] = path.name
    doc.metadata["source_path"] = str(path)
    if page is not None and doc.metadata.get("page") is None:
        doc.metadata["page"] = page
    doc.metadata["mtime"] = int(path.stat().st_mtime)

def load_pdf(path: Path) -> List[Document]:
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    out = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, d.metadata.get("page"))
        out.append(d)
    return out

def load_html(path: Path) -> List[Document]:
    # BSHTMLLoader extracts visible text; override metadata + normalize
    loader = BSHTMLLoader(str(path))
    docs = loader.load()
    out = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        # try to keep per-file granularity; BSHTMLLoader may not set page
        assign_common_metadata(d, path, None)
        out.append(d)
    return out

def load_docx(path: Path) -> List[Document]:
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    out = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, None)
        out.append(d)
    return out

def load_text_like(path: Path) -> List[Document]:
    loader = TextLoader(str(path), autodetect_encoding=True)
    docs = loader.load()
    out = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, None)
        out.append(d)
    return out

def load_csv(path: Path) -> List[Document]:
    """
    Loads CSV as one Document per row, including header mapping in content.
    CSVLoader defaults: delimiter=",", encoding="utf-8"
    """
    loader = CSVLoader(str(path))
    docs = loader.load()
    out = []
    for d in docs:
        d.page_content = normalize_text(d.page_content)
        assign_common_metadata(d, path, None)
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

def chunk_documents(raw_docs: List[Document],
                    chunk_size: int,
                    chunk_overlap: int,
                    min_chars: int) -> List[Document]:
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

def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def batched(iterable: Iterable, n: int) -> Iterable[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

# -------------------- Ingest Core --------------------
def ingest_once(
    docs_dir: Path,
    db_dir: Path,
    embed_model: str,
    rebuild: bool = False,
    dry_run: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> Dict[str, Any]:

    ensure_dirs(docs_dir)
    ensure_dirs(db_dir)

    if rebuild and not dry_run:
        # wipe existing DB dir
        for p in db_dir.glob("**/*"):
            try:
                if p.is_file():
                    p.unlink()
            except Exception:
                pass
        for p in sorted(db_dir.glob("**/*"), reverse=True):
            try:
                if p.is_dir():
                    p.rmdir()
            except Exception:
                pass
        ensure_dirs(db_dir)
        log.warning("Rebuild mode: existing DB wiped.")

    log.info(f"Using embeddings: {embed_model}")
    embeddings = OllamaEmbeddings(model=embed_model)

    vectordb = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    files = discover_files(docs_dir)
    if not files:
        log.warning(f"No supported files found in {docs_dir.resolve()}")
        return {"added": 0, "skipped": 0, "total_chunks": 0, "files": 0}

    # load docs per file type
    raw_docs: List[Document] = []
    for path in tqdm(files, desc="Loading files"):
        try:
            suf = path.suffix.lower()
            if suf == ".pdf":
                raw_docs.extend(load_pdf(path))
            elif suf in (".html", ".htm"):
                raw_docs.extend(load_html(path))
            elif suf == ".docx":
                raw_docs.extend(load_docx(path))
            elif suf in (".txt", ".md", ".markdown"):
                raw_docs.extend(load_text_like(path))
            elif suf == ".csv":
                raw_docs.extend(load_csv(path))
        except Exception as e:
            log.error(f"Failed to load {path}: {e}")

    if not raw_docs:
        log.warning("No documents loaded after parsing.")
        return {"added": 0, "skipped": 0, "total_chunks": 0, "files": len(files)}

    chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap, min_chars)
    if not chunks:
        log.warning("No chunks produced (check chunking params / min_chars).")
        return {"added": 0, "skipped": 0, "total_chunks": 0, "files": len(files)}

    ids = [make_chunk_id(c) for c in chunks]

    # find existing ids (batched)
    existing = set()
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
    # write in batches
    docs_batches = list(batched(to_add_docs, 256))
    ids_batches = list(batched(to_add_ids, 256))
    for batch_docs, batch_ids in zip(docs_batches, ids_batches):
        try:
            Chroma.add_documents(vectordb, documents=batch_docs, ids=batch_ids)  # explicit to be clear
            added += len(batch_docs)
        except Exception as e:
            log.error(f"Error adding batch: {e}")

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
    }

# -------------------- Watch Mode (polling) --------------------
def build_mtime_index(docs_dir: Path) -> Dict[str, float]:
    idx: Dict[str, float] = {}
    for f in discover_files(docs_dir):
        try:
            idx[str(f)] = f.stat().st_mtime
        except Exception:
            pass
    return idx

def watch_and_ingest(
    docs_dir: Path,
    db_dir: Path,
    embed_model: str,
    interval: int,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
):
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
            # We keep the same dedup logic; only new/changed content will be added
            summary = ingest_once(
                docs_dir=docs_dir,
                db_dir=db_dir,
                embed_model=embed_model,
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
    p.add_argument("--docs", default=DEFAULT_DOCS_DIR, help=f"Docs directory (default: {DEFAULT_DOCS_DIR})")
    p.add_argument("--db", default=DEFAULT_DB_DIR, help=f"Chroma DB directory (default: {DEFAULT_DB_DIR})")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=f"Ollama embedding model (default: {DEFAULT_EMBED_MODEL})")
    p.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the DB")
    p.add_argument("--dry-run", action="store_true", help="Do everything except write to DB")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Chunk size (default: {DEFAULT_CHUNK_SIZE})")
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help=f"Chunk overlap (default: {DEFAULT_CHUNK_OVERLAP})")
    p.add_argument("--min-chars", type=int, default=DEFAULT_MIN_CHARS, help=f"Drop chunks shorter than this (default: {DEFAULT_MIN_CHARS})")
    p.add_argument("--watch", action="store_true", help="Watch for file changes and ingest incrementally (polling)")
    p.add_argument("--interval", type=int, default=DEFAULT_WATCH_INTERVAL, help=f"Watch poll interval seconds (default: {DEFAULT_WATCH_INTERVAL})")
    return p.parse_args()

def main():
    args = parse_args()
    docs_dir = Path(args.docs)
    db_dir = Path(args.db)

    log.info(f"Docs dir: {docs_dir.resolve()}")
    log.info(f"DB dir:   {db_dir.resolve()}")
    log.info(f"Embed:    {args.embed_model}")

    try:
        if args.watch:
            # Prime the DB once before watching
            summary = ingest_once(
                docs_dir=docs_dir,
                db_dir=db_dir,
                embed_model=args.embed_model,
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
                embed_model=args.embed_model,
                interval=args.interval,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                min_chars=args.min_chars,
            )
        else:
            summary = ingest_once(
                docs_dir=docs_dir,
                db_dir=db_dir,
                embed_model=args.embed_model,
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
    except Exception as e:
        log.exception("Ingest failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
