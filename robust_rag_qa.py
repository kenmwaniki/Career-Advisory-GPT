#!/usr/bin/env python3
"""
robust_rag_qa.py — an industry-grade RAG CLI for Ollama + LangChain + Chroma

Highlights
- Argparse CLI with sane defaults and env overrides
- Choice of retrieval mode: similarity or MMR (diverse)
- Optional context compression (LLM-based) for lower token cost
- Structured citations (prints Sources: with file + page when available)
- One-shot mode (default) and interactive REPL mode (--repl)
- Fast fail with clear diagnostics if DB/models are missing
- Safe generation caps (num_predict, stop sequences), and temperature flag
- Optional JSON output (--json) for programmatic use
- Streaming output (--stream) for responsive UX

Prereqs
  pip install "langchain>=0.2" langchain-community langchain-ollama chromadb
  # Ensure Ollama is running:  ollama serve
  # Ensure models exist:       ollama pull mxbai-embed-large; ollama pull llama3:instruct
  # Ensure you have ingested docs into a Chroma db at --db (default: ./chroma_db)

Example
  python robust_rag_qa.py "What does the refund policy say?"
  python robust_rag_qa.py --repl
  python robust_rag_qa.py --k 6 --mmr --num-predict 256 "Summarize chapter 3"
  python robust_rag_qa.py --json "Key SLAs?"
  python robust_rag_qa.py --stream "Give 5 bullets from the guide"
"""
import os
import sys
import json
import signal
import argparse
from datetime import datetime
from typing import List, Optional, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

# RunnablePassthrough moved in newer LangChain; support both locations
try:
    from langchain_core.runnables import RunnablePassthrough
except ImportError:  # pragma: no cover
    from langchain.schema.runnable import RunnablePassthrough

# Optional contextual compression (can be toggled via --compress)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# ------------------------- Defaults & ENV -------------------------
DEF_DB_DIR = os.getenv("RAG_DB_DIR", "chroma_db")
DEF_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "mxbai-embed-large")
DEF_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "llama3:instruct")
DEF_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
DEF_NUM_PREDICT = int(os.getenv("RAG_NUM_PREDICT", "256"))
DEF_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
DEF_STOP = os.getenv("RAG_STOP", "").split("||") if os.getenv("RAG_STOP") else []

PROMPT_TMPL = (
    "You are a precise assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say: 'I don't know.'\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer (concise; cite like [1], [2] when referring to items):"
)

# ------------------------- Utilities -------------------------

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def fail(msg: str, code: int = 1) -> None:
    log(f"ERROR: {msg}")
    sys.exit(code)


def format_docs(docs) -> str:
    """Join retrieved docs into a single context string with indices & source hints."""
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        p = f", p.{page}" if page is not None else ""
        parts.append(f"[{i}] ({src}{p})\n{d.page_content}")
    return "\n\n".join(parts)


def summarize_sources(docs) -> List[str]:
    items = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        p = f", p.{page}" if page is not None else ""
        items.append(f"[{i}] {src}{p}")
    return items


# ------------------------- Core Builder -------------------------

def build_retriever(db_dir: str, embed_model: str, k: int, mmr: bool,
                    compress: bool, llm_model: str, temperature: float,
                    num_predict: int):
    if not os.path.isdir(db_dir):
        fail(f"Chroma DB directory not found: {db_dir}. Did you run ingestion?")

    embeddings = OllamaEmbeddings(model=embed_model)
    vectordb = Chroma(persist_directory=db_dir, embedding_function=embeddings)

    if mmr:
        base = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": k, "fetch_k": max(12, 3 * k), "lambda_mult": 0.5},
        )
    else:
        base = vectordb.as_retriever(search_kwargs={"k": k})

    if not compress:
        return base

    compressor = LLMChainExtractor.from_llm(
        ChatOllama(model=llm_model, temperature=temperature, num_predict=num_predict)
    )
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base)


def build_chain(retriever, llm_model: str, temperature: float, num_predict: int, stops: List[str]):
    llm = ChatOllama(
        model=llm_model,
        temperature=temperature,
        num_predict=num_predict,
        stop=stops if stops else None,
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TMPL)
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm)
    return chain


# ------------------------- Execution Paths -------------------------

def answer_once(question: str, chain, retriever, json_out: bool, stream: bool) -> None:
    # fetch docs explicitly for printing sources
    docs = retriever.invoke(question)
    context = format_docs(docs)
    messages = ChatPromptTemplate.from_template(PROMPT_TMPL).format_messages(question=question, context=context)

    result_text: str
    if stream:
        # Stream tokens for responsiveness
        result_text = ""
        for chunk in chain[-1].stream(messages):  # last stage is the LLM
            piece = getattr(chunk, "content", str(chunk))
            result_text += piece
            print(piece, end="", flush=True)
        print()  # newline after stream
    else:
        result = chain.invoke(question)
        result_text = getattr(result, "content", str(result))
        print(result_text)

    if json_out:
        payload = {
            "answer": result_text,
            "sources": summarize_sources(docs),
        }
        print(json.dumps(payload, ensure_ascii=False))
    else:
        if docs:
            print("\nSources:")
            for item in summarize_sources(docs):
                print("  ", item)
        else:
            print("\n[No sources found — your DB may be empty or the query mismatched]")


def repl(chain, retriever, json_out: bool, stream: bool) -> None:
    print("RAG REPL ready. Empty line to quit.")
    while True:
        try:
            q = input("\nAsk: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not q:
            print("Bye!")
            break
        print("\n--- Answer ---\n")
        try:
            answer_once(q, chain, retriever, json_out=json_out, stream=stream)
        except KeyboardInterrupt:
            print("\n[Interrupted]")
        except Exception as e:
            log(f"Error: {e}")


# ------------------------- Main -------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG over Chroma using Ollama models.")
    p.add_argument("question", nargs="*", help="Question to ask (omit when using --repl)")
    p.add_argument("--db", default=DEF_DB_DIR, help=f"Chroma DB directory (default: {DEF_DB_DIR})")
    p.add_argument("--embed-model", default=DEF_EMBED_MODEL, help=f"Embedding model (default: {DEF_EMBED_MODEL})")
    p.add_argument("--llm-model", default=DEF_LLM_MODEL, help=f"LLM model (default: {DEF_LLM_MODEL})")
    p.add_argument("--k", type=int, default=DEF_TOP_K, help=f"Top-K chunks to retrieve (default: {DEF_TOP_K})")
    p.add_argument("--mmr", action="store_true", help="Use MMR (diverse) retrieval")
    p.add_argument("--compress", action="store_true", help="Use LLM-based context compression")
    p.add_argument("--num-predict", type=int, default=DEF_NUM_PREDICT, help=f"Max tokens to generate (default: {DEF_NUM_PREDICT})")
    p.add_argument("--temperature", type=float, default=DEF_TEMPERATURE, help=f"Temperature (default: {DEF_TEMPERATURE})")
    p.add_argument("--stop", action="append", default=DEF_STOP, help="Add a stop sequence; repeatable")
    p.add_argument("--json", dest="json_out", action="store_true", help="Emit JSON with answer & sources")
    p.add_argument("--repl", action="store_true", help="Interactive loop mode")
    p.add_argument("--stream", action="store_true", help="Stream tokens as they generate")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    retriever = build_retriever(
        db_dir=args.db,
        embed_model=args.embed_model,
        k=args.k,
        mmr=args.mmr,
        compress=args.compress,
        llm_model=args.llm_model,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )

    chain = build_chain(
        retriever=retriever,
        llm_model=args.llm_model,
        temperature=args.temperature,
        num_predict=args.num_predict,
        stops=args.stop,
    )

    # One-shot (default) or REPL
    if args.repl:
        repl(chain, retriever, json_out=args.json_out, stream=args.stream)
        return

    question = " ".join(args.question).strip()
    if not question:
        fail("Provide a question or use --repl")

    answer_once(question, chain, retriever, json_out=args.json_out, stream=args.stream)


if __name__ == "__main__":
    # Graceful Ctrl+C on Windows/Linux/macOS
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")

