# rag_qa_advanced.py (invoke-based, no deprecation warnings)
import os
from datetime import datetime

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate

# RunnablePassthrough import path varies by version
try:
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    from langchain.schema.runnable import RunnablePassthrough

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

DB_DIR = "chroma_db"
EMBED_MODEL = "mxbai-embed-large"   # you have this
LLM_MODEL = "llama3:instruct"       # you pulled this
TOP_K = int(os.getenv("RAG_TOP_K", "6"))
FETCH_K = int(os.getenv("RAG_FETCH_K", "24"))
MMR_LAMBDA = float(os.getenv("RAG_MMR_LAMBDA", "0.5"))

def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ---- Base vector store / retriever (MMR) ----
log(f"Loading DB='{DB_DIR}' with embeddings='{EMBED_MODEL}' and LLM='{LLM_MODEL}'")
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

base_retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": MMR_LAMBDA},
)

# ---- Query expansion (MultiQueryRetriever) ----
llm_for_tools = ChatOllama(model=LLM_MODEL, temperature=0.0, num_predict=128)
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm_for_tools,
    prompt=ChatPromptTemplate.from_template(
        "You are a query rewriter. Given the user's question, produce 4 diverse, "
        "concise reformulations that might retrieve better results. "
        "Avoid synonyms-only rephrases; vary entities and phrasing.\n\n"
        "Question: {question}\n\nRewrites:"
    ),
)

# ---- Context compression (keep only relevant lines) ----
compressor = LLMChainExtractor.from_llm(
    llm=ChatOllama(model=LLM_MODEL, temperature=0.0, num_predict=256)
)
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_retriever,
)

# ---- Final answering LLM ----
llm_answer = ChatOllama(model=LLM_MODEL, temperature=0.2, num_predict=384)

prompt = ChatPromptTemplate.from_template(
    "You are a precise assistant. Use ONLY the provided context to answer."
    " If the answer is not in the context, say: \"I don't know.\""
    " Prefer exact figures, program names, and proper nouns as written."
    " If the question has a typo, infer the likely intended name but still rely on context.\n\n"
    "Question: {question}\n\n"
    "Context (compressed & relevant):\n{context}\n\n"
    "Answer (concise, include source bracket numbers when citing):"
)

def format_docs(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag = f" p.{page}" if page is not None else ""
        parts.append(f"[{i}] ({src}{tag}) {d.page_content.strip()}")
    return "\n".join(parts)

def retrieve(question: str):
    # ✅ Use invoke instead of get_relevant_documents
    docs = compressed_retriever.invoke(question)
    log(f"Retrieved {len(docs)} compressed chunks.")
    if docs:
        for i, d in enumerate(docs[:3], 1):
            print(f"  -> [{i}] source={d.metadata.get('source','?')} page={d.metadata.get('page','?')}")
    else:
        log("No relevant context retrieved. Add more docs and re-run `python ingest.py`.")
    return docs

def ask(question: str) -> str:
    docs = retrieve(question)
    context = format_docs(docs)
    messages = prompt.format_messages(question=question, context=context)
    resp = llm_answer.invoke(messages)
    return getattr(resp, "content", str(resp))

if __name__ == "__main__":
    print("Advanced local RAG ready. Ask questions (empty line to quit).")
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
            print(ask(q) or "[No text in response]")
        except Exception as e:
            print(f"[ERROR] {e}")
