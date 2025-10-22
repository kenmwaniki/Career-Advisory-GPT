#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
University Crawler → (docs folder) → rag_ingest.ingest_once → Chroma

What it does:
- Async crawl (robots-aware, sitemap-first, BFS fallback)
- Per-site config (include/exclude, limits)
- Saves each page as .md into your docs dir
- After crawling, calls your existing ingest_once() to index into Chroma

Defaults (Windows):
  --docs-dir  C:\Users\ZBOOK\local-rag-llama\docs
  --db-dir    C:\Users\ZBOOK\local-rag-llama\chroma_db
  --embed-model mxbai-embed-large

Usage:
  python crawler.py --config sites.yaml --ingest-mode at_end
"""

import asyncio
import hashlib
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Set, Dict
from urllib.parse import urljoin, urldefrag, urlparse
import argparse
import xml.etree.ElementTree as ET

import aiohttp
from aiohttp import ClientTimeout
from aiolimiter import AsyncLimiter
import yaml
from bs4 import BeautifulSoup
from urllib import robotparser

# <<< IMPORTANT: this imports your existing ingester >>>
from ingest import ingest_once  # make sure rag_ingest.py is in the same folder or on PYTHONPATH

# --------------------------- Logging ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("crawler")

# --------------------------- Models ---------------------------
@dataclass
class SiteConfig:
    name: str
    base_url: str
    allowed_domains: List[str] = field(default_factory=list)
    include_paths: List[str] = field(default_factory=lambda: [r".*"])  # regexes
    exclude_paths: List[str] = field(default_factory=list)             # regexes
    headers: Dict[str, str] = field(default_factory=dict)
    rate_limit_per_sec: float = 1.0
    max_pages: int = 200
    max_depth: int = 3
    use_sitemap: bool = True

@dataclass
class PageDoc:
    url: str
    site: str
    title: Optional[str]
    description: Optional[str]
    text: Optional[str]
    lang: Optional[str]
    published_at: Optional[str]
    fetched_at: str
    hash: str

# --------------------------- State (SQLite) ---------------------------
class StateDB:
    def __init__(self, path: str = "crawl_state.db"):
        self.path = path
        self._init()

    def _init(self):
        conn = sqlite3.connect(self.path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seen (
                    url TEXT PRIMARY KEY,
                    fetched_at TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    url TEXT PRIMARY KEY,
                    last_error TEXT,
                    last_attempt TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def is_seen(self, url: str) -> bool:
        conn = sqlite3.connect(self.path)
        try:
            cur = conn.execute("SELECT 1 FROM seen WHERE url = ?", (url,))
            return cur.fetchone() is not None
        finally:
            conn.close()

    def mark_seen(self, url: str):
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO seen(url, fetched_at) VALUES (?, ?)",
                (url, datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

    def mark_failure(self, url: str, err: str):
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO failures(url, last_error, last_attempt) VALUES (?, ?, ?)",
                (url, err[:250], datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

# --------------------------- Helpers ---------------------------
def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = urljoin(base, href)
    href, _frag = urldefrag(href)
    if not href.lower().startswith(("http://", "https://")):
        return None
    return href

def url_in_allowed_domains(url: str, allowed: List[str]) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in allowed)

def path_allowed(url: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    path = urlparse(url).path or "/"
    if exclude_patterns and any(re.search(p, path) for p in exclude_patterns):
        return False
    return any(re.search(p, path) for p in include_patterns) if include_patterns else True

def pick_meta(soup: BeautifulSoup, names: Iterable[str]) -> Optional[str]:
    for n in names:
        tag = soup.find("meta", attrs={"name": n}) or soup.find("meta", attrs={"property": n})
        if tag and tag.get("content"):
            return tag["content"].strip()
    return None

def guess_lang(soup: BeautifulSoup) -> Optional[str]:
    html = soup.find("html")
    if html and html.get("lang"):
        return html["lang"].strip()
    return pick_meta(soup, ["og:locale", "language"])

def guess_published(soup: BeautifulSoup) -> Optional[str]:
    val = pick_meta(soup, [
        "article:published_time", "og:updated_time", "pubdate", "publishdate",
        "dc.date", "date", "dcterms.date"
    ])
    if val:
        return val
    t = soup.find("time")
    if t and t.get("datetime"):
        return t["datetime"].strip()
    return None

def extract_main_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript", "iframe", "svg"]):
        tag.decompose()

    candidates = []
    for selector in ["article", "main", "[role=main]", ".content", ".post", ".article", ".entry-content"]:
        el = soup.select_one(selector)
        if el:
            candidates.append(el)

    if not candidates:
        candidates = soup.select("p")

    if not candidates:
        return soup.get_text(" ", strip=True)

    def text_len(el):
        return len(el.get_text(" ", strip=True))

    best = max(candidates, key=text_len)
    text = best.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()

async def fetch_text(session: aiohttp.ClientSession, url: str, headers: Dict[str, str], timeout: int = 30) -> str:
    async with session.get(url, headers=headers, timeout=ClientTimeout(total=timeout), allow_redirects=True) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" not in ctype and "xml" not in ctype:
            raise aiohttp.ClientResponseError(r.request_info, r.history, status=415, message=f"Unsupported content-type: {ctype}")
        return await r.text(errors="ignore")

async def head_ok(session: aiohttp.ClientSession, url: str, headers: Dict[str, str]) -> bool:
    try:
        async with session.head(url, headers=headers, timeout=ClientTimeout(total=15), allow_redirects=True) as r:
            return r.status < 400
    except Exception:
        return False

async def try_fetch_robots(session: aiohttp.ClientSession, base_url: str, headers: Dict[str, str]) -> Optional[robotparser.RobotFileParser]:
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    try:
        txt = await fetch_text(session, robots_url, headers=headers, timeout=15)
        rp = robotparser.RobotFileParser()
        rp.parse(txt.splitlines())
        rp.set_url(robots_url)
        return rp
    except Exception:
        return None

async def discover_sitemaps(session: aiohttp.ClientSession, base_url: str, headers: Dict[str, str]) -> List[str]:
    parsed = urlparse(base_url)
    candidates = [
        f"{parsed.scheme}://{parsed.netloc}/sitemap.xml",
        f"{parsed.scheme}://{parsed.netloc}/sitemap_index.xml",
        f"{parsed.scheme}://{parsed.netloc}/sitemap-index.xml",
    ]
    results = []
    for u in candidates:
        if await head_ok(session, u, headers=headers):
            results.append(u)
    return results

def parse_sitemap(xml_text: str) -> List[str]:
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))
    except Exception:
        return []
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls: List[str] = []
    for loc in root.findall(".//sm:sitemap/sm:loc", ns):
        urls.append(loc.text.strip())
    for loc in root.findall(".//sm:url/sm:loc", ns):
        urls.append(loc.text.strip())
    # de-dupe
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u); out.append(u)
    return out

async def iter_sitemap_urls(session: aiohttp.ClientSession, site: SiteConfig) -> Iterable[str]:
    sitemaps = await discover_sitemaps(session, site.base_url, headers=site.headers)
    if not sitemaps:
        return []
    queue = list(sitemaps)
    seen = set(queue)
    urls = []
    while queue:
        sm = queue.pop(0)
        try:
            xml = await fetch_text(session, sm, headers=site.headers, timeout=30)
            found = parse_sitemap(xml)
            for u in found:
                if u.endswith(".xml") or "/sitemap" in u:
                    if u not in seen:
                        seen.add(u); queue.append(u)
                else:
                    urls.append(u)
        except Exception as e:
            log.warning("Failed to parse sitemap %s: %s", sm, e)
    return urls

def default_headers(user_agent: str) -> Dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.7",
    }

# --------------------------- Markdown sink ---------------------------
def write_markdown(docs_dir: Path, doc: PageDoc) -> Path:
    ymd = datetime.utcnow().strftime("%Y-%m-%d")
    base = f"{ymd}_{doc.site}_{doc.hash[:10]}.md"
    safe = "".join(c for c in base if c.isalnum() or c in ("-", "_", ".", " ")).rstrip()
    p = docs_dir / safe
    lines = [
        f"# {doc.title or '(untitled)'}",
        "",
        f"- **URL:** {doc.url}",
        f"- **Site:** {doc.site}",
        f"- **Published:** {doc.published_at or 'n/a'}",
        f"- **Fetched:** {doc.fetched_at}",
        f"- **Lang:** {doc.lang or 'n/a'}",
        "",
        f"**Description:** {doc.description or ''}",
        "",
        doc.text or ""
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p

# --------------------------- Crawler core ---------------------------
class SiteCrawler:
    def __init__(self, site: SiteConfig, state: StateDB, ua: str, docs_dir: Path, ingest_mode: str, embed_model: str, db_dir: Path):
        self.site = site
        self.state = state
        self.ua = ua
        self.docs_dir = docs_dir
        self.ingest_mode = ingest_mode
        self.embed_model = embed_model
        self.db_dir = db_dir

        self.domain_limiter = AsyncLimiter(max_rate=site.rate_limit_per_sec, time_period=1.0)
        self.robots: Optional[robotparser.RobotFileParser] = None

        self._include = [re.compile(p) for p in site.include_paths]
        self._exclude = [re.compile(p) for p in site.exclude_paths]

        self.headers = default_headers(ua)
        self.headers.update(site.headers or {})

    def allowed_by_robots(self, url: str) -> bool:
        if not self.robots:
            return True
        return self.robots.can_fetch(self.ua, url)

    def allowed_by_rules(self, url: str) -> bool:
        if not url_in_allowed_domains(url, self.site.allowed_domains or [urlparse(self.site.base_url).hostname or ""]):
            return False
        return path_allowed(url, [p.pattern for p in self._include], [p.pattern for p in self._exclude])

    async def fetch_and_extract(self, session: aiohttp.ClientSession, url: str) -> Optional[PageDoc]:
        async with self.domain_limiter:
            try:
                html = await fetch_text(session, url, headers=self.headers)
            except Exception as e:
                self.state.mark_failure(url, str(e))
                log.debug("Fetch failed %s: %s", url, e)
                return None

        soup = BeautifulSoup(html, "lxml")

        title = (pick_meta(soup, ["og:title", "twitter:title"]) or (soup.title.string.strip() if soup.title and soup.title.string else None))
        description = pick_meta(soup, ["description", "og:description", "twitter:description"])
        lang = guess_lang(soup)
        published = guess_published(soup)
        text = extract_main_text(soup)
        doc_hash = sha1((title or "") + "\n" + (text or ""))

        return PageDoc(
            url=url,
            site=self.site.name,
            title=title,
            description=description,
            text=text,
            lang=lang,
            published_at=published,
            fetched_at=datetime.utcnow().isoformat(),
            hash=doc_hash
        )

    async def discover_links(self, session: aiohttp.ClientSession, url: str, html: str) -> List[str]:
        soup = BeautifulSoup(html, "lxml")
        out = []
        for a in soup.find_all("a", href=True):
            u = normalize_url(url, a["href"])
            if not u:
                continue
            if not self.allowed_by_rules(u):
                continue
            out.append(u)
        seen, uniq = set(), []
        for u in out:
            if u not in seen:
                seen.add(u); uniq.append(u)
        return uniq

    async def crawl(self, session: aiohttp.ClientSession, max_concurrency: int):
        self.robots = await try_fetch_robots(session, self.site.base_url, headers=self.headers)
        queue: asyncio.Queue = asyncio.Queue()

        seeds = []
        if self.site.use_sitemap:
            try:
                sm_urls = await iter_sitemap_urls(session, self.site)
                seeds = [u for u in sm_urls if self.allowed_by_rules(u)]
                log.info("[%s] Discovered %d URLs from sitemap(s).", self.site.name, len(seeds))
            except Exception as e:
                log.warning("[%s] sitemap discovery failed: %s", self.site.name, e)

        if not seeds:
            seeds = [self.site.base_url]

        added = 0
        for u in seeds:
            if not self.state.is_seen(u) and self.allowed_by_robots(u) and self.allowed_by_rules(u):
                await queue.put((u, 0))
                added += 1
        if added == 0:
            await queue.put((self.site.base_url, 0))

        processed = 0
        in_flight: Set[str] = set()
        sem = asyncio.Semaphore(max_concurrency)

        async def worker():
            nonlocal processed
            async with sem:
                while not queue.empty() and processed < self.site.max_pages:
                    url, depth = await queue.get()
                    try:
                        if self.state.is_seen(url):
                            continue
                        if not self.allowed_by_robots(url) or not self.allowed_by_rules(url):
                            continue

                        try:
                            html = await fetch_text(session, url, headers=self.headers)
                        except Exception as e:
                            self.state.mark_failure(url, str(e))
                            continue

                        # Extract + save to docs/
                        doc = await self.fetch_and_extract(session, url)
                        if doc and (doc.title or doc.text or doc.description):
                            md_path = write_markdown(self.docs_dir, doc)
                            log.info("[%s] wrote %s", self.site.name, md_path)

                            # Optional immediate ingest (slower overall)
                            if self.ingest_mode == "on_each":
                                try:
                                    summary = ingest_once(
                                        docs_dir=self.docs_dir,
                                        db_dir=self.db_dir,
                                        embed_model=self.embed_model,
                                        rebuild=False,
                                        dry_run=False,
                                    )
                                    log.info("[%s] immediate ingest: added=%s skipped=%s",
                                             self.site.name, summary.get("added"), summary.get("skipped"))
                                except Exception as pe:
                                    log.warning("[%s] Immediate ingest failed for %s: %s", self.site.name, url, pe)

                            self.state.mark_seen(url)
                            processed += 1

                        # Link discovery
                        if depth < self.site.max_depth and processed < self.site.max_pages:
                            next_links = await self.discover_links(session, url, html)
                            for nxt in next_links:
                                if not self.state.is_seen(nxt) and nxt not in in_flight and self.allowed_by_robots(nxt) and self.allowed_by_rules(nxt):
                                    in_flight.add(nxt)
                                    await queue.put((nxt, depth + 1))
                    finally:
                        queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(max_concurrency)]
        await queue.join()
        for w in workers:
            w.cancel()
        log.info("[%s] Done. Processed %d pages.", self.site.name, processed)

# --------------------------- CLI & main ---------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="University crawler → docs → ingest → Chroma")
    p.add_argument("--config", required=True, help="YAML file with sites list")
    p.add_argument("--state-db", default="crawl_state.db", help="SQLite file for de-dup")
    p.add_argument("--user-agent", default="ISP-CareerGPT-Crawler/1.0", help="Crawler User-Agent")
    p.add_argument("--concurrency", type=int, default=8, help="Global max concurrency")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")

    # Integration with your local RAG
    p.add_argument("--docs-dir", default=r"C:\Users\ZBOOK\local-rag-llama\docs")
    p.add_argument("--db-dir", default=r"C:\Users\ZBOOK\local-rag-llama\chroma_db")
    p.add_argument("--embed-model", default="mxbai-embed-large")
    p.add_argument("--ingest-mode", choices=["none", "at_end", "on_each"], default="at_end",
                   help="When to call ingest_once (none|at_end|on_each)")

    return p.parse_args()

async def main():
    args = parse_args()

    docs_dir = Path(args.docs_dir); docs_dir.mkdir(parents=True, exist_ok=True)
    db_dir = Path(args.db_dir);     db_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sites: List[SiteConfig] = []
    for s in cfg.get("sites", []):
        sites.append(SiteConfig(
            name=s["name"],
            base_url=s["base_url"],
            allowed_domains=s.get("allowed_domains") or [],
            include_paths=s.get("include_paths") or [r".*"],
            exclude_paths=s.get("exclude_paths") or [],
            headers=s.get("headers") or {},
            rate_limit_per_sec=float(s.get("rate_limit_per_sec", 1.0)),
            max_pages=int(s.get("max_pages", 200)),
            max_depth=int(s.get("max_depth", 3)),
            use_sitemap=bool(s.get("use_sitemap", True)),
        ))

    state = StateDB(args.state_db)
    timeout = ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2, ssl=False)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=True) as session:
        tasks = []
        for site in sites:
            crawler = SiteCrawler(
                site=site,
                state=state,
                ua=args.user_agent,
                docs_dir=docs_dir,
                ingest_mode=args.ingest_mode,
                embed_model=args.embed_model,
                db_dir=db_dir,
            )
            tasks.append(asyncio.create_task(crawler.crawl(session, max_concurrency=args.concurrency)))

        await asyncio.gather(*tasks, return_exceptions=False)

    # Batch ingest at the end (recommended)
    if args.ingest_mode == "at_end":
        log.info("Starting batch ingest into Chroma...")
        summary = ingest_once(
            docs_dir=docs_dir,
            db_dir=db_dir,
            embed_model=args.embed_model,
            rebuild=False,
            dry_run=False,
        )
        log.info("Ingest summary: %s", summary)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user")
