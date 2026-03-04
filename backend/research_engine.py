import time
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
import json
from datetime import datetime


import ollama
from memory_engine import store_item

MODEL = "deepseek-r1:8b"

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SokratesResearchBot/1.0"
HEADERS = {"User-Agent": USER_AGENT}

DEFAULT_MAX_RESULTS = 8      # search results to consider
DEFAULT_MAX_PAGES = 4        # pages to actually fetch
REQUEST_TIMEOUT = 12


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Normalize whitespace
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def web_search(queries, max_results=DEFAULT_MAX_RESULTS):
    results = []
    with DDGS() as ddgs:
        for q in queries:
            for r in ddgs.text(q, max_results=max_results):
                # r has keys like: title, href, body
                results.append({
                    "query": q,
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
    # Deduplicate by url
    seen = set()
    deduped = []
    for r in results:
        u = r["url"]
        if u and u not in seen:
            seen.add(u)
            deduped.append(r)
    return deduped


def fetch_page(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return _clean_text(resp.text)


def _extract_research_intent(text: str) -> dict:
    """Stage 1: extract structured intent from a single user message."""
    import json, re

    prompt = [
        {
            "role": "system",
            "content": (
                "Extract the research intent from the user message.\n"
                "Return ONLY a JSON object with these fields:\n"
                "  topic       - main topic in 3-8 words\n"
                "  subtopics   - list of 2-4 specific subtopics or angles\n"
                "  key_terms   - list of 3-6 domain-specific terms\n"
                "  time_filter - one of: 'latest', 'historical', 'any'\n"
                "  domain      - e.g. 'medicine', 'finance', 'software', 'science', 'general'\n"
                "  goal        - one of: 'understand', 'compare', 'evaluate', 'find_data', 'how_to'\n"
                "No markdown, no explanation."
            )
        },
        {"role": "user", "content": text[:2000]}
    ]

    raw = ollama.chat(model=MODEL, messages=prompt)["message"]["content"]

    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass

    return {"topic": text[:80], "subtopics": [], "key_terms": [],
            "time_filter": "any", "domain": "general", "goal": "understand"}


def _build_query_prompt(intent: dict, n: int) -> str:
    time_instruction = {
        "latest":     "Include recency modifiers like '2024', '2025', 'recent', 'new study'.",
        "historical": "Include historical or archival modifiers where relevant.",
        "any":        "Mix timeless and current queries."
    }.get(intent.get("time_filter", "any"), "")

    goal_instruction = {
        "understand": "Focus on explainers and foundational resources.",
        "compare":    "Focus on comparisons and 'X vs Y' queries.",
        "evaluate":   "Focus on critiques, pros/cons, and expert opinions.",
        "find_data":  "Focus on datasets, statistics, and empirical studies.",
        "how_to":     "Focus on tutorials and implementation guides."
    }.get(intent.get("goal", "understand"), "")

    return (
        f"Generate exactly {n} highly specific web search queries about:\n\n"
        f"TOPIC: {intent.get('topic', '')}\n"
        f"SUBTOPICS: {', '.join(intent.get('subtopics', []))}\n"
        f"KEY TERMS: {', '.join(intent.get('key_terms', []))}\n"
        f"DOMAIN: {intent.get('domain', 'general')}\n\n"
        f"- {time_instruction}\n"
        f"- {goal_instruction}\n"
        "- Each query must be 6-14 words — specific enough to return focused results.\n"
        "- Cover different angles (overview, deep-dive, recent, compare, critical, data).\n"
        "- No vague queries like 'what is X'. No repeated angles.\n"
        "- Return ONLY a valid JSON array of strings. No markdown, no explanation."
    )


def generate_research_queries(conversation_text: str, n=6):
    import json, re

    intent = _extract_research_intent(conversation_text)

    prompt = [
        {"role": "system", "content": _build_query_prompt(intent, n)},
        {"role": "user", "content": conversation_text[:2000]}
    ]
    raw = ollama.chat(model=MODEL, messages=prompt)["message"]["content"]

    def _parse(text: str, limit: int) -> list:
        text = text.strip()
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(q).strip() for q in result if str(q).strip()][:limit]
        except Exception:
            pass
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return [str(q).strip() for q in result if str(q).strip()][:limit]
            except Exception:
                pass
        qs = re.findall(r'"([^"]{10,})"', text)
        if qs:
            return qs[:limit]
        return [
            re.sub(r"^\s*[\d\.\-\*\[\w\]]+\s*", "", ln).strip()
            for ln in text.splitlines()
            if ln.strip() and len(ln.strip()) > 10
        ][:limit]

    queries = _parse(raw, n)

    # Dedup by first-5-word prefix
    seen, deduped = set(), []
    for q in queries:
        prefix = " ".join(q.lower().split()[:5])
        if prefix not in seen:
            seen.add(prefix)
            deduped.append(q)

    # Pad if needed
    topic = intent.get("topic", conversation_text[:60])
    for fb in [f"{topic} in depth", f"{topic} recent studies 2025", f"{topic} limitations evidence"]:
        if len(deduped) >= n:
            break
        prefix = " ".join(fb.lower().split()[:5])
        if prefix not in seen:
            seen.add(prefix)
            deduped.append(fb)

    return deduped


def summarize_sources(topic_context: str, sources: list):
    # Keep the LLM input bounded
    source_blurbs = []
    for s in sources:
        source_blurbs.append(
            f"TITLE: {s['title']}\nURL: {s['url']}\nEXTRACT:\n{s['text'][:2500]}"
        )
    joined = "\n\n---\n\n".join(source_blurbs)

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a rigorous research assistant.\n"
                "Given the topic context and web source extracts, produce:\n"
                "1) Key findings (bulleted)\n"
                "2) Tensions / disagreements between sources\n"
                "3) Follow-up questions\n"
                "4) Practical next steps (if relevant)\n"
                "Always include citations as URLs after the bullet that uses them."
            )
        },
        {"role": "user", "content": f"TOPIC CONTEXT:\n{topic_context}\n\nSOURCES:\n{joined}"}
    ]
    return ollama.chat(model=MODEL, messages=prompt)["message"]["content"]


def run_research(conversation_text: str, max_pages=DEFAULT_MAX_PAGES):
    queries = generate_research_queries(conversation_text)
    search_results = web_search(queries)

    # Fetch only first N pages
    fetched = []
    for r in search_results[:max_pages]:
        url = r["url"]
        try:
            text = fetch_page(url)
            fetched.append({
                "title": r["title"],
                "url": url,
                "text": text
            })
            time.sleep(1.0)  # be polite
        except Exception as e:
            # skip failures
            continue

    if not fetched:
        report = "No pages could be fetched. Try again with fewer restrictions or different queries."
        return report, queries, search_results, fetched

    report = summarize_sources(conversation_text, fetched)

    # 1) Report als Episode speichern (hochwertig, kompakt)
    store_item(
        text="RESEARCH REPORT:\n" + report,
        tag="research",
        kind="memory",
        importance=0.75,
        meta={"source": "research_engine"}
    )

    # 2) Quellenliste (nur URLs + Titel, kein riesiger Textdump)
    for f in fetched:
        store_item(
            text=f"SOURCE: {f['title']} — {f['url']}",
            tag="research_source",
            kind="memory",
            importance=0.55,
            meta={"source": "web", "url": f["url"]}
        )

    with open("last_research.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "conversation_context": conversation_text[:6000],
                "queries": queries,
                "report": report
            },
            f,
            indent=2
        )
    return report, queries, search_results, fetched