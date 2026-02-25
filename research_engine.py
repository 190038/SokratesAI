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


def generate_research_queries(conversation_text: str, n=6):
    prompt = [
        {
            "role": "system",
            "content": (
                "You generate web research queries.\n"
                f"Return ONLY a JSON array of {n} short search queries.\n"
                "No markdown, no explanation."
            )
        },
        {"role": "user", "content": conversation_text}
    ]
    r = ollama.chat(model=MODEL, messages=prompt)["message"]["content"]

    # super simple fallback if JSON is messy
    # try to extract lines if JSON parse fails
    import json, re
    try:
        return json.loads(r)
    except:
        # extract quoted strings as a fallback
        qs = re.findall(r'"([^"]+)"', r)
        if qs:
            return qs[:n]
        # last resort: split lines
        return [ln.strip("- ").strip() for ln in r.splitlines() if ln.strip()][:n]


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