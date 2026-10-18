#!/usr/bin/env python3
"""Collect recent science-paper candidates from public, key-free feeds.

This first-stage collector does not summarize or send Telegram messages yet.
It uses only Python's standard library and public arXiv / Europe PMC endpoints.
"""
from datetime import date, timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import re
import xml.etree.ElementTree as ET

USER_AGENT = "ScienceBriefingBot/0.1 (personal research digest)"
MAX_PER_SOURCE = 8

ARXIV_FEEDS = {
    "Physics & astronomy": "https://rss.arxiv.org/rss/astro-ph",
    "AI & computer science": "https://rss.arxiv.org/rss/cs.AI",
    "Machine learning": "https://rss.arxiv.org/rss/cs.LG",
    "Biology": "https://rss.arxiv.org/rss/q-bio",
    "Physics": "https://rss.arxiv.org/rss/physics",
    "Economics & society": "https://rss.arxiv.org/rss/econ.GN",
}

EUROPE_PMC_TOPICS = {
    "Climate & environment": '(TITLE_ABS:climate OR TITLE_ABS:biodiversity OR TITLE_ABS:"climate change")',
    "Biology & medicine": '(TITLE_ABS:biology OR TITLE_ABS:medical OR TITLE_ABS:health OR TITLE_ABS:genome)',
    "Neuroscience & psychology": '(TITLE_ABS:neuroscience OR TITLE_ABS:psychology OR TITLE_ABS:cognition OR TITLE_ABS:brain)',
    "Archaeology & human history": '(TITLE_ABS:archaeology OR TITLE_ABS:"ancient DNA" OR TITLE_ABS:prehistoric OR TITLE_ABS:archaeological)',
}

def fetch(url):
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=25) as response:
        return response.read()

def clean(text):
    return re.sub(r"\\s+", " ", (text or "")).strip()

def get_arxiv(category, url):
    items = []
    try:
        root = ET.fromstring(fetch(url))
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns)[:MAX_PER_SOURCE]:
            title = clean(entry.findtext("atom:title", default="", namespaces=ns))
            link = entry.findtext("atom:id", default="", namespaces=ns)
            published = clean(entry.findtext("atom:published", default="", namespaces=ns))[:10]
            summary = clean(entry.findtext("atom:summary", default="", namespaces=ns))
            if title and link:
                items.append({
                    "category": category, "title": title, "url": link,
                    "date": published, "source": "arXiv preprint",
                    "abstract": summary, "status": "PREPRINT — not necessarily peer-reviewed",
                })
    except Exception as exc:
        print(f"WARNING: arXiv feed failed ({category}): {exc}")
    return items

def get_europe_pmc(category, topic_query, start_date, end_date):
    query = f"({topic_query}) AND FIRST_PDATE:[{start_date} TO {end_date}]"
    params = urlencode({
        "query": query, "format": "json", "pageSize": MAX_PER_SOURCE,
        "sort_date": "y", "resultType": "core",
    })
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + params
    items = []
    try:
        data = json.loads(fetch(url).decode("utf-8"))
        for result in data.get("resultList", {}).get("result", []):
            title = clean(result.get("title"))
            doi = result.get("doi")
            source = result.get("source", "")
            article_id = result.get("id", "")
            link = f"https://doi.org/{doi}" if doi else (
                f"https://europepmc.org/article/{source}/{article_id}" if source and article_id else ""
            )
            if not title or not link:
                continue
            pub_date = result.get("firstPublicationDate", "")
            is_preprint = source.upper() == "PPR" or result.get("isPreprint", "N") == "Y"
            items.append({
                "category": category, "title": title, "url": link,
                "date": pub_date, "source": "Europe PMC",
                "abstract": clean(result.get("abstractText", "")),
                "status": "PREPRINT — not necessarily peer-reviewed" if is_preprint else "Publication record; peer-review status not independently verified",
            })
    except Exception as exc:
        print(f"WARNING: Europe PMC query failed ({category}): {exc}")
    return items

def main():
    today = date.today()
    start = today - timedelta(days=14)
    print(f"SCIENCE BRIEFING — candidate collection: {start} to {today}")
    print("This is an unranked candidate list, not the final weekly briefing.\n")

    items = []
    for category, url in ARXIV_FEEDS.items():
        items.extend(get_arxiv(category, url))
    for category, query in EUROPE_PMC_TOPICS.items():
        items.extend(get_europe_pmc(category, query, start.isoformat(), today.isoformat()))

    deduped = {}
    for item in items:
        key = item["url"].lower()
        if key not in deduped:
            deduped[key] = item

    candidates = sorted(deduped.values(), key=lambda x: x.get("date", ""), reverse=True)
    print(f"Collected {len(candidates)} unique candidates.\n")
    for index, item in enumerate(candidates[:40], 1):
        print(f"{index}. [{item['category']}] {item['title']}")
        print(f"   Date: {item['date'] or 'date not supplied'} | Source: {item['source']}")
        print(f"   Evidence status: {item['status']}")
        print(f"   Link: {item['url']}")
        if item["abstract"]:
            excerpt = item["abstract"][:500].rstrip()
            print(f"   Abstract: {excerpt}{'…' if len(item['abstract']) > 500 else ''}")
        print()

    if not candidates:
        raise SystemExit("No candidates collected. Check the public feed/API endpoints in the workflow log.")

if __name__ == "__main__":
    main()
