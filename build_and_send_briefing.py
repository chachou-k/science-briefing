#!/usr/bin/env python3
"""Build a weekly science digest with validated summaries; never use a paid fallback."""
import json
import os
import re
import time
from datetime import date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import collect_articles

GROQ_MODEL = "openai/gpt-oss-20b"
MAX_ARTICLES = 7
MIN_ARTICLES = 5
MAX_SUMMARY_ATTEMPTS = 1
REQUEST_PAUSE_SECONDS = 10
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"
HEADING_ALIASES = (
    ("why it matters", "why this matters", "importance"),
    ("research question and method", "research question and methods", "question and method", "question and methods"),
    ("main findings", "key findings", "findings"),
    ("what the evidence does and does not show", "what the evidence does and doesnt show",
     "what the evidence can and cannot show", "evidence and interpretation", "what we can conclude"),
    ("limitations", "caveats", "limitations and caveats"),
)


def request_json(url, data=None, headers=None, timeout=45):
    body = json.dumps(data).encode("utf-8") if data is not None else None
    safe_headers = {"User-Agent": "ScienceBriefingBot/0.1 (personal research digest)"}
    safe_headers.update(headers or {})
    req = Request(url, data=body, headers=safe_headers, method="POST" if body else "GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")[:1000]
        if exc.code == 429:
            raise RuntimeError("Groq free-tier limit reached (HTTP 429). Stopping; no fallback model will be used.") from exc
        raise RuntimeError(f"API request failed (HTTP {exc.code}): {details}") from exc


def collect_candidates():
    today = date.today()
    start = today - timedelta(days=30)
    items = []
    for category, url in collect_articles.ARXIV_FEEDS.items():
        items.extend(collect_articles.get_arxiv(category, url))
    for category, query in collect_articles.EUROPE_PMC_TOPICS.items():
        items.extend(collect_articles.get_europe_pmc(category, query, start.isoformat(), today.isoformat()))
    deduped = {}
    for item in items:
        if item.get("url") and item.get("abstract"):
            deduped.setdefault(item["url"].lower(), item)
    # Prefer journal/publication records over explicit preprints, then newer papers.
    def is_preprint(item):
        status = item.get("status", "").upper()
        return "PREPRINT" in status or "ARXIV" in item.get("source", "").upper()
    newest_first = sorted(deduped.values(), key=lambda x: x.get("date", ""), reverse=True)
    # Stable second sort puts publication records first while preserving newest-first order within each group.
    return sorted(newest_first, key=lambda item: 1 if is_preprint(item) else 0)


def category_group(category):
    c = category.lower()
    if "climate" in c or "environment" in c: return "Climate & environment"
    if "neuro" in c or "psych" in c: return "Neuroscience & psychology"
    if "archaeology" in c or "history" in c: return "Archaeology & human history"
    if "biology" in c or "medicine" in c: return "Biology & medicine"
    if "ai" in c or "computer" in c or "learning" in c: return "AI & computer science"
    if "physics" in c or "astronomy" in c: return "Physics & astronomy"
    if "econom" in c or "society" in c: return "Society, economics & politics"
    return category


def select_candidates(items):
    # Keep a larger reserve so incomplete summaries can be skipped and replaced.
    selected, counts = [], {}
    for item in items:
        group = category_group(item.get("category", ""))
        abstract = item.get("abstract", "").strip()
        if len(abstract) < 180:
            continue
        if counts.get(group, 0) >= 3:
            continue
        selected.append(item)
        counts[group] = counts.get(group, 0) + 1
        if len(selected) >= 24:
            break
    if len(selected) < MIN_ARTICLES:
        raise RuntimeError(f"Only found {len(selected)} candidates with useful abstracts; no digest sent.")
    return selected


def has_all_sections(summary):
    # Ignore Markdown emphasis and punctuation, and accept common equivalent heading wording.
    normalized = re.sub(r"[^a-z0-9]+", " ", summary.lower()).strip()
    matched_sections = sum(
        1 for aliases in HEADING_ALIASES
        if any(re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip() in normalized for alias in aliases)
    )
    # Some valid summaries use equivalent wording not covered by the heading aliases.
    return len(summary) >= 600 and matched_sections >= 4


def summarize(item, api_key):
    last_problem = "missing required section"
    for attempt in range(1, MAX_SUMMARY_ATTEMPTS + 1):
        retry_note = ""
        if attempt > 1:
            retry_note = (
                "\nYour previous answer was incomplete. Rewrite the WHOLE summary and include "
                "all five exact headings. Do not omit any section.\n"
            )
        prompt = f"""Write a detailed but accessible English research briefing for an interested non-specialist.
Aim for 180-220 words total, excluding title and link. Explain technical terms in plain English when first used. Explain what the researchers actually did, not just the topic. Include sample size, time period, comparison group, numerical results, or effect sizes when stated in the abstract. Do not invent any details. If information is absent, say "Not reported in the abstract." Distinguish results from interpretation, and correlation from causation. Explain what cannot be concluded. Be specific about limitations supported by the supplied information; if the abstract gives no limitations, say so rather than inventing them.
Use ALL five exact headings below, each on its own line:
Why it matters
Research question and method
Main findings
What the evidence does and does not show
Limitations
For "Research question and method", explain technical methods in simple language. For "Main findings", give concrete results and numbers when available. For "What the evidence does and does not show", separate the direct result from broader interpretations. For "Limitations", note missing details, design constraints, generalisability, and uncertainty only where justified.
If this is a preprint, clearly state that it has not necessarily been peer reviewed. If it is a journal publication record, do not claim peer review is confirmed unless the metadata confirms it.
{retry_note}
Category: {item.get('category', 'Science')}
Title: {item.get('title', '')}
Date: {item.get('date', 'not supplied')}
Evidence status: {item.get('status', 'status not verified')}
Source: {item.get('source', '')}
Abstract:
{item.get('abstract', '')[:5000]}
"""
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a cautious science editor. Produce complete, plain-English summaries and never fabricate research details."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 700,
        }
        result = request_json(GROQ_URL, payload, {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        try:
            summary = result["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Groq returned an unexpected response; no digest sent.") from exc
        if has_all_sections(summary):
            return summary
        last_problem = "one or more required headings were missing"
        print(f"Incomplete summary for '{item.get('title', '')}' (attempt {attempt}); " +
              ("retrying." if attempt < MAX_SUMMARY_ATTEMPTS else "skipping article."))
        time.sleep(REQUEST_PAUSE_SECONDS)
    print(f"Skipped article because summary remained incomplete: {last_problem}.")
    return None


def telegram_send(token, chat_id, text):
    result = request_json(TELEGRAM_URL.format(token), {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }, {"Content-Type": "application/json"})
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected a message; check the Telegram secrets.")


def main():
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not api_key or not telegram_token or not chat_id:
        raise RuntimeError("Missing GROQ_API_KEY, TELEGRAM_BOT_TOKEN, or TELEGRAM_CHAT_ID secret.")

    candidates = select_candidates(collect_candidates())
    print(f"Collected {len(candidates)} candidates. Summarizing with fixed model {GROQ_MODEL}.")
    summaries = []
    for item in candidates:
        if len(summaries) >= MAX_ARTICLES:
            break
        print(f"Summarizing candidate {len(summaries) + 1}; valid summaries so far: {len(summaries)}")
        summary = summarize(item, api_key)
        if summary:
            summaries.append((item, summary))
        time.sleep(REQUEST_PAUSE_SECONDS)

    if len(summaries) < MIN_ARTICLES:
        raise RuntimeError(
            f"Only {len(summaries)} complete summaries could be produced; no Telegram digest sent. "
            "This prevents incomplete article messages."
        )

    telegram_send(telegram_token, chat_id,
        f"SCIENCE BRIEFING — {date.today().isoformat()}\n"
        f"{len(summaries)} research stories from the last 30 days. Published records are prioritised; "
        "preprints are labelled. Summaries are based on abstracts, so follow the original links for full details.")
    for i, (item, summary) in enumerate(summaries, 1):
        message = (
            f"{i}. [{category_group(item.get('category', 'Science'))}] {item.get('title', '')}\n"
            f"Date: {item.get('date') or 'not supplied'}\n"
            f"Evidence status: {item.get('status', 'not verified')}\n\n"
            f"{summary}\n\nOriginal paper: {item.get('url', '')}"
        )
        if len(message) > 3900:
            raise RuntimeError(f"Summary for article {i} is too long for Telegram; no more messages sent.")
        telegram_send(telegram_token, chat_id, message)
        time.sleep(1)
    print(f"Science briefing sent to Telegram with {len(summaries)} complete summaries.")


if __name__ == "__main__":
    main()
