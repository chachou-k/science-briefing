#!/usr/bin/env python3
"""Build a small weekly science digest using Groq's API and send it to Telegram.

Safety choices: one fixed model, no paid-model fallback, at most seven requests,
and fail closed on API/quota errors. Requires GROQ_API_KEY and Telegram secrets.
"""
import json
import os
import time
from datetime import date, timedelta
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode

import collect_articles

GROQ_MODEL = "openai/gpt-oss-20b"
MAX_ARTICLES = 7
MIN_ARTICLES = 5
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"


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
    start = today - timedelta(days=14)
    items = []
    for category, url in collect_articles.ARXIV_FEEDS.items():
        items.extend(collect_articles.get_arxiv(category, url))
    for category, query in collect_articles.EUROPE_PMC_TOPICS.items():
        items.extend(collect_articles.get_europe_pmc(category, query, start.isoformat(), today.isoformat()))
    deduped = {}
    for item in items:
        if item.get("url") and item.get("abstract"):
            deduped.setdefault(item["url"].lower(), item)
    return sorted(deduped.values(), key=lambda x: x.get("date", ""), reverse=True)


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


def select_articles(items):
    # Require an abstract and favour varied topics. Avoid flooding the digest with one feed.
    selected, counts = [], {}
    for item in items:
        group = category_group(item.get("category", ""))
        if counts.get(group, 0) >= 2:
            continue
        abstract = item.get("abstract", "").strip()
        if len(abstract) < 180:
            continue
        selected.append(item)
        counts[group] = counts.get(group, 0) + 1
        if len(selected) >= MAX_ARTICLES:
            break
    if len(selected) < MIN_ARTICLES:
        raise RuntimeError(f"Only found {len(selected)} candidates with useful abstracts; stopping rather than sending a weak digest.")
    return selected


def summarize(item, api_key):
    prompt = f"""Write a careful, accessible English research briefing for a curious non-specialist.
Use ONLY the information supplied below. Never invent sample sizes, statistics, methods, results, or limitations. If the abstract does not report a detail, say it is not stated in the abstract.
Use these short headings:
Why it matters
Research question and method
Main findings (include numbers/sample sizes/effect sizes only when provided)
What the evidence does and does not show
Limitations
Aim for 120-170 words total. Distinguish findings from interpretation and avoid causal claims unless the study supports them. If this is a preprint, explicitly say it has not necessarily been peer reviewed.

Category: {item.get('category', 'Science')}
Title: {item.get('title', '')}
Date: {item.get('date', 'not supplied')}
Evidence status: {item.get('status', 'status not verified')}
Source: {item.get('source', '')}
Abstract:
{item.get('abstract', '')[:7000]}
"""
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a cautious science editor. Be accurate, readable, and transparent about uncertainty."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 650,
    }
    result = request_json(GROQ_URL, payload, {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Groq returned an unexpected response; stopping without sending a digest.") from exc


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

    articles = select_articles(collect_candidates())
    print(f"Selected {len(articles)} candidates. Summarizing with fixed model {GROQ_MODEL}.")
    summaries = []
    for i, item in enumerate(articles, 1):
        print(f"Summarizing article {i}/{len(articles)}")
        summary = summarize(item, api_key)
        summaries.append((item, summary))
        time.sleep(1)

    # Send only after every summary succeeded, so quota/API failures don't yield a half-written digest.
    telegram_send(telegram_token, chat_id,
        f"SCIENCE BRIEFING — {date.today().isoformat()}\n"
        f"{len(summaries)} research stories from the last 14 days. Summaries are based on abstracts; follow the original links for full details.")
    for i, (item, summary) in enumerate(summaries, 1):
        message = (
            f"{i}. [{category_group(item.get('category', 'Science'))}] {item.get('title', '')}\n"
            f"Date: {item.get('date') or 'not supplied'}\n"
            f"Evidence status: {item.get('status', 'not verified')}\n\n"
            f"{summary}\n\nOriginal paper: {item.get('url', '')}"
        )
        # Telegram message limit is 4096 characters; trim only if the model unexpectedly exceeds it.
        if len(message) > 3900:
            message = message[:3850] + "\n\n[Trimmed to fit Telegram.]\n" + item.get("url", "")
        telegram_send(telegram_token, chat_id, message)
        time.sleep(1)
    print("Science briefing sent to Telegram.")


if __name__ == "__main__":
    main()
