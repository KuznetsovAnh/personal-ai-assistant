from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any

import requests

from core.settings import settings
from core.search.format import clean_search_text

logger = logging.getLogger(__name__)

_VN_TZ = timezone(timedelta(hours=7))

# ── Content cleaning ──────────────────────────────────────────────────────────

def _clean_result_text(text: str) -> str:
    """Remove scraped garbage (emails, instructions, metadata) from search content."""
    return clean_search_text(text)


def _clean_item(item: dict[str, Any]) -> dict[str, Any]:
    """Apply content cleaning to a search result item."""
    item["content"] = _clean_result_text(item.get("content", ""))
    item["snippet"] = _clean_result_text(item.get("snippet", ""))
    return item


# ── DDG ──────────────────────────────────────────────────────────────────────


def _ddg_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """DuckDuckGo text search via ddgs library."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results, region="vn-vn"))

    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for r in results:
        url = (r.get("href") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        items.append(_clean_item({
            "title": (r.get("title") or "").strip(),
            "url": url,
            "content": (r.get("body") or "").strip(),
            "snippet": (r.get("body") or "").strip()[:300],
            "source": _extract_domain(url),
            "published_date": "",
        }))
    return items


def _ddg_news_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """DuckDuckGo news search via ddgs library."""
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.news(query, max_results=max_results, region="vn-vn"))

    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for r in results:
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        items.append(_clean_item({
            "title": (r.get("title") or "").strip(),
            "url": url,
            "content": (r.get("body") or "").strip(),
            "snippet": (r.get("body") or "").strip()[:300],
            "source": r.get("source") or _extract_domain(url),
            "published_date": r.get("date", ""),
        }))
    return items


# ── Tavily ────────────────────────────────────────────────────────────────────


_TAVILY_SEARCH_URL = "https://api.tavily.com/search"


def _tavily_search(
    query: str,
    max_results: int = 5,
    freshness_hours: int | None = None,
) -> list[dict[str, Any]]:
    if not settings.tavily_api_key:
        raise RuntimeError("missing_tavily_key")

    payload: dict[str, Any] = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
        "include_raw_content": False,
    }
    if freshness_hours:
        payload["time_range"] = _freshness_to_tavily_range(freshness_hours)

    try:
        resp = requests.post(_TAVILY_SEARCH_URL, json=payload, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Tavily API error: %s", e)
        raise

    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for r in data.get("results", []):
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        items.append(_clean_item({
            "title": (r.get("title") or "").strip(),
            "url": url,
            "content": (r.get("content") or "").strip(),
            "snippet": (r.get("content") or "").strip()[:300],
            "source": _extract_domain(url),
            "published_date": r.get("published_date", ""),
        }))
    return items


def _freshness_to_tavily_range(hours: int) -> str:
    if hours <= 1:
        return "1h"
    if hours <= 24:
        return "1d"
    if hours <= 168:
        return "7d"
    return "30d"


# ── Exa ──────────────────────────────────────────────────────────────────────


_EXA_SEARCH_URL = "https://api.exa.ai/search"


def _exa_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    if not settings.exa_api_key:
        raise RuntimeError("missing_exa_key")

    payload: dict[str, Any] = {
        "query": query,
        "numResults": max(1, min(max_results, 10)),
        "type": "auto",
        "contents": {
            "text": {"maxCharacters": 2500},
            "highlights": {"numSentences": 3},
        },
    }

    try:
        resp = requests.post(
            _EXA_SEARCH_URL,
            headers={
                "x-api-key": settings.exa_api_key,
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Exa API error: %s", e)
        raise

    items: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for r in data.get("results", []):
        if not isinstance(r, dict):
            continue
        url = (r.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        text = (r.get("text") or "").strip()
        highlights = r.get("highlights") or []
        if not text and isinstance(highlights, list):
            text = " ".join(str(h).strip() for h in highlights if str(h).strip())
        items.append(_clean_item({
            "title": (r.get("title") or url or "Untitled").strip(),
            "url": url,
            "content": text,
            "snippet": text[:300],
            "source": _extract_domain(url),
            "published_date": r.get("publishedDate") or r.get("published_date", ""),
        }))
    return items


# ── Utilities ────────────────────────────────────────────────────────────────


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return url