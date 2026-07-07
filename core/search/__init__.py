"""
Unified search module.

Usage:
    from core.search import search, search_news

    result = search("công nghệ AI", max_results=5)
    result = search_news("thế giới", max_results=5)
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

from core.search.providers import _ddg_search, _ddg_news_search, _tavily_search, _exa_search
from core.search.format import format_search_response, format_news_response

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = 5, freshness_hours: int | None = None) -> str:
    """Unified web search: DDG → Tavily → Exa fallback.

    Returns pre-formatted text with (via [Source](URL)) citations.
    """
    clean = (query or "").strip()
    if not clean:
        return "Vui lòng nhập truy vấn tìm kiếm hợp lệ."

    items = _try_providers(clean, max_results, freshness_hours)
    if not items:
        return f"Không tìm thấy kết quả cho '{query}'."

    return format_search_response(items, query, max_results)


def search_news(query: str, max_results: int = 5, freshness_hours: int | None = None) -> str:
    """Unified news search: DDG News → DDG Web → Tavily → Exa fallback.

    Returns pre-formatted news text with blank-line spacing.
    """
    clean = (query or "").strip()
    if not clean:
        return "Vui lòng nhập truy vấn tìm kiếm hợp lệ."

    search_query = _rewrite_news_query(clean)

    # Try DDG news first
    try:
        items = _ddg_news_search(search_query, max_results * 3)
        items = _filter_news_items(items, clean)
        if items:
            return format_news_response(items, query, max_results)
    except Exception as e:
        logger.warning("DDG news search failed: %s", e)

    # Fall back to general search providers
    items = _try_providers(search_query, max_results * 2, freshness_hours)
    items = _filter_news_items(items, clean)
    if not items:
        return f"Không tìm thấy tin tức cho '{query}'."

    return format_news_response(items, query, max_results)


def search_raw(query: str, max_results: int = 5, freshness_hours: int | None = None) -> list[dict[str, Any]]:
    """Return raw search results as dicts (for programmatic use)."""
    return _try_providers(query, max_results, freshness_hours)


def _try_providers(query: str, max_results: int, freshness_hours: int | None = None) -> list[dict[str, Any]]:
    """Try DDG → Tavily → Exa, returning first non-empty result list."""
    items: list[dict[str, Any]] = []

    # 1. DDG
    try:
        items = _ddg_search(query, max_results + 3)
        if len(items) >= 2:
            return items
    except Exception as e:
        logger.warning("DDG search failed: %s", e)

    # 2. Tavily
    if not items or len(items) < 2:
        try:
            items = _tavily_search(query, max_results, freshness_hours)
            if len(items) >= 2:
                return items
        except Exception as e:
            if str(e) != "missing_tavily_key":
                logger.warning("Tavily search failed: %s", e)

    # 3. Exa
    if not items or len(items) < 2:
        try:
            items = _exa_search(query, max_results)
        except Exception as e:
            if str(e) != "missing_exa_key":
                logger.warning("Exa search failed: %s", e)

    return items


def _normalize_text(text: str) -> str:
    lowered = (text or "").lower()
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"[^a-z0-9\s]", " ", stripped)


def _is_world_news_query(query: str) -> bool:
    normalized = _normalize_text(query)
    return any(marker in normalized for marker in ("the gioi", "quoc te", "world", "international"))


def _rewrite_news_query(query: str) -> str:
    if _is_world_news_query(query):
        return "tin thế giới quốc tế mới nhất hôm nay chiến sự ngoại giao chính trị"
    return query


def _filter_news_items(items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    if not _is_world_news_query(query):
        return items

    blacklist = (
        "gia xang", "gia dau", "gia vang", "chung khoan", "co phieu",
        "tai san", "giau nhat", "ty phu", "elon musk", "nang nong",
        "thoi tiet", "tp ho chi minh", "trong nuoc", "msn",
    )
    world_signals = (
        "nga", "ukraine", "my", "trung quoc", "israel", "gaza", "iran",
        "eu", "nato", "phap", "anh", "duc", "nhat ban", "han quoc",
        "dong nam a", "chau au", "chau phi", "trump", "putin", "zelensky",
        "chien su", "chien tranh", "ngoai giao", "ten lua", "uav",
        "xung dot", "bau cu", "tong thong", "thu tuong", "quoc te",
        "the gioi", "world", "international", "russia", "china",
        "united states", "middle east",
    )

    filtered: list[dict[str, Any]] = []
    for item in items:
        text = _normalize_text(
            f"{item.get('title', '')} {item.get('content', '')} {item.get('snippet', '')} {item.get('source', '')} {item.get('url', '')}"
        )
        if any(term in text for term in blacklist):
            continue
        if any(term in text for term in world_signals):
            filtered.append(item)
    return filtered
