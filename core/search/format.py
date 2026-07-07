from __future__ import annotations

import re
from datetime import datetime, timezone, timedelta
from typing import Any

_VN_TZ = timezone(timedelta(hours=7))

# Defense-in-depth: catch any remaining scraped garbage
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_GOOGLE_PREFS_URL_RE = re.compile(r"https?://www\.google\.com/preferences\S*")
_GOOGLE_GARBAGE_MARKERS = [
    "làm nguồn", "trên google", "bước", "nhấp vào nút",
    "bấm vào ô", "mở trực tiếp", "google.com/preferences",
    "ưu tiên trên", "tại trang vừa mở", "tìm dòng",
    "xem hướng dẫn", "chọn báo", "chọn .+? làm nguồn",
]
_VIETNAMESE_CHARS = set("ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")
_VIETNAMESE_WORDS = {
    "và", "của", "trong", "người", "ngày", "hôm", "nay", "tin", "tức",
    "thế", "giới", "việt", "nam", "cho", "với", "đã", "đang", "không",
}
_ENGLISH_WORDS = {
    "the", "and", "of", "to", "in", "for", "on", "with", "says", "said",
    "president", "government", "attack", "news", "latest", "today",
}


def _is_garbage_sentence(s: str) -> bool:
    s_lower = s.lower()
    return any(
        re.search(marker, s_lower) if marker.startswith("chọn") else marker in s_lower
        for marker in _GOOGLE_GARBAGE_MARKERS
    )


def _clean_content(text: str) -> str:
    """Strip emails, Google News setup garbage, and scraped metadata."""
    return clean_search_text(text)


def clean_search_text(text: str) -> str:
    """Strip emails, Google News setup garbage, and scraped metadata."""
    text = _EMAIL_RE.sub("", text)
    text = _GOOGLE_PREFS_URL_RE.sub("", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    clean = [s for s in sentences if not _is_garbage_sentence(s)]
    text = " ".join(clean).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_language(text: str) -> str:
    lower = (text or "").lower()
    if any(ch in _VIETNAMESE_CHARS for ch in lower):
        return "vi"
    words = set(re.findall(r"[a-z]+", lower))
    vi_hits = len(words & _VIETNAMESE_WORDS)
    en_hits = len(words & _ENGLISH_WORDS)
    return "vi" if vi_hits > en_hits else "en"


def _looks_like_language(text: str, target_lang: str) -> bool:
    if not text:
        return False
    detected = _detect_language(text)
    return detected == target_lang


def _prefer_language(items: list[dict[str, Any]], query: str, max_results: int) -> list[dict[str, Any]]:
    target_lang = _detect_language(query)
    if target_lang != "vi":
        return items

    preferred: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    for item in items:
        text = f"{item.get('title', '')} {item.get('content', '')} {item.get('snippet', '')}"
        if _looks_like_language(text, target_lang):
            preferred.append(item)
        else:
            fallback.append(item)
    if preferred:
        return preferred
    return fallback


def _dedup_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate items by URL."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        url = (item.get("url") or "").strip()
        if url and url not in seen:
            seen.add(url)
            result.append(item)
    return result


def format_search_response(items: list[dict[str, Any]], query: str, max_results: int = 5) -> str:
    """Format search results into user-ready text with source citations.

    Each item becomes: ``{idx}. {title}: {content} (via [{source}]({url}))``
    """
    items = _prefer_language(_dedup_items(items), query, max_results)
    if not items:
        return ""

    now = datetime.now(_VN_TZ)
    parts: list[str] = [
        f"Kết quả cho '{query}' — {now.strftime('%d/%m/%Y %H:%M')}:",
        "",
    ]

    for idx, item in enumerate(items[:max_results], start=1):
        title = (item.get("title") or "").strip()
        content = _clean_content((item.get("content") or item.get("snippet") or "").strip())
        url = (item.get("url") or "#").strip()
        source = item.get("source") or _extract_domain(url)

        citation = f"(via [{source}]({url}))" if url and url != "#" else ""

        line = f"{idx}. {title}"
        if content:
            short = content[:500]
            line += f": {short}"
        if citation:
            line += f" {citation}"
        parts.append(line)

    return "\n".join(parts)


def format_news_response(items: list[dict[str, Any]], query: str, max_results: int = 5) -> str:
    """Format news results with headline-numbered layout + blank line between items."""
    items = _prefer_language(_dedup_items(items), query, max_results)
    if not items:
        return ""

    now = datetime.now(_VN_TZ)
    current_date = now.strftime("%d/%m/%Y")
    parts: list[str] = [
        f"Tin tức về '{query}' — ngày {current_date}:",
        "",
    ]

    for idx, item in enumerate(items[:max_results], start=1):
        title = (item.get("title") or "").strip()
        content = _clean_content((item.get("content") or item.get("snippet") or "").strip())
        url = (item.get("url") or "#").strip()
        source = item.get("source") or _extract_domain(url)

        published = item.get("published_date", "")
        date_str = ""
        if published and published != "unknown":
            try:
                dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                date_str = f" ({dt.strftime('%d/%m')})"
            except (ValueError, TypeError):
                pass

        citation = f"(via [{source}]({url}))" if url and url != "#" else ""

        entry = f"{idx}. {title}{date_str}"
        if content:
            entry += f": {content[:600]}"
        if citation:
            entry += f" {citation}"
        parts.append(entry)
        parts.append("")

    return "\n".join(parts).strip()


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return url
