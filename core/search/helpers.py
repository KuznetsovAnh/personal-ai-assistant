import re
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, urlunparse, parse_qsl, urlencode

from bs4 import BeautifulSoup


class HtmlParser:
    _CONTENT_SELECTORS = [
        "article",
        "[role=main]",
        ".entry-content", ".post-content", ".article-content",
        ".article-body", ".post-body",
    ]

    @staticmethod
    def extract_main_text(html_text: str) -> str:
        if not html_text:
            return ""
        soup = BeautifulSoup(html_text, 'lxml')
        for tag in soup(["script", "style", "noscript", "svg", "form", "button", "header", "footer", "nav", "aside"]):
            tag.decompose()

        for selector in HtmlParser._CONTENT_SELECTORS:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=' ')
                text = re.sub(r"\b(cookie policy|accept cookies|subscribe|advertisement|all rights reserved)\b", " ", text, flags=re.IGNORECASE)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) >= 150:
                    return text

        body = soup.body
        if body:
            candidates = []
            for child in body.find_all(['div', 'section', 'article', 'main'], recursive=False):
                text = child.get_text(separator=' ')
                text = re.sub(r"\b(cookie policy|accept cookies|subscribe|advertisement|all rights reserved)\b", " ", text, flags=re.IGNORECASE)
                text = re.sub(r"\s+", " ", text).strip()
                if len(text) >= 150:
                    candidates.append(text)
            if candidates:
                return max(candidates, key=len)

            text = body.get_text(separator=' ')
            text = re.sub(r"\b(cookie policy|accept cookies|subscribe|advertisement|all rights reserved)\b", " ", text, flags=re.IGNORECASE)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        return ""


class DateParser:
    _MONTH_NAMES = (
        r'(?:January|February|March|April|May|June|July|August|'
        r'September|October|November|December|'
        r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
    )
    _VIET_DATE_RX = re.compile(
        r'(?P<day>\d{1,2})\s*[/\-]\s*(?P<month>\d{1,2})\s*[/\-]\s*(?P<year>\d{4})'
    )
    _VIET_DATE_TEXT_RX = re.compile(
        r'(?:ngày\s+)?(?P<day>\d{1,2})\s*[/\-]\s*(?P<month>\d{1,2})\s*[/\-]\s*(?P<year>\d{4})',
        re.IGNORECASE
    )
    _VIET_TEXT_DATE_RX = re.compile(
        r'(?:(?:ngày|day)\s+)?(?P<day>\d{1,2})\s*'
        r'(?:tháng|thg|month|th|/)\s*(?P<month>\d{1,2})\s*'
        r'(?:năm|year|,)?\s*(?P<year>\d{4})',
        re.IGNORECASE
    )
    _VIET_MONTH_DAY_YEAR_RX = re.compile(
        r'(?:tháng|thg|month)\s+(?P<month>\d{1,2})\s+'
        r'(?P<day>\d{1,2})\s*,?\s*'
        r'(?:năm|year)?\s*(?P<year>\d{4})',
        re.IGNORECASE
    )
    _EN_MONTH_DAY_YEAR_RX = re.compile(
        rf'{_MONTH_NAMES}\s+(?P<day>\d{{1,2}})\s*,?\s+(?P<year>\d{{4}})',
        re.IGNORECASE
    )

    @staticmethod
    def extract_date(text: str) -> Optional[datetime]:
        import dateparser
        if not text:
            return None

        text_stripped = text.strip()

        m = DateParser._VIET_DATE_TEXT_RX.search(text_stripped)
        if m:
            try:
                return datetime(int(m.group('year')), int(m.group('month')), int(m.group('day')))
            except ValueError:
                pass

        m = DateParser._VIET_TEXT_DATE_RX.search(text_stripped)
        if m:
            try:
                return datetime(int(m.group('year')), int(m.group('month')), int(m.group('day')))
            except ValueError:
                pass

        m = DateParser._VIET_MONTH_DAY_YEAR_RX.search(text_stripped)
        if m:
            try:
                return datetime(int(m.group('year')), int(m.group('month')), int(m.group('day')))
            except ValueError:
                pass

        m = DateParser._EN_MONTH_DAY_YEAR_RX.search(text_stripped)
        if m:
            try:
                month_str = m.group(0).split()[0]
                month_map = {
                    'january':1,'february':2,'march':3,'april':4,'may':5,'june':6,
                    'july':7,'august':8,'september':9,'october':10,'november':11,'december':12,
                    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
                }
                month = month_map.get(month_str.lower())
                if month:
                    return datetime(int(m.group('year')), month, int(m.group('day')))
            except (ValueError, KeyError):
                pass

        try:
            return dateparser.parse(text_stripped, languages=['vi', 'en'])
        except Exception:
            return None


class TextProcessor:
    @staticmethod
    def remove_diacritics(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    @staticmethod
    def canonicalize_search_query(query: str) -> str:
        from core.search.constants import SEARCH_CACHE_PHRASE_ALIASES, SEARCH_CACHE_TOKEN_ALIASES, SEARCH_CACHE_STOPWORDS
        lowered = (query or "").strip().lower()
        lowered = lowered.replace("[force fallback]", " ")
        lowered = TextProcessor.remove_diacritics(lowered)
        for src, dst in SEARCH_CACHE_PHRASE_ALIASES:
            lowered = re.sub(rf"\b{re.escape(src)}\b", dst, lowered)
        lowered = re.sub(r"[^a-z0-9_\s]", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        if not lowered:
            return ""
        tokens = []
        for token in lowered.split(" "):
            normalized_token = SEARCH_CACHE_TOKEN_ALIASES.get(token, token)
            if not normalized_token or normalized_token in SEARCH_CACHE_STOPWORDS:
                continue
            if len(normalized_token) <= 1:
                continue
            tokens.append(normalized_token)
        if not tokens:
            return lowered
        canonical_tokens = sorted(set(tokens))
        return " ".join(canonical_tokens[:32])

    @staticmethod
    def normalize_text_for_match(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", (text or "").lower())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def normalize_query_tokens(text: str) -> List[str]:
        normalized = unicodedata.normalize("NFKD", (text or "").lower())
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        tokens = [t for t in normalized.split() if len(t) > 2]
        return tokens

    @staticmethod
    def query_overlap_count(query: str, text: str) -> int:
        q_tokens = set(TextProcessor.normalize_query_tokens(query))
        if not q_tokens:
            return 0
        t_tokens = set(TextProcessor.normalize_query_tokens(text))
        return len(q_tokens.intersection(t_tokens))

    @staticmethod
    def query_coverage_score(query: str, text: str) -> float:
        q_tokens = TextProcessor.normalize_query_tokens(query)
        if not q_tokens:
            return 0.0

        normalized_text = TextProcessor.normalize_text_for_match(text)
        text_tokens = set(normalized_text.split())
        unique_query_tokens = list(dict.fromkeys(q_tokens))

        token_hits = sum(1 for token in unique_query_tokens if token in text_tokens)
        token_ratio = token_hits / len(unique_query_tokens)

        score = 0.0
        if token_ratio >= 0.9:
            score += 1.45
        elif token_ratio >= 0.75:
            score += 1.1
        elif token_ratio >= 0.55:
            score += 0.8
        elif token_ratio >= 0.35:
            score += 0.45
        elif token_ratio > 0:
            score += 0.2

        phrase_hits = 0
        phrase_total = 0
        max_n = min(4, len(q_tokens))
        padded_text = f" {normalized_text} "

        for n in range(max_n, 1, -1):
            for i in range(0, len(q_tokens) - n + 1):
                phrase = " ".join(q_tokens[i:i + n]).strip()
                if len(phrase) < 7:
                    continue
                phrase_total += 1
                if f" {phrase} " in padded_text:
                    phrase_hits += 1

        if phrase_total > 0:
            phrase_ratio = phrase_hits / phrase_total
            if phrase_ratio >= 0.5:
                score += 1.25
            elif phrase_ratio >= 0.3:
                score += 0.85
            elif phrase_hits > 0:
                score += 0.45

        if phrase_hits > 0 and token_ratio >= 0.6:
            score += 0.35

        return round(score, 3)


class UrlUtils:
    _BLOCKED_DOMAINS = {
        "pinterest.com", "pinterest.ca", "facebook.com", "instagram.com",
        "tiktok.com", "reddit.com", "x.com", "twitter.com",
        "shopee", "lazada", "amazon", "tiki",
    }

    @staticmethod
    def normalize_domain(url: str) -> str:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc or parsed.path.split("/")[0]
            return domain.lower().replace("www.", "").strip()
        except Exception:
            return (url or "").lower().replace("www.", "").strip()

    @staticmethod
    def organization_domain(domain: str) -> str:
        d = (domain or "").strip().lower()
        if not d:
            return ""
        labels = [x for x in d.split(".") if x]
        if len(labels) < 2:
            return d
        if len(labels) >= 3 and ".".join(labels[-2:]) in {"com.vn", "gov.vn", "org.vn", "edu.vn", "net.vn"}:
            return ".".join(labels[-3:])
        return ".".join(labels[-2:])

    @staticmethod
    def normalize_url(url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url.strip())
            if parsed.scheme not in {"http", "https"}:
                return ""
            query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
            filtered = []
            for k, v in query_pairs:
                lk = (k or "").lower()
                if lk.startswith("utm_") or lk in {"fbclid", "gclid", "igshid", "mc_cid", "mc_eid"}:
                    continue
                filtered.append((k, v))
            normalized = parsed._replace(
                scheme="https",
                netloc=(parsed.netloc or "").lower(),
                query=urlencode(filtered, doseq=True),
                fragment=""
            )
            clean = urlunparse(normalized)
            return clean[:-1] if clean.endswith("/") else clean
        except Exception:
            return ""

    @staticmethod
    def is_blocked_domain(url: str) -> bool:
        domain = UrlUtils.normalize_domain(url)
        if not domain:
            return True
        return any(token in domain for token in UrlUtils._BLOCKED_DOMAINS)
