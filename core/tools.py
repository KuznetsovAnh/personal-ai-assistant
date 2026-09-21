"""Pure tool functions for the personal assistant.

These functions are registered as Pydantic-AI tools via @agent.tool_plain
in agent.py. They have NO framework dependency so they stay easy to test.

Also exposed via MCP server in mcp_server.py.
"""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, TypeVar
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from core.settings import settings
try:
    from core.search.evidence_crawler import fetch_text_evidence
except ModuleNotFoundError as exc:
    if exc.name not in {"core.search", "core.search.evidence_crawler"}:
        raise

    def fetch_text_evidence(url: str, max_chars: int = 8000, timeout_sec: int = 5, depth: int = 1) -> str:
        """Fallback page evidence fetcher when core.search is not installed."""
        _ = depth
        if re.search(r"\.(?:jpe?g|png|gif|webp|pdf|zip)(?:[?#].*)?$", url or "", flags=re.IGNORECASE):
            return ""
        response = requests.get(
            url,
            timeout=timeout_sec,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.8,vi;q=0.7",
            },
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if any(kind in content_type for kind in ("image/", "application/pdf")):
            return ""
        text = _extract_main_text(response.text)
        return text[:max_chars]

import zoneinfo

VN_TZ_NAME = "Asia/Ho_Chi_Minh"
HTTP_SCHEME = "http://"
HTTPS_SCHEME = "https://"

CITY_HONG_KONG = "Hong Kong"
CITY_NEW_YORK = "New York"
CITY_LOS_ANGELES = "Los Angeles"
CITY_NEW_DELHI = "New Delhi"
CITY_DA_NANG = "Da Nang"
CITY_CAN_THO = "Can Tho"
CITY_HAI_PHONG = "Hai Phong"
CITY_DA_LAT = "Da Lat"
CITY_VUNG_TAU = "Vung Tau"

DOMAIN_VNEXPRESS = "vnexpress.net"
DOMAIN_TUOITRE = "tuoitre.vn"
DOMAIN_THANHNIEN = "thanhnien.vn"
DOMAIN_DANTRI = "dantri.com.vn"
DOMAIN_VIETNAMNET = "vietnamnet.vn"
DOMAIN_VOV = "vov.vn"
DOMAIN_VTV = "vtv.vn"
DOMAIN_LAODONG = "laodong.vn"
DOMAIN_BAODAUTU = "baodautu.vn"
DOMAIN_CAFEF = "cafef.vn"
DOMAIN_VIETSTOCK = "vietstock.vn"
DOMAIN_VNECONOMY = "vneconomy.vn"

_VN_TZ = zoneinfo.ZoneInfo(VN_TZ_NAME)
_VN_WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

logger = logging.getLogger(__name__)


class DDGRateLimitError(RuntimeError):
    """Raised when DuckDuckGo blocks/rate-limits search requests."""


def get_current_datetime() -> str:
    """Return the current system date and time in Vietnam timezone."""
    now_vn = datetime.now(_VN_TZ)
    weekday = _VN_WEEKDAYS[now_vn.weekday()]
    return now_vn.strftime(f"%H:%M:%S, {weekday}, ngày %d/%m/%Y (giờ Việt Nam, UTC+7)")


def save_memory(fact: str) -> str:
    """Save an important fact about the user (e.g., name, preferences)."""
    return f"Đã ghi nhớ thông tin: {fact}"


def translate_text(text: str, target_lang: str) -> str:
    """Translate text to the target language."""
    return f"Bản dịch ({target_lang}): {text}"


def search(query: str, max_results: int = 5, freshness_hours: int | None = None) -> str:
    """Unified search: DDG → Tavily → Exa fallback. Returns formatted text with source citations."""
    from core.search import search as search_module_fn
    return search_module_fn(query, max_results=max_results, freshness_hours=freshness_hours)


def knowledge_search(query: str, _topic: str = "") -> str:
    """Search in the local knowledge base."""
    normalized = (query or "").strip().lower()
    entries = {
        "python": (
            "Python là một ngôn ngữ lập trình bậc cao, thông dịch, đa mục đích, nổi bật nhờ cú pháp dễ đọc "
            "và hệ sinh thái thư viện rất lớn. Python thường được dùng trong phát triển web, tự động hóa, "
            "khoa học dữ liệu, trí tuệ nhân tạo, giáo dục lập trình và viết script hệ thống."
        ),
        "hà nội": (
            "Hà Nội là thủ đô của Việt Nam, nằm ở khu vực Đồng bằng sông Hồng và là trung tâm chính trị, "
            "văn hóa, giáo dục quan trọng của cả nước. Thành phố có lịch sử lâu đời với nhiều di tích như "
            "Hoàng thành Thăng Long, Văn Miếu - Quốc Tử Giám và Hồ Gươm."
        ),
        "albert einstein": (
            "Albert Einstein là nhà vật lý lý thuyết nổi tiếng, người phát triển thuyết tương đối hẹp và "
            "thuyết tương đối rộng. Công thức E=mc² của Einstein mô tả mối liên hệ giữa khối lượng và năng lượng, "
            "và ông nhận giải Nobel Vật lý năm 1921 nhờ giải thích hiệu ứng quang điện."
        ),
        "vinai": "VinAI là viện nghiên cứu trí tuệ nhân tạo hàng đầu tại Việt Nam, tập trung vào nghiên cứu ứng dụng AI, thị giác máy tính, xử lý ngôn ngữ tự nhiên và các sản phẩm công nghệ thông minh.",
    }
    for key, value in entries.items():
        if key in normalized:
            return value
    return "Không tìm thấy thông tin trong cơ sở dữ liệu cục bộ cho chủ đề này. Bạn có thể dùng web_search để tra cứu nguồn cập nhật hơn."


def _currency_lookup_key(value: str) -> str:
    return re.sub(r"\s+", " ", _normalize_ascii_words(value or "")).strip()


def _supported_currency_catalog() -> list[dict[str, Any]]:
    if not settings.exchangerate_api_key:
        return []

    def fetch_codes() -> list[dict[str, Any]]:
        url = f"https://v6.exchangerate-api.com/v6/{settings.exchangerate_api_key}/codes"
        response = _session.get(url, timeout=8)
        response.raise_for_status()
        payload = response.json()
        catalog: list[dict[str, Any]] = []
        for row in payload.get("supported_codes") or []:
            if not isinstance(row, list) or len(row) < 2:
                continue
            code = str(row[0]).strip().upper()
            name = str(row[1]).strip()
            if not re.fullmatch(r"[A-Z]{3}", code):
                continue
            key = _currency_lookup_key(name)
            catalog.append(
                {
                    "code": code,
                    "name": name,
                    "key": key,
                }
            )
        return catalog

    try:
        return _cached("exchange-rate:supported-codes", fetch_codes, ttl_seconds=86400)
    except Exception:
        logger.warning("Exchange-rate supported-codes lookup failed", exc_info=True)
        return []


def _supported_currency_codes() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in _supported_currency_catalog():
        code = item["code"]
        mapping[code.lower()] = code
        mapping[_currency_lookup_key(code)] = code
        mapping[item["key"]] = code
    return mapping


def _resolve_currency_code(value: str) -> str:
    raw = (value or "").strip()
    upper = raw.upper()
    if re.fullmatch(r"[A-Z]{3}", upper):
        return upper

    key = _currency_lookup_key(raw)
    if not key:
        return ""

    supported = _supported_currency_codes()
    if key in supported:
        return supported[key]
    for name_key, code in supported.items():
        if len(key) >= 4 and (key in name_key or name_key in key):
            return code
    return ""


def _format_exchange_rate(rate: float, target: str) -> str:
    if target in {"VND", "JPY", "KRW", "IDR"} or abs(rate) >= 1000:
        return f"{rate:,.0f}"
    decimals = 6 if abs(rate) < 1 else 4
    return f"{rate:,.{decimals}f}".rstrip("0").rstrip(".")


def get_exchange_rate(currency_from: str, currency_to: str) -> str:
    """Get a live currency exchange rate."""
    base = _resolve_currency_code(currency_from)
    target = _resolve_currency_code(currency_to)
    now = datetime.now(_VN_TZ).strftime("%d/%m/%Y %H:%M")
    if not base or not target:
        return "Mình chưa xác định được mã tiền tệ cần quy đổi. Hãy gọi tool với mã ISO 4217 như USD, EUR, CNY, JPY, RUB, AUD."
    if base == target:
        return f"Cập nhật {now} (giờ Việt Nam): 1 {base} = 1 {target}."
    if not settings.exchangerate_api_key:
        return "Mình chưa có API key tỷ giá nên chưa thể lấy tỷ giá real-time đáng tin cậy."

    try:
        url = f"https://v6.exchangerate-api.com/v6/{settings.exchangerate_api_key}/pair/{base}/{target}"
        response = _session.get(url, timeout=8)
        response.raise_for_status()
        payload = response.json()
        rate = payload.get("conversion_rate")
        if rate is None:
            raise ValueError(f"missing conversion_rate: {payload}")
        rate_f = float(rate)
        shown = _format_exchange_rate(rate_f, target)
        return f"Cập nhật {now} (giờ Việt Nam): 1 {base} ≈ {shown} {target}."
    except Exception:
        logger.warning("Exchange-rate API failed for %s/%s", base, target, exc_info=True)
        return f"Mình chưa lấy được tỷ giá real-time cho {base}/{target} lúc này. Vui lòng thử lại sau."


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.
    Allowed operators: +, -, *, /, ** (power), % (modulo), and parentheses.
    """
    cleaned = expression.strip()
    if not cleaned:
        return "Không thể thực hiện phép tính: biểu thức trống."
    try:
        allowed_chars = set("0123456789+-*/.%() ")
        if not all(c in allowed_chars for c in cleaned):
            return "Không thể thực hiện phép tính: biểu thức không hợp lệ."
        res = eval(cleaned, {"__builtins__": None}, {})
        if isinstance(res, float):
            res_str = f"{res:.3f}"
        else:
            res_str = str(res)
        return f"{cleaned} = {res_str}"
    except Exception as e:
        return f"Không thể thực hiện phép tính: {str(e)}"


# ── Generic retry + fallback wrapper ─────────────────────────────────────────
T = TypeVar("T")


def _safe_tool_call(
    func: Callable[..., T],
    *args: Any,
    retries: int = 1,
    delay: float = 0.5,
    fallback: T | None = None,
    tool_name: str = "",
    **kwargs: Any,
) -> T:
    """Call *func* with automatic retry + friendly fallback on failure."""
    for attempt in range(1 + retries):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            name = tool_name or func.__name__
            logger.warning(
                "[%s] attempt %d/%d failed: %s",
                name, attempt + 1, 1 + retries, exc,
            )
            if attempt < retries:
                time.sleep(delay)

    if fallback is not None:
        return fallback  # type: ignore[return-value]
    name = tool_name or func.__name__
    return f"Xin lỗi, không thể thực hiện {name} lúc này. Vui lòng thử lại sau."  # type: ignore[return-value]


# ── TTL Cache ────────────────────────────────────────────────────────────────
_cache_store: Dict[str, tuple[Any, float]] = {}
_cache_lock = threading.Lock()


def _cached(key: str, func: Callable[..., T], ttl_seconds: int, *args: Any, **kwargs: Any) -> T:
    """Return cached result if fresh, otherwise call *func* and cache it."""
    now = time.time()
    with _cache_lock:
        if key in _cache_store:
            result, cached_at = _cache_store[key]
            if now - cached_at < ttl_seconds:
                logger.debug("[cache-hit] %s (age %.1fs)", key, now - cached_at)
                return result  # type: ignore[return-value]

    result = func(*args, **kwargs)

    with _cache_lock:
        _cache_store[key] = (result, time.time())
    return result


_HCM = "Ho Chi Minh City"

# Shared session with retry logic and connection pooling
_retry_strategy = Retry(
    total=2,
    backoff_factor=0.2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
_adapter = HTTPAdapter(max_retries=_retry_strategy, pool_connections=10, pool_maxsize=20)
_session = requests.Session()
_session.mount(HTTPS_SCHEME, _adapter)
_session.mount(HTTP_SCHEME, _adapter)


_EVIDENCE_TOTAL_TIMEOUT = 15  # hard ceiling (seconds) for ALL page-evidence fetches combined


class _DaemonPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor whose worker threads are daemon threads.

    A worker can get stuck in a blocking OS call (e.g. DNS resolution via
    ``socket.getaddrinfo``) that no per-request timeout can interrupt. Daemon
    workers keep such a hang from blocking the main thread or leaking a
    non-daemon thread in a long-lived agent.
    """

    def _adjust_thread_count(self) -> None:
        # Mirrors ThreadPoolExecutor._adjust_thread_count but spawns daemon threads.
        if self._idle_semaphore.acquire(timeout=0):
            return

        import concurrent.futures.thread as _cft
        import weakref

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        num_threads = len(self._threads)
        if num_threads < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, num_threads)
            t = threading.Thread(
                name=thread_name,
                target=_cft._worker,
                args=(
                    weakref.ref(self, weakref_cb),
                    self._work_queue,
                    self._initializer,
                    self._initargs,
                ),
                daemon=True,
            )
            t.start()
            self._threads.add(t)
            _cft._threads_queues[t] = self._work_queue


_REQUEST_TIMEOUT = 12  # seconds
_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_EXA_SEARCH_URL = "https://api.exa.ai/search"
_TAVILY_BROAD_MIN_RESULTS = 25
_TAVILY_BROAD_MAX_RESULTS = 35
_MIN_PRERERANK_K = 3
_MAX_PRERERANK_K = 10

WEATHER_CODE_MAP = {
    0: "trời quang",
    1: "ít mây",
    2: "có mây",
    3: "nhiều mây",
    45: "sương mù",
    48: "sương mù đóng băng",
    51: "mưa phùn nhẹ",
    53: "mưa phùn vừa",
    55: "mưa phùn dày",
    61: "mưa nhẹ",
    63: "mưa vừa",
    65: "mưa to",
    80: "mưa rào nhẹ",
    81: "mưa rào vừa",
    82: "mưa rào mạnh",
    95: "dông",
    96: "dông kèm mưa đá nhẹ",
    99: "dông kèm mưa đá mạnh",
}

_CITY_TIMEZONE_MAP = {
    "Tokyo": "Asia/Tokyo",
    "Beijing": "Asia/Shanghai",
    "Shanghai": "Asia/Shanghai",
    "Seoul": "Asia/Seoul",
    "Bangkok": "Asia/Bangkok",
    "Singapore": "Asia/Singapore",
    CITY_HONG_KONG: "Asia/Hong_Kong",
    "Taipei": "Asia/Taipei",
    "London": "Europe/London",
    "Paris": "Europe/Paris",
    "Berlin": "Europe/Berlin",
    "Moscow": "Europe/Moscow",
    CITY_NEW_YORK: "America/New_York",
    CITY_LOS_ANGELES: "America/Los_Angeles",
    "Chicago": "America/Chicago",
    "Toronto": "America/Toronto",
    "Sydney": "Australia/Sydney",
    "Melbourne": "Australia/Melbourne",
    "Dubai": "Asia/Dubai",
    "Mumbai": "Asia/Kolkata",
    CITY_NEW_DELHI: "Asia/Kolkata",
    "Jakarta": "Asia/Jakarta",
    "Cairo": "Africa/Cairo",
    # ── Việt Nam cities → đúng timezone ──
    "Hanoi": VN_TZ_NAME,
    "Ho Chi Minh City": VN_TZ_NAME,
    CITY_DA_NANG: VN_TZ_NAME,
    "Hue": VN_TZ_NAME,
    CITY_CAN_THO: VN_TZ_NAME,
    CITY_HAI_PHONG: VN_TZ_NAME,
    "Nha Trang": VN_TZ_NAME,
    CITY_DA_LAT: VN_TZ_NAME,
    CITY_VUNG_TAU: VN_TZ_NAME,
    "Phu Quoc": VN_TZ_NAME,
}


def normalize_location(location: str) -> str:
    """Normalize Vietnamese location names to English for geocoding API."""
    cleaned = location.strip()
    lower = cleaned.lower()
    # Remove common Vietnamese question suffixes
    trailing = [
        "là bao nhiêu độ", "bao nhiêu độ", "là bao nhiêu",
        "như thế nào", "thế nào", "ra sao", "hôm nay",
        "ngày mai", "hiện tại", "bây giờ", "lúc này",
        "đang là bao nhiêu", "đang thế nào", "đang ra sao",
        "có mưa không", "có nắng không", "có gió không",
        "nhiệt độ", "thời tiết", "weather", "temperature",
    ]
    for phrase in trailing:
        if lower.endswith(phrase):
            cleaned = cleaned[: len(cleaned) - len(phrase)].strip()
            lower = cleaned.lower()
    cleaned = cleaned.strip(" ?!,.:")
    if not cleaned:
        cleaned = "TPHCM"

    normalized = cleaned.lower()
    alias_map = {
        # ── Việt Nam ──
        "tphcm": _HCM,
        "tp hcm": _HCM,
        "tp.hcm": _HCM,
        "tp hồ chí minh": _HCM,
        "thành phố hồ chí minh": _HCM,
        "thành phố hồ chí minhh": _HCM,
        "sài gòn": _HCM,
        "sai gon": _HCM,
        "hcm": _HCM,
        "hcmc": _HCM,
        "hồ chí minhh": _HCM,
        "hà nội": "Hanoi",
        "ha noi": "Hanoi",
        "đà nẵng": CITY_DA_NANG,
        "da nang": CITY_DA_NANG,
        "huế": "Hue",
        "hue": "Hue",
        "cần thơ": CITY_CAN_THO,
        "can tho": CITY_CAN_THO,
        "hải phòng": CITY_HAI_PHONG,
        "hai phong": CITY_HAI_PHONG,
        "nha trang": "Nha Trang",
        "đà lạt": CITY_DA_LAT,
        "da lat": CITY_DA_LAT,
        "vũng tàu": CITY_VUNG_TAU,
        "vung tau": CITY_VUNG_TAU,
        "biên hòa": "Bien Hoa",
        "bien hoa": "Bien Hoa",
        "quy nhơn": "Quy Nhon",
        "quy nhon": "Quy Nhon",
        "buôn ma thuột": "Buon Ma Thuot",
        "bình dương": "Binh Duong",
        "long an": "Long An",
        "thái nguyên": "Thai Nguyen",
        "nam định": "Nam Dinh",
        "vinh": "Vinh",
        "thanh hóa": "Thanh Hoa",
        "nghệ an": "Nghe An",
        "phú quốc": "Phu Quoc",

        # ── Châu Á ──
        "tokyo": "Tokyo",
        "đông kinh": "Tokyo",
        "osaka": "Osaka",
        "kyoto": "Kyoto",
        "bắc kinh": "Beijing",
        "bac kinh": "Beijing",
        "beijing": "Beijing",
        "thượng hải": "Shanghai",
        "thuong hai": "Shanghai",
        "shanghai": "Shanghai",
        "quảng châu": "Guangzhou",
        "quang chau": "Guangzhou",
        "thâm quyến": "Shenzhen",
        "tham quyen": "Shenzhen",
        "hồng kông": CITY_HONG_KONG,
        "hong kong": CITY_HONG_KONG,
        "đài bắc": "Taipei",
        "dai bac": "Taipei",
        "taipei": "Taipei",
        "seoul": "Seoul",
        "xơ un": "Seoul",
        "busan": "Busan",
        "bangkok": "Bangkok",
        "băng cốc": "Bangkok",
        "bang coc": "Bangkok",
        "singapore": "Singapore",
        "xin ga po": "Singapore",
        "kuala lumpur": "Kuala Lumpur",
        "jakarta": "Jakarta",
        "manila": "Manila",
        "new delhi": CITY_NEW_DELHI,
        "niu đê li": CITY_NEW_DELHI,
        "mumbai": "Mumbai",
        "phnom penh": "Phnom Penh",
        "phnôm pênh": "Phnom Penh",
        "viêng chăn": "Vientiane",
        "vientiane": "Vientiane",
        "yangon": "Yangon",
        "dubai": "Dubai",

        # ── Châu Âu ──
        "london": "London",
        "luân đôn": "London",
        "luan don": "London",
        "paris": "Paris",
        "pa ri": "Paris",
        "berlin": "Berlin",
        "béc lin": "Berlin",
        "bec lin": "Berlin",
        "madrid": "Madrid",
        "ma đrít": "Madrid",
        "roma": "Rome",
        "rome": "Rome",
        "la mã": "Rome",
        "amsterdam": "Amsterdam",
        "moscow": "Moscow",
        "mát xcơ va": "Moscow",
        "mat xco va": "Moscow",
        "mạc tư khoa": "Moscow",
        "mac tu khoa": "Moscow",
        "vienna": "Vienna",
        "viên": "Vienna",
        "zurich": "Zurich",
        "prague": "Prague",
        "praha": "Prague",
        "warsaw": "Warsaw",
        "vác sa va": "Warsaw",
        "istanbul": "Istanbul",
        "athens": "Athens",
        "a ten": "Athens",
        "barcelona": "Barcelona",
        "munich": "Munich",
        "lisbon": "Lisbon",
        "stockholm": "Stockholm",
        "oslo": "Oslo",
        "helsinki": "Helsinki",
        "copenhagen": "Copenhagen",
        "brussels": "Brussels",

        # ── Châu Mỹ ──
        "new york": CITY_NEW_YORK,
        "niu oóc": CITY_NEW_YORK,
        "niu ước": CITY_NEW_YORK,
        "nữu ước": CITY_NEW_YORK,
        "nuu uoc": CITY_NEW_YORK,
        "los angeles": CITY_LOS_ANGELES,
        "la": CITY_LOS_ANGELES,
        "chicago": "Chicago",
        "si ca gô": "Chicago",
        "san francisco": "San Francisco",
        "washington": "Washington",
        "hoa thịnh đốn": "Washington",
        "hoa thinh don": "Washington",
        "seattle": "Seattle",
        "miami": "Miami",
        "houston": "Houston",
        "boston": "Boston",
        "las vegas": "Las Vegas",
        "toronto": "Toronto",
        "vancouver": "Vancouver",
        "mexico city": "Mexico City",
        "são paulo": "Sao Paulo",
        "sao paulo": "Sao Paulo",
        "buenos aires": "Buenos Aires",

        # ── Châu Đại Dương ──
        "sydney": "Sydney",
        "xít ni": "Sydney",
        "melbourne": "Melbourne",
        "auckland": "Auckland",

        # ── Châu Phi ──
        "cairo": "Cairo",
        "cai rô": "Cairo",
        "cape town": "Cape Town",
        "lagos": "Lagos",
        "nairobi": "Nairobi",
    }
    return alias_map.get(normalized, cleaned.strip())


def geocode_location(location: str) -> Dict[str, Any]:
    """Geocode a location name to lat/lon via Open-Meteo API."""
    response = _session.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": normalize_location(location),
            "count": 1,
            "language": "vi",
            "format": "json",
        },
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"Không tìm thấy địa điểm: {location}")
    return results[0]


def _get_weather_raw(location: str) -> str:
    """Internal: fetch weather data (no cache/retry wrapping)."""
    clean_location = normalize_location(location)
    place = geocode_location(clean_location)

    # ── Xác định timezone cho thành phố ──
    city_name = place.get("name", clean_location)
    tz_name = _CITY_TIMEZONE_MAP.get(clean_location,
              _CITY_TIMEZONE_MAP.get(city_name, VN_TZ_NAME))

    response = _session.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
            "timezone": tz_name,
            "forecast_days": 1,
        },
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    current = data.get("current", {})
    daily = data.get("daily", {})
    daily_code = daily.get("weather_code", [current.get("weather_code")])[0]

    resolved = ", ".join(
        str(item)
        for item in [place.get("name"), place.get("admin1"), place.get("country")]
        if item
    )
    weather_text = WEATHER_CODE_MAP.get(current.get("weather_code"), "không rõ")
    daily_weather = WEATHER_CODE_MAP.get(daily_code, "không rõ")

    # ── Hiển thị giờ LOCAL của thành phố đó ──
    try:
        local_tz = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        local_tz = _VN_TZ
    now_local = datetime.now(local_tz)
    time_str = now_local.strftime("%H:%M %d/%m/%Y")
    tz_label = tz_name.split("/")[-1].replace("_", " ")

    return (
        f"Thời tiết tại {resolved} (cập nhật lúc {time_str} giờ {tz_label}): "
        f"hiện tại trời {weather_text}, "
        f"nhiệt độ {current.get('temperature_2m')}°C, "
        f"cảm giác như {current.get('apparent_temperature')}°C, "
        f"độ ẩm {current.get('relative_humidity_2m')}%, "
        f"gió {current.get('wind_speed_10m')} km/h, "
                f"lượng mưa {current.get('precipitation')} mm. "
        f"Dự báo hôm nay: {daily_weather}, "
        f"thấp nhất {daily.get('temperature_2m_min', [None])[0]}°C, "
        f"cao nhất {daily.get('temperature_2m_max', [None])[0]}°C, "
        f"khả năng mưa {daily.get('precipitation_probability_max', [None])[0]}%."
    )


def get_weather(location: str) -> str:
    """Lấy thời tiết hiện tại và dự báo hôm nay theo địa điểm.

    Args:
        location: Tên thành phố hoặc địa điểm cần lấy thời tiết.

    Returns:
        Chuỗi mô tả thời tiết bằng tiếng Việt.
    """
    cache_key = f"weather:{normalize_location(location).lower()}"
    return _cached(
        cache_key,
        lambda: _safe_tool_call(
            _get_weather_raw, location,
            retries=1, delay=0.5,
            fallback=f"Không thể lấy dữ liệu thời tiết cho {location} lúc này. Vui lòng thử lại sau.",
            tool_name="get_weather",
        ),
        ttl_seconds=600,  # cache 10 phút
    )


# ══════════════════════════════════════════════════════════════════════════════
# HTML / SNIPPET UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def strip_html_tags(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    cleaned = re.sub(r"<[^>]*>", " ", raw_html)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


_BOILERPLATE_SNIPPET_MARKERS = [
    "cookie",
    "privacy",
    "điều khoản",
    "đăng nhập",
    "sign in",
    "all rights reserved",
    "subscribe",
    "newsletter",
]


def _is_boilerplate_snippet(text: str) -> bool:
    """Return True if text looks like website boilerplate rather than content."""
    lower = text.strip().lower()
    if len(lower) < 24:
        return True
    return any(marker in lower for marker in _BOILERPLATE_SNIPPET_MARKERS)


# Vietnamese boilerplate/filler sentence patterns to strip from snippets
# These are common meta-text phrases that add no informational value
_SNIPPET_BOILERPLATE_PATTERNS = [
    # Time/meta stamps (full sentence)
    r"(?i)(bài viết|bản tin|tin tức|thông tin)\s+(được\s+)?(đăng|cập nhật|công bố|chia sẻ)\s+(vào lúc|lúc|ngày|từ|vào)\s+[\d:\-/.,\s]+",
    r"(?i)(đăng|cập nhật|công bố)\s+(lúc|vào lúc|ngày)\s+[\d:\-/.,\s]+",
    # Author byline + timestamp: "Dũng Huỳnh 26/05/2026, 18:43" or "Tên Tác Giả DD/MM/YYYY HH:MM"
    # Only match 2 capitalized words immediately followed by date+time (not longer sequences)
    r"(?i)(?:^|(?<=\s))(?:[A-ZÀ-Ỹ][a-zà-ỹ]+\s+)[A-ZÀ-Ỹ][a-zà-ỹ]+\s+\d{1,2}[/\.]\d{1,2}[/\.]\d{4}\s*[,\.]?\s*\d{1,2}:\d{2}",
    # Section label with count: "0 Thời sự", "5 Bình luận"
    r"(?i)\b\d+\s+(thời sự|kinh tế|thể thao|giải trí|thế giới|pháp luật|sức khỏe|giáo dục|công nghệ|du lịch|xe|đời sống)\b",
    # Discussion prompt with count: "Cùng luận bàn 0"
    r"(?i)(cùng\s+luận\s+bàn|thảo\s+luận|bàn\s+luận)\s+\d+",
    # Author role labels
    r"(?i)\b(Nhà\s+báo|PV|Phóng\s+viên|Nhà\s+ngoại\s+giao|Chuyên\s+gia|Tiến\s+sỹ|Giáo\s+sư)\b",
    r"(?i)và\s+\d+\s+tác\s+giả\s+khác",
    # Source labels
    r"(?i)(theo\s+ghi\s+nhận\s+của|theo\s+báo\s+cáo|theo\s+đánh\s+giá|theo\s+thống\s+kê)",
    # Filler intro: "Đây là X đáng chúý/quan trọng..." (allow any words between type and adjective)
    r"(?i)đây là\s+(kết quả|thông tin|nhóm thông tin|nhóm trận|bài viết|bản tin|tin tức|nội dung|điểm nhấn|khung giờ|trận)\s+.*?(nổi bật|đáng chú ý|quan trọng|mới nhất|chính|mới được cập nhật|dành cho|vì|có)",
    r"(?i)(nội dung|bài viết|bản tin)\s+(bài\s+)?(này|viết)\s+(cung cấp|tập trung|đề cập|nói|chia sẻ|cập nhật)\s+(thông tin\s+)?(về|cho|liên quan)",
    r"(?i)nội dung\s+bài\s+viết\s+(cung cấp|tập trung|đề cập|chia sẻ|cho biết)",
    # Content focus meta
    r"(?i)(nội dung|bài viết|bản tin)\s+(tập trung|chủ yếu)\s+(vào\s+)?(việc\s+)?(tổng hợp|nắm nhanh|cung cấp|đưa tin|tóm tắt)\s+[^\.\,]*",
    # Filler outro phrases (full sentence)
    r"(?i)(để biết thêm chi tiết|xem thêm|đọc thêm|mời bạn đọc|các bạn có thể)\s+[^\.\,]*",
    r"(?i)(theo dõi|đón đọc)\s+(thêm|tiếp|trên)\s+[^\.\,]*",
    # Vague result intro (full sentence, no specific facts)
    r"(?i)(ghi nhận|cho thấy|đáng chú ý|nổi bật)\s+(nhiều\s+)?(kết quả|diễn biến|thông tin|thay đổi|tác động)\s+(đáng chú ý|quan trọng|nổi bật|mới)",
    r"(?i)(loạt|chuỗi|nhiều)\s+(trận|kết quả|diễn biến|tin tức|thông tin)\s+(được\s+)?(cập nhật|ghi nhận|diễn ra)\s+(trong|ở|vào)\s+(ngày|đêm|rạng sáng|sáng)",
    r"(?i)(loạt|chuỗi|nhiều)\s+(trận|kết quả|diễn biến|tin tức|thông tin)\s+.*?(có|ghi nhận)\s+nhiều\s+(kết quả|diễn biến|tin tức)\s+(đáng chú ý|nổi bật|quan trọng)",
    # Market/intro meta
    r"(?i)(thị trường|làng\s+chuyển\s+nhượng)\s+.*?(ghi nhận|có|thấy)\s+nhiều\s+(tin|diễn biến|thông tin)\s+(đáng chú ý|nổi bật|quan trọng|nóng)\s+(liên quan|liên quan đến|về)",
    # Generic news boilerplate (full sentence)
    r"(?i)(trang tin|kênh tin tức|báo điện tử)\s+(cập nhật|cung cấp|đưa tin)\s+(nhanh chóng|liên tục|newest|24h)",
    r"(?i)(thông tin|tin tức)\s+(trên|từ|của)\s+\w+\s+(cho biết|cho hay|viết|đăng tải)",
    # Source attribution repeated at end (already cited in SOURCE field)
    r"(?i)\(theo\s+[^)]+\)\s*$",
    # Redundant meta descriptions (full sentence)
    r"(?i)(tổng hợp|điểm tin|điểm báo|bản tin)\s+(từ|các|trên)\s+[^\.\,]*",
    # Self-referential LLM-style meta commentary (full sentence)
    r"(?i)(đây là|nội dung này)\s+(thông tin|kết quả|bản tin|bài viết|dữ liệu)\s+(liên quan|thuộc|phù hợp|quan trọng)\s+(với|cho|về|đến)",
    # Category nav sequences: "Kinh doanh Tiêu dùng Thứ hai, 01/06/2026 - 09:00 :"
    r"(?i)^(kinh doanh|tiêu dùng|thời sự|trong nước|quốc tế|thể thao|giải trí|sức khỏe|đời sống|giáo dục|văn hóa|pháp luật)(?:\s+(kinh doanh|tiêu dùng|thời sự|trong nước|quốc tế|thể thao|giải trí|sức khỏe|đời sống|giáo dục|văn hóa|pháp luật))*\s+(thứ\s+[hai|ba|bốn|năm|sáu|bảy|cn]|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*\d{1,2}[/\.]\d{1,2}[/\.]\d{4}\s*[-–—:]+\s*\d{1,2}:\d{2}",
    # Broken sentence fragments: "Tại số ChatToday hôm nay của báo ."
    r"(?i)^(tại\s+|trong\s+số|ở\s+|về\s+|của\s+)\w+\s+hôm\s+nay\s+của\s+báo\s*\.?\s*$",
    r"(?i)^\w+\s+\(\w+\)\s*[-–—:]\s*$",  # "(Dân trí) -" standalone
]

# Substring patterns for in-sentence boilerplate prefix removal
_SNIPPET_BOILERPLATE_PREFIX_PATTERNS = [
    # Leading > artifact (blockquote or navigation marker)
    (r"(?i)^>\s*", ""),
    # Transition + meta intro: "Ngoài ra, bản tin X đề cập việc " -> keep content after "việc "
    (r"(?i)^(ngoài ra|bên cạnh đó|cùng với đó|đồng thời)\s*[,;]?\s*(bản tin|tin tức|thông tin|loạt tin)\s+.*?(đề cập|nhắc tới|nói về|cho biết|đưa tin)\s+(việc\s+|về\s+|thêm\s+)?", ""),
    # Meta intro: "Bản tin X cho biết " -> keep content after
    (r"(?i)^(bản tin|tin tức|loạt tin|thông tin)\s+.*?(cho biết|cho hay|viết|đăng tải|cập nhật|đề cập|nói)\s+(:\s+|rằng\s+|việc\s+)?", ""),
]

# Substring patterns for in-sentence boilerplate suffix removal
# These match trailing meta phrases that should be stripped from end of sentence
_SNIPPET_BOILERPLATE_SUFFIX_PATTERNS = [
    # Time window meta at end of sentence: ", vẫn nằm trong khung 48 giờ gần nhất so với hiện tại."
    (r"(?i)\s*[,;]\s*(vẫn|nằm|thuộc)\s+(nằm\s+)?(trong|ở|ở trong)\s+(khung|khoảng|phạm vi)\s+\d+\s*(giờ|phút|ngày|tuần)\s+(gần nhất|mới nhất|qua|trước|hiện tại)\s*(so với hiện tại)?\s*\.?\s*$", "."),
    # Vague relevance/importance at end
    (r"(?i)\s*[,;]\s*(vì|do|nên)\s+(liên quan|thuộc|phù hợp|quan trọng|ảnh hưởng)\s+(trực tiếp\s+)?(đến|với|cho)\s+[^\.\,]*\.?\s*$", "."),
    # Meta commentary about content focus: ", nội dung tập trung vào việc..."
    (r"(?i)\s*[,;]\s*(nội dung|bài viết|bản tin)\s+(tập trung|chủ yếu)\s+(vào\s+)?(việc\s+)?(tổng hợp|nắm nhanh|cung cấp)\s+[^\.\,]*\.?\s*$", "."),
    # "Đây là X đáng chú ý với..." at end
    (r"(?i)\s*[,;]\s*đây là\s+(điểm nhấn|trận|thông tin)\s+(đáng chú ý|quan trọng|nổi bật)\s+(với|cho|của)\s+[^\.\,]*\.?\s*$", "."),
]

# OPTIMIZATION 4: Pre-compile regex patterns for performance
_SNIPPET_BOILERPLATE_COMPILED = []
_SNIPPET_PREFIX_COMPILED = []
_SNIPPET_SUFFIX_COMPILED = []


def _clean_snippet_content(text: str) -> str:
    """Remove boilerplate/filler sentences from snippet content, keeping core news facts.

    CRITICAL: Must remove image captions, leading digits, UI artifacts, VOV weather widgets
    so LLM gets CLEAN data for summarization.
    """
    if not text or len(text) < 30:
        return text

    # Step 0: Remove leading markdown/blockquote prefix
    text = re.sub(r'^>\s+', '', text)

    # Step 0a: CRITICAL - Strip VOV weather widget content
    # VOV injects: "Hà Nội Đặt mặc định Xem Cao Bằng Đặt mặc định Xem..."
    # Must strip BEFORE other cleaning to avoid polluting the content
    text = _strip_vov_weather_widget(text)
    if not text or len(text) < 30:
        return text

    text = _strip_news_leadin_noise(text)
    if not text or len(text) < 30:
        return text

    # Step 0b: CRITICAL - Remove image caption artifacts at START of content
    # Pattern: "0 (Ảnh: AP) Đợt tấn công..." → "Đợt tấn công..."
    text = re.sub(r'^(?:\d+\s+)+(?:\(Ảnh:\s*\w+\)|\(\s*Photo:\s*[^)]*\))\s*', '', text)
    # Pattern: "0 Người dân bàng hoàng..." → "Người dân bàng hoàng..."
    text = re.sub(r'^\d+\s+(?=[A-ZÀ-Ỹ])', '', text)
    # Pattern: "1 of 4 | ..." or "(AP Photo/John) ..."
    text = re.sub(r'^\d+\s+of\s+\d+\s*\|?\s*', '', text)
    # Pattern: "Latest videos AP Top Stories June 2 ..." (AP junk header)
    text = re.sub(r'^(?:Latest videos\s+)?AP\s+Top\s+Stories\s+[A-Z][a-z]+\s+\d+\s*', '', text)
    # Pattern: ", [monthFull] [day], [year] STARKE, Fla. (AP) — " (AP dateline junk)
    text = re.sub(r'^,\s*[A-Z][a-z]+\s+\d+,\s*\d{4}\s+[A-Z][A-Z\s,]+\(AP\)\s*—\s*', '', text)
    # Pattern: "Starke, Fla. (AP) — " (AP dateline without date)
    text = re.sub(r'^[A-Z][a-z]+,\s*[A-Z]{2}\s+\(AP\)\s*—\s*', '', text)

    # Step 0b: Remove (Ảnh: AP) / (Photo: ...) / (AP Photo/...) anywhere in text
    text = re.sub(r'\s*\(Ảnh:\s*\w+\)\s*', ' ', text)
    text = re.sub(r'\s+ẢNH:\s*[A-ZÀ-ỸA-Za-z0-9/ ._-]{2,40}\s+', ' ', text)
    text = re.sub(r'\s*\(\s*(?:Photo|Image|AP\s+Photo|AFP|Reuters|Getty)[^)]*\)\s*', ' ', text)

    # Step 0c: Remove title duplication at start of content
    # "Tin tức thế giới 2-6: Ông Trump... Tin tức thế giới 2-6: Ông Trump..."
    title_dup_pattern = r'^(.{5,80}?[:\s])\s*\1'
    text = re.sub(title_dup_pattern, r'\1', text, flags=re.IGNORECASE)

    # Step 0c: Replace bullet/separator artifacts with proper sentence breaks
    # Sites often use · or • or | to separate snippets → convert to periods
    text = re.sub(r'\s*[·•|]\s*', '. ', text)

    # Step 1: Remove standalone UI element text (not full sentences)
    # Catches: "Chia sẻ", "Theo dõi", "0 Trở lại chủ đề", "Bình luận" etc.
    text = re.sub(
        r'(?i)\b(chia sẻ|theo dõi|trở lại chủ đề|bình luận|chuyên mục|xem thêm|xem nhanh|đọc nhanh|tóm tắt nhanh|in bài|gửi email|cỡ chữ|thời gian đọc|đăng ký|like|follow)\b',
        ' ',
        text,
    )
    # Remove standalone digits followed by UI text: "0 Trở lại", "5 Bình luận"
    text = re.sub(r'(?i)\d+\s+(trở lại|bình luận|chia sẻ|comment|share)\b', ' ', text)
    # Remove section labels with count: "0 Thời sự", "5 Bình luận", "0 Phát triển"
    text = re.sub(r'(?i)\b\d+\s+(thời sự|kinh tế|thể thao|giải trí|thế giới|pháp luật|sức khỏe|giáo dục|công nghệ|du lịch|xe|đời sống|phát triển)\b', ' ', text)
    # Remove discussion prompts: "Cùng luận bàn 0"
    text = re.sub(r'(?i)(cùng\s+luận\s+bàn|thảo\s+luận|bàn\s+luận)\s+\d+\b', ' ', text)
    # Remove author role labels
    text = re.sub(r'(?i)\b(Nhà\s+báo|PV|Phóng\s+viên|Nhà\s+ngoại\s+giao|Chuyên\s+gia|Tiến\s+sỹ|Giáo\s+sư|Thứ\s+trưởng|Bộ\s+trưởng)\b', ' ', text)
    text = re.sub(r'(?i)và\s+\d+\s+tác\s+giả\s+khác\b', ' ', text)
    # Remove source labels
    text = re.sub(r'(?i)(theo\s+ghi\s+nhận\s+của|theo\s+báo\s+cáo|theo\s+đánh\s+giá|theo\s+thống\s+kế)\s+', ' ', text)
    # Remove ALL CAPS author names (2+ words): "NGỌC AN", "BẢO NGỌC", "NGUYỄN VĂN A"
    # Use explicit uppercase class to avoid matching lowercase with diacritics (e.g. ừ is in À-Ỹ range)
    _UPPER = r'A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ'
    text = re.sub(rf'\b[{_UPPER}]{{2,}}\s+[{_UPPER}][{_UPPER} ]*(?:\s+[{_UPPER}][{_UPPER} ]*)*\b', ' ', text)
    # Remove author byline: "Dũng Huỳnh 26/05/2026, 18:43" or "Nguyên Nga - Quang Thuần 01/06/2026 05:59"
    # MUST run BEFORE timestamp removal, or the date+time anchor disappears
    # Pattern: dash-separated authors (multi-word names) followed by date+time
    text = re.sub(r'[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+\s*(?:[-–—]\s*[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)+)*\s+\d{1,2}[/\.]\d{1,2}[/\.]\d{4}\s*[,\.]?\s*\d{1,2}:\d{2}', ' ', text)
    # Pattern: single author (2 words) followed by date+time
    text = re.sub(r'[A-ZÀ-Ỹ][a-zà-ỹ]+\s+[A-ZÀ-Ỹ][a-zà-ỹ]+\s+\d{1,2}[/\.]\d{1,2}[/\.]\d{4}\s*[,\.]?\s*\d{1,2}:\d{2}', ' ', text)
    # Remove timestamps: "01/06/2026 09:07 GMT+7", "26/05/2026, 18:43"
    # FIX #3: Also remove parentheses-wrapped timestamps to avoid "( :00)" artifacts
    text = re.sub(r'\(\s*\d{1,2}[/\.]\d{1,2}(?:[/\.]\d{2,4})?\s*[,\.]?\s*\d{1,2}:\d{2}(?:\s*GMT\+?\d+)?\s*\)', ' ', text)
    # FIX #3b: Also handle "( :00)" artifact — broken timestamp parse leaves "( :XX)"
    text = re.sub(r'\(\s*:\d{2}\)', ' ', text)
    text = re.sub(r'\(\s*:00\s*\)', ' ', text)
    text = re.sub(r'\d{1,2}[/\.]\d{1,2}(?:[/\.]\d{2,4})?\s*[,\.]?\s*\d{1,2}:\d{2}(?:\s*GMT\+?\d+)?', ' ', text)
    # Remove leftover "GMT+7" / "GMT+8" if timestamp was partially matched
    text = re.sub(r'\bGMT\+\d+\b', ' ', text)
    # Remove media company names that appear as artifacts: "NKKTech Global Media April 17, 2026"
    text = re.sub(r'(?i)\b(?:NKKTech|Global Media|Associated Press|Agence France-Presse|Reuters|Bloomberg|TechCrunch|The Verge|CNN|BBC|VnExpress|Tuổi Trẻ|Dân Trí|VietnamNet|Thanh Niên)\s+(?:April|May|June|July|August|September|October|November|December|January|February|March|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|tháng)\s+\d{1,2},?\s+\d{4}', ' ', text)
    # Remove newspaper names used as source labels at END of content
    # Matches: "...Thanh Niên, đến ngày 31.5." or "...Báo Mới, ngày 1/6"
    text = re.sub(r'[\s,]+(?:Thanh\s*Niên|Tuổi\s*Trẻ|VnExpress|Dân\s*Trí|Lao\s*Động|Tiền\s*Phong)\s*(?:,\s*(?:đến\s+)?ngày\s+\d{1,2}[/\.]?\d{0,2}\.?)?\s*$', ' ', text, flags=re.IGNORECASE)
    # Remove markdown artifacts: "*Việt Nam", "-19 Boys'"
    text = re.sub(r'[*\-]\s*([A-ZÀ-Ỹ])', r'\1', text)
    # Fix markdown-damaged alphanumeric tokens: "-19" → "U19", "-2" → "U2"
    text = re.sub(r'(?i)(?:^|(?<=\s))[-–—]\s*(\d{1,2}\s*(?:boy|girl|team|u\d|giải))', r'U\1', text)
    # Remove truncated sentence fragments at end (text ending mid-word with short lowercase word + period)
    # Only remove truly truncated patterns like "chuyển.", "tổ ch." — NOT codes like "E10."
    text = re.sub(r'\b[a-zà-ỹ]{1,3}\.\s*$', ' ', text)
    # Remove navigation-menu-like category sequences at start: "Kinh tế Kinh tế xanh..."
    category_kw = r'(?:kinh tế|thể thao|giải trí|thế giới|trong nước|quốc tế|pháp luật|sức khỏe|giáo dục|du lịch|khoa học|công nghệ|xe|bất động sản|văn hóa|đời sống|an ninh|quân sự|chính trị|doanh nghiệp|ngân hàng|chứng khoán|xã hội|lao động|chính sách|phát triển|môi trường|biển đảo|kinh doanh|tiêu dùng|bóng đá|bóng rổ|quần vợt|cầu lông|bơi lội|điền kinh|võ thuật|tennis)'
    text = re.sub(rf"^\s*(?:{category_kw})(?:\s+(?:xanh|số|24h|online|mới|nóng|-\s*\w+))?(?:\s+(?:,?\s*|-?\s*)?{category_kw})+\s*", " ", text, flags=re.IGNORECASE)
    # Remove category nav + timestamp: "Kinh doanh Tiêu dùng Thứ hai, 01/06/2026 - 09:00 :"
    text = re.sub(r'(?i)^(?:kinh doanh|tiêu dùng|thời sự|trong nước|quốc tế|thể thao|giải trí|sức khỏe|đời sống|giáo dục|văn hóa|pháp luật)(?:\s+(?:kinh doanh|tiêu dùng|thời sự|trong nước|quốc tế|thể thao|giải trí|sức khỏe|đời sống|giáo dục|văn hóa|pháp luật))*\s+(?:thứ\s+(?:hai|ba|bốn|năm|sáu|bảy|cn)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*\d{1,2}[/\.]\d{1,2}[/\.]\d{4}\s*[-–—:]+\s*\d{1,2}:\d{2}\s*:?\s*', ' ', text)
    # Standalone day+timestamp: "Thứ hai, 01/06/2026 - 09:00 :"
    text = re.sub(r'(?i)^(?:thứ\s+(?:hai|ba|bốn|năm|sáu|bảy|cn)|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\s*,?\s*\d{1,2}[/\.]\d{1,2}[/\.]\d{4}\s*[-–—:]+\s*\d{1,2}:\d{2}\s*:?\s*', ' ', text)
    # Remove author byline with newspaper: "Minh Huyền (Dân trí) -" or "Tên Tác Giả (Báo) -"
    text = re.sub(r'[A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*\s+\([^)]+\)\s*[-–—:]+\s*', ' ', text)
    # Remove AP/Reuters image caption artifacts: "1 of 4 | ...", "2 of 2 | ...", "(Photo: ...)"
    # Pattern with pipe: "1 of 4 | Some caption text"
    text = re.sub(r'\d+\s+of\s+\d+\s*\|\s*[^\n]*', ' ', text)
    # Pattern without pipe: "1 of 2 Some caption text" (AP sometimes omits the pipe)
    text = re.sub(r'\b\d+\s+of\s+\d+\s+', ' ', text)
    text = re.sub(r'\(\d+\s+of\s+\d+\)', ' ', text)
    # Remove photo credits: "(AP Photo/...)", "(Photo: ...)", "(Image: ...)", "(AFP/...)"
    text = re.sub(r'\((?:AP\s+Photo|Photo|Image|AFP|Reuters|Getty\s+Images?|Handout)[^)]*\)', ' ', text)
    # Remove UI action sequences: "Copy Link copied", "--> -->", "Print Email X Bluesky"
    text = re.sub(r'Copy\s+Link\s+copied', ' ', text)
    text = re.sub(r'-->\s*-->', ' ', text)
    text = re.sub(r'(?:Print|Email|X|Bluesky|Flipboard|Reddit|Share|Save|Bookmark|Send)\s+(?:Print|Email|X|Bluesky|Flipboard|Reddit|Share|Save|Bookmark|Send)?', ' ', text)
    # Remove "Leer en español" / language switch links
    text = re.sub(r'(?i)Leer\s+en\s+español', ' ', text)
    text = re.sub(r'(?i)Read\s+in\s+\w+', ' ', text)
    # Remove AP template artifacts: "By , [monthFull] [day], [year]"
    text = re.sub(r'By\s*,?\s*\[monthFull\]\s*\[day\],\s*\[year\]', ' ', text)
    # Remove "By Updated [hour]:[minute]" / placeholder template artifacts
    text = re.sub(r'By\s+Updated\s+\[hour\]:\[minute\][^,\.]*[,\.]?\s*', ' ', text)
    text = re.sub(r'By\s*,?\s*(?:[A-ZÀ-Ỹ][a-zà-ỹ]+\s+){0,3}\[monthFull\]\s*\[day\],\s*\[year\]', ' ', text)
    # Remove "Updated [hour]:[minute] [AMPM] [timezone]" template artifacts
    text = re.sub(r'Updated\s+\[hour\]:\[minute\]\s+\[AMPM\]\s+\[timezone\]', ' ', text)
    text = re.sub(r'\[(?:hour|minute|AMPM|timezone|monthFull|day|year)\]', ' ', text, flags=re.IGNORECASE)
    # Remove conservative leading byline names: "Việt Dũng - ..." or "Việt Dũng Theo ..."
    text = re.sub(r'^(?:[A-ZÀ-Ỹ][a-zà-ỹ]+\s+){1,3}[A-ZÀ-Ỹ][a-zà-ỹ]+\s*(?:\([^)]+\))?\s*(?:[-–—:]|Theo\b)\s*', ' ', text)

    # Remove markdown artifacts from scraped content: "## Read Next", "**bold**", etc.
    text = re.sub(r'#{1,6}\s+', ' ', text)  # Markdown headers
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic

    # Remove "Read Next" / "Related" / "More" navigation headers
    text = re.sub(r'(?i)\b(Read Next|Related|More on this|You may also like|Sponsored|Advertisement)\b', ' ', text)

    # Remove category labels: "Asia Pacificcategory" → "Asia Pacific"
    # Handle concatenated: "Pacificcategory" → "Pacific" (no space before)
    # Handle spaced: "Sports category" → "Sports" (with space before)
    # Handle mid-word: "BusinesscategoryEuro" → "BusinessEuro"
    # Remove standalone "category" or "categories" with or without leading space
    text = re.sub(r'(?i)\s*categories?\b', ' ', text)
    # Also handle concatenated form where category is stuck to another word
    text = re.sub(r'(?i)categor[a-z]*', '', text)

    # Remove time-ago patterns: "4 giờ trước", "2 days ago", "3 hours ago", "5 mins ago"
    text = re.sub(r'(?i)\d+\s+(gi\u1edd|phút|giờ|ng\u00e0y|tu\u1ea7n|th\u00e1ng|n\u0103m)\s+tr\u01b0\u1edbc\b', ' ', text)
    text = re.sub(r'(?i)\d+\s+(second|minute|hour|day|week|month|year)s?\s+ago\b', ' ', text)

    # Remove leading timestamp patterns: "01/06 19:30" at start of content
    text = re.sub(r'^\d{1,2}[/\.]\d{1,2}\s+\d{1,2}:\d{2}\s+', ' ', text)

    # Strip Vietnamese meta-text sentences (start, end, or anywhere).
    meta_sentence_patterns = [
        r"(?:Bản tin được đăng lúc|Bài viết được đăng lúc|Đăng lúc|Cập nhật lúc)\s+[^.\n]+\s*[,.\n]",
        r"(?:Đăng ngày|Publication date)\s+[^.\n]+\s*[,.\n]",
        r"Đây là nhóm thông tin đáng chú ý[^.\n]+[.\n]",
        r"Nội dung bài viết[^.\n]+[.\n]",
        r"toàn bộ thông tin chính[^.\n]+[.\n]",
        r"(?:theo|via)\s+(?:VietnamNet|Tuổi\s*Trẻ|Dân\s*Trí|VnExpress|Thanh\s*Niên|Lao\s*Động|Tiền\s*Phong|Báo\s*Mới|Báo\s+Đất\s+Việt|Báo\s+Mới)\s*[,.\n]",
        r"Ngoài ra,\s*bản\s+tin\s+[^\n]+đề cập việc\s+",
        r"trong thời gian gần đây\s*,?\s*cư dân mạng[^.\n]+[.\n]",
        r"vẫn nằm trong khung 48 giờ gần nhất\s*[,.\n]",
        r"Mời độc giả theo dõi thêm[^.\n]*[.\n]",
        r"Mời quý độc giả[^.\n]*[.\n]",
        r"Xem thêm[^.\n]*[.\n]",
        r"Read more[^.\n]*[.\n]",
        r"Subscribe to[^.\n]*[.\n]",
        r"Click here to[^.\n]*[.\n]",
        r"For more information[^.\n]*[.\n]",
        r"This article was originally published[^.\n]*[.\n]",
    ]
    for pat in meta_sentence_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Collapse extra whitespace and remove leading byline/source lead-in that may remain after UI/template cleanup.
    text = re.sub(r'\s+', ' ', text).strip()
    text = _strip_news_leadin_noise(text)
    text = re.sub(
        r'^(?:[A-ZÀ-Ỹ][a-zà-ỹ]+\s+){1,3}[A-ZÀ-Ỹ][a-zà-ỹ]+\s+(?=(Ngoại\s+trưởng|Tổng\s+thống|Thủ\s+tướng|Bộ\s+|Quân\s+|Mỹ\s+|Nga\s+|Iran\s+|Ukraine\s+|Israel\s+))',
        ' ',
        text,
    ).strip()

    # Remove interview subheadings/prompts: "Thưa ông,", "Theo đó,", "Chưa ghi nhận tiêu cực từ người dân"
    text = re.sub(r'(?i)\b(thưa\s+ông|thưa\s+bà|theo\s+đó|chưa\s+ghi\s+nhận\s+tiêu\s+cực|nhận\s+định\s+trên)\b\s*,?\s*', ' ', text)
    # Remove interview question headings: "Niềm tin khu vực tư nhân tăng Thưa ông,"
    text = re.sub(r'(?i)\b(niềm tin|thị trường|kết quả|đánh giá|nhận định|phản hồi|báo cáo)\s+(khu vực\s+)?(tư nhân|người dân|doanh nghiệp|chuyên gia|nhà đầu tư)\s+(tăng|giảm|thay đổi|cho biết|phản ánh|lạc quan|bi quan)\s+thưa\s+(ông|bà)\b\s*,?\s*', ' ', text)
    # Clean up resulting double spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # If text became too short after UI cleanup, return original
    if len(text) < 30:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []

    for sentence in sentences:
        # Step 1: Check if entire sentence is boilerplate -> remove (pre-compiled)
        is_boilerplate = False
        for pattern in _SNIPPET_BOILERPLATE_COMPILED:
            if pattern.search(sentence):
                is_boilerplate = True
                break
        if is_boilerplate:
            continue

        # Step 2: Try to strip boilerplate prefix from sentence (pre-compiled)
        cleaned_sentence = sentence
        for pattern, replacement in _SNIPPET_PREFIX_COMPILED:
            new_sentence = pattern.sub(replacement, cleaned_sentence).strip()
            # Only apply if it actually shortened the sentence and left meaningful content
            if new_sentence and len(new_sentence) < len(cleaned_sentence) and len(new_sentence) >= 10:
                cleaned_sentence = new_sentence
                break  # Apply only first matching prefix pattern

        # Step 3: Try to strip boilerplate suffix from sentence (pre-compiled)
        for pattern, replacement in _SNIPPET_SUFFIX_COMPILED:
            new_sentence = pattern.sub(replacement, cleaned_sentence).strip()
            # Only apply if it actually shortened the sentence and left meaningful content
            if new_sentence and len(new_sentence) < len(cleaned_sentence) and len(new_sentence) >= 10:
                cleaned_sentence = new_sentence
                break  # Apply only first matching suffix pattern

        kept.append(cleaned_sentence)

    if not kept:
        # If everything was boilerplate, return original (better than empty)
        return text

    result = " ".join(kept).strip()

    # Safety: if we lost too much content, return original
    # But be lenient for short snippets: keep if result >= 20 chars
    if len(result) < len(text) * 0.2 and len(result) < 20:
        return text

    return result


def _decode_ddg_redirect_url(href: str) -> str:
    """Decode DuckDuckGo redirect URL to actual target URL."""
    if not href:
        return ""
    parsed = urlparse(href)
    if "duckduckgo.com" not in parsed.netloc:
        return href

    query = parse_qs(parsed.query)
    uddg = query.get("uddg", [])
    if uddg:
        return unquote(uddg[0])
    return href


def _normalize_search_query(query: str) -> str:
    """Normalize query text while preserving Vietnamese accents."""
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    normalized = re.sub(r"[​‌‍﻿]", "", normalized)
    return normalized.strip(" \t\n\r,;:")


def _contains_phrase_or_token(text: str, keyword: str) -> bool:
    """Match phrase normally; short alnum tokens require word boundaries."""
    hay = (text or "").lower()
    needle = (keyword or "").lower().strip()
    if not hay or not needle:
        return False
    if " " in needle:
        return needle in hay

    alnum_len = len(re.sub(r"[^a-z0-9]", "", needle))
    if alnum_len <= 3:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", hay))
    return needle in hay


def _query_has_any_keyword(query: str, keywords: list[str]) -> bool:
    return any(_contains_phrase_or_token(query, kw) for kw in keywords)


def _query_has_realtime_hint(query: str) -> bool:
    lower = query.lower()
    hints = ["hôm nay", "mới nhất", "latest", "breaking", "vừa", "mới ra", "24h", "24 giờ", "recent", "hiện tại", "current", "now"]
    return any(hint in lower for hint in hints)


def _build_time_filter_intent(query: str, is_news: bool) -> Dict[str, str]:
    """Build internal time-filter intent before sending to providers."""
    lower = (query or "").lower()
    if "7 ngày" in lower or "tuần này" in lower or "past week" in lower:
        return {"sort_by": "date", "time_range": "past_7d"}
    if is_news or _query_has_realtime_hint(query):
        return {"sort_by": "date", "time_range": "past_24h"}
    return {"sort_by": "relevance", "time_range": "past_48h"}


# ══════════════════════════════════════════════════════════════════════════════
# SEARCH KEYWORD DETECTION
# ══════════════════════════════════════════════════════════════════════════════

_PRICE_KEYWORDS = [
    "giá xăng", "giá vàng", "giá dầu", "giá gas", "giá điện",
    "giá bao nhiêu", "bao nhiêu tiền", "giá cả",
]

_STOCK_KEYWORDS = [
    "cổ phiếu", "chứng khoán", "mã chứng khoán", "thị trường chứng khoán",
    "vn-index", "vnindex", "vn30", "hnx", "hose", "upcom",
    "giá cổ phiếu", "stock", "cổ phần",
]


def _is_price_query(query: str) -> bool:
    """Check if query is about prices."""
    return _query_has_any_keyword(query, _PRICE_KEYWORDS)


def _is_stock_query(query: str) -> bool:
    """Check if query is about stocks/securities."""
    return _query_has_any_keyword(query, _STOCK_KEYWORDS)


def extract_stock_ticker(query: str) -> Optional[str]:
    """Extract an explicit stock symbol from user text when one is present."""
    text = (query or "").strip()
    for pattern in (
        r"\(([A-Z]{1,8}(?:\.[A-Z]{1,4})?)\)",
        r"\b(?:ma|mã|ticker|symbol)\s+([A-Z]{1,8}(?:\.[A-Z]{1,4})?)\b",
        r"\b([A-Z]{2,5}(?:\.[A-Z]{1,4})?)\b",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# NEWS DOMAIN LISTS (for Tavily include_domains filtering)
# ══════════════════════════════════════════════════════════════════════════════

_VN_NEWS_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "nhandan.vn", DOMAIN_VOV, DOMAIN_VTV,
    DOMAIN_LAODONG, "tienphong.vn", "plo.vn", "nld.com.vn",
    DOMAIN_BAODAUTU, DOMAIN_CAFEF, DOMAIN_VIETSTOCK, DOMAIN_VNECONOMY,
]

_VN_TECH_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "genk.vn", "tinhte.vn", DOMAIN_VOV, DOMAIN_VTV,
]

_GAME_OFFICIAL_DOMAINS = [
    "hoyolab.com", "hoyoverse.com", "mihoyo.com", "gryphline.com", "hypergryph.com",
    "riotgames.com", "playvalorant.com", "leagueoflegends.com",
    "playstation.com", "xbox.com", "nintendo.com", "epicgames.com",
    "arknights.global", "yostar.co.jp", "hypergryph.com", "steampowered.com",
]

_GAME_REPUTABLE_DOMAINS = [
    "gamek.vn", "genk.vn", "ign.com", "gamespot.com", "pcgamer.com", "polygon.com",
    "prydwen.gg", "arknights.wiki.gg", "wiki.gg", "game8.co",
    "honkai-star-rail.fandom.com", "genshin-impact.fandom.com", "arknights.fandom.com", "fandom.com",
]

_GAME_SOCIAL_DOMAINS = [
    "x.com", "twitter.com", "reddit.com",
]

_VN_GAME_DOMAINS = [
    *_GAME_OFFICIAL_DOMAINS,
    *_GAME_REPUTABLE_DOMAINS,
    *_GAME_SOCIAL_DOMAINS,
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, DOMAIN_VOV, DOMAIN_VTV,
]

_VN_SPORT_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "bongda.com.vn", "bongdaplus.vn", "thethao247.vn",
]

_VN_FINANCE_DOMAINS = [
    DOMAIN_CAFEF, DOMAIN_VIETSTOCK, DOMAIN_VNECONOMY, DOMAIN_BAODAUTU,
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
]

_VN_ENTERTAINMENT_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, DOMAIN_VTV, DOMAIN_VOV, DOMAIN_LAODONG,
]

# NEW: Additional category domains for diverse topics
_VN_LAW_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "plo.vn", "tienphong.vn", "nld.com.vn",
    DOMAIN_LAODONG, DOMAIN_VTV,
]

_VN_MILITARY_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "quocphongvietnam.vn", "biendong.net",
    DOMAIN_VOV, DOMAIN_VTV, "nhandan.vn",
]

_VN_SCIENCE_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "khoahoc.tv", "vatlyvietnam.org",
    DOMAIN_VOV, DOMAIN_VTV, "tiasang.com.vn",
]

_VN_EDUCATION_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "giaoduc.net.vn", "giaoducthoidai.vn",
    DOMAIN_VOV, DOMAIN_VTV, "nhandan.vn",
]

_VN_HEALTH_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "suckhoedoisong.vn", "thaythuocvietnam.vn",
    DOMAIN_VOV, DOMAIN_VTV, "nld.com.vn",
]

_VN_TRAVEL_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "vietnam.travel", "dulichvietnam.com.vn",
    DOMAIN_VOV, DOMAIN_VTV, "kenh14.vn",
]

_VN_CULTURE_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, DOMAIN_VTV, DOMAIN_VOV, "vanhoanghean.vn",
    "giaoduc.net.vn", "nhandan.vn",
]

_VN_LIFESTYLE_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "kenh14.vn", "elle.vn", "dep.com.vn",
    DOMAIN_VOV, DOMAIN_VTV, "nld.com.vn",
]

_VN_YOUTH_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    "kenh14.vn", "gamek.vn", "zingnews.vn", "vtv.vn",
    "dantri.com.vn", "tuoitre.vn",
]

_INTL_NEWS_DOMAINS = [
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, DOMAIN_VOV, DOMAIN_VTV,
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "bloomberg.com", "ft.com", "wsj.com",
    "france24.com", "dw.com", "aljazeera.com",
]

_VN_PRIMARY_NEWS_DOMAINS = {
    DOMAIN_VNEXPRESS, DOMAIN_TUOITRE, DOMAIN_THANHNIEN, DOMAIN_DANTRI,
    DOMAIN_VIETNAMNET, "nhandan.vn", DOMAIN_VOV, DOMAIN_VTV,
    DOMAIN_LAODONG, "tienphong.vn", "plo.vn", "nld.com.vn",
    DOMAIN_BAODAUTU, DOMAIN_CAFEF, DOMAIN_VIETSTOCK, DOMAIN_VNECONOMY,
}

_INTL_PRIMARY_NEWS_DOMAINS = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "bloomberg.com", "ft.com", "wsj.com",
    "france24.com", "dw.com", "aljazeera.com",
}


def _get_domains_for_category(category: str) -> list[str]:
    """Return appropriate domain list for news category."""
    domain_map = {
        "international": _INTL_NEWS_DOMAINS,
        "tech":          _VN_TECH_DOMAINS,
        "game":          _VN_GAME_DOMAINS,
        "sport":         _VN_SPORT_DOMAINS,
        "finance":       _VN_FINANCE_DOMAINS,
        "entertainment": _VN_ENTERTAINMENT_DOMAINS,
        "domestic":      _VN_NEWS_DOMAINS,
        # Specialized categories with clear boundaries
        "science":       _VN_SCIENCE_DOMAINS,
        "education":     _VN_EDUCATION_DOMAINS,
        "health":        _VN_HEALTH_DOMAINS,
        "travel":        _VN_TRAVEL_DOMAINS,
        "culture":       _VN_CULTURE_DOMAINS,
        "lifestyle":     _VN_LIFESTYLE_DOMAINS,
        "youth":         _VN_YOUTH_DOMAINS,
        # Note: law and military are merged into domestic (see _detect_news_category)
    }
    return domain_map.get(category, _VN_NEWS_DOMAINS)


def _normalize_domain(url_or_domain: str) -> str:
    """Normalize URL/domain to bare lowercase netloc without www."""
    value = (url_or_domain or "").strip().lower()
    if not value:
        return ""
    if "://" in value:
        try:
            value = urlparse(value).netloc.lower()
        except Exception:
            pass
    return value.replace("www.", "")


def _is_vn_news_domain(domain: str) -> bool:
    normalized = _normalize_domain(domain)
    return normalized.endswith(".vn") or normalized in _VN_PRIMARY_NEWS_DOMAINS


def _is_intl_news_domain(domain: str) -> bool:
    normalized = _normalize_domain(domain)
    return normalized in _INTL_PRIMARY_NEWS_DOMAINS


def _news_source_score(domain: str, category: str) -> int:
    """Higher score means more preferred source for the category."""
    normalized = _normalize_domain(domain)

    if category == "game":
        if normalized in _GAME_OFFICIAL_DOMAINS:
            return 80
        # VN reputable game sources — check BEFORE generic reputable
        if normalized.endswith(".vn") and normalized in _GAME_REPUTABLE_DOMAINS:
            return 70
        if normalized in _GAME_REPUTABLE_DOMAINS:
            return 65
        if normalized in _GAME_SOCIAL_DOMAINS:
            return 45
        if normalized in _VN_PRIMARY_NEWS_DOMAINS:
            return 35
        if normalized.endswith(".vn"):
            return 20
        return 0

    if category == "international":
        if normalized in _INTL_PRIMARY_NEWS_DOMAINS:
            return 45
        if normalized in _VN_PRIMARY_NEWS_DOMAINS:
            return 30
        if normalized.endswith(".vn"):
            return 20
        return 0

    # Non-intl, non-game categories: prioritize VN sources
    if normalized in _VN_PRIMARY_NEWS_DOMAINS:
        return 40
    cat_domains = _get_domains_for_category(category)
    if normalized in cat_domains:
        return 40  # Category-specific VN domain
    if normalized.endswith(".vn"):
        return 25
    if normalized in _INTL_PRIMARY_NEWS_DOMAINS:
        return 10  # Low priority for non-intl categories; VN sources preferred
    return 0


def _max_international_sources(category: str, max_items: int) -> int:
    if category == "international":
        return max(1, max_items // 2)
    return max(1, max_items // 3)


def _try_select_news_item(
    item: Dict[str, Any],
    selected: List[Dict[str, Any]],
    source_counts: Dict[str, int],
    category: str,
    max_per_source: int,
    max_intl: int,
    intl_count: int,
) -> tuple[bool, int]:
    domain = _normalize_domain(item.get("url", ""))
    if not domain:
        return False, intl_count
    if source_counts.get(domain, 0) >= max_per_source:
        return False, intl_count

    is_intl = _is_intl_news_domain(domain)
    if is_intl and intl_count >= max_intl:
        return False, intl_count
    is_game_source = category == "game" and domain in {*_GAME_OFFICIAL_DOMAINS, *_GAME_REPUTABLE_DOMAINS, *_GAME_SOCIAL_DOMAINS}
    if category != "international" and not (_is_vn_news_domain(domain) or is_intl or is_game_source):
        return False, intl_count

    selected.append(item)
    source_counts[domain] = source_counts.get(domain, 0) + 1
    if is_intl:
        intl_count += 1
    return True, intl_count


def _limit_news_source_repetition(
    results: List[Dict[str, Any]],
    max_items: int,
    category: str,
    max_per_source: int = 2,
) -> List[Dict[str, Any]]:
    """Prefer reputable VN sources, allow only limited source repetition."""
    if not results:
        return []

    sorted_items = sorted(
        results,
        key=lambda item: _news_source_score(item.get("url", ""), category),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    max_intl = _max_international_sources(category, max_items)
    intl_count = 0

    for item in sorted_items:
        if len(selected) >= max_items:
            break
        _, intl_count = _try_select_news_item(
            item,
            selected,
            source_counts,
            category,
            max_per_source,
            max_intl,
            intl_count,
        )

    if len(selected) >= max_items:
        return selected[:max_items]

    for item in sorted_items:
        if len(selected) >= max_items:
            break
        if item in selected:
            continue

        domain = _normalize_domain(item.get("url", ""))
        if not domain or source_counts.get(domain, 0) >= max_per_source:
            continue

        selected.append(item)
        source_counts[domain] = source_counts.get(domain, 0) + 1

    return selected[:max_items]


def _detect_news_category(query: str) -> str:
    """Detect news category from query for domain filtering.

    Categories merged based on real Vietnamese news organization:
    - Law (Pháp luật) + Military (Quân sự) → merged into domestic
      (because law/military news in VN is usually Politics/Current Affairs)
    - Other categories kept separate (clear boundaries)
    """
    q = query.lower()

    # International (check first — world news, diplomacy, military world)
    intl_keywords = [
        "thế giới", "quốc tế", "world", "international", "global",
        "mỹ", "trung quốc", "nga", "ukraine", "israel", "gaza",
        "châu âu", "châu á", "nato", "trump", "biden", "putin",
        "ngoại giao", "liên hợp quốc", "lhq",
    ]
    if _query_has_any_keyword(query, intl_keywords):
        return "international"

    # Game
    game_keywords = [
        "honkai", "genshin", "wuthering", "wuwa", "zenless", "zzz",
        "arknights", "endfield", "enfield", "fgo",
        "mobile legends", "liên quân", "valorant", "game", "gaming", "esports", "esport", "trò chơi",
        "patch", "update game", "banner", "gacha", "reroll", "tier list",
        "thể thao điện tử",
    ]
    if _query_has_any_keyword(query, game_keywords):
        return "game"

    # Health (check before science — health queries often contain "research", "study")
    health_keywords = [
        "sức khỏe", "y tế", "bệnh", "thuốc", "vaccine", "dịch bệnh",
        "bệnh viện", "phòng bệnh", "chăm sóc sức khỏe", "bác sĩ",
        "ung thư", "tim mạch", "tiểu đường", "covid", "sốt xuất huyết",
        "tay chân miệng", "dinh dưỡng", "thực phẩm chức năng",
        # English keywords
        "health", "medical", "hospital", "doctor", "patient", "treatment",
        "pandemic", "outbreak", "medicine", "pharma", "drug", "therapy",
    ]
    if _query_has_any_keyword(query, health_keywords):
        return "health"

    # Science
    science_keywords = [
        "khoa học", "khám phá", "nghiên cứu", "phát minh", "vũ trụ",
        "hành tinh", "mặt trăng", "sao hỏa", "nasa", "môi trường",
        "biến đổi khí hậu", "năng lượng tái tạo", "khí hậu",
        "sinh học", "vật lý", "hóa học", "thiên văn",
        # English keywords
        "science", "research", "discovery", "breakthrough", "planet", "moon",
        "mars", "space", "astronomy", "climate change", "renewable energy",
        "physics", "chemistry", "biology", "invention", "laboratory",
    ]
    if _query_has_any_keyword(query, science_keywords):
        return "science"

    # Sport (check BEFORE education — "thể thao học sinh" should be sport, not education)
    sport_keywords = ["thể thao", "bóng đá", "bóng rổ", "tennis", "sport", "football", "giải đấu", "trận đấu", "bóng chuyền", "bơi lội"]
    if _query_has_any_keyword(query, sport_keywords):
        return "sport"

    # Sport (check BEFORE education — "thể thao học sinh" should be sport, not education)
    sport_keywords = ["thể thao", "bóng đá", "bóng rổ", "tennis", "sport", "football", "giải đấu", "trận đấu", "bóng chuyền", "bơi lội"]
    if _query_has_any_keyword(query, sport_keywords):
        return "sport"

    # Education
    education_keywords = [
        "giáo dục", "tuyển sinh", "thi cử", "đại học", "cao đẳng",
        "du học", "học phí", "học bổng", "trường học", "giảng viên",
        "sinh viên", "học sinh", "kỳ thi", "tốt nghiệp", "lớp 10",
        "thpt quốc gia", "đánh giá năng lực", "điểm chuẩn",
    ]
    if _query_has_any_keyword(query, education_keywords):
        return "education"

    # Travel
    travel_keywords = [
        "du lịch", "khách sạn", "vé máy bay", "điểm đến", "lễ hội du lịch",
        "du lịch việt nam", "đà lạt", "nha trang", "hạ long", "phú quốc",
        "sapa", "huế", "hội an", "đà nẵng", "tour du lịch",
        # English keywords
        "travel", "tourism", "hotel", "flight", "destination", "resort",
        "vacation", "trip", "backpack", "holiday", "visit vietnam",
    ]
    if _query_has_any_keyword(query, travel_keywords):
        return "travel"

    # Culture
    culture_keywords = [
        "văn hóa", "nghệ thuật", "di sản", "lễ hội", "truyền thống",
        "sách", "văn học", "sân khấu", "múa", "hát", "tuồng", "chèo",
        "ca trù", "quan họ", "di tích", "bảo tàng", "văn hóa việt",
        # English keywords
        "cultural", "heritage", "festival", "traditional", "museum",
        "literature", "performing arts", "folk song", "unesco",
    ]
    if _query_has_any_keyword(query, culture_keywords):
        return "culture"

    # Lifestyle
    lifestyle_keywords = [
        "đời sống", "phong cách sống", "gia đình", "nấu ăn", "món ngon",
        "nhà cửa", "tình yêu", "hôn nhân", "mẹo vặt", "cuộc sống",
        "ẩm thực", "thời trang", "làm đẹp", "decor", "nội thất",
    ]
    if _query_has_any_keyword(query, lifestyle_keywords):
        return "lifestyle"

    # Youth
    youth_keywords = [
        "nhịp sống trẻ", "giới trẻ", "xu hướng", "hot trend", "gen z",
        "mạng xã hội", "tiktok", "facebook", "youtube", "streamer",
        "idol", "fandom", "kpop", "vpop", "teen",
    ]
    if _query_has_any_keyword(query, youth_keywords):
        return "youth"

    # Tech
    tech_keywords = ["công nghệ", "tech", "technology", "ai", "smartphone", "laptop", "điện thoại", "sản phẩm mới", "xe điện", "xe hơi"]
    if _query_has_any_keyword(query, tech_keywords):
        return "tech"

    # Finance
    finance_keywords = ["kinh tế", "tài chính", "chứng khoán", "ngân hàng", "finance", "stock", "gdp", "doanh nghiệp", "khởi nghiệp", "bất động sản"]
    if _query_has_any_keyword(query, finance_keywords):
        return "finance"

    # Entertainment
    entertainment_keywords = ["giải trí", "showbiz", "phim", "âm nhạc", "ca sĩ", "movie", "music", "mv", "sao việt", "nghệ sĩ"]
    if _query_has_any_keyword(query, entertainment_keywords):
        return "entertainment"

    # Default: domestic
    # Includes: Politics (Chính trị), Law (Pháp luật), Military (Quân sự), Society (Xã hội), Current Affairs (Thời sự)
    # Because in VN media, law/military/politics news all live under "Thời sự/Chính trị"
    return "domestic"


# ══════════════════════════════════════════════════════════════════════════════
# TAVILY SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _tavily_search_depth(is_price: bool, is_stock: bool, is_news: bool, query: str = "") -> str:
    if is_price or is_stock or is_news:
        return "advanced"
    # Game queries (banner, character, gacha) cần advanced depth
    game_hints = ["arknights", "endfield", "enfield", "honkai", "genshin", "wuthering", "banner", "gacha"]
    if any(hint in query.lower() for hint in game_hints):
        return "advanced"
    return "basic"


def _tavily_result_limit(is_news: bool, is_price: bool, is_stock: bool, max_results: int) -> int:
    base = max(max_results + 10, _TAVILY_BROAD_MIN_RESULTS)
    if is_news:
        base = max(base, 25)
    if is_price or is_stock:
        base = max(base, 20)
    return min(base, _TAVILY_BROAD_MAX_RESULTS)


def _build_tavily_search_query(query: str, is_news: bool) -> str:
    lower = query.lower()
    if "banner" in lower and any(game in lower for game in ["honkai", "genshin", "wuthering", "arknights", "endfield", "enfield"]):
        return query

    needs_date = is_news or _query_has_realtime_hint(query) or "tin tức" in lower
    if not needs_date:
        return query

    date_str = datetime.now(_VN_TZ).strftime('%d/%m/%Y')
    refined = query if date_str in query else f"{query} ngày {date_str}"
    if "mới nhất" not in lower and "latest" not in lower:
        refined = f"{refined} mới nhất"
    return refined


def _build_tavily_request(
    query: str,
    is_price: bool,
    is_stock: bool = False,
    max_results: int = 15,
    is_news: bool = False,
    freshness_hours: int | None = None,
) -> dict:
    """Build Tavily API request payload."""
    news_category = _detect_news_category(query) if is_news else ""
    time_intent = _build_time_filter_intent(query, is_news)
    payload: dict[str, Any] = {
        "api_key": settings.tavily_api_key,
        "query": _build_tavily_search_query(query, is_news),
        "exclude_domains": ["youtube.com", "tiktok.com", "facebook.com", "instagram.com", "reddit.com"],
        "search_depth": _tavily_search_depth(is_price, is_stock, is_news, query),
        "include_answer": is_price or is_stock,
        "max_results": _tavily_result_limit(is_news, is_price, is_stock, max_results),
        "include_raw_content": False,
    }

    focused_domains = _focused_game_domains(query)
    if focused_domains and not is_news:
        payload["include_domains"] = focused_domains + [
            d for d in [*_GAME_REPUTABLE_DOMAINS, *_GAME_SOCIAL_DOMAINS]
            if d not in focused_domains
        ]

    if is_news:
        payload["topic"] = "news"
        time_range = time_intent.get("time_range")
        if freshness_hours is not None:
            payload["days"] = max(1, min(7, (int(freshness_hours) + 23) // 24))
        else:
            payload["days"] = 7 if time_range == "past_7d" else (1 if time_range == "past_24h" else 2)
        payload["include_answer"] = False
        payload["include_raw_content"] = False

        include_domains = _get_domains_for_category(news_category)
        if news_category == "game":
            focused_domains = _focused_game_domains(query)
            if focused_domains:
                include_domains = focused_domains + [
                    d for d in [*_GAME_REPUTABLE_DOMAINS, *_GAME_SOCIAL_DOMAINS]
                    if d not in focused_domains
                ]
        payload["include_domains"] = include_domains

        date_str = datetime.now(_VN_TZ).strftime("%d/%m/%Y")
        if date_str not in payload["query"]:
            payload["query"] = f"{payload['query']} ngày {date_str}"
        if time_intent.get("sort_by") == "date" and "mới nhất" not in payload["query"].lower():
            payload["query"] = f"{payload['query']} mới nhất"

    return payload


_JUNK_NEWS_PATTERNS = [
    "trang tin cập nhật", "kênh cung cấp", "trang tin tức", "cập nhật nhanh chóng",
    "theo dõi tin tức hàng giờ", "nội dung bao gồm", "đọc báo online", "tin mới nhất 24h",
    "cập nhật liên tục", "các trang báo lớn", "người dùng dễ dàng", "mời bạn đọc",
    "báo điện tử", "thông tin đầy đủ và hấp dẫn", "chi tiết các tin tức", "nắm bắt xu hướng",
    "xem thêm tại", "click để xem", "nhấn vào đây", "báo tuổi trẻ - tin tức mới nhất",
    "tin nhanh, tin nóng", "tin tức và dữ liệu kinh tế", "tin tức mới nhất trên các lĩnh vực",
    "đăng nhập xin chào", "đăng xuất", "facebook youtube tiktok", "tin tức move",
    "download the app", "newsletters subscribe", "information you can trust",
    "our standards", "subscriber agreement", "non-commercial use",
    "podcast youtube cần biết rao vặt", "cài đặt tài khoản tin đã lưu", "tất cả chuyên mục",
    "tin đã lưu bình luận của bạn", "lịch sử giao dịch", "vào tuổi trẻ sao",
]

_NON_NEWS_PAGE_PATTERNS = [
    "join the club", "sign in now", "link copied to clipboard", "now playing",
    "official launch trailer", "overview", "playlists", "user reviews", "videos - ign",
    "trailers", "images & screenshots", "learn more",
]

_VAGUE_TITLE_PATTERNS = [
    "điểm tin", "điểm báo", "điểm loạt", "toàn cảnh", "tổng hợp",
    "vấn đề nóng", "nóng trong ngày", "bản tin", "nhìn lại", "cập nhật nhanh", "đọc nhanh",
]

_GENERIC_PAGE_TITLE_PATTERNS = [
    "trang chủ", "tin mới nhất", "tin nhanh 24h", "tin kinh tế", "nhịp sống kinh tế",
    "tuổi trẻ online", "gamek", "genk", "pc gamer", "vnexpress", "vneconomy",
    "thông tin mới nhất", "kênh tin game", "tin tức game", "mobile apps",
]

_NEWS_SIGNAL_KEYWORDS = ["chính phủ", "quốc hội", "giao thông", "điện", "xăng", "lãi suất", "gdp", "ra mắt", "phát hành", "cập nhật", "sự kiện", "nghị định", "quyết định", "luật", "ban hành", "phê duyệt", "ký", "thắng", "đánh bại", "tuyên bố", "công bố", "khởi tố", "bắt giữ", "thượng đỉnh", "nasa", "war", "phát hiện", "hành tinh", "vũ trụ", "chấn động", "nổ súng", "tai nạn", "hỏa hoạn", "lũ lụt", "bão", "động đất"]


def _has_specific_title_signal(title_lower: str) -> bool:
    has_number = bool(re.search(r"\b\d+([\.,]\d+)?\b", title_lower))
    has_keyword = any(keyword in title_lower for keyword in _NEWS_SIGNAL_KEYWORDS)
    return has_number or has_keyword


def _is_homepage_or_section_result(title: str, content: str, url: str) -> bool:
    title_lower = title.lower().strip()
    content_lower = content.lower().strip()
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/").lower()
    except Exception:
        path = ""

    if not path:
        return True

    # Detect category/section pages by shallow path + file extension
    # e.g. "kinh-te.htm", "the-thao.html", "gioi-sao/" 
    section_extensions = re.search(r"\.(htm|html|php|asp)(\?.*)?$", path)
    shallow_section = len(path.split("/")) <= 1 and not re.search(r"\d{6,}|20\d{2}", path)
    is_section_file = bool(section_extensions) and shallow_section
    
    # Only treat as section page if title/content lack specific news signals
    if is_section_file:
        if not _has_specific_title_signal(title_lower) and not _has_specific_title_signal(content_lower):
            return True

    generic_title = any(pattern in title_lower for pattern in _GENERIC_PAGE_TITLE_PATTERNS)
    lacks_specifics = not _has_specific_title_signal(title_lower) and not _has_specific_title_signal(content_lower)
    if shallow_section and generic_title and lacks_specifics:
        return True
    if len(content_lower) > 450 and sum(marker in content_lower for marker in _BOILERPLATE_SNIPPET_MARKERS) >= 2:
        return True
    return False


def _is_stale_by_url(url: str, now: datetime) -> bool:
    try:
        path = urlparse(url).path
    except Exception:
        path = url
    candidates = re.findall(r"(20\d{2})[/\-_]?(\d{2})[/\-_]?(\d{2})", path)
    candidates += [(year, month, day) for day, month, year in re.findall(r"(\d{2})(\d{2})(20\d{2})", path)]
    ages: list[int] = []
    for year_s, month_s, day_s in candidates:
        try:
            article_date = datetime(int(year_s), int(month_s), int(day_s), tzinfo=_VN_TZ)
            ages.append((now - article_date).days)
        except (ValueError, OverflowError):
            continue
    return bool(ages and min(ages) > 4)


def _is_stale_by_years(combined: str, title: str, now: datetime) -> bool:
    """Check if article is stale based on year references.
    
    VERY CONSERVATIVE: Only reject if the year appears to be a publication date
    (e.g., "2022" standalone in title or as the ONLY date indicator).
    Years mentioned in phrases like "năm 2024" are context, not publication dates.
    """
    years_found = re.findall(r"\b(20[1-2]\d)\b", combined)
    if not years_found:
        return False

    current_year = now.year
    last_year = current_year - 1
    two_years_ago = current_year - 2
    
    # If ANY reference is current year or last year, NOT stale
    has_recent = any(int(y) >= last_year for y in years_found)
    if has_recent:
        return False
    
    # Count how many year references exist
    unique_years = set(years_found)
    
    # If there are multiple different years, it's likely a news article with context
    # Don't filter articles that discuss multiple time periods
    if len(unique_years) >= 2:
        return False
    
    # Only filter if there's a single old year reference AND it appears 
    # to be a publication date indicator (in title, or standalone at start)
    title_years = re.findall(r"\b(20[1-2]\d)\b", title)
    if title_years:
        # Year in title is more likely to be stale indicator
        title_stale = all(int(y) < two_years_ago for y in title_years)
        if title_stale:
            logger.debug("Filtered stale news by title year: %s", title[:60])
            return True
        return False
    
    # Year only in content — likely context, not publication date
    # Don't filter unless the year is 3+ years old
    three_years_ago = current_year - 3
    all_very_old = all(int(y) < three_years_ago for y in years_found)
    if all_very_old:
        logger.debug("Filtered very stale news (years: %s): %s", years_found, title[:60])
        return True
    
    return False


def _is_stale_by_dates(combined: str, title: str, now: datetime) -> bool:
    """Check if article is stale based on date patterns in content.
    
    Only filter if there are CLEAR date references (DD/MM/YYYY format with year).
    Skip ambiguous short date patterns that could be stock prices, percentages, etc.
    """
    # Only use full date patterns (DD/MM/YYYY) — these are unambiguous
    full_date_patterns = re.findall(r"(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})", combined)
    
    # Also check for Vietnamese date format: "ngày DD tháng MM năm YYYY"
    vn_date_patterns = re.findall(r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", combined)
    
    ages: list[int] = []
    
    for day_s, month_s, year_s in full_date_patterns:
        try:
            article_date = datetime(int(year_s), int(month_s), int(day_s), tzinfo=_VN_TZ)
            ages.append((now - article_date).days)
        except (ValueError, OverflowError):
            continue
    
    for day_s, month_s, year_s in vn_date_patterns:
        try:
            article_date = datetime(int(year_s), int(month_s), int(day_s), tzinfo=_VN_TZ)
            ages.append((now - article_date).days)
        except (ValueError, OverflowError):
            continue
    
    # Skip short date patterns like "1.300" (stock prices) or "1/2" (fractions)
    # These cause false positives for finance, sports, etc. content
    
    if not ages:
        return False
    if min(ages) <= 3:
        return False

    all_old = all(age > 4 for age in ages)
    if all_old:
        logger.debug("Filtered old article (ages=%s): %s", ages[:5], title[:60])
        return True
    return False


def _is_junk_news_result(title: str, content: str, url: str = "", allow_english_international: bool = False, now: datetime | None = None) -> bool:
    """Detect junk/boilerplate results OR stale/generic news snippets."""
    combined = f"{title} {content} {url}".lower()
    title_lower = title.lower().strip()

    # FIX #1: Filter weather forecast pages — VOV weather junk
    if _is_vov_weather_junk(content, title):
        return True
    if "bbc.co.uk/sounds" in combined or re.search(r"(?i)^live\s+news\s*$", title_lower):
        return True
    if len(re.findall(r"\b\d+\s+of\s+\d+\b", combined)) >= 2:
        return True
    if "photo-gallery" in combined or "top-photos-day" in combined:
        return True

    # FIX #2: Filter English content for Vietnamese news queries
    # EXCEPTION: Allow English title if URL is from Vietnamese news source
    # (many VN sites syndicate AP/Reuters with English titles but Vietnamese content)
    if not allow_english_international:
        if _is_english_content(content) and _is_english_content(title):
            # Both title AND content are English → filter
            return True
        if _is_english_content(content) and not _is_vn_news_url(url):
            # Content is English AND not from VN source → filter
            return True

    if _is_homepage_or_section_result(title, content, url) and not allow_english_international:
        return True
    if any(pattern in combined for pattern in _JUNK_NEWS_PATTERNS):
        return True
    if any(pattern in combined for pattern in _NON_NEWS_PAGE_PATTERNS):
        return True

    domain = _normalize_domain(url)
    game_domains = {*_GAME_OFFICIAL_DOMAINS, *_GAME_REPUTABLE_DOMAINS, *_GAME_SOCIAL_DOMAINS}
    game_terms = ["honkai", "genshin", "wuthering", "arknights", "endfield", "banner", "patch", "gacha", "operator", "character"]
    if domain in game_domains and any(term in combined for term in game_terms):
        return False

    is_vague_title = any(pat in title_lower for pat in _VAGUE_TITLE_PATTERNS)
    if is_vague_title and not _has_specific_title_signal(title_lower):
        return True
    # Don't reject short content if title is specific (has numbers, dates, or news keywords)
    if len(content.strip()) < 80:
        if allow_english_international or _has_specific_title_signal(title_lower) or re.search(r"\d+", title_lower):
            return False  # Title/source is specific enough, accept despite short content
        return True

    # TASK 2: Whitelist (Kim bài miễn tử) cho RSS Uy tín (Tin từ vnexpress, tuoitre, dantri, thanhnien, vneconomy, vtv...)
    trusted_vn_rss_domains = {"vnexpress.net", "tuoitre.vn", "dantri.com.vn", "thanhnien.vn", "vneconomy.vn", "vtv.vn", "vietnamnet.vn", "plo.vn"}
    if _normalize_domain(url) in trusted_vn_rss_domains and "rss" in url.lower():
        # Bypass zombie filter cho các RSS nội địa vì bài đã được BTV duyệt
        pass
    else:
        now_val = now or datetime.now(_VN_TZ)
        if _is_stale_by_url(url, now_val):
            return True
        if _is_stale_by_years(combined, title, now_val):
            return True
        if _is_stale_by_dates(combined, title, now_val):
            return True

    return False


def _get_best_content(item: dict, prefer_raw: bool) -> str:
    """Extract best content from a search result item."""
    content = item.get("content", "")
    raw = item.get("raw_content", "")

    if prefer_raw and raw and len(raw) > len(content):
        return raw[:800]

    if raw and len(raw) > len(content) * 1.5 and len(raw) > 200:
        return raw[:2000]

    return content


def _extract_domain(url: str) -> str:
    """Extract domain from URL safely."""
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def _content_limit(is_general: bool, is_stock: bool) -> int:
    if is_general:
        return 3000  
    if is_stock:
        return 1500
    return 2500  


def _result_limit(is_general: bool, is_stock: bool, requested_max_results: int) -> int:
    safe_max = max(1, requested_max_results)
    if is_stock:
        return min(safe_max, 6)
    if is_general:
        return min(safe_max, 15)
    return min(safe_max, 12)


def _truncate_snippet(content: str, content_limit: int) -> str:
    snippet = content
    if len(content) > content_limit:
        snippet = content[:content_limit].rsplit(" ", 1)[0]
    if snippet.endswith((".", "!", "?", "...")):
        return snippet
    return f"{snippet}..."


def _filter_search_items(
    raw_results: list[Any],
    prefer_raw: bool,
    should_filter_junk: bool,
) -> List[Dict[str, Any]]:
    filtered_results: List[Dict[str, Any]] = []
    for item in raw_results:
        title = item.get("title", "").strip()
        url = item.get("url") or item.get("href") or ""
        snippet = (item.get("snippet") or _get_best_content(item, prefer_raw)).strip()
        source = _extract_domain(url)

        if not title or not snippet:
            continue
        if should_filter_junk and _is_junk_news_result(title, snippet, url):
            logger.debug("Filtered junk news result: %s", title[:60])
            continue

        filtered_results.append({
            "title": title,
            "url": url,
            "content": snippet,
            "snippet": snippet,
            "source": source,
            "published_date": item.get("published_date") or item.get("publishedAt") or item.get("date") or "unknown",
            "score": item.get("score", 0),
            "provider": item.get("provider", "search"),
        })

    return filtered_results


def _select_search_items(
    filtered_results: List[Dict[str, Any]],
    query: str,
    is_news: bool,
    result_limit: int,
) -> List[Dict[str, Any]]:
    if not is_news:
        return filtered_results[:result_limit]
    return _limit_news_source_repetition(
        filtered_results,
        max_items=result_limit,
        category=_detect_news_category(query),
        max_per_source=2,
    )


_RERANK_STOPWORDS = {
    "la", "là", "va", "và", "cho", "voi", "với", "trong", "gồm", "nhung", "những", "cua", "của",
    "the", "for", "and", "with", "from", "into", "what", "when", "where", "who", "how", "current",
    "latest", "today", "news", "tin", "moi", "mới", "nhat", "nhất", "hom", "hôm", "nay", "ai",
}


def _tokenize_rerank_text(text: str) -> set[str]:
    tokens = {
        tok for tok in re.findall(r"[a-zA-ZÀ-ỹ0-9]+", (text or "").lower())
        if len(tok) >= 2
    }
    return {tok for tok in tokens if tok not in _RERANK_STOPWORDS}


_GAME_BRAND_TOKENS = {
    "honkai", "genshin", "wuthering", "valorant", "fortnite", "minecraft", "pubg", "roblox",
    "arknights", "endfield", "zenless", "zzz", "wuwa", "fgo",
}


def _query_anchor_tokens(query: str) -> set[str]:
    return {tok for tok in _tokenize_rerank_text(query) if len(tok) >= 4}


def _query_brand_tokens(query: str) -> set[str]:
    return _query_anchor_tokens(query) & _GAME_BRAND_TOKENS


def _metadata_relevance_score(query: str, item: Dict[str, Any], is_news: bool) -> float:
    title = (item.get("title") or "").lower()
    snippet = (item.get("content") or item.get("snippet") or "").lower()
    domain = _normalize_domain(item.get("url", ""))

    q_tokens = _tokenize_rerank_text(query)
    title_tokens = _tokenize_rerank_text(title)
    body_tokens = _tokenize_rerank_text(snippet)
    title_overlap = len(q_tokens & title_tokens)
    body_overlap = len(q_tokens & body_tokens)

    score = title_overlap * 2.4 + body_overlap * 1.2
    score += min(len(snippet) / 260.0, 2.0)
    if query.lower().strip() and query.lower().strip() in f"{title} {snippet}":
        score += 2.0

    if domain:
        score += 0.4
    if domain.endswith(".gov") or domain.endswith(".gov.vn") or "official" in domain:
        score += 2.0
    if is_news:
        category = _detect_news_category(query)
        score += _news_source_score(domain, category) / 45.0

    freshness_blob = f"{title} {snippet} {item.get('published_date', '')}".lower()
    now = datetime.now(_VN_TZ)
    today_markers = ["hôm nay", "mới nhất", "latest", "today", "update", now.strftime("%d/%m/%Y"), now.strftime("%Y-%m-%d")]
    if _query_has_realtime_hint(query) and any(h in freshness_blob for h in today_markers):
        score += 1.8

    return score


def _prerank_search_items(
    filtered_results: List[Dict[str, Any]],
    query: str,
    is_news: bool,
) -> List[Dict[str, Any]]:
    if not filtered_results:
        return []

    anchors = _query_anchor_tokens(query)
    brand_tokens = _query_brand_tokens(query)
    if is_news and anchors:
        anchored_results: List[Dict[str, Any]] = []
        for item in filtered_results:
            item_tokens = _tokenize_rerank_text(f"{item.get('title', '')} {item.get('content', '')}")
            if brand_tokens:
                if brand_tokens & item_tokens:
                    anchored_results.append(item)
                continue

            overlap_count = len(anchors & item_tokens)
            if overlap_count >= min(2, len(anchors)):
                anchored_results.append(item)

        if anchored_results:
            filtered_results = anchored_results

    ranked = sorted(
        filtered_results,
        key=lambda item: _metadata_relevance_score(query, item, is_news),
        reverse=True,
    )
    return ranked


def _dynamic_prerank_k(result_limit: int) -> int:
    if result_limit <= 3:
        return _MIN_PRERERANK_K
    if result_limit <= 5:
        return 5
    if result_limit <= 10:
        return min(8, result_limit)
    return _MAX_PRERERANK_K


def _markdown_excerpt(text: str, limit: int = 2500) -> str:
    """Extract meaningful sentences from text, ensuring full sentences are preserved."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    # First split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    # Pick complete sentences that are meaningful (> 20 chars), keeping order
    picked: list[str] = []
    total_len = 0
    for s in sentences:
        s_stripped = s.strip()
        if len(s_stripped) < 20:
            continue
        # Only add if it fits within limit
        if total_len + len(s_stripped) + 2 <= limit:
            picked.append(s_stripped)
            total_len += len(s_stripped) + 2
        else:
            break  # Stop when we hit the limit (sentence boundary preserved)
    if not picked and cleaned:
        # Fallback: return first N chars at sentence boundary
        first_period = cleaned[:limit].rfind(".")
        if first_period > limit * 0.3:
            picked = [cleaned[:first_period + 1]]
        else:
            picked = [cleaned[:limit]]
    return " ".join(picked).strip()


def _async_enrich_selected_items(
    selected_results: List[Dict[str, Any]],
    result_limit: int,
) -> List[Dict[str, Any]]:
    if not selected_results:
        return selected_results

    # OPTIMIZATION 3: Reduced deep-read for news (4 instead of 10)
    # Check if this is a news call by looking for "NỘI DUNG" context
    is_news_call = any("PUBLISHED:" in str(item) or "NỘI DUNG" in str(item) for item in selected_results[:3])
    if is_news_call:
        enrich_k = min(len(selected_results), min(4, result_limit))  # Was 10
    else:
        enrich_k = min(len(selected_results), _dynamic_prerank_k(result_limit))

    if enrich_k <= 0:
        return selected_results

    enriched = [dict(item) for item in selected_results]
    targets: list[tuple[int, str]] = []
    for idx, item in enumerate(enriched[:enrich_k]):
        url = (item.get("url") or "").strip()
        if not url.startswith("http"):
            continue
        targets.append((idx, url))

    if not targets:
        return enriched

    # OPTIMIZATION 3b: Shorter timeout for news (4s instead of 8s)
    page_timeout = 4 if is_news_call else 8
    executor = _DaemonPoolExecutor(max_workers=min(4, len(targets)))
    try:
        future_map = {
            executor.submit(_fetch_page_evidence, url, page_timeout): idx
            for idx, url in targets
        }
        for future in as_completed(future_map, timeout=_EVIDENCE_TOTAL_TIMEOUT):
            idx = future_map[future]
            try:
                evidence = future.result()
            except Exception:
                continue
            evidence_text = evidence.get("text", "") if isinstance(evidence, dict) else str(evidence or "")
            if not evidence_text:
                continue
            enriched[idx]["evidence"] = _markdown_excerpt(evidence_text) if not any(
                hint in enriched[idx].get("url", "").lower() for hint in
                ["game8", "prydwen", "fandom", "wiki", "hoyolab", "gryphline"]
            ) else evidence_text[:2500]
            enriched[idx]["fetched_at"] = evidence.get("fetched_at", "") if isinstance(evidence, dict) else ""
    except TimeoutError:
        logger.warning("Page evidence enrichment exceeded %ss budget", _EVIDENCE_TOTAL_TIMEOUT)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return enriched


def _item_mentions_brand(item: Dict[str, Any], brands: set[str]) -> bool:
    if not brands:
        return True
    blob = (
        f"{item.get('title', '')} {item.get('content', '')} {item.get('url', '')}"
    ).lower()
    return any(brand in blob for brand in brands)


def _format_search_results(
    data: dict,
    query: str,
    is_price: bool,
    is_stock: bool = False,
    is_news: bool = False,
    max_results: int = 5,
) -> str:
    """Format Tavily search results into a clean, LLM-friendly string."""
    answer = data.get("answer", "")
    raw_results = data.get("results", [])

    parts: List[str] = []
    if answer and not is_news:
        parts.append(f"Trả lời: {answer}")

    is_general = not is_price and not is_stock
    is_game = any(hint in query.lower() for hint in ["arknights", "endfield", "enfield", "honkai", "genshin", "banner", "gacha"])
    content_limit = _content_limit(is_general, is_stock)
    result_limit = _result_limit(is_general, is_stock, max_results)
    prefer_raw = is_price or is_stock or is_news or is_game
    should_filter_junk = is_news or is_game  # Filter junk for game too

    filtered_results = _filter_search_items(raw_results, prefer_raw, should_filter_junk)
    preranked_results = _prerank_search_items(filtered_results, query, is_news)
    selected_results = _select_search_items(preranked_results, query, is_news, result_limit)
    selected_results = _async_enrich_selected_items(selected_results, result_limit)

    brand_tokens = _query_brand_tokens(query)
    if brand_tokens:
        brand_matched = [item for item in selected_results if _item_mentions_brand(item, brand_tokens)]
        if brand_matched:
            selected_results = brand_matched
        else:
            return f"Không tìm thấy kết quả đáng tin cậy cho '{query}' từ nguồn chính thống hiện tại."

    now = datetime.now(_VN_TZ)
    fetched_default = now.strftime("%Y-%m-%d %H:%M:%S UTC+7")
    for idx, item in enumerate(selected_results, start=1):
        title = item["title"]
        url = item.get("url", "")
        snippet = (item.get("snippet") or item.get("content", "")).strip()
        source = item.get("source") or _extract_domain(url)
        published = item.get("published_date") or "unknown"
        fetched_at = item.get("fetched_at") or fetched_default

        # Build clean result block: NO legacy line, just structured fields
        block = (
            f"[KQ{idx}] TITLE: {title}\n"
            f"      SOURCE: {source}\n"
            f"      URL: {url}\n"
            f"      PUBLISHED: {published}\n"
            f"      FETCHED: {fetched_at}\n"
        )

        # Use best available content: evidence > raw_content > snippet
        evidence = (item.get("evidence") or "").strip()
        if evidence:
            evidence_limit = 3000 if is_game else 2500  # News needs full context for LLM summarization
            cleaned_evidence = _clean_snippet_content(evidence)
            # ALWAYS truncate at sentence boundary (never word boundary)
            if len(cleaned_evidence) > evidence_limit:
                truncated = cleaned_evidence[:evidence_limit]
                last_period = truncated.rfind(".")
                last_question = truncated.rfind("?")
                last_exclaim = truncated.rfind("!")
                cut_pos = max(last_period, last_question, last_exclaim)
                # Always use sentence boundary if found (lowered threshold from 50% to 20%)
                if cut_pos > evidence_limit * 0.2:
                    truncated_evidence = truncated[:cut_pos + 1]
                else:
                    # If no sentence boundary found at all, just use what we have
                    truncated_evidence = truncated
            else:
                truncated_evidence = cleaned_evidence
            block += f"      NỘI DUNG: {truncated_evidence}"
        elif snippet:
            cleaned_snippet = _clean_snippet_content(snippet)
            # ALWAYS truncate at sentence boundary (never word boundary)
            if len(cleaned_snippet) > content_limit:
                truncated = cleaned_snippet[:content_limit]
                last_period = truncated.rfind(".")
                last_question = truncated.rfind("?")
                last_exclaim = truncated.rfind("!")
                cut_pos = max(last_period, last_question, last_exclaim)
                if cut_pos > content_limit * 0.2:
                    block += f"      SNIPPET: {truncated[:cut_pos + 1]}"
                else:
                    block += f"      SNIPPET: {truncated}"
            else:
                block += f"      SNIPPET: {cleaned_snippet}"
        else:
            block += f"      SNIPPET: (no content)"

        parts.append(block)

    if not parts:
        return f"Không tìm thấy kết quả cho: {query}"

    time_intent = _build_time_filter_intent(query, is_news or _query_has_realtime_hint(query))
    header = (
        f"Kết quả tìm kiếm cho '{query}':\n"
        f"Current datetime: {fetched_default}\n"
        f"Search policy: sort_by={time_intent.get('sort_by', 'relevance')}, time_range={time_intent.get('time_range', 'past_48h')}\n"
        f"Raw metadata candidates: {len(raw_results)}\n"
        f"Selected/evidence results: {len(selected_results)}\n"
    )
    return header + "\n\n".join(parts)


def _tavily_timeout_seconds(is_price: bool, is_stock: bool, is_news: bool) -> int:
    if is_price or is_stock:
        return 20
    if is_news:
        return 15
    return 12


def _web_search_tavily(query: str, max_results: int = 5, is_news: bool = False, freshness_hours: int | None = None) -> str:
    """Internal: call Tavily API provider with broader fallback."""
    if not settings.tavily_api_key:
        raise RuntimeError("missing_tavily_key")

    is_price = _is_price_query(query)
    is_stock = _is_stock_query(query)
    tavily_timeout = _tavily_timeout_seconds(is_price, is_stock, is_news)

    try:
        # First attempt with domain restrictions
        response = _session.post(
            _TAVILY_SEARCH_URL,
            json=_build_tavily_request(query, is_price, is_stock, max_results, is_news=is_news, freshness_hours=freshness_hours),
            timeout=tavily_timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Check if we got enough results
        results_count = len(data.get("results", []))

        # If too few results and this is a news search, try broader search without domain restrictions
        if results_count < 3 and is_news:
            logger.info(f"Tavily domain-restricted search returned only {results_count} results, trying broader search")
            broader_payload = _build_tavily_request(query, is_price, is_stock, max_results, is_news=is_news, freshness_hours=freshness_hours)
            # Remove domain restrictions for broader search
            broader_payload.pop("include_domains", None)

            try:
                broader_response = _session.post(
                    _TAVILY_SEARCH_URL,
                    json=broader_payload,
                    timeout=tavily_timeout,
                )
                broader_response.raise_for_status()
                broader_data = broader_response.json()

                # Use broader results if we got more
                broader_count = len(broader_data.get("results", []))
                if broader_count > results_count:
                    logger.info(f"Broader search returned {broader_count} results, using that instead")
                    data = broader_data
            except Exception as inner_e:
                logger.error(f"Tavily broader search fallback failed: {inner_e}")
                # We can still keep the domain-restricted data if it exists

        return _format_search_results(data, query, is_price, is_stock, is_news=is_news, max_results=max_results)

    except Exception as e:
        logger.error(f"Tavily API request failed: {e}")
        raise e


# ══════════════════════════════════════════════════════════════════════════════
# PAGE EVIDENCE EXTRACTION (for DDG deep read)
# ══════════════════════════════════════════════════════════════════════════════

def _build_exa_request(query: str, max_results: int = 5) -> dict:
    """Build Exa search payload and keep it compatible with Exa text results."""
    payload: dict[str, Any] = {
        "query": _normalize_search_query(query),
        "numResults": max(1, min(max_results, 10)),
        "type": "auto",
        "contents": {
            "text": {"maxCharacters": 2500},
            "highlights": {"numSentences": 3},
        },
    }
    focused_domains = _focused_game_domains(query)
    if focused_domains:
        payload["includeDomains"] = focused_domains
    return payload


def _normalize_exa_result(item: dict[str, Any]) -> dict[str, Any]:
    url = str(item.get("url") or "").strip()
    title = str(item.get("title") or url or "Untitled").strip()
    text = str(item.get("text") or "").strip()
    highlights = item.get("highlights") or []
    if not text and isinstance(highlights, list):
        text = " ".join(str(part).strip() for part in highlights if str(part).strip())
    return {
        "title": title,
        "url": url,
        "source": f"Exa: {_extract_domain(url)}" if url else "Exa",
        "content": text,
        "snippet": text,
        "published_date": item.get("publishedDate") or item.get("published_date") or "unknown",
        "fetched_at": datetime.now(_VN_TZ).strftime("%Y-%m-%d %H:%M:%S UTC+7"),
    }


def _web_search_exa(query: str, max_results: int = 5) -> str:
    """Internal: call Exa as an optional AI search provider and return [KQ] blocks."""
    cleaned = _normalize_search_query(query)
    if not cleaned:
        return "Vui lòng nhập truy vấn tìm kiếm hợp lệ."
    if not settings.exa_api_key:
        raise RuntimeError("missing_exa_key")

    try:
        response = _session.post(
            _EXA_SEARCH_URL,
            headers={
                "x-api-key": settings.exa_api_key,
                "Content-Type": "application/json",
            },
            json=_build_exa_request(cleaned, max_results=max_results),
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") if isinstance(data, dict) else []
        normalized = [_normalize_exa_result(item) for item in results if isinstance(item, dict)]
        return _format_search_results(
            {"results": normalized},
            cleaned,
            is_price=False,
            is_stock=False,
            is_news=False,
            max_results=max_results,
        )
    except Exception as e:
        logger.error("Exa API request failed: %s", e)
        raise e


def _extract_main_text(html_text: str) -> str:
    """Extract main text from HTML, removing boilerplate.
    
    Handles article, main, section, div content areas and tables.
    """
    raw_input = (html_text or "").strip()
    if raw_input.startswith(("http://", "https://")) and _STATIC_EVIDENCE_URL_PATTERN.search(raw_input):
        return ""

    def _match_closing_tag(html: str, open_start: int, tag: str) -> int:
        """Find the position of the matching closing tag for a given opening tag.
        Properly handles nested tags of the same type."""
        end_tag = f"</{tag}>"
        end_tag_lower = end_tag.lower()
        open_tag_pattern = re.compile(rf"<{tag}[>\s]", flags=re.IGNORECASE)
        depth = 1
        pos = open_start + 1
        while pos < len(html) and depth > 0:
            next_open = open_tag_pattern.search(html, pos)
            next_close = html.find(end_tag, pos)
            if next_close < 0:
                return len(html)
            open_pos = next_open.start() if next_open else len(html)
            if open_pos < next_close:
                depth += 1
                pos = open_pos + 1
            else:
                depth -= 1
                if depth == 0:
                    return next_close + len(end_tag)
                pos = next_close + len(end_tag)
        return len(html)

    text = html_text or ""
    text = html.unescape(text)
    text = re.sub(r"<script[\s\S]*?</script>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<style[\s\S]*?</style>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<noscript[\s\S]*?</noscript>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<(svg|form|button|iframe|figure|caption)[\s\S]*?</\1>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<nav[\s\S]*?</nav>", " ", text, flags=re.IGNORECASE)
    # Remove common navigation/menu divs by class/id (VN sites use divs, not <nav>)
    nav_div_pattern = r'<div[^>]*(?:class|id)\s*=\s*["\'][^"\']*(?:menu|navigation|sidebar|breadcrumb|category-list|related-news|most-read|trending|hot-news|popular|widget|social-share|share-bar|author-bio|article-tags|tags-list|comment-section|newsletter|breadcrumb-wrap|top-nav|main-menu|cat-list|topic-list)[^"\']*["\'][^>]*>[\s\S]*?</div>'
    text = re.sub(nav_div_pattern, " ", text, flags=re.IGNORECASE)

    # Try multiple content container patterns (game wikis often use div-based layouts).
    # Use proper nested tag matching to handle nested divs.
    best_text = ""
    # mw-parser-output (MediaWiki) - use proper div depth matching
    m = re.search(r'(<div[^>]*\bclass\s*=\s*["\'][^"\']*mw-parser-output[^"\']*["\'][^>]*>)', text, flags=re.IGNORECASE)
    if m:
        end = _match_closing_tag(text, m.start(), "div")
        candidate = text[m.end():end - len("</div>")] if end > 0 else ""
        if len(candidate) > len(best_text):
            best_text = candidate
    # article/main tag
    article_matches = list(re.finditer(r'<(article|main)[^>]*>([\s\S]*?)</\1>', text, flags=re.IGNORECASE | re.DOTALL))
    for match in article_matches:
        candidate = match.group(2)
        if len(candidate) > len(best_text):
            best_text = candidate
    # div.content patterns (class or id CONTAINING content/article key words - non-greedy prefix)
    for m in re.finditer(r'<div[^>]*(?:class|id)\s*=\s*["\'][^"\']*?(?:content|article|main-content|entry-content|post-content|page-content|article-body)[^"\']*["\']', text, flags=re.IGNORECASE):
        end = _match_closing_tag(text, m.start(), "div")
        start = m.end()
        candidate = text[start:end - len("</div>")] if end > 0 else ""
        if len(candidate) > len(best_text):
            best_text = candidate
    if best_text:
        text = best_text

    # Remove remaining boilerplate tags
    for block_tag in ("header", "footer", "aside"):
        text = re.sub(rf"<{block_tag}[^>]*>[\s\S]*?</{block_tag}>", " ", text, flags=re.IGNORECASE)

    # Extract data-name attributes from character tooltips (game wiki operator icons).
    # Replace the entire tag with just the operator name so it survives HTML tag stripping.
    text = re.sub(
        r'<[^>]*\bdata-name\s*=\s*(["\'])([^>]+?)\1[^>]*>',
        r' \2 ',
        text,
    )

    # Extract table content: convert table rows to text lines with pipe separators
    def _table_to_lines(table_html: str) -> str:
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, flags=re.IGNORECASE | re.DOTALL)
        if not rows:
            return table_html
        lines = []
        for row in rows:
            cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, flags=re.IGNORECASE | re.DOTALL)
            cell_texts = []
            for c in cells:
                ct = re.sub(r'<[^>]+>', ' ', c)
                ct = re.sub(r'\s+', ' ', ct).strip()
                if ct:
                    cell_texts.append(ct)
            if cell_texts:
                lines.append(" | ".join(cell_texts))
        return "\n".join(lines)

    text = re.sub(
        r'<table[^>]*>(.*?)</table>',
        lambda m: "\n" + _table_to_lines(m.group(0)) + "\n",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(r"<[^>]+>", " ", text)

    # Aggressive boilerplate removal
    boilerplate_patterns = [
        r"\b(cookie policy|accept cookies|subscribe|advertisement|all rights reserved|privacy policy|terms of service|sign in|sign up|log in|log out|register now)\b",
        r"\b(comments?\s*\d+|share this|tweet|facebook|instagram|youtube|telegram|whatsapp|pinterest|linkedin)\b",
        r"\b(related (?:articles?|posts?|news|stories)|you may also like|recommended for you|most popular|trending now)\b",
        r"\b(sponsored|promoted|ad|advertisement|brought to you by|in partnership with)\b",
    ]
    for bp in boilerplate_patterns:
        text = re.sub(bp, " ", text, flags=re.IGNORECASE)

    # Vietnamese UI/boilerplate patterns
    vn_boilerplate_patterns = [
        r"\b(chia sẻ|like|thích|yêu thích)\b",
        r"\b(theo dõi|follow|đăng ký)\b",
        r"\b(trở lại chủ đề|trở lại|về trang chủ|quay lại|back to top)\b",
        r"\b(bình luận|comment|gửi bình luận)\b",
        r"\b(chuyên mục|chủ đề|category)\b",
        r"\b(xem thêm|read more|đọc tiếp|tiếp tục)\b",
        r"\b(in bài|gửi email|gửi cho bạn|forward)\b",
        r"\b(cỡ chữ|font size|tăng chữ|giảm chữ)\b",
        r"\b(\d+\s+trở lại chủ đề|\d+\s+bình luận|\d+\s+comment|\d+\s+chia sẻ)\b",
        r"\b(thời gian đọc|phút đọc|min read)\b",
        r"\b(nhận xét|đánh giá|rate)\b",
        r"\b(báo lỗi|report|phản ánh)\b",
        # Section labels with count: "0 Thời sự", "5 Bình luận", "Cùng luận bàn 0"
        r"\b\d+\s+(thời sự|kinh tế|thể thao|giải trí|thế giới|pháp luật|sức khỏe|giáo dục|công nghệ|du lịch|xe|đời sống)\b",
        r"\b(cùng\s+luận\s+bàn|thảo\s+luận|bàn\s+luận)\s+\d+\b",
        # Author role labels: "Nhà ngoại giao", "PV Thanh Niên", "và 1 tác giả khác"
        r"\b(Nhà\s+báo|PV|Phóng\s+viên|Nhà\s+ngoại\s+giao|Chuyên\s+gia|Tiến\s+sĩ|Giáo\s+sư|TS|GS|PGS)\b",
        r"\bvà\s+\d+\s+tác\s+giả\s+khác\b",
        # Source labels: "Theo ghi nhận của PV", "Theo báo cáo"
        r"\b(theo\s+ghi\s+nhận\s+của|theo\s+báo\s+cáo|theo\s+đánh\s+giá|theo\s+thống\s+kê)\b",
    ]
    for bp in vn_boilerplate_patterns:
        text = re.sub(bp, " ", text, flags=re.IGNORECASE)

    # Remove navigation-menu-like text sequences at start of text
    # VN sites often have category lists like: "Kinh tế Thể thao Giải trí..." at the top
    category_keywords = r'(?:kinh tế|thể thao|giải trí|thế giới|trong nước|quốc tế|pháp luật|sức khỏe|giáo dục|du lịch|khoa học|công nghệ|xe|bất động sản|văn hóa|đời sống|an ninh|quân sự|chính trị|doanh nghiệp|ngân hàng|chứng khoán|xã hội|lao động|việc làm|môi trường|biển đảo|chính sách|phát triển)'
    # Match 2+ category keywords (with optional modifiers) at start, then stop before actual content
    nav_prefix = rf"^\s*(?:{category_keywords})(?:\s+(?:xanh|số|24h|online|mới|nóng|cập nhật|-\s*\w+))?(?:\s+(?:,?\s*|-?\s*)?{category_keywords})+\s*"
    text = re.sub(nav_prefix, " ", text, flags=re.IGNORECASE)

    # Strip > prefix from lines (blockquote artifacts)
    text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n>\s*", "\n", text)
    # Also strip > after spaces (when blockquote content ends up mid-text)
    text = re.sub(r"\s+>\s+", " ", text)
    text = re.sub(r"^\s*>\s+", "", text)

    # Clean up resulting double/triple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


_STATIC_EVIDENCE_URL_PATTERN = re.compile(
    r"\.(?:jpe?g|png|gif|webp|pdf|zip)(?:[?#].*)?$",
    flags=re.IGNORECASE,
)


def _fetch_page_evidence(url: str, timeout_sec: int = 5) -> Dict[str, str]:
    """Fetch and extract main content from a URL.
    
    Returns up to 8000 chars for detailed extraction.
    """
    fetched_at = datetime.now(_VN_TZ).strftime("%Y-%m-%d %H:%M:%S UTC+7")
    if _STATIC_EVIDENCE_URL_PATTERN.search(url or ""):
        return {"text": "", "fetched_at": fetched_at}
    links: list[dict[str, str]] = []
    try:
        response = _session.get(
            url,
            timeout=timeout_sec,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            },
        )
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if any(kind in content_type for kind in ("image/", "application/pdf")):
            return {"text": "", "links": [], "fetched_at": fetched_at}
        html_text = response.text
        text = _extract_main_text(html_text)[:8000]
        links = _extract_article_links_from_html(url, html_text)
    except Exception:
        text = fetch_text_evidence(url, max_chars=8000, timeout_sec=timeout_sec, depth=1)
    return {"text": text, "links": links, "fetched_at": fetched_at}


def _extract_article_links_from_html(base_url: str, html_text: str, limit: int = 8) -> list[dict[str, str]]:
    """Extract likely article links from a news/category page."""
    base_domain = urlparse(base_url).netloc.lower().removeprefix("www.")
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"<a\b[^>]*\bhref\s*=\s*(['\"])(?P<href>.*?)\1[^>]*>(?P<title>[\s\S]*?)</a>",
        html_text or "",
        flags=re.IGNORECASE,
    ):
        href = html.unescape(match.group("href")).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        full_url = urljoin(base_url, href).split("#", 1)[0]
        parsed = urlparse(full_url)
        domain = parsed.netloc.lower().removeprefix("www.")
        if domain != base_domain:
            continue
        path = parsed.path.lower().strip("/")
        if not path or path.endswith((".rss", ".epi")):
            continue
        last_part = path.rsplit("/", 1)[-1]
        slug = re.sub(r"\.(?:html?|epi|rss)$", "", last_part)
        if slug in {"the-gioi", "tin-the-gioi", "quoc-te", "tin-quoc-te", "news", "world", "international"}:
            continue
        article_signal = bool(re.search(r"(?:-\d{5,}|/\d{4}/\d{2}/|-\d+\.html|\.html?)$", path))
        if not article_signal:
            continue
        title = re.sub(r"<[^>]+>", " ", match.group("title"))
        title = " ".join(html.unescape(title).split()).strip(" -–—:;,.")
        if len(title) < 20 or len(title) > 180:
            continue
        title_norm = unicodedata.normalize("NFD", title.lower())
        title_norm = "".join(ch for ch in title_norm if unicodedata.category(ch) != "Mn")
        if any(term in title_norm for term in ("xem them", "doc tiep", "video", "podcast", "rss")):
            continue
        url_key = full_url.rstrip("/").lower()
        if url_key in seen:
            continue
        seen.add(url_key)
        rows.append({"title": title, "url": full_url})
        if len(rows) >= limit:
            break
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# DUCKDUCKGO SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _score_and_rank_records(query: str, records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Score and rank search records based on query relevance and snippet quality."""
    query_tokens = set(re.findall(r"\w+", query.lower()))

    for rec in records:
        score = 0.0
        title = rec.get("title", "").lower()
        snippet = rec.get("snippet", "").lower()
        href = rec.get("href", "").lower()

        overlap = sum(1 for tok in query_tokens if len(tok) > 2 and (tok in title or tok in snippet))
        score += overlap * 0.5

        if len(snippet) > 100:
            score += 0.5
        if len(snippet) > 200:
            score += 0.5

        if any(tag in href for tag in [
            ".gov", ".edu", ".vn", "wikipedia",
            DOMAIN_VNEXPRESS, DOMAIN_DANTRI, DOMAIN_TUOITRE,
        ]):
            score += 1.0

        rec["_score"] = score  # type: ignore[assignment]

    return sorted(records, key=lambda x: x.get("_score", 0), reverse=True)


def _domain_diversify_records(records: List[Dict[str, str]], limit: int) -> List[Dict[str, str]]:
    """Ensure results come from diverse domains."""
    diverse: List[Dict[str, str]] = []
    used_domains: set[str] = set()

    for rec in records:
        url = rec.get("href", "")
        try:
            domain = urlparse(url).netloc.replace("www.", "").lower()
        except Exception:
            domain = url

        if domain not in used_domains:
            diverse.append(rec)
            used_domains.add(domain)
            if len(diverse) >= limit:
                return diverse

    # Fill remaining spots if needed
    for rec in records:
        if len(diverse) >= limit:
            break
        if rec not in diverse:
            diverse.append(rec)

    return diverse[:limit]


_DDG_RESULT_PATTERN = re.compile(
    r'<a[^>]*class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>'
    r'(?P<title>[^<]*(?:<[^/][^>]*>[^<]*)*)</a>.*?'
    r'(?:<a[^>]*class="result__snippet"[^>]*>'
    r'(?P<snippet_a>[^<]*(?:<[^/][^>]*>[^<]*)*)</a>'
    r'|<div[^>]*class="result__snippet"[^>]*>'
    r'(?P<snippet_div>[^<]*(?:<[^/][^>]*>[^<]*)*)</div>)?',
    re.IGNORECASE | re.DOTALL,
)


def _ddg_headers() -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }


def _fetch_ddg_page(query: str, headers: Dict[str, str]) -> str:
    search_url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    # TASK 3: Hạ nhiệt DuckDuckGo - Ép timeout=5s để nhảy nhanh sang nguồn khác nếu mạng nghẽn
    response = _session.get(search_url, headers=headers, timeout=5)
    page = response.text
    if response.status_code != 202 and 'class="result__snippet"' in page:
        return page

    lite_response = _session.post(
        "https://lite.duckduckgo.com/lite/",
        data={"q": query},
        headers=headers,
        timeout=5,
    )
    lite_page = lite_response.text
    if lite_response.status_code == 202 or 'class="result-snippet"' not in lite_page:
        raise DDGRateLimitError("DDG rate-limited (status 202), use Tavily fallback")
    return lite_page


def _ddg_broad_limit(max_results: int) -> int:
    return min(max(max_results + 10, _TAVILY_BROAD_MIN_RESULTS), _TAVILY_BROAD_MAX_RESULTS)


def _parse_ddg_items_lite(page: str, max_results: int) -> List[Dict[str, str]]:
    parsed_items: List[Dict[str, str]] = []
    anchor_pattern = re.compile(r'<a[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.IGNORECASE | re.DOTALL)
    for match in anchor_pattern.finditer(page):
        href = (match.group("href") or "").strip()
        if not href:
            continue
        if not (href.startswith("http") or "uddg=" in href):
            continue

        title = strip_html_tags(match.group("title") or "")
        if len(title) < 8:
            continue

        tail = page[match.end():match.end() + 450]
        snippet_match = re.search(
            r'(?:result__snippet|result-snippet)[^>]*>(?P<snippet>.*?)</(?:a|div|td)>',
            tail,
            flags=re.IGNORECASE | re.DOTALL,
        )
        snippet = strip_html_tags((snippet_match.group("snippet") if snippet_match else "") or "")

        parsed_items.append({"title": title, "snippet": snippet, "href": href})
        if len(parsed_items) >= _ddg_broad_limit(max_results):
            break
    return parsed_items


def _parse_ddg_items(page: str, max_results: int) -> List[Dict[str, str]]:
    parsed_items: List[Dict[str, str]] = []
    for match in _DDG_RESULT_PATTERN.finditer(page):
        title = strip_html_tags(match.group("title") or "")
        snippet = strip_html_tags(match.group("snippet_a") or match.group("snippet_div") or "")
        href = (match.group("href") or "").strip()
        if not title:
            continue
        parsed_items.append({"title": title, "snippet": snippet, "href": href})
        if len(parsed_items) >= _ddg_broad_limit(max_results):
            break

    if parsed_items:
        return parsed_items
    return _parse_ddg_items_lite(page, max_results)


def _build_ddg_result_lines(
    selected_items: List[Dict[str, Any]],
    is_general: bool,
    is_stock: bool,
) -> List[str]:
    lines: List[str] = []
    content_limit = _content_limit(is_general, is_stock)

    for idx, item in enumerate(selected_items, start=1):
        title = (item.get("title") or "").strip()
        href = (item.get("url") or "").strip()
        snippet = (item.get("snippet") or item.get("content", "")).strip()
        evidence = (item.get("evidence", "") or "").strip()
        source = item.get("source") or _extract_domain(href)
        published = item.get("published_date") or "unknown"
        fetched_at = item.get("fetched_at") or datetime.now(_VN_TZ).strftime("%Y-%m-%d %H:%M:%S UTC+7")

        block = (
            f"[KQ{idx}] TITLE: {title}\n"
            f"      SOURCE: {source}\n"
            f"      URL: {href}\n"
            f"      PUBLISHED: {published}\n"
            f"      FETCHED: {fetched_at}\n"
        )
        if evidence:
            cleaned_evidence = _clean_snippet_content(evidence)
            block += f"      NỘI DUNG: {cleaned_evidence[:1200].rsplit(' ', 1)[0] if len(cleaned_evidence) > 1200 else cleaned_evidence}"
        elif snippet:
            cleaned_snippet = _clean_snippet_content(snippet)
            block += f"      SNIPPET: {cleaned_snippet[:content_limit]}"
        else:
            block += f"      SNIPPET: (no content)"
        lines.append(block)

    return lines


def _web_search_duckduckgo(
    query: str,
    max_results: int = 10,
    allow_deep_read: bool = True,
) -> str:
    """Primary provider: DuckDuckGo HTML search.

    Pipeline:
    1) Fetch broad metadata candidates (title/url/snippet)
    2) Pre-rerank by relevance + source quality
    3) Dynamic-K async enrichment for top candidates
    4) Return concise source-cited records
    """
    _ = allow_deep_read
    try:
        page = _fetch_ddg_page(query, _ddg_headers())
        parsed_items = _parse_ddg_items(page, max_results)
        if not parsed_items:
            raise RuntimeError("DDG returned no valid results")

        is_price = _is_price_query(query)
        is_stock = _is_stock_query(query)
        is_news = _should_use_news_mode(query)
        is_general = not is_price and not is_stock

        ranked_records = _score_and_rank_records(query, parsed_items)
        broad_records = _domain_diversify_records(ranked_records, _ddg_broad_limit(max_results))

        search_items: List[Dict[str, Any]] = []
        for rec in broad_records:
            url = _decode_ddg_redirect_url(rec.get("href", ""))
            title = (rec.get("title") or "").strip()
            snippet = (rec.get("snippet") or "").strip()
            if not title or not snippet:
                continue
            search_items.append({
                "title": title,
                "url": url,
                "content": snippet,
                "snippet": snippet,
                "source": _extract_domain(url),
                "published_date": "unknown",
                "provider": "duckduckgo",
            })

        if not search_items:
            raise RuntimeError("DDG parsed results are empty")

        filtered_items = _filter_search_items(
            raw_results=search_items,
            prefer_raw=False,
            should_filter_junk=is_news,
        )
        preranked_items = _prerank_search_items(filtered_items, query, is_news)

        result_limit = _result_limit(is_general, is_stock, max_results)
        selected_items = _select_search_items(preranked_items, query, is_news, result_limit)
        selected_items = _async_enrich_selected_items(selected_items, result_limit)

        brand_tokens = _query_brand_tokens(query)
        if brand_tokens:
            brand_matched = [item for item in selected_items if _item_mentions_brand(item, brand_tokens)]
            if brand_matched:
                selected_items = brand_matched
            elif not selected_items:
                return f"Không tìm thấy kết quả đáng tin cậy cho '{query}' từ nguồn chính thống hiện tại."

        return "\n".join(_build_ddg_result_lines(selected_items, is_general, is_stock))
    except DDGRateLimitError:
        raise
    except Exception as e:
        raise RuntimeError(f"DDG Search failed: {str(e)}") from e


def _should_use_news_mode(query: str) -> bool:
    lower = query.lower()
    # Game queries (banner, character, gacha) cần tìm wiki/guide, không phải news article
    game_markers = ["honkai", "genshin", "wuthering", "wuwa", "zenless", "zzz",
                    "arknights", "endfield", "enfield", "fgo",
                    "banner", "gacha", "reroll", "tier list", "rate up", "rate-up"]
    if any(marker in lower for marker in game_markers):
        return False

    news_hints = [
        "tin", "news", "mới nhất", "hôm nay", "breaking", "cập nhật",
        "patch", "event", "release",
    ]
    if any(hint in lower for hint in news_hints):
        return True
    return False


def _build_focused_game_query(query: str) -> str:
    lower = query.lower()
    is_endfield = ("endfield" in lower or "enfield" in lower) and "-endfield" not in lower

    if "honkai" in lower:
        return "Honkai Star Rail current banner warp characters rate up Prydwen Game8"
    if "genshin" in lower:
        return "Genshin Impact current banner characters rate up Prydwen Game8"

    if "zenless" in lower or "zzz" in lower:
        return "Zenless Zone Zero current banner characters rate up Prydwen Game8"

    if "wuthering" in lower or "wuwa" in lower:
        return "Wuthering Waves current banner characters rate up Prydwen Game8"

    if is_endfield:
        return "Arknights Endfield current event banner Fest of Brilliance character list"

    if "arknights" in lower or "arknight" in lower:
        return "Arknights current headhunting banner rate up operators list"

    if "fgo" in lower or "fate" in lower:
        return "Fate Grand Order current banner servants rate up Game8"

    return f"{query} current upcoming events"


def _focused_game_domains(query: str) -> list[str]:
    lower = query.lower()
    # Bắt luôn cả trường hợp user gõ thiếu chữ "d" thành "enfield"
    is_endfield = ("endfield" in lower or "enfield" in lower) and "-endfield" not in lower
    
    if "honkai" in lower:
        return ["hoyolab.com", "hoyoverse.com", "mihoyo.com", "game8.co", "prydwen.gg", "fandom.com"]
    if "genshin" in lower:
        return ["hoyolab.com", "hoyoverse.com", "mihoyo.com", "game8.co", "prydwen.gg", "fandom.com"]

    if "zenless" in lower or "zzz" in lower:
        return ["hoyolab.com", "hoyoverse.com", "mihoyo.com", "game8.co", "prydwen.gg", "fandom.com"]

    if "wuthering" in lower or "wuwa" in lower:
        return ["game8.co", "prydwen.gg", "fandom.com", "wutheringwaves.wiki.gg"]

    if is_endfield:
        return ["endfield.gryphline.com", "endfield.hypergryph.com", "endfield.wiki.gg", "game8.co", "prydwen.gg", "fandom.com"]

    if "arknights" in lower or "arknight" in lower:
        return ["arknights.wiki.gg", "game8.co", "prydwen.gg", "fandom.com"]

    if "fgo" in lower or "fate" in lower:
        return ["game8.co", "fandom.com"]
        
    return []


def _extract_search_result_links(search_output: str, limit: int) -> list[dict[str, str]]:
    """Extract cited links from core.search formatted output."""
    if not search_output or "http" not in search_output:
        return []

    links: list[dict[str, str]] = []
    seen: set[str] = set()
    citation_rx = re.compile(
        r"(?m)^\s*\d+\.\s+(?P<body>.*?)\s+\(via\s+\[(?P<source>[^\]]+)\]\((?P<url>https?://[^)]+)\)\)"
    )
    for match in citation_rx.finditer(search_output):
        url = match.group("url").strip()
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if (
            not parsed.scheme.startswith("http")
            or url in seen
            or _STATIC_EVIDENCE_URL_PATTERN.search(url)
            or any(skip in domain for skip in ("youtube.com", "youtu.be", "facebook.com", "x.com", "twitter.com"))
        ):
            continue

        body = match.group("body").strip()
        links.append({
            "title": body.split(": ", 1)[0].strip(),
            "source": (match.group("source") or _extract_domain(url)).strip(),
            "url": url,
        })
        seen.add(url)
        if len(links) >= limit:
            break

    return links


def _compact_page_evidence(text: str, limit: int = 2600) -> str:
    compact = " ".join(html.unescape(text or "").replace("~~", "").split())
    if len(compact) <= limit:
        return compact

    chunk = compact[:limit]
    cut = max(chunk.rfind("."), chunk.rfind("?"), chunk.rfind("!"))
    if cut > limit * 0.25:
        return chunk[:cut + 1]
    return chunk


def _priority_evidence_links(query: str) -> list[dict[str, str]]:
    lower = (query or "").lower()
    if "arknights" not in lower or "endfield" in lower or "enfield" in lower:
        return []
    year = datetime.now(_VN_TZ).strftime("%Y")
    return [
        {
            "title": f"Arknights Headhunting Banners {year}",
            "source": "arknights.wiki.gg",
            "url": f"https://arknights.wiki.gg/wiki/Headhunting/Banners/{year}",
        },
        {
            "title": "Arknights Current Events",
            "source": "arknights.wiki.gg",
            "url": "https://arknights.wiki.gg/wiki/Event",
        },
        {
            "title": "Arknights Upcoming Events",
            "source": "arknights.wiki.gg",
            "url": "https://arknights.wiki.gg/wiki/Event/Upcoming",
        },
    ]


def _append_page_evidence(search_output: str, query: str = "", max_links: int = 3) -> str:
    """Append readable page content to core search output when available."""
    priority_links = _priority_evidence_links(query)
    regular_links = _extract_search_result_links(search_output, max_links)
    links: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in priority_links + regular_links:
        url = item["url"]
        if url in seen:
            continue
        links.append(item)
        seen.add(url)
        if len(links) >= max_links:
            break
    if not links:
        return search_output

    evidence_blocks: list[str] = [""] * len(links)
    executor = _DaemonPoolExecutor(max_workers=min(3, len(links)))
    try:
        try:
            future_map = {
                executor.submit(_fetch_page_evidence, item["url"], 5): (idx, item)
                for idx, item in enumerate(links)
            }
        except Exception:
            # Executor unavailable (e.g. interpreter shutting down) — skip evidence.
            return search_output
        for future in as_completed(future_map, timeout=_EVIDENCE_TOTAL_TIMEOUT):
            idx, item = future_map[future]
            try:
                fetched = future.result()
            except Exception:
                continue
            text = fetched.get("text", "") if isinstance(fetched, dict) else str(fetched or "")
            child_links = fetched.get("links", []) if isinstance(fetched, dict) else []
            evidence_limit = 8000 if item["source"].lower() == "arknights.wiki.gg" else 2600
            evidence = _compact_page_evidence(text, limit=evidence_limit)
            if len(evidence) < 180:
                continue
            links_text = ""
            if child_links:
                links_text = "\n      LINK BAI VIET:\n" + "\n".join(
                    f"      - {link['title']}: {link['url']}"
                    for link in child_links[:8]
                )
            evidence_blocks[idx] = (
                f"[DOC{idx + 1}] TITLE: {item['title']}\n"
                f"      SOURCE: {item['source']}\n"
                f"      URL: {item['url']}\n"
                f"      NOI DUNG: {evidence}{links_text}"
            )
    except TimeoutError:
        logger.warning(
            "Page evidence fetch exceeded %ss budget; using partial evidence",
            _EVIDENCE_TOTAL_TIMEOUT,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    evidence_blocks = [
        re.sub(r"^\[DOC\d+\]", f"[DOC{idx}]", block)
        for idx, block in enumerate((block for block in evidence_blocks if block), start=1)
    ]
    if not evidence_blocks:
        return search_output

    return f"{search_output}\n\nNOI DUNG DA DOC TU CAC LINK:\n" + "\n\n".join(evidence_blocks)


_NEWS_QUERY_MARKERS = (
    "tin tức", "tin tuc", "tin mới", "tin moi", "tin nóng", "tin nong",
    "tin hot", "tin nổi bật", "tin noi bat", "bản tin", "ban tin",
    "thời sự", "thoi su", "breaking news", "headline", "headlines",
    "tin thế giới", "tin the gioi", "tin quốc tế", "tin quoc te",
    "tin trong nước", "tin trong nuoc", "tin hot nhất", "tin nóng nhất",
)


def _looks_like_news_query(query: str) -> bool:
    """Lightweight news-intent detection used to route web_search internally."""
    normalized = unicodedata.normalize("NFD", (query or "").lower())
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return any(marker in normalized for marker in _NEWS_QUERY_MARKERS)


def _is_ddg_blocked_error(exc: BaseException) -> bool:
    """True when a DDG exception matches the configured TLS/connection block signature."""
    signature = (settings.search_ddg_blocked_signature or "").strip().lower()
    return bool(signature) and signature in str(exc).lower()


_ddg_blocked_count = 0
_ddg_circuit_lock = threading.Lock()


def _record_ddg_block() -> None:
    """Count a blocking DDG failure and warn once the circuit is open."""
    global _ddg_blocked_count
    with _ddg_circuit_lock:
        _ddg_blocked_count += 1
        if _ddg_blocked_count >= settings.search_ddg_failure_threshold:
            logger.warning(
                "DDG blocked repeatedly (%d) — skipping DDG, using Tavily + Exa",
                _ddg_blocked_count,
            )


def _ddg_should_skip() -> bool:
    with _ddg_circuit_lock:
        return _ddg_blocked_count >= settings.search_ddg_failure_threshold


def _web_search_api_fallback(query: str, max_results: int) -> list[dict]:
    """Combine Exa + Tavily API results for the best coverage when DDG fails."""
    from core.search.providers import _exa_search, _tavily_search

    combined: list[dict] = []
    for provider in (_exa_search, _tavily_search):
        try:
            combined.extend(provider(query, max_results))
        except Exception as exc:
            if str(exc) not in ("missing_tavily_key", "missing_exa_key"):
                logger.warning("Fallback search provider failed: %s", exc)
    return combined


def _web_search_run(query: str, max_results: int, is_news: bool) -> str:
    """DDG-first search; on DDG failure, fall back to Tavily + Exa combined.

    Repeated DDG blocks trip a circuit breaker that skips DDG entirely so the
    user is not left waiting on a provider Yahoo is actively rejecting.
    """
    from core.search import _filter_news_items, _rewrite_news_query
    from core.search.format import format_news_response, format_search_response
    from core.search.providers import _ddg_news_search, _ddg_search

    search_query = _rewrite_news_query(query) if is_news else query
    items: list[dict] = []

    if not _ddg_should_skip():
        try:
            if is_news:
                items = _ddg_news_search(search_query, max_results * 3)
            else:
                items = _ddg_search(search_query, max_results + 3)
        except Exception as exc:
            if _is_ddg_blocked_error(exc):
                _record_ddg_block()
            logger.warning("DDG search failed: %s", exc)
            items = []

        if is_news:
            items = _filter_news_items(items, query)

    if len(items) >= 2:
        if is_news:
            return format_news_response(items, query, max_results)
        return format_search_response(items, query, max_results)

    items = _web_search_api_fallback(search_query, max_results)
    if is_news:
        items = _filter_news_items(items, query)

    if not items:
        return f"Không tìm thấy {'tin tức' if is_news else 'kết quả'} cho '{query}'."
    if is_news:
        return format_news_response(items, query, max_results)
    return format_search_response(items, query, max_results)


def web_search(query: str, max_results: int = 5) -> str:
    """Web search and news: realtime search, article filtering, source links, and concise summaries.

    Use for any real-time, time-sensitive, or news request: prices (xăng/vàng/
    crypto), people, definitions, current events, breaking news, 'tin tức',
    politics, sports, world or Vietnam headlines. Routes news-like queries to a
    dedicated news backend, otherwise a general web backend (DDG → Tavily +
    Exa), and returns formatted, deduplicated results with source URLs. Skips
    DuckDuckGo automatically after repeated blocking failures.
    """
    clean = (query or "").strip()
    if not clean:
        return "Vui lòng nhập truy vấn tìm kiếm hợp lệ."

    is_news = _looks_like_news_query(clean)
    result = _web_search_run(clean, max_results, is_news)
    max_links = min(3, max(1, max_results)) if is_news else min(5, max(1, max_results))
    return _append_page_evidence(result, clean, max_links=max_links)


def _get_yahoo_stock_meta(ticker: str) -> Optional[dict[str, Any]]:
    symbol = re.sub(r"[^A-Z0-9.^=-]", "", (ticker or "").strip().upper())
    if not symbol:
        return None

    response = _session.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{quote_plus(symbol)}",
        params={"range": "1d", "interval": "1m"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=8,
    )
    response.raise_for_status()
    result = (response.json().get("chart", {}).get("result") or [None])[0]
    if not result:
        return None

    meta = result.get("meta") or {}
    price = meta.get("regularMarketPrice")
    if price is None:
        closes = (((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or [])
        price = next((value for value in reversed(closes) if value is not None), None)
    if price is None:
        return None
    meta["regularMarketPrice"] = price
    return meta


def _format_money(value: float, currency: str) -> str:
    decimals = 0 if currency in {"VND", "JPY", "KRW", "IDR"} else 2
    return f"{value:,.{decimals}f} {currency}".strip()


def _get_yahoo_stock_quote(ticker: str) -> Optional[str]:
    symbol = re.sub(r"[^A-Z0-9.^=-]", "", (ticker or "").strip().upper())
    meta = _get_yahoo_stock_meta(symbol)
    if not meta:
        return None

    price = meta.get("regularMarketPrice")
    if price is None:
        return None

    currency = meta.get("currency") or ""
    exchange = meta.get("fullExchangeName") or meta.get("exchangeName") or ""
    market_ts = meta.get("regularMarketTime")
    market_time = ""
    if market_ts:
        market_time = datetime.fromtimestamp(int(market_ts), _VN_TZ).strftime("%d/%m/%Y %H:%M")

    previous = meta.get("chartPreviousClose") or meta.get("previousClose")
    change = ""
    if previous:
        delta = float(price) - float(previous)
        percent = (delta / float(previous)) * 100 if float(previous) else 0
        change = f", thay \u0111\u1ed5i {delta:+,.2f} ({percent:+.2f}%) so v\u1edbi gi\u00e1 \u0111\u00f3ng c\u1eeda tr\u01b0\u1edbc"

    fetched_at = datetime.now(_VN_TZ).strftime("%d/%m/%Y %H:%M")
    time_part = f"c\u1eadp nh\u1eadt th\u1ecb tr\u01b0\u1eddng {market_time}" if market_time else f"l\u1ea5y l\u00fac {fetched_at}"
    venue = f" tr\u00ean {exchange}" if exchange else ""
    source = f"https://finance.yahoo.com/quote/{quote_plus(symbol)}"
    company = meta.get("longName") or meta.get("shortName") or ""
    prefix = f"{company} ({symbol})" if company and company.upper() != symbol else symbol
    return (
        f"{prefix}{venue}: {_format_money(float(price), currency)} ({time_part}, gi\u1edd VN{change}).\n"
        f"Ngu\u1ed3n: Yahoo Finance - {source}"
    )


def _looks_like_stock_symbol(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9]{1,8}(?:\.[A-Z]{1,4})?", (text or "").strip().upper()))


def _normalize_ascii_words(text: str) -> str:
    lowered = (text or "").lower().replace("đ", "d")
    no_marks = "".join(
        ch for ch in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(ch) != "Mn"
    )
    return re.sub(r"[^a-z0-9\s]", " ", no_marks)


def _clean_stock_lookup_query(query: str) -> str:
    filler = {
        "gia", "co", "phieu", "chung", "khoan", "hom", "nay", "la", "bao", "nhieu",
        "cua", "tap", "doan", "cong", "ty", "ma", "ticker", "symbol", "stock", "price",
        "latest", "today", "hien", "tai", "bao", "nhieu",
    }
    original_tokens = re.findall(r"[\w.]+", query or "", flags=re.UNICODE)
    kept: list[str] = []
    for token in original_tokens:
        normalized = _normalize_ascii_words(token).strip()
        if not normalized or normalized in filler:
            continue
        kept.append(token)
    return " ".join(kept).strip()


def _stock_lookup_variants(*queries: str) -> list[str]:
    variants: list[str] = []
    for query in queries:
        clean = re.sub(r"\s+", " ", (query or "").strip())
        ascii_clean = re.sub(r"\s+", " ", _normalize_ascii_words(clean)).strip()
        compact_ascii = ascii_clean.replace(" ", "")
        for variant in (clean, ascii_clean, compact_ascii):
            if variant and variant not in variants:
                variants.append(variant)
    return variants


_STOCK_COMPANY_WORDS = {
    "corp", "corporation", "company", "co", "inc", "ltd", "limited", "group",
    "holdings", "holding", "jsc", "joint", "stock", "capital", "plc", "sa",
}


def _stock_text_tokens(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", _normalize_ascii_words(text or "")))
    meaningful = {token for token in tokens if token not in _STOCK_COMPANY_WORDS and len(token) > 1}
    return meaningful or {token for token in tokens if len(token) > 1}


def _stock_market_suffix_hints(query: str) -> list[str]:
    normalized = _normalize_ascii_words(query or "")
    rules = [
        (("viet nam", "vietnam", "hose", "hnx", "upcom", "vnd"), ["VN"]),
        (("russia", "nga", "moscow", "moex", "rub", "ruble"), ["ME"]),
        (("australia", "asx", "aud"), ["AX"]),
        (("london", "lse", "uk", "gbp"), ["L"]),
        (("canada", "tsx", "cad"), ["TO", "V"]),
        (("japan", "tokyo", "jpy"), ["T"]),
        (("hong kong", "hkex", "hkd"), ["HK"]),
    ]
    hints: list[str] = []
    for keywords, suffixes in rules:
        if any(keyword in normalized for keyword in keywords):
            hints.extend(suffix for suffix in suffixes if suffix not in hints)
    return hints


def _score_stock_match(query: str, item: dict[str, Any], suffix_hints: list[str]) -> int:
    symbol = str(item.get("symbol") or "").strip().upper()
    if not symbol:
        return -100
    query_tokens = _stock_text_tokens(query)
    item_text = " ".join(
        str(item.get(key) or "")
        for key in ("symbol", "shortname", "shortName", "longname", "longName", "displayName", "name")
    )
    item_tokens = _stock_text_tokens(item_text)
    base_symbol = symbol.split(".", 1)[0].lower()
    score = 0
    if base_symbol in query_tokens:
        score += 80
    overlap = query_tokens & item_tokens
    if query_tokens:
        score += int(80 * len(overlap) / len(query_tokens))
        if query_tokens <= item_tokens:
            score += 40
    query_phrase = " ".join(sorted(query_tokens))
    item_phrase = " ".join(sorted(item_tokens))
    if query_phrase and query_phrase in item_phrase:
        score += 30
    suffix = symbol.rsplit(".", 1)[1] if "." in symbol else ""
    if suffix_hints:
        if suffix in suffix_hints:
            score += 20
        elif suffix:
            score -= 15
    if not overlap and base_symbol not in query_tokens:
        score -= 60
    return score


def _resolve_yahoo_stock_symbol(query: str, context_query: str = "") -> Optional[str]:
    clean = re.sub(r"\s+", " ", (query or "").strip())
    if not clean:
        return None
    response = _session.get(
        "https://query1.finance.yahoo.com/v1/finance/search",
        params={"q": clean, "quotes_count": 8, "news_count": 0, "enableFuzzyQuery": "true"},
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
        timeout=8,
    )
    response.raise_for_status()
    quotes = response.json().get("quotes") or []
    suffix_hints = _stock_market_suffix_hints(f"{context_query} {clean}")
    best_symbol = None
    best_score = 0
    for item in quotes:
        symbol = str(item.get("symbol") or "").strip().upper()
        quote_type = str(item.get("quoteType") or item.get("typeDisp") or "").upper()
        if not symbol or quote_type not in {"EQUITY", "ETF"}:
            continue
        score = _score_stock_match(clean, item, suffix_hints)
        if score > best_score:
            best_symbol = symbol
            best_score = score
    return best_symbol if best_score >= 60 else None


def _stock_symbols_from_text(text: str, suffix_hints: list[str]) -> list[str]:
    found: list[str] = []

    def add(symbol: str) -> None:
        symbol = re.sub(r"[^A-Z0-9.]", "", (symbol or "").upper())
        if not symbol or symbol in found:
            return
        found.append(symbol)

    for match in re.finditer(r"\b([A-Z0-9]{1,8})\.(VN|ME|AX|L|TO|V|T|HK|SS|SZ|KS|KQ)\b", text):
        add(f"{match.group(1)}.{match.group(2)}")

    exchange_suffix = {
        "HOSE": "VN", "HSX": "VN", "HNX": "VN", "UPCOM": "VN",
        "MOEX": "ME", "MCX": "ME", "ASX": "AX", "LSE": "L", "TSX": "TO", "HKEX": "HK",
    }
    exchange_pattern = r"\b(HOSE|HSX|HNX|UPCOM|MOEX|MCX|ASX|LSE|TSX|HKEX)\s*[:\-]?\s*([A-Z0-9]{1,8})\b"
    for exchange, symbol in re.findall(exchange_pattern, text, flags=re.IGNORECASE):
        add(f"{symbol}.{exchange_suffix[exchange.upper()]}")

    marker_pattern = r"\b(?:ticker|symbol|stock code|m[aã]\s*c[oổ]\s*phi[eế]u)\s*[:\-]?\s*([A-Z0-9]{1,8})\b"
    for symbol in re.findall(marker_pattern, text, flags=re.IGNORECASE):
        for suffix in suffix_hints:
            add(f"{symbol}.{suffix}")
        add(symbol)

    for symbol in re.findall(r"\(([A-Z0-9]{2,8})\)", text):
        if len(symbol) < 2:
            continue
        for suffix in suffix_hints:
            add(f"{symbol}.{suffix}")
        add(symbol)

    return found


def _resolve_stock_symbol_from_web(lookup_query: str, context_query: str) -> Optional[str]:
    suffix_hints = _stock_market_suffix_hints(f"{context_query} {lookup_query}")
    query_tokens = _stock_text_tokens(lookup_query)
    compact_query = _normalize_ascii_words(lookup_query).replace(" ", "")
    try:
        evidence = web_search(f'"{lookup_query}" stock ticker symbol exchange Yahoo Finance', max_results=5)
    except Exception as exc:
        logger.warning("Stock symbol web resolve failed for %s: %s", lookup_query, exc)
        return None

    normalized_evidence = _normalize_ascii_words(evidence)
    best_symbol = None
    best_score = 0
    for symbol in _stock_symbols_from_text(evidence, suffix_hints):
        base_symbol = symbol.split(".", 1)[0]
        if "." not in symbol and len(base_symbol) <= 2:
            continue
        try:
            meta = _get_yahoo_stock_meta(symbol)
        except Exception:
            continue
        if not meta:
            continue
        item = {
            "symbol": symbol,
            "shortName": meta.get("shortName"),
            "longName": meta.get("longName"),
        }
        score = _score_stock_match(lookup_query, item, suffix_hints)
        symbol_key = symbol.lower().replace(".", " ")
        symbol_url_key = f"finance yahoo com quote {symbol_key}"
        symbol_pattern = rf"(?<![a-z0-9]){re.escape(symbol_key)}(?![a-z0-9])"
        for match in re.finditer(symbol_pattern, normalized_evidence):
            start = max(0, match.start() - 160)
            end = min(len(normalized_evidence), match.end() + 160)
            nearby = normalized_evidence[start:end]
            nearby_tokens = set(re.findall(r"[a-z0-9]+", nearby))
            if query_tokens & nearby_tokens:
                score += 90
            if compact_query and compact_query in nearby.replace(" ", ""):
                score += 90
            if symbol_url_key in nearby:
                score += 40
        if score > best_score:
            best_symbol = symbol
            best_score = score
    return best_symbol if best_score >= 60 else None


def get_stock_price(ticker: str) -> str:
    """Resolve a company name or ticker dynamically, then fetch its latest quote."""
    clean_query = re.sub(r"\s+", " ", (ticker or "").strip())
    lookup_query = _clean_stock_lookup_query(clean_query)
    if not clean_query:
        return "Vui l\u00f2ng cung c\u1ea5p m\u00e3 c\u1ed5 phi\u1ebfu h\u1ee3p l\u1ec7."

    candidates: list[str] = []
    explicit = extract_stock_ticker(clean_query)
    if explicit:
        candidates.append(explicit)
        if "." not in explicit and re.fullmatch(r"[A-Z]{2,6}", explicit):
            candidates.append(f"{explicit}.VN")
    if _looks_like_stock_symbol(clean_query.upper()):
        candidates.append(clean_query.upper())

    resolve_queries = [candidate for candidate in candidates]
    resolve_queries.extend(_stock_lookup_variants(lookup_query, clean_query))
    for resolve_query in resolve_queries:
        try:
            resolved = _resolve_yahoo_stock_symbol(resolve_query, clean_query)
            if resolved:
                candidates.append(resolved)
        except Exception as exc:
            logger.warning("Yahoo symbol resolve failed for %s: %s", resolve_query, exc)
    if lookup_query:
        resolved = _resolve_stock_symbol_from_web(lookup_query, clean_query)
        if resolved:
            candidates.append(resolved)

    seen: set[str] = set()
    for symbol in candidates:
        if symbol in seen:
            continue
        seen.add(symbol)
        try:
            direct_quote = _get_yahoo_stock_quote(symbol)
            if direct_quote:
                return direct_quote
        except Exception as exc:
            logger.warning("Yahoo stock quote failed for %s: %s", symbol, exc)

    return f"M\u00ecnh ch\u01b0a x\u00e1c \u0111\u1ecbnh \u0111\u01b0\u1ee3c m\u00e3 giao d\u1ecbch ch\u00ednh x\u00e1c cho '{clean_query}', n\u00ean ch\u01b0a th\u1ec3 tr\u1ea3 gi\u00e1 c\u1ed5 phi\u1ebfu \u0111\u00e1ng tin c\u1eady."


# ══════════════════════════════════════════════════════════════════════════════
# MUSIC PLAYER TOOL
# ══════════════════════════════════════════════════════════════════════════════
# Opens music in the user's NORMAL browser (default profile, with login/extensions).
# Window control (dedicated window for smoother behavior):
#   - Play:   Ctrl+N to open a NEW Chromium window, then navigate to URL
#   - Switch: close current music window + open a NEW one
#   - Stop:   close the music window
#   - Pause:  media key (OS-level, toggles pause/play)
#   - Resume: media key (same toggle)

_music_lock = threading.Lock()
_music_is_active: bool = False
_music_is_paused: bool = False
_music_song_name: Optional[str] = None
_music_window_hwnd: Optional[int] = None
_music_window_hwnds: List[int] = []


def _search_youtube_url(query: str) -> Optional[str]:
    """Search YouTube for a song and return the first video URL."""
    try:
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        resp = _session.get(
            search_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36"
                ),
                "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        match = re.search(r'"videoId"\s*:\s*"([a-zA-Z0-9_-]{11})"', resp.text)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    except Exception:
        pass
    return None


def _resolve_music_url(query: str) -> tuple[str, bool]:
    """Resolve a music query to a URL. Returns (url, is_direct_match)."""
    if query.startswith((HTTP_SCHEME, HTTPS_SCHEME, "www.")):
        return query, True
    found_url = _search_youtube_url(query)
    if found_url:
        return found_url, True
    return f"https://www.youtube.com/results?search_query={quote_plus(query)}", False


def _find_browser_exe() -> Optional[str]:
    """Find Edge or Chrome executable on Windows."""
    import platform

    if platform.system() != "Windows":
        return None

    candidates: List[str] = []
    for env in ["ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"]:
        base = os.environ.get(env, "")
        if base:
            candidates.extend([
                os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                os.path.join(base, "Google", "Chrome", "Application", "chrome.exe"),
            ])

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def _get_window_process_name(hwnd: int) -> str:
    """Return lowercase process executable name for a window handle."""
    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(int(hwnd), ctypes.byref(pid))
        if not pid.value:
            return ""

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        process_handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            int(pid.value),
        )
        if not process_handle:
            return ""

        try:
            buf_len = ctypes.wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(buf_len.value)
            if not kernel32.QueryFullProcessImageNameW(
                process_handle,
                0,
                buf,
                ctypes.byref(buf_len),
            ):
                return ""
            return os.path.basename(buf.value).lower()
        finally:
            kernel32.CloseHandle(process_handle)
    except Exception:
        return ""


def _is_chromium_window(hwnd: Optional[int]) -> bool:
    """Return True only for Chromium top-level browser windows."""
    if not hwnd:
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd_i = int(hwnd)
        if not user32.IsWindow(hwnd_i):
            return False

        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd_i, class_buf, 256)
        if class_buf.value != "Chrome_WidgetWin_1":
            return False

        return _get_window_process_name(hwnd_i) in {"chrome.exe", "msedge.exe", "brave.exe"}
    except Exception:
        return False


def _get_chromium_windows() -> List[int]:
    """Get visible top-level Chromium windows (Chrome/Edge/Brave)."""
    import platform

    if platform.system() != "Windows":
        return []

    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.wintypes.HWND,
            ctypes.wintypes.LPARAM,
        )

        windows: List[int] = []

        def _enum_callback(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd) and _is_chromium_window(int(hwnd)):
                windows.append(int(hwnd))
            return len(windows) < 10000

        user32.EnumWindows(WNDENUMPROC(_enum_callback), 0)
        return windows
    except Exception:
        return []


def _get_foreground_hwnd() -> Optional[int]:
    """Return current foreground window handle."""
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        return int(hwnd) if hwnd else None
    except Exception:
        return None


def _set_clipboard_text(text: str) -> bool:
    """Set Unicode text to clipboard."""
    try:
        import ctypes
        import ctypes.wintypes

        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        msvcrt = ctypes.cdll.msvcrt

        if not user32.OpenClipboard(None):
            return False

        try:
            user32.EmptyClipboard()
            data = text + "\0"
            data_bytes = data.encode("utf-16-le")
            byte_count = len(data_bytes)

            hglobal = kernel32.GlobalAlloc(GMEM_MOVEABLE, byte_count)
            if not hglobal:
                return False

            locked = kernel32.GlobalLock(hglobal)
            if not locked:
                kernel32.GlobalFree(hglobal)
                return False

            try:
                msvcrt.memcpy(locked, data_bytes, byte_count)
            finally:
                kernel32.GlobalUnlock(hglobal)

            if not user32.SetClipboardData(CF_UNICODETEXT, hglobal):
                kernel32.GlobalFree(hglobal)
                return False
            return True
        finally:
            user32.CloseClipboard()
    except Exception:
        return False


def _send_ctrl_key(hwnd: int, vk_key: int) -> bool:
    """Send Ctrl+<key> to a specific window."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetForegroundWindow(int(hwnd))
        time.sleep(0.08)

        VK_CONTROL = 0x11
        user32.keybd_event(VK_CONTROL, 0, 0, 0)
        user32.keybd_event(vk_key, 0, 0, 0)
        time.sleep(0.03)
        user32.keybd_event(vk_key, 0, 2, 0)
        user32.keybd_event(VK_CONTROL, 0, 2, 0)
        return True
    except Exception:
        return False


def _send_enter(hwnd: int) -> bool:
    """Send Enter key to a specific window."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.SetForegroundWindow(int(hwnd))
        time.sleep(0.05)

        VK_RETURN = 0x0D
        user32.keybd_event(VK_RETURN, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(VK_RETURN, 0, 2, 0)
        return True
    except Exception:
        return False


def _navigate_chromium_window_to_url(hwnd: int, url: str) -> bool:
    """Navigate a Chromium window to URL via Ctrl+L, Ctrl+V, Enter."""
    if not _send_ctrl_key(hwnd, 0x4C):  # Ctrl+L
        return False
    if not _set_clipboard_text(url):
        return False
    if not _send_ctrl_key(hwnd, 0x56):  # Ctrl+V
        return False
    return _send_enter(hwnd)


def _find_window_for_pid(pid: int, timeout_seconds: float = 4.0) -> Optional[int]:
    """Find a visible top-level window belonging to the given PID."""
    import platform

    if platform.system() != "Windows":
        return None

    try:
        import ctypes
        import ctypes.wintypes

        user32 = ctypes.windll.user32
        WNDENUMPROC = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.wintypes.HWND,
            ctypes.wintypes.LPARAM,
        )

        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            found_hwnd: list[Optional[int]] = [None]

            def _enum_callback(hwnd, _lparam, found_hwnd_ref=found_hwnd):
                if not user32.IsWindowVisible(hwnd):
                    return True
                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True

                proc_id = ctypes.wintypes.DWORD()
                user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc_id))
                if int(proc_id.value) != int(pid):
                    return True

                found_hwnd_ref[0] = int(hwnd)
                return False

            user32.EnumWindows(WNDENUMPROC(_enum_callback), 0)
            if found_hwnd[0] is not None:
                return found_hwnd[0]
            time.sleep(0.1)
    except Exception:
        return None

    return None


def _wait_for_new_chromium_window(old_set: set[int], timeout_seconds: float = 3.0) -> Optional[int]:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        current = _get_chromium_windows()
        created = [h for h in current if h not in old_set]
        if created:
            return created[-1]
        time.sleep(0.08)
    return None


def _try_open_music_via_ctrl_n(url: str, existing_windows: list[int]) -> tuple[bool, Optional[int]]:
    if not existing_windows:
        return False, None

    foreground = _get_foreground_hwnd()
    base_hwnd = foreground if foreground in existing_windows else existing_windows[0]
    if not _send_ctrl_key(base_hwnd, 0x4E):
        return False, None

    new_hwnd = _wait_for_new_chromium_window(set(existing_windows))
    if not new_hwnd:
        return False, None
    if not _navigate_chromium_window_to_url(new_hwnd, url):
        return False, None
    return True, new_hwnd


def _try_open_music_via_browser_exe(url: str) -> tuple[bool, Optional[int]]:
    browser = _find_browser_exe()
    if not browser:
        return False, None

    old_set = set(_get_chromium_windows())
    try:
        proc = subprocess.Popen(
            [browser, "--new-window", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        hwnd = _find_window_for_pid(proc.pid)
        if hwnd:
            return True, hwnd
        return True, _wait_for_new_chromium_window(old_set)
    except Exception:
        return False, None


def _open_in_music_window(url: str) -> tuple[bool, Optional[int]]:
    """Open music in NEW window using Ctrl+N on Chromium browsers."""
    existing_windows = _get_chromium_windows()

    opened, hwnd = _try_open_music_via_ctrl_n(url, existing_windows)
    if opened:
        return True, hwnd

    opened, hwnd = _try_open_music_via_browser_exe(url)
    if opened:
        return True, hwnd

    try:
        os.startfile(url)  # type: ignore[attr-defined]
        return True, None
    except Exception:
        return False, None


def _is_window_alive(hwnd: Optional[int]) -> bool:
    """Check if a native window handle is still valid/alive."""
    if not hwnd:
        return False

    try:
        import ctypes
        return bool(ctypes.windll.user32.IsWindow(int(hwnd)))
    except Exception:
        return False


def _window_title(hwnd: int) -> str:
    """Return lowercase window title for matching."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        length = user32.GetWindowTextLengthW(int(hwnd))
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(int(hwnd), buf, length + 1)
        return buf.value.lower()
    except Exception:
        return ""


def _find_youtube_windows() -> List[int]:
    """Find visible Chromium windows that look like YouTube pages."""
    result: List[int] = []
    for hwnd in _get_chromium_windows():
        title = _window_title(hwnd)
        if "youtube" in title:
            result.append(hwnd)
    return result


def _normalize_match_text(text: str) -> str:
    """Normalize text for loose title matching."""
    lowered = (text or "").lower().strip().replace("đ", "d")
    lowered = "".join(
        ch for ch in unicodedata.normalize("NFD", lowered)
        if unicodedata.category(ch) != "Mn"
    )
    lowered = re.sub(r"[^0-9a-z\s]", " ", lowered)
    return " ".join(lowered.split())


def _build_song_match_keys(song_hint: str | None = None) -> List[str]:
    """Build normalized keyword candidates for title matching."""
    keys: List[str] = []
    for raw in [_music_song_name, song_hint]:
        key = _normalize_match_text(raw or "")
        if key and key not in keys:
            keys.append(key)
    return keys


def _remember_music_window(hwnd: Optional[int]) -> None:
    """Track a music window handle for later close operations."""
    global _music_window_hwnds

    if not hwnd or not _is_chromium_window(hwnd):
        return

    hwnd_i = int(hwnd)
    if hwnd_i not in _music_window_hwnds:
        _music_window_hwnds.append(hwnd_i)


def _cleanup_tracked_music_windows() -> None:
    """Drop invalid/closed/non-Chromium tracked windows."""
    global _music_window_hwnds

    _music_window_hwnds = [
        h for h in _music_window_hwnds
        if _is_window_alive(h) and _is_chromium_window(h)
    ]


def _is_youtube_window(hwnd: Optional[int]) -> bool:
    """Return True for tracked music Chromium windows or title-indicated YouTube windows."""
    if not hwnd or not _is_window_alive(hwnd) or not _is_chromium_window(hwnd):
        return False

    hwnd_i = int(hwnd)
    if (_music_window_hwnd and hwnd_i == int(_music_window_hwnd)) or hwnd_i in _music_window_hwnds:
        return True

    title = _window_title(hwnd_i)
    return "youtube" in title


def _collect_music_window_candidates() -> List[int]:
    base_candidates: List[int] = []
    if _music_window_hwnd and _is_youtube_window(_music_window_hwnd):
        base_candidates.append(int(_music_window_hwnd))

    for hwnd in _music_window_hwnds:
        if _is_youtube_window(hwnd) and hwnd not in base_candidates:
            base_candidates.append(hwnd)
    return base_candidates


def _score_music_window_candidate(hwnd: int, index: int, match_keys: List[str]) -> int:
    score = 0
    if _music_window_hwnd and hwnd == _music_window_hwnd:
        score += 1000

    title_key = _normalize_match_text(_window_title(hwnd))
    for key in match_keys:
        if not key or key not in title_key:
            continue
        score += 120
        if title_key.startswith(key):
            score += 20

    score += max(0, 50 - index)
    return score


def _rank_music_window_candidates(song_hint: str | None = None) -> List[int]:
    """Rank close candidates with deterministic priority and song-title scoring."""
    _cleanup_tracked_music_windows()
    base_candidates = _collect_music_window_candidates()
    match_keys = _build_song_match_keys(song_hint)

    scored = [
        (_score_music_window_candidate(hwnd, idx, match_keys), hwnd)
        for idx, hwnd in enumerate(base_candidates)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [hwnd for _score, hwnd in scored]


def _send_alt_f4_to_hwnd(hwnd: int) -> bool:
    """Close a specific Chromium window by hwnd without global keyboard shortcuts."""
    import platform

    if platform.system() != "Windows":
        return False

    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd_i = int(hwnd)
        if not user32.IsWindow(hwnd_i):
            return True
        if not _is_chromium_window(hwnd_i):
            return False

        WM_SYSCOMMAND = 0x0112
        SC_CLOSE = 0xF060
        WM_CLOSE = 0x0010

        user32.PostMessageW(hwnd_i, WM_SYSCOMMAND, SC_CLOSE, 0)

        deadline = time.time() + 1.1
        while time.time() < deadline:
            if not user32.IsWindow(hwnd_i):
                return True
            time.sleep(0.04)

        user32.PostMessageW(hwnd_i, WM_CLOSE, 0, 0)

        deadline = time.time() + 1.1
        while time.time() < deadline:
            if not user32.IsWindow(hwnd_i):
                return True
            time.sleep(0.04)

        return not bool(user32.IsWindow(hwnd_i))
    except Exception:
        return False


def _is_music_window_match(hwnd: int, tracked_set: set[int], match_keys: List[str]) -> bool:
    if hwnd in tracked_set:
        return True
    if not match_keys:
        return False
    title_key = _normalize_match_text(_window_title(hwnd))
    return any(key in title_key for key in match_keys)


def _sync_tracked_music_window_state() -> None:
    global _music_window_hwnd, _music_window_hwnds

    _cleanup_tracked_music_windows()
    _music_window_hwnds = [h for h in _music_window_hwnds if _is_window_alive(h)]
    if _music_window_hwnd and not _is_window_alive(_music_window_hwnd):
        _music_window_hwnd = None


def _close_current_or_matching_music_window(
    song_hint: str | None = None,
    close_all_matches: bool = False,
) -> bool:
    """Close tracked/song-matched music windows; optionally close all matched ones."""
    global _music_window_hwnd, _music_window_hwnds

    candidates = _rank_music_window_candidates(song_hint=song_hint)
    if not candidates:
        return False

    match_keys = _build_song_match_keys(song_hint)
    tracked_set = set(_music_window_hwnds)
    if _music_window_hwnd:
        tracked_set.add(int(_music_window_hwnd))

    closed = False
    for hwnd in candidates:
        if not _is_music_window_match(hwnd, tracked_set, match_keys):
            continue
        if not _send_alt_f4_to_hwnd(hwnd):
            continue

        closed = True
        if _music_window_hwnd and hwnd == _music_window_hwnd:
            _music_window_hwnd = None
        if not close_all_matches:
            break

    _sync_tracked_music_window_state()
    return closed


def _close_other_tracked_music_windows(active_hwnd: int | None) -> None:
    """Close any leftover tracked windows except the current active one."""
    global _music_window_hwnds

    _cleanup_tracked_music_windows()
    keep = int(active_hwnd) if active_hwnd else None

    for hwnd in _music_window_hwnds:
        if keep and hwnd == keep:
            continue
        if _is_window_alive(hwnd):
            _send_alt_f4_to_hwnd(hwnd)

    _cleanup_tracked_music_windows()
    _music_window_hwnds = [
        h for h in _music_window_hwnds
        if _is_window_alive(h) and (not keep or h == keep)
    ]


def _reset_music_state() -> None:
    """Reset all in-memory music states."""
    global _music_is_active, _music_is_paused, _music_song_name, _music_window_hwnd, _music_window_hwnds

    _music_is_active = False
    _music_is_paused = False
    _music_song_name = None
    _music_window_hwnd = None
    _music_window_hwnds = []


def _refresh_music_state_from_window() -> bool:
    """Sync state with real window lifecycle; return True if still active."""
    global _music_is_active, _music_is_paused, _music_song_name, _music_window_hwnd, _music_window_hwnds

    if not _music_is_active:
        return False

    _cleanup_tracked_music_windows()

    if _music_window_hwnd and _is_window_alive(_music_window_hwnd):
        _remember_music_window(_music_window_hwnd)
        return True

    if _music_window_hwnds:
        _music_window_hwnd = _music_window_hwnds[0]
        return True

    _music_is_active = False
    _music_is_paused = False
    _music_song_name = None
    _music_window_hwnd = None
    return False


def play_music(url_or_query: str) -> str:
    """Play music from a URL or a YouTube search query."""
    global _music_is_active, _music_song_name, _music_is_paused, _music_window_hwnd

    query = url_or_query.strip()
    if not query:
        return "Bạn cho mình biết tên bài hát hoặc gửi link nhạc nhé."

    url, is_direct = _resolve_music_url(query)

    if not query.startswith((HTTP_SCHEME, HTTPS_SCHEME)):
        next_song_name = query
    else:
        next_song_name = "nhạc"

    with _music_lock:
        _refresh_music_state_from_window()

        if _music_is_active:
            closed_old = _close_current_or_matching_music_window(
                song_hint=_music_song_name,
                close_all_matches=False,
            )
            if not closed_old:
                return (
                    "Mình chưa đóng được đúng bài cũ nên chưa thể chuyển bài chính xác. "
                    "Bạn thử lại giúp mình nhé."
                )
            _reset_music_state()
            time.sleep(0.2)

        windows_before_open = set(_get_chromium_windows())
        opened, music_hwnd = _open_in_music_window(url)
        if not opened:
            return "Mình chưa mở được nhạc lúc này. Bạn thử lại sau nhé."

        if not music_hwnd:
            deadline = time.time() + 2.5
            while time.time() < deadline:
                current = _get_chromium_windows()
                created = [h for h in current if h not in windows_before_open]
                if created:
                    music_hwnd = created[-1]
                    break
                time.sleep(0.08)

        if not music_hwnd:
            return (
                "Mình đã gửi lệnh mở nhạc nhưng chưa nhận diện được đúng cửa sổ mới, "
                "nên dừng an toàn để tránh tắt nhầm tab khác. Bạn thử lại giúp mình nhé."
            )

        _music_is_active = True
        _music_is_paused = False
        _music_song_name = next_song_name
        _music_window_hwnd = music_hwnd
        _remember_music_window(music_hwnd)

        if is_direct:
            return (
                f"Đang mở bài {_music_song_name} cho bạn trong cửa sổ mới. "
                "Bạn có thể nói tạm dừng nhạc, tiếp tục phát, tắt nhạc hoặc chuyển bài nhé."
            )

        return "Mình đã mở YouTube trong cửa sổ mới cho bạn. Hãy chọn bài hát bạn muốn nghe nhé."


def _send_media_key_pause() -> bool:
    """Send media play/pause key on Windows. This is a TOGGLE: pause↔play."""
    import platform

    if platform.system() != "Windows":
        return False

    try:
        import ctypes
        VK_MEDIA_PLAY_PAUSE = 0xB3
        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_MEDIA_PLAY_PAUSE, 0, 2, 0)
        return True
    except Exception:
        return False


def _close_music_target_window() -> bool:
    global _music_window_hwnd

    if not (_music_window_hwnd and _is_youtube_window(_music_window_hwnd)):
        return False
    closed = _send_alt_f4_to_hwnd(int(_music_window_hwnd))
    if closed:
        _music_window_hwnd = None
    return closed


def _close_ranked_music_match(song_hint: str | None, match_keys: List[str]) -> bool:
    global _music_window_hwnd

    for hwnd in _rank_music_window_candidates(song_hint=song_hint):
        if not _is_youtube_window(hwnd):
            continue

        title_key = _normalize_match_text(_window_title(hwnd))
        if match_keys and not any(key in title_key for key in match_keys):
            continue
        if not _send_alt_f4_to_hwnd(hwnd):
            continue

        if _music_window_hwnd and hwnd == _music_window_hwnd:
            _music_window_hwnd = None
        return True

    return False


def stop_music(song_hint: str | None = None) -> str:
    """Turn music OFF by closing the tracked music window (TẮT NHẠC / ĐÓNG NHẠC).

    Fully ends playback and closes the window. Use for 'tắt nhạc', 'đóng nhạc',
    'dừng hẳn', 'tắt bài hát'. For a temporary mid-song pause ('tạm dừng nhạc',
    'dừng nhạc', 'ngừng phát giữa chừng'), call pause_music() instead.
    """
    with _music_lock:
        if not _refresh_music_state_from_window():
            return "Hiện không có nhạc đang phát."

        song = song_hint or _music_song_name or "nhạc"
        match_keys = _build_song_match_keys(song_hint)

        closed = _close_music_target_window() or _close_ranked_music_match(song_hint, match_keys)
        if closed:
            _reset_music_state()
            return f"Đã tắt và đóng cửa sổ nhạc bài {song}."

        return (
            f"Mình chưa xác định được đúng cửa sổ YouTube của bài {song} để tắt an toàn. "
            "Bạn thử nói rõ tên bài hoặc đóng thủ công giúp mình nhé."
        )


def pause_music() -> str:
    """Pause the current song mid-play without closing the window (TẠM DỪNG NHẠC / DỪNG NHẠC).

    Interrupts playback at the current point; the window stays open and can be
    resumed. Use for 'tạm dừng nhạc', 'dừng nhạc', 'ngừng phát', 'dừng bài hát
    giữa chừng'. To fully close the window ('tắt nhạc', 'đóng nhạc'), call
    stop_music() instead.
    """
    global _music_is_paused

    with _music_lock:
        if not _refresh_music_state_from_window():
            return "Hiện không có nhạc đang phát để tạm dừng."

        if _music_is_paused:
            song = _music_song_name or "nhạc"
            return f"Bài {song} đang tạm dừng rồi. Nói tiếp tục phát để nghe tiếp nhé."

        if _send_media_key_pause():
            _music_is_paused = True
            song = _music_song_name or "nhạc"
            return f"Đã tạm dừng bài {song}. Nói tiếp tục phát nhạc để nghe tiếp nhé."

        return "Mình chưa tạm dừng được. Bạn bấm nút pause trên trình duyệt giúp mình nhé."


def resume_music() -> str:
    """Resume a paused song in the tracked music window (TIẾP TỤC PHÁT / PHÁT TIẾP)."""
    global _music_is_paused

    with _music_lock:
        if not _refresh_music_state_from_window():
            return "Hiện không có nhạc nào để tiếp tục phát. Nói mở nhạc kèm tên bài hát nhé."

        if not _music_is_paused:
            song = _music_song_name or "nhạc"
            return f"Bài {song} đang phát rồi mà."

        if _send_media_key_pause():
            _music_is_paused = False
            song = _music_song_name or "nhạc"
            return (
                f"Đã tiếp tục phát bài {song}. "
                "Bạn có thể nói tạm dừng nhạc, tắt nhạc hoặc chuyển bài nhé."
            )

        return "Mình chưa tiếp tục phát được. Bạn bấm nút play trên trình duyệt giúp mình nhé."


def is_music_active() -> bool:
    """Check whether music is currently playing/paused in tracked music window."""
    with _music_lock:
        return _refresh_music_state_from_window()

