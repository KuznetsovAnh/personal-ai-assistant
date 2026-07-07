"""Personal Assistant Agent powered by Pydantic-AI.

This module wires together:
- A pydantic_ai.Agent with OpenAI-compatible provider
- Tool functions (weather, web search, memory, etc.)
- Memory system for conversation persistence
- Streaming + parallel processing for sub-10s responses

Routing strategy (fastest path first):
   1. Music       â†’ direct tool call, no LLM
   2. Stock       â†’ direct stock quote
   3. Weather     â†’ direct weather tool
   4. Memory save â†’ direct persist
   5. General     â†’ single LLM agent with tools
"""

from __future__ import annotations

import logging
import re
import time
import unicodedata
import difflib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core.settings import settings
from core.memory import AssistantMemory
from core.models import AssistantResponse
from core.tools import (
    get_weather, save_memory, web_search, _fetch_page_evidence,
    get_current_datetime, calculate,
    translate_text, knowledge_search,
    get_exchange_rate,
    play_music, stop_music, pause_music, resume_music,
    get_stock_price, extract_stock_ticker,
)

logger = logging.getLogger(__name__)

_VN_TZ = timezone(timedelta(hours=7))
_VN_WEEKDAYS = ["Thá»© Hai", "Thá»© Ba", "Thá»© TÆ°", "Thá»© NÄƒm", "Thá»© SÃ¡u", "Thá»© Báº£y", "Chá»§ Nháº­t"]


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# SYSTEM PROMPT
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _get_current_datetime_str() -> str:
    """Get current datetime string in Vietnamese for system prompt injection."""
    now = datetime.now(_VN_TZ)
    weekday = _VN_WEEKDAYS[now.weekday()]
    return (
        f"Thá»i Ä‘iá»ƒm hiá»‡n táº¡i: {now.strftime('%H:%M:%S')}, "
        f"{weekday}, ngÃ y {now.strftime('%d')} thÃ¡ng {now.strftime('%m')} nÄƒm {now.strftime('%Y')} "
        f"(giá» Viá»‡t Nam, UTC+7). "
        f"ÄÃ‚Y LÃ€ THá»œI GIAN THá»°C, KHÃ”NG PHáº¢I TÆ¯Æ NG LAI."
    )


SYSTEM_PROMPT_TEMPLATE = (
    "You are an intelligent personal AI assistant. Respond naturally in the user's language. "
    "{current_datetime} "
    "Use built-in knowledge only for stable facts. For current or time-sensitive "
    "information such as prices, stock quotes, news, office holders, events, weather, "
    "exchange rates, or post-April-2024 facts, call the appropriate tool first. "
    "Tool results are evidence, but do not paste raw tool output. Summarize it into a clean answer. "
    "Do not invent numbers, dates, names, banners, prices, or sources. "
    "When a tool has no useful data, say that clearly instead of guessing. "
    "Ignore search results that are clearly category pages, RSS feeds, videos, spam, or unrelated snippets. "
    "For news, use article-like results with concrete URLs; if there are not enough clean items, say how many you found. "
    "For prices or schedules, answer the exact user intent first, then add brief source context. "
    "For exchange-rate questions, infer the user's intended currencies from context, convert them to ISO 4217 codes, "
    "then call get_exchange_rate with those codes. If the user uses a nonstandard currency phrase but the country context is clear, "
    "use the country's official currency code; if it is ambiguous, ask for clarification instead of guessing. "
    "Keep citations/source links that are present, and only use facts found in the tool output. "
    "Avoid mechanical raw-search formats such as 'Káº¿t quáº£ cho ...' unless the user explicitly asks for raw search results. "
    "For music commands: play uses play_music; stop/close uses stop_music; pause uses pause_music; resume uses resume_music. "
    "Tools: get_weather, web_search, get_current_datetime, calculate, "
    "save_memory, translate_text, knowledge_search, get_exchange_rate, "
    "play_music, stop_music, pause_music, resume_music, get_stock_price."
)


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# KEYWORD TABLES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

KW_TIN_TUC = "tin tá»©c"
KW_TIN_MOI = "tin má»›i"
KW_TIN_NONG = "tin nÃ³ng"
KW_TIN_HOM_NAY = "tin hÃ´m nay"
KW_HOM_NAY = "hÃ´m nay"
KW_THE_GIOI = "tháº¿ giá»›i"
_WORLD_TOPIC_MARKERS = [KW_THE_GIOI, "quá»‘c táº¿", "world", "international"]
KW_TONG_THONG = "tá»•ng thá»‘ng"
KW_THU_TUONG = "thá»§ tÆ°á»›ng"
REGEX_NEWS_NUMBERED_ITEM = r'(?m)^\s*(?:[1-9]|1[0-5])\.\s+\S+'

WEATHER_KEYWORDS = [
    "thá»i tiáº¿t", "weather", "nhiá»‡t Ä‘á»™", "mÆ°a", "Ä‘á»™ áº©m", "giÃ³", "dá»± bÃ¡o",
]

_WEATHER_EXCLUDE_KEYWORDS = [
    KW_TIN_TUC, KW_TIN_MOI, KW_TIN_NONG, KW_TIN_HOM_NAY, "news",
    KW_THE_GIOI, "thá»i sá»±", "chÃ­nh trá»‹", "kinh táº¿",
]

_DIRECT_NEWS_KEYWORDS = [
    KW_TIN_TUC, "tin má»›i nháº¥t", KW_TIN_MOI, KW_TIN_NONG, KW_TIN_HOM_NAY,
    "tÃ³m táº¯t tin tá»©c", "tÃ³m táº¯t tin", "cÃ³ gÃ¬ má»›i", "hÃ´m nay cÃ³ gÃ¬",
    "news", "tin tá»©c hÃ´m nay",
]

_NEWS_FILLER_WORDS = [
    "láº¥y cho tÃ´i", "cho tÃ´i", "cho mÃ¬nh", "láº¥y", "tÃ¬m", "xem",
    "liá»‡t kÃª", "Ä‘Æ°a cho", "gá»­i cho", "ká»ƒ cho", "tÃ¬m cho",
    "get me", "give me", "show me", "find me", "tell me",
    "hÃ£y", "giÃºp", "giÃ¹m", "dÃ¹m", "Ä‘i", "nÃ o", "nhÃ©", "áº¡",
    "vÃ i", "má»™t sá»‘", "nhá»¯ng", "cÃ¡c", "cÃ¡i", "vá»", "táº¡i", "á»Ÿ",
    "hÃ´m nay", "má»›i nháº¥t", "hot nháº¥t", "hot",
    "today", "latest", "top", "most",
]

REALTIME_KEYWORDS = [
    # â”€â”€ GiÃ¡ cáº£ â”€â”€
    "giÃ¡ xÄƒng", "giÃ¡ vÃ ng", "giÃ¡ dáº§u", "giÃ¡ gas", "giÃ¡ Ä‘iá»‡n",
    "tá»· giÃ¡", "exchange rate",
    "bitcoin", "crypto", "BTC", "ETH",
    "giÃ¡ bao nhiÃªu", "bao nhiÃªu tiá»n",
    # â”€â”€ Chá»©ng khoÃ¡n â”€â”€
    "chá»©ng khoÃ¡n", "VN-Index", "VNIndex", "stock",
    "cá»• phiáº¿u", "mÃ£ chá»©ng khoÃ¡n", "thá»‹ trÆ°á»ng chá»©ng khoÃ¡n",
    "VN30", "HNX", "HOSE", "UPCOM",
    "giÃ¡ cá»• phiáº¿u", "cá»• pháº§n",
    # â”€â”€ Tin tá»©c â”€â”€
    KW_TIN_TUC, "news", KW_TIN_MOI, KW_TIN_NONG, KW_TIN_HOM_NAY,
    "tÃ³m táº¯t", "tÃ³m táº¯t tin",
    # â”€â”€ ChÃ­nh trá»‹ â”€â”€
    KW_TONG_THONG, KW_THU_TUONG, "phÃ³ tá»•ng thá»‘ng", "chá»§ tá»‹ch nÆ°á»›c",
    "tá»•ng bÃ­ thÆ°", "bá»™ trÆ°á»Ÿng", "thá»‘ng Ä‘á»‘c",
    "hiá»‡n táº¡i lÃ  ai", "lÃ  ai", "ai lÃ ",
    # â”€â”€ Sá»± kiá»‡n â”€â”€
    "tÃ¬nh hÃ¬nh", "chiáº¿n sá»±", "chiáº¿n tranh",
    "ngoáº¡i giao", "trá»«ng pháº¡t", "cáº¥m váº­n",
    # â”€â”€ NhÃ¢n váº­t ná»•i tiáº¿ng â”€â”€
    "JD Vance", "Trump", "Putin", "Zelensky",
    "TÃ´ LÃ¢m", "Pháº¡m Minh ChÃ­nh", "LÆ°Æ¡ng CÆ°á»ng",
    "Friedrich Merz", "Olaf Scholz",
    "Elon Musk", "Mark Zuckerberg",
    # â”€â”€ NhÃ¢n váº­t fiction + factual â”€â”€
    "nhÃ¢n váº­t", "character",
    "trong phim", "trong game", "trong truyá»‡n", "trong anime",
    "nghÄ©a lÃ  gÃ¬", "cÃ³ nghÄ©a lÃ  gÃ¬",
    "who is", "what is",
]

_STOCK_KEYWORDS_AGENT = [
    "cá»• phiáº¿u", "chá»©ng khoÃ¡n", "mÃ£ chá»©ng khoÃ¡n", "thá»‹ trÆ°á»ng chá»©ng khoÃ¡n",
    "vn-index", "vnindex", "vn30", "hnx", "hose", "upcom",
    "giÃ¡ cá»• phiáº¿u", "stock", "cá»• pháº§n",
]

MUSIC_KEYWORDS = [
    "má»Ÿ nháº¡c", "phÃ¡t nháº¡c", "báº­t nháº¡c", "nghe nháº¡c", "play music",
    "chÆ¡i nháº¡c", "má»Ÿ bÃ i", "phÃ¡t bÃ i", "báº­t bÃ i",
    "mo nhac", "phat nhac", "bat nhac", "nghe nhac", "choi nhac",
    "mo bai", "phat bai", "bat bai", "play song",
]
MUSIC_STOP_KEYWORDS = [
    "táº¯t nháº¡c", "táº¯t bÃ i", "táº¯t bÃ i hÃ¡t", "táº¯t phÃ¡t nháº¡c",
    "Ä‘Ã³ng nháº¡c", "Ä‘Ã³ng tab nháº¡c",
    "tat nhac", "tat bai", "tat bai hat", "dong nhac", "dong tab nhac",
]
MUSIC_PAUSE_KEYWORDS = [
    "dá»«ng nháº¡c", "ngá»«ng nháº¡c", "ngÆ°ng nháº¡c", "stop music",
    "dá»«ng bÃ i", "ngá»«ng phÃ¡t", "dá»«ng phÃ¡t nháº¡c",
    "khÃ´ng nghe ná»¯a", "ngá»«ng phÃ¡t nháº¡c", "pause music",
    "dá»«ng bÃ i hÃ¡t",
    "dung nhac", "ngung nhac", "dung bai", "ngung phat",
    "dung phat nhac", "khong nghe nua", "dung bai hat",
    "tam dung nhac", "tam dung bai", "tam dung bai hat",
]
MUSIC_RESUME_KEYWORDS = [
    "tiáº¿p tá»¥c phÃ¡t", "phÃ¡t tiáº¿p", "nghe tiáº¿p", "tiáº¿p tá»¥c nghe",
    "resume music", "tiáº¿p tá»¥c bÃ i", "má»Ÿ láº¡i nháº¡c", "báº­t láº¡i nháº¡c",
    "phÃ¡t láº¡i nháº¡c", "phÃ¡t láº¡i bÃ i",
    "tiep tuc phat", "phat tiep", "nghe tiep", "tiep tuc nghe",
    "tiep tuc bai", "mo lai nhac", "bat lai nhac", "phat lai nhac",
    "phat lai bai", "tiep tuc choi nhac",
]
MUSIC_SWITCH_KEYWORDS = [
    "chuyá»ƒn sang bÃ i", "chuyá»ƒn sang nháº¡c", "chuyá»ƒn bÃ i", "chuyá»ƒn nháº¡c",
    "Ä‘á»•i sang bÃ i", "Ä‘á»•i sang nháº¡c", "Ä‘á»•i bÃ i", "Ä‘á»•i nháº¡c",
    "nghe bÃ i khÃ¡c", "phÃ¡t bÃ i khÃ¡c",
    "switch song", "bÃ i khÃ¡c",
    "chuyen sang bai", "chuyen sang nhac", "chuyen bai", "chuyen nhac",
    "doi sang bai", "doi sang nhac", "doi bai", "doi nhac",
    "nghe bai khac", "phat bai khac",
]

_WEATHER_TRAILING_PHRASES = [
    "lÃ  bao nhiÃªu Ä‘á»™", "bao nhiÃªu Ä‘á»™", "lÃ  bao nhiÃªu",
    "nhÆ° tháº¿ nÃ o", "tháº¿ nÃ o", "ra sao", KW_HOM_NAY,
    "ngÃ y mai", "hiá»‡n táº¡i", "bÃ¢y giá»", "lÃºc nÃ y",
    "Ä‘ang lÃ  bao nhiÃªu", "Ä‘ang tháº¿ nÃ o", "Ä‘ang ra sao",
    "cÃ³ mÆ°a khÃ´ng", "cÃ³ náº¯ng khÃ´ng", "cÃ³ giÃ³ khÃ´ng",
]

_ANALYSIS_SIGNALS = [
    "áº£nh hÆ°á»Ÿng", "táº¡i sao", "vÃ¬ sao", "nguyÃªn nhÃ¢n",
    "so sÃ¡nh", "liÃªn quan", "nÃªn khÃ´ng", "cÃ³ nÃªn",
    "giáº£i thÃ­ch", "phÃ¢n tÃ­ch", "dá»± Ä‘oÃ¡n", "dá»± bÃ¡o tuáº§n",
    "xu hÆ°á»›ng", "triá»ƒn vá»ng", "Ä‘Ã¡nh giÃ¡", "nháº­n xÃ©t",
    "khÃ¡c nhau", "giá»‘ng nhau", "má»‘i quan há»‡",
    "lá»i khuyÃªn", "tÆ° váº¥n", "gá»£i Ã½", "recommend",
    "anh huong", "tai sao", "vi sao", "nguyen nhan",
    "so sanh", "lien quan", "co nen", "phan tich",
    "giai thich", "du doan", "xu huong",
]

MEMORY_PATTERNS = [
    r"\btÃ´i tÃªn lÃ \b",
    r"\bmÃ¬nh tÃªn lÃ \b",
    r"\btÃªn mÃ¬nh lÃ \b",
    r"\bhÃ£y nhá»› ráº±ng\b",
    r"\bnhá»› ráº±ng\b",
    r"\bsá»Ÿ thÃ­ch cá»§a tÃ´i lÃ \b",
    r"\btÃ´i thÃ­ch\b",
    r"\bgá»i tÃ´i lÃ \b",
    r"\bcá»© gá»i tÃ´i\b",
    r"\bbiá»‡t danh cá»§a tÃ´i\b",
    r"\bnickname cá»§a tÃ´i\b",
]

_SHORT_KEYWORD_THRESHOLD = 4


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# QUERY REWRITING TABLES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

_SEARCH_PREFIX_TO_STRIP = [
    "cho tÃ´i há»i", "cho mÃ¬nh há»i", "tÃ´i muá»‘n há»i", "mÃ¬nh muá»‘n há»i",
    "báº¡n Æ¡i", "Ãª", "hey", "há»i chÃºt", "há»i xÃ­u",
    "tÃ´i muá»‘n biáº¿t", "mÃ¬nh muá»‘n biáº¿t", "cho tÃ´i biáº¿t", "cho mÃ¬nh biáº¿t",
    "báº¡n cÃ³ biáº¿t", "ai biáº¿t", "giÃºp tÃ´i tÃ¬m", "giÃºp mÃ¬nh tÃ¬m",
    "tÃ¬m giÃºp tÃ´i", "tÃ¬m giÃºp mÃ¬nh", "search giÃºp",
    "tÃ¬m hiá»ƒu vá»", "tÃ¬m hiá»ƒu", "tra cá»©u", "tra giÃºp",
    "nÃ³i cho tÃ´i", "nÃ³i cho mÃ¬nh", "ká»ƒ cho tÃ´i", "ká»ƒ cho mÃ¬nh",
]

_FICTION_CONTEXT_CLUES = {
    "trong phim": "movie character",
    "trong truyá»‡n": "manga novel character",
    "trong game": "game character",
    "trong anime": "anime character",
    "trong series": "TV series character",
    "the boys": "The Boys TV series character wiki",
    "marvel": "Marvel character wiki",
    "dc comics": "DC Comics character wiki",
    "one piece": "One Piece character wiki",
    "naruto": "Naruto character wiki",
    "attack on titan": "Attack on Titan character wiki",
    "demon slayer": "Demon Slayer character wiki",
    "jujutsu kaisen": "Jujutsu Kaisen character wiki",
    "dragon ball": "Dragon Ball character wiki",
    "harry potter": "Harry Potter character wiki",
    "game of thrones": "Game of Thrones character wiki",
    "lord of the rings": "Lord of the Rings character wiki",
    "star wars": "Star Wars character wiki",
    "league of legends": "League of Legends champion wiki",
    "valorant": "Valorant agent wiki",
    "dota": "Dota 2 hero wiki",
}

_FACTUAL_PATTERNS = [
    r'\blÃ  ai\b', r'\blÃ  gÃ¬\b', r'\bnghÄ©a lÃ  gÃ¬\b',
    r'\bai lÃ \b', r'\bwho is\b', r'\bwhat is\b',
    r'\bnhÃ¢n váº­t\b', r'\bcharacter\b',
    r'\btiá»ƒu sá»­\b', r'\bbiography\b',
    r'\bthá»§ tÆ°á»›ng\b', r'\btá»•ng thá»‘ng\b', r'\bchá»§ tá»‹ch\b',
    r'\bthá»§ Ä‘Ã´\b', r'\bdÃ¢n sá»‘\b', r'\bdiá»‡n tÃ­ch\b',
    r'\bsinh nÄƒm\b', r'\btuá»•i\b.*\bbao nhiÃªu\b',
    r'\bcÃ´ng thá»©c\b', r'\bÄ‘á»‹nh nghÄ©a\b',
    r'\btÃ¡c giáº£\b', r'\bngÆ°á»i sÃ¡ng láº­p\b', r'\bfounder\b',
]

_WHO_WHAT_PATTERNS = [
    r'\blÃ  ai\b', r'\blÃ  gÃ¬\b', r'\bwho is\b', r'\bwhat is\b',
    r'\bnghÄ©a lÃ  gÃ¬\b', r'\bcÃ³ nghÄ©a lÃ \b',
    r'\bnhÃ¢n váº­t\b.*\blÃ \b', r'\bcharacter\b',
]

_BASIC_KNOWLEDGE_CLUES = [
    "toÃ¡n", "váº­t lÃ½", "hÃ³a há»c", "sinh há»c", "lá»‹ch sá»­",
    "cÃ´ng thá»©c", "Ä‘á»‹nh lÃ½", "phÆ°Æ¡ng trÃ¬nh", "nguyÃªn tá»‘",
    "math", "physics", "chemistry", "formula",
]

_COUNTRY_NAME_MAP = {
    "má»¹": "United States", "hoa ká»³": "United States",
    "anh": "United Kingdom", "phÃ¡p": "France",
    "Ä‘á»©c": "Germany", "nháº­t": "Japan", "nháº­t báº£n": "Japan",
    "hÃ n": "South Korea", "hÃ n quá»‘c": "South Korea",
    "trung quá»‘c": "China", "trung": "China",
    "nga": "Russia", "áº¥n Ä‘á»™": "India",
    "Ãºc": "Australia", "canada": "Canada",
    "hungary": "Hungary", "ba lan": "Poland",
    "ukraine": "Ukraine", "thÃ¡i lan": "Thailand",
    "indonesia": "Indonesia", "singapore": "Singapore",
    "philippines": "Philippines", "malaysia": "Malaysia",
    "lÃ o": "Laos", "campuchia": "Cambodia",
    "myanmar": "Myanmar", "Ã½": "Italy", "italia": "Italy",
    "tÃ¢y ban nha": "Spain", "bá»“ Ä‘Ã o nha": "Portugal",
    "hÃ  lan": "Netherlands", "bá»‰": "Belgium",
    "thá»¥y sÄ©": "Switzerland", "thá»¥y Ä‘iá»ƒn": "Sweden",
    "na uy": "Norway", "Ä‘an máº¡ch": "Denmark",
    "pháº§n lan": "Finland", "Ã¡o": "Austria",
    "hy láº¡p": "Greece", "thá»• nhÄ© ká»³": "Turkey",
    "ai cáº­p": "Egypt", "iran": "Iran", "iraq": "Iraq",
    "israel": "Israel", "áº£ ráº­p": "Saudi Arabia",
    "brazil": "Brazil", "argentina": "Argentina",
    "mexico": "Mexico", "cuba": "Cuba",
    "triá»u tiÃªn": "North Korea", "báº¯c triá»u tiÃªn": "North Korea",
}

_STOPWORDS_VN = {
    "cho", "tÃ´i", "há»i", "mÃ¬nh", "báº¡n", "Æ¡i", "lÃ ", "gÃ¬",
    "ai", "cá»§a", "trong", "vÃ ", "cÃ³", "Ä‘Æ°á»£c", "nÃ y", "Ä‘Ã³",
    "má»™t", "cÃ¡c", "nhá»¯ng", "vá»›i", "Ä‘á»ƒ", "khi", "thÃ¬", "mÃ ",
    "nhÆ°ng", "hay", "hoáº·c", "náº¿u", "vÃ¬", "do", "bá»Ÿi",
    "hÃ£y", "Ä‘i", "nhÃ©", "nÃ o", "áº¡", "váº­y", "tháº¿",
    "nÄƒm", "nay", "hiá»‡n", "táº¡i", "hÃ´m",
}


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# AGENT DEPENDENCIES
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@dataclass
class AgentDeps:
    """Dependencies injected into the Pydantic-AI agent at runtime."""
    memory: AssistantMemory


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# AGENT BUILDER
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def _build_openai_compatible_model() -> OpenAIChatModel:
    provider = OpenAIProvider(
        base_url=settings.base_url.rstrip("/") + "/v1",
        api_key=settings.api_key,
    )
    return OpenAIChatModel(
        model_name=settings.model,
        provider=provider,
    )


def _build_pydantic_agent() -> Agent[AgentDeps, str]:
    """Create the Pydantic-AI Agent with OpenAI-compatible provider and tools."""
    model = _build_openai_compatible_model()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        current_datetime=_get_current_datetime_str()
    )

    agent = Agent(
        model=model,
        deps_type=AgentDeps,
        instructions=system_prompt,
        model_settings={
            "temperature": min(settings.llm_temperature, 0.1),
            "max_tokens": max(settings.llm_max_completion_tokens, 4096),
            "extra_body": {"enable_thinking": False},
        },
    )

    @agent.instructions
    def add_memory_context(ctx: RunContext[AgentDeps]) -> str:
        return ctx.deps.memory.get_context_system_prompt()

    agent.tool_plain(get_weather)
    agent.tool_plain(web_search)
    agent.tool_plain(save_memory)
    agent.tool_plain(get_current_datetime)
    agent.tool_plain(calculate)
    agent.tool_plain(translate_text)
    agent.tool_plain(knowledge_search)
    agent.tool_plain(get_exchange_rate)
    agent.tool_plain(play_music)
    agent.tool_plain(stop_music)
    agent.tool_plain(pause_music)
    agent.tool_plain(resume_music)
    agent.tool_plain(get_stock_price)

    return agent


def _build_formatter_agent() -> Agent[AgentDeps, str]:
    """Create a no-tool agent for formatting already-fetched evidence."""
    model = _build_openai_compatible_model()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        current_datetime=_get_current_datetime_str()
    )
    return Agent(
        model=model,
        deps_type=AgentDeps,
        instructions=system_prompt,
        model_settings={
            "temperature": min(settings.llm_temperature, 0.1),
            "max_tokens": min(settings.llm_max_completion_tokens, 2048),
            "extra_body": {"enable_thinking": False},
        },
    )

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN AGENT CLASS
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class PersonalAssistantAgent:
    def __init__(self) -> None:
        self.memory = AssistantMemory()
        self._agent = _build_pydantic_agent()
        self._formatter_agent = _build_formatter_agent()
        self.memory.sync_music_state_to_tools()

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # TEXT NORMALIZATION HELPERS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    @staticmethod
    def _normalize_intent_text(text: str) -> str:
        """Remove Vietnamese diacritics and special chars for robust keyword matching."""
        lowered = (text or "").lower().replace("đ", "d")
        no_diacritics = "".join(
            ch for ch in unicodedata.normalize("NFD", lowered)
            if unicodedata.category(ch) != "Mn"
        )
        normalized = re.sub(r"[^a-z0-9\s]", " ", no_diacritics)
        return re.sub(r"\s+", " ", normalized).strip()

    @staticmethod
    def _has_keyword_match(
        text: str,
        keywords: list[str],
        short_threshold: int = _SHORT_KEYWORD_THRESHOLD,
    ) -> bool:
        """Check if any keyword matches in text with word boundary protection."""
        normalized_text = PersonalAssistantAgent._normalize_intent_text(text)
        if not normalized_text:
            return False

        for kw in keywords:
            nkw = PersonalAssistantAgent._normalize_intent_text(kw)
            if not nkw:
                continue

            if len(nkw) <= short_threshold:
                pattern = r'(?<![a-z0-9])' + re.escape(nkw) + r'(?![a-z0-9])'
                if re.search(pattern, normalized_text):
                    return True
            else:
                if nkw in normalized_text:
                    return True

        return False

    @staticmethod
    def _is_news_response(text: str) -> bool:
        """Detect if text is a numbered news list (e.g. '1. ...\\n\\n2. ...')."""
        text = text or ""
        numbered_items = re.findall(REGEX_NEWS_NUMBERED_ITEM, text)
        item_markers = re.findall(r'(?m)^\s*(?:[-*]\s+|\d+\.\s+)', text)
        has_news_citation = bool(
            re.search(r'\((?:theo|Theo|via|Via)\s+[^)]+\)', text)
            or re.search(r'https?://\S+', text)
        )
        return has_news_citation and (len(numbered_items) >= 1 or len(item_markers) >= 2)

    @staticmethod
    def _normalize_answer(answer_text: str, max_chars: int = 0) -> str:
        """Clean and trim answer, preserving paragraph breaks for news."""
        answer_text = (answer_text or "").replace("~~", "")
        limit = max_chars if max_chars > 0 else settings.max_response_chars

        is_news = PersonalAssistantAgent._is_news_response(answer_text)
        is_structured = is_news or bool(
            re.search(r'(?m)^\s*(?:[-*]\s+|\d+\.\s+|[^\n:]{1,60}:\s*)', answer_text)
        )

        if is_structured:
            normalized = re.sub(r'\s+(?=(?:[1-9]|1[0-5])\.\s+\S+)', '\n\n', answer_text)
            normalized = re.sub(r'\n{3,}', '\n\n', normalized)
            lines = normalized.split('\n')
            cleaned_lines = [' '.join(line.split()) for line in lines]
            compact = '\n'.join(cleaned_lines)
            compact = re.sub(r'\n{3,}', '\n\n', compact).strip()
        else:
            compact = " ".join(answer_text.split())

        if len(compact) <= limit:
            return compact

        truncated = compact[:limit]

        if is_news:
            item_starts = list(re.finditer(r'(?m)^\d+\.\s', truncated))
            if len(item_starts) >= 2:
                candidate = truncated[:item_starts[-1].start()].rstrip()
                if re.search(r'(?m)^\d+\.\s', candidate):
                    return candidate

            last_double_nl = truncated.rfind('\n\n')
            if last_double_nl > len(truncated) // 4:
                candidate = truncated[:last_double_nl].rstrip()
                if re.search(r'(?m)^\d+\.\s', candidate):
                    return candidate

        for sep in [".", "!", "?"]:
            last_idx = truncated.rfind(sep)
            if last_idx > len(truncated) // 3:
                return truncated[:last_idx + 1]
        space_idx = truncated.rfind(" ")
        if space_idx > 0:
            return truncated[:space_idx].rstrip(" ,.;:") + "."
        return truncated.rstrip(" ,.;:") + "."

    @staticmethod
    def _is_analytical_query(text: str) -> bool:
        """Detect if query is analytical/complex â†’ needs full LLM reasoning."""
        return PersonalAssistantAgent._has_keyword_match(text, _ANALYSIS_SIGNALS)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # HISTORY FILTERING
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    @staticmethod
    def _filter_history_for_api(messages: List[ModelMessage]) -> List[ModelMessage]:
        """Remove messages containing tool-call or tool-return parts."""
        clean: List[ModelMessage] = []
        for msg in messages:
            has_tool_part = any(
                getattr(part, "part_kind", "") in ("tool-call", "tool-return")
                for part in msg.parts
            )
            if not has_tool_part:
                clean.append(msg)
        return clean

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # LLM RUNNER WITH RETRY
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _run_llm_with_retry(
        self,
        prompt: str,
        message_history: List[ModelMessage] | None = None,
    ):
        """Run LLM agent with automatic retry on history errors."""
        history = message_history if message_history is not None else []
        try:
            return self._agent.run_sync(
                prompt,
                deps=AgentDeps(memory=self.memory),
                message_history=history,
            )
        except Exception:
            if not history:
                raise
            logger.warning("LLM failed with history (%d msgs), retrying with empty history", len(history))
            return self._agent.run_sync(
                prompt,
                deps=AgentDeps(memory=self.memory),
                message_history=[],
            )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # QUERY TYPE DETECTION + REWRITING
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _answer_from_search(
        self,
        *,
        user_text: str,
        search_query: str,
        raw_data: str,
        tool_name: str,
        task_hint: str = "",
        max_chars: int | None = None,
    ) -> str:
        """Format search evidence with one no-tool LLM call."""
        prompt = (
            f"NgÆ°á»i dÃ¹ng há»i: {user_text}\n"
            f"Truy váº¥n Ä‘Ã£ dÃ¹ng: {search_query}\n"
            f"Tool Ä‘Ã£ dÃ¹ng: {tool_name}\n\n"
            "Dá»® LIá»†U TOOL TRáº¢ Vá»€:\n"
            f"{raw_data}\n\n"
            f"{task_hint}\n\n"
            "HÃ£y tráº£ lá»i tá»± nhiÃªn báº±ng Ä‘Ãºng ngÃ´n ngá»¯ cá»§a ngÆ°á»i dÃ¹ng. "
            "Chá»‰ dÃ¹ng dá»¯ liá»‡u trong pháº§n tool tráº£ vá»; khÃ´ng tá»± bá»‹a sá»‘ liá»‡u, ngÃ y, tÃªn banner, giÃ¡ hoáº·c nguá»“n. "
            "KhÃ´ng paste raw list mÃ¡y mÃ³c kiá»ƒu 'Káº¿t quáº£ cho...' hoáº·c 'Tin tá»©c vá»...'. "
            "Bá» qua káº¿t quáº£ rÃµ rÃ ng lÃ  rÃ¡c, trang chuyÃªn má»¥c, RSS, video, hoáº·c khÃ´ng Ä‘Ãºng intent. "
            "Vá»›i cÃ¢u há»i cÃ³ 'hÃ´m nay', 'hiá»‡n táº¡i', 'current' hoáº·c 'má»›i nháº¥t', khÃ´ng trÃ¬nh bÃ y dá»¯ liá»‡u cÅ© nhÆ° thÃ´ng tin hiá»‡n táº¡i; náº¿u chá»‰ cÃ³ dá»¯ liá»‡u cÅ© thÃ¬ nÃ³i rÃµ lÃ  chÆ°a tháº¥y dá»¯ liá»‡u hÃ´m nay. "
            "Náº¿u dá»¯ liá»‡u khÃ´ng Ä‘á»§ Ä‘á»ƒ tráº£ lá»i chÃ­nh xÃ¡c, nÃ³i rÃµ Ä‘iá»u Ä‘Ã³ vÃ  nÃªu nguá»“n gáº§n nháº¥t há»¯u Ã­ch. "
            "Giá»¯ link/source quan trá»ng náº¿u cÃ³. Vá»›i cÃ¢u há»i tin tá»©c, trÃ¬nh bÃ y gá»n theo tá»«ng dÃ²ng tin."
        )
        result = self._formatter_agent.run_sync(
            prompt,
            deps=AgentDeps(memory=self.memory),
            message_history=[],
        )
        return self._normalize_answer(result.output, max_chars=max_chars or settings.max_response_chars)

    @staticmethod
    def _user_prefers_vietnamese(text: str) -> bool:
        normalized = PersonalAssistantAgent._normalize_intent_text(text)
        vi_markers = {
            "la", "gi", "cua", "toi", "minh", "hien", "tai", "hom", "nay",
            "bao", "nhieu", "su", "kien", "gia", "tin", "tuc", "cho",
        }
        return bool(set(normalized.split()) & vi_markers)

    @staticmethod
    def _count_numbered_items(text: str) -> int:
        return len(re.findall(r"(?:^|\n)\s*\d+\.\s+", text or ""))

    @staticmethod
    def _is_world_news_request(text: str) -> bool:
        normalized = PersonalAssistantAgent._normalize_intent_text(text)
        return "the gioi" in normalized or "quoc te" in normalized or "world" in normalized

    @staticmethod
    def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
        return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text) for term in terms)

    @staticmethod
    def _is_world_news_match(text: str) -> bool:
        normalized = PersonalAssistantAgent._normalize_intent_text(text)
        world_terms = (
            "quoc te", "the gioi", "world", "international", "ngoai giao",
            "bien gioi", "lanh tho tranh chap", "chien tranh", "xung dot",
            "chien su", "quan su", "phong khong", "ten lua", "hat nhan",
            "ukraine", "ukraina", "nga", "russia", "my", "trung quoc", "china",
            "nhat ban", "japan", "han quoc", "trieu tien", "syria", "damascus",
            "iran", "israel", "palestine", "gaza", "campuchia", "thai lan",
            "phap", "france", "duc", "germany", "anh", "britain", "india",
            "an do", "indonesia", "asean", "eu", "nato", "lien hop quoc",
            "kiev", "kyiv", "kharkov", "kharkiv", "senkaku", "dieu ngu",
        )
        if not PersonalAssistantAgent._contains_any_term(normalized, world_terms):
            return False

        domestic_terms = (
            "tp hcm", "tphcm", "ha noi", "dong thap", "tuyen quang", "ubnd",
            "dang uy", "ban dang", "dai hoi doan", "can bo", "khai tru dang",
            "huan chuong", "chu tich nuoc", "doanh nghiep viet nam",
            "kinh te viet nam", "dong von fdi", "thu truong co quan",
            "co quan dai dien", "bo ngoai giao", "dai su quan", "tong lanh su quan",
            "viet nam o nuoc ngoai",
            "trong nuoc", "tin trong nuoc", "thoi su trong nuoc", "tin thoi su trong nuoc",
        )
        return not PersonalAssistantAgent._contains_any_term(normalized, domestic_terms)

    @staticmethod
    def _is_sports_news_request(text: str) -> bool:
        normalized = PersonalAssistantAgent._normalize_intent_text(text)
        return "the thao" in normalized or "sports" in normalized or "sport" in normalized

    @staticmethod
    def _news_output_looks_stale(text: str, max_age_days: int = 7) -> bool:
        dates = re.findall(r"\((\d{1,2})/(\d{1,2})\)", text or "")
        if not dates:
            return False
        today = datetime.now(_VN_TZ).date()
        stale_count = 0
        for day_s, month_s in dates:
            try:
                pub_date = datetime(today.year, int(month_s), int(day_s), tzinfo=_VN_TZ).date()
            except ValueError:
                continue
            if (today - pub_date).days > max_age_days:
                stale_count += 1
        return stale_count >= max(1, len(dates) // 2)

    @staticmethod
    def _limit_numbered_answer(text: str, max_items: int) -> str:
        lines = (text or "").splitlines()
        output: list[str] = []
        count = 0
        for line in lines:
            if re.match(r"\s*\d+\.\s+", line):
                count += 1
                if count > max_items:
                    continue
            elif count > max_items:
                if line.strip():
                    continue
            output.append(line)
        return "\n".join(output).strip()

    @staticmethod
    def _extract_result_rows(raw_data: str, max_items: int) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for line in (raw_data or "").splitlines():
            match = re.match(r"\s*(\d+)\.\s+(.+)", line)
            if not match:
                continue
            body = match.group(2).strip()
            source = ""
            url = ""
            source_match = re.search(r"\(via\s+\[([^\]]+)\]\(([^)]+)\)\)", body)
            if source_match:
                source = source_match.group(1).strip()
                url = source_match.group(2).strip()
                body = body[:source_match.start()].strip()
            else:
                plain_source = re.search(r"\(via\s+([^)]+)\)", body, flags=re.IGNORECASE)
                if plain_source:
                    source = plain_source.group(1).strip()
                    body = body[:plain_source.start()].strip()
            title, _, snippet = body.partition(": ")
            rows.append({
                "title": PersonalAssistantAgent._clean_search_fragment(title),
                "snippet": PersonalAssistantAgent._clean_search_fragment(snippet),
                "source": PersonalAssistantAgent._clean_source_label(source),
                "url": url.rstrip(".,); "),
            })
            if len(rows) >= max_items:
                break
        return rows

    @staticmethod
    def _clean_search_fragment(text: str) -> str:
        text = (text or "").replace("~~", "")
        text = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", "", text)
        text = re.sub(r"\b(?:\d+\s+)?(?:seconds?|minutes?|hours?|days?|weeks?)\s+ago\s*[-â€“â€”]\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:Chá»n|Chon)\s+[^.]{0,80}?(?:Google|tÃ¬m kiáº¿m|tim kiem)[^.]*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bXem\s+hÆ°á»›ng\s+dáº«n\.?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text)
        return text.strip(" -â€“â€”;:,.")

    @staticmethod
    def _clean_source_label(source: str) -> str:
        source = PersonalAssistantAgent._clean_search_fragment(source)
        source = re.sub(r"\s+on\s+MSN$", "", source, flags=re.IGNORECASE)
        source = source.replace("www.", "").strip()
        return source

    @staticmethod
    def _row_date_is_stale(row: dict[str, str], max_age_days: int = 7) -> bool:
        text = f"{row.get('title', '')} {row.get('snippet', '')}"
        dates = re.findall(r"(?<!\d)(\d{1,2})/(\d{1,2})(?!\d)", text)
        url_dates = re.findall(r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)", row.get("url", ""))
        today = datetime.now(_VN_TZ).date()
        for year_s, month_s, day_s in url_dates:
            try:
                pub_date = datetime(int(year_s), int(month_s), int(day_s), tzinfo=_VN_TZ).date()
            except ValueError:
                continue
            if pub_date > today:
                return True
            if (today - pub_date).days <= max_age_days:
                return False
        if url_dates:
            return True
        if not dates:
            return bool(re.search(r"\b(?:1|one)\s+week\s+ago\b|\b\d+\s+weeks\s+ago\b", text, flags=re.IGNORECASE))
        for day_s, month_s in dates:
            try:
                pub_date = datetime(today.year, int(month_s), int(day_s), tzinfo=_VN_TZ).date()
            except ValueError:
                continue
            if pub_date > today:
                continue
            if (today - pub_date).days <= max_age_days:
                return False
        return True

    @staticmethod
    def _news_subject_tokens(user_norm: str) -> list[str]:
        stopwords = {
            "lay", "cho", "toi", "minh", "tim", "xem", "hay", "giup",
            "tin", "tuc", "bai", "ban", "muc", "dieu", "nhung", "cac",
            "hom", "nay", "moi", "nhat", "hot", "top", "latest", "today",
            "news", "get", "give", "show", "find", "tell", "most", "hottest",
        }
        tokens = []
        for token in re.findall(r"\b[a-z0-9]+\b", user_norm):
            if len(token) < 3 or token.isdigit() or token in stopwords:
                continue
            if token not in tokens:
                tokens.append(token)
        return tokens[:5]

    @staticmethod
    def _news_topic_phrase(user_text: str) -> str:
        normalized = PersonalAssistantAgent._normalize_intent_text(user_text)
        stopwords = {
            "lay", "cho", "toi", "minh", "tim", "xem", "hay", "giup",
            "tin", "tuc", "bai", "ban", "muc", "dieu", "nhung", "cac",
            "hom", "nay", "moi", "nhat", "hot", "top", "latest", "today",
            "news", "get", "give", "show", "find", "tell", "most", "hottest",
        }
        tokens = [
            token for token in re.findall(r"\b[a-z0-9]+\b", normalized)
            if len(token) >= 2 and not token.isdigit() and token not in stopwords
        ]
        return " ".join(tokens[:4])

    @staticmethod
    def _news_domains(rows: list[dict[str, str]]) -> set[str]:
        return {
            urlparse(row.get("url", "")).netloc.lower().removeprefix("www.")
            for row in rows
            if row.get("url")
        }

    @staticmethod
    def _news_diversity_target(max_results: int) -> int:
        return min(3, max_results)

    @staticmethod
    def _news_needs_more_sources(rows: list[dict[str, str]], max_results: int) -> bool:
        if len(rows) < max_results:
            return True
        return len(PersonalAssistantAgent._news_domains(rows)) < PersonalAssistantAgent._news_diversity_target(max_results)

    @staticmethod
    def _news_row_matches_request(user_text: str, row: dict[str, str]) -> bool:
        user_norm = PersonalAssistantAgent._normalize_intent_text(user_text)
        row_norm = PersonalAssistantAgent._normalize_intent_text(
            f"{row.get('title', '')} {row.get('snippet', '')} {row.get('source', '')}"
        )

        wants_recent = any(marker in user_norm for marker in ("tin tuc", "hom nay", "hot nhat", "moi nhat"))
        if wants_recent and PersonalAssistantAgent._row_date_is_stale(row):
            return False
        if wants_recent:
            years = re.findall(r"\b(20\d{2})\b", row_norm)
            current_year = str(datetime.now(_VN_TZ).year)
            if years and current_year not in years:
                return False

        if "bong da" in user_norm or "football" in user_norm or "soccer" in user_norm:
            bad_terms = ["bong chay", "baseball", "phim", "hoat hinh", "dien anh", "judo"]
            if any(term in row_norm for term in bad_terms):
                return False
            good_terms = [
                "bong da", "football", "soccer", "world cup", "fifa", "uefa",
                "cau thu", "tran dau", "doi tuyen", "chuyen nhuong", "ronaldo", "messi",
            ]
            return any(term in row_norm for term in good_terms)

        if PersonalAssistantAgent._is_sports_news_request(user_text):
            bad_terms = ["giao duc", "diem thi", "liet si", "khai tru dang", "phap luat", "kinh doanh"]
            if any(term in row_norm for term in bad_terms):
                return False
            good_terms = [
                "the thao", "bong da", "football", "soccer", "world cup", "fifa", "uefa",
                "tennis", "cau thu", "tran dau", "doi tuyen", "chuyen nhuong", "ronaldo",
                "messi", "lich thi dau", "giai dau", "vo dich",
            ]
            return any(term in row_norm for term in good_terms)

        if PersonalAssistantAgent._is_world_news_request(user_text):
            soft_excludes = [
                "gia xang", "gia vang", "gia tieu", "nong san", "thi truong", "hang hoa",
                "co phieu", "chung khoan", "ty phu", "giau nhat", "tai san ca nhan",
                "the thao", "bong da", "world cup", "fifa", "cau thu", "tran dau", "lich thi dau",
            ]
            has_soft_exclude = any(term in row_norm for term in soft_excludes)
            if has_soft_exclude:
                return False
            if not PersonalAssistantAgent._is_world_news_match(
                f"{row.get('title', '')} {row.get('snippet', '')} {row.get('url', '')}"
            ):
                return False

        if not PersonalAssistantAgent._is_world_news_request(user_text):
            subject_tokens = PersonalAssistantAgent._news_subject_tokens(user_norm)
            if subject_tokens and not any(token in row_norm for token in subject_tokens):
                return False

        return True

    @staticmethod
    def _is_article_like_news_row(row: dict[str, str]) -> bool:
        url = (row.get("url") or "").strip().lower()
        title_norm = PersonalAssistantAgent._normalize_intent_text(row.get("title", ""))
        snippet_norm = PersonalAssistantAgent._normalize_intent_text(row.get("snippet", ""))
        if not url.startswith("http"):
            return False
        if any(host in url for host in ["youtube.com", "youtu.be", "google.com/search", "news.google.com"]):
            return False

        parsed = urlparse(url)
        path = parsed.path.lower().strip("/")
        if len(title_norm.split()) < 4:
            return False
        if title_norm.startswith(("danh sach", "tra cuu", "huong dan")):
            return False
        if "/video/" in f"/{path}/" or path.startswith("video"):
            return False
        if "/event/" in f"/{path}/" or "/su-kien/" in f"/{path}/" or path.startswith("media-"):
            return False
        last_part = path.rsplit("/", 1)[-1]
        slug = re.sub(r"\.(?:html?|epi|rss)$", "", last_part)
        if re.search(r"\.html?$", last_part) and not re.search(r"\d", slug) and len(path.split("/")) <= 3:
            return False
        category_slugs = {
            "the-gioi", "tin-the-gioi", "quoc-te", "tin-quoc-te", "tin-tuc-quoc-te",
            "world", "international", "news", "tin-tuc", "russia",
        }
        if not path or path.endswith((".rss", ".epi")) or slug in category_slugs:
            return False
        has_article_signal = bool(re.search(r"(?:-\d{5,}|/\d{4}/\d{2}/|/\d{4}/|-\d+\.html|\.html?|\.htm)$", path))
        if "/" not in path and not has_article_signal:
            return False
        if re.search(r"-\d{1,3}$", slug):
            generic_title_markers = [
                "tin the gioi", "the gioi 24h", "quoc te hom nay", "tin tuc quoc te",
                "thoi su quoc te", "tin nong quoc te", "tin tuc the thao", "the thao 24h",
            ]
            if any(marker in title_norm for marker in generic_title_markers):
                return False
        if re.search(r"-c\d+\.html?$", last_part) and not re.search(r"-d\d+\.html?$", last_part):
            return False

        generic_title_markers = [
            "tin the gioi", "the gioi 24h", "quoc te hom nay", "tin tuc quoc te",
            "phan tich tinh hinh", "doc bao", "cap nhat tin tuc",
            "tin the thao moi nhat", "tin the thao 24h", "tin bong da", "lich thi dau bong da",
        ]
        if any(marker in title_norm for marker in generic_title_markers) and (
            len(snippet_norm.split()) < 18 or "moi-nhat" in path or "24h" in path
        ):
            return False

        if not has_article_signal:
            return False
        return True

    @staticmethod
    def _news_rows_for_answer(raw_news: str, user_text: str, max_results: int) -> list[dict[str, str]]:
        scan_limit = max(max_results * 12, 60)
        rows = PersonalAssistantAgent._extract_result_rows(raw_news, scan_limit)
        rows = PersonalAssistantAgent._extract_doc_link_rows(raw_news, scan_limit) + rows
        candidates: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        seen_topics: set[str] = set()

        for row in rows:
            if not row.get("title"):
                continue
            if not PersonalAssistantAgent._is_article_like_news_row(row):
                continue
            if not PersonalAssistantAgent._news_row_matches_request(user_text, row):
                continue
            url_key = row.get("url", "").split("?", 1)[0].rstrip("/").lower()
            topic = PersonalAssistantAgent._news_topic_from_title(row["title"])
            topic_key = PersonalAssistantAgent._normalize_intent_text(topic)
            if url_key and url_key in seen_urls:
                continue
            if topic_key and topic_key in seen_topics:
                continue
            summary = PersonalAssistantAgent._clean_search_fragment(row.get("snippet") or row["title"])
            if not summary:
                continue
            candidates.append({
                "topic": topic,
                "summary": summary,
                "source": row.get("source") or "nguá»“n tÃ¬m kiáº¿m",
                "url": row.get("url", ""),
                "title": row["title"],
            })
            if url_key:
                seen_urls.add(url_key)
            if topic_key:
                seen_topics.add(topic_key)
        if len(candidates) < max_results:
            for row in PersonalAssistantAgent._extract_doc_link_rows(raw_news, scan_limit):
                url_key = row.get("url", "").split("?", 1)[0].rstrip("/").lower()
                topic = PersonalAssistantAgent._news_topic_from_title(row["title"])
                topic_key = PersonalAssistantAgent._normalize_intent_text(topic)
                if url_key and url_key in seen_urls:
                    continue
                if topic_key and topic_key in seen_topics:
                    continue
                if not PersonalAssistantAgent._news_row_matches_request(user_text, row):
                    continue
                candidates.append({
                    "topic": topic,
                    "summary": PersonalAssistantAgent._clean_search_fragment(row.get("snippet") or row["title"]),
                    "source": row.get("source") or "nguon tim kiem",
                    "url": row.get("url", ""),
                    "title": row["title"],
                })
                if url_key:
                    seen_urls.add(url_key)
                if topic_key:
                    seen_topics.add(topic_key)
        return PersonalAssistantAgent._diversify_news_rows(candidates, max_results)

    @staticmethod
    def _diversify_news_rows(rows: list[dict[str, str]], max_results: int) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        selected_urls: set[str] = set()
        domain_counts: dict[str, int] = {}
        topic_sets = [
            PersonalAssistantAgent._topic_token_set(f"{row.get('topic', '')} {row.get('summary', '')} {row.get('url', '')}")
            for row in rows
        ]
        max_per_domain = 1

        for strict in (True, False):
            for row, token_set in zip(rows, topic_sets):
                if len(selected) >= max_results:
                    return selected
                url_key = row.get("url", "").split("?", 1)[0].rstrip("/").lower()
                if url_key and url_key in selected_urls:
                    continue
                domain = urlparse(row.get("url", "")).netloc.lower().removeprefix("www.") or row.get("source", "")
                if strict and domain_counts.get(domain, 0) >= max_per_domain:
                    continue
                if strict and token_set and any(
                    PersonalAssistantAgent._topic_overlap(token_set, prev) >= 0.5
                    for prev in (
                        PersonalAssistantAgent._topic_token_set(f"{item.get('topic', '')} {item.get('summary', '')} {item.get('url', '')}")
                        for item in selected
                    )
                ):
                    continue
                selected.append(row)
                if url_key:
                    selected_urls.add(url_key)
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
        return selected[:max_results]

    @staticmethod
    def _topic_token_set(text: str) -> set[str]:
        normalized = PersonalAssistantAgent._normalize_intent_text(text)
        ignored = {"http", "https", "www", "html", "htm", "com", "net", "org"}
        tokens = [
            token for token in normalized.split()
            if len(token) >= 4 and not token.isdigit() and token not in ignored
        ]
        if not tokens:
            return set()
        counts = {token: tokens.count(token) for token in set(tokens)}
        return {token for token in counts if counts[token] <= max(1, len(tokens) // 3)}

    @staticmethod
    def _topic_overlap(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / max(1, min(len(left), len(right)))

    @staticmethod
    def _extract_doc_link_rows(raw_news: str, max_items: int) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        source = ""
        doc_title = ""
        for line in (raw_news or "").splitlines():
            title_match = re.match(r"\s*\[DOC\d+\]\s+TITLE:\s*(.+)", line)
            if title_match:
                doc_title = PersonalAssistantAgent._clean_search_fragment(title_match.group(1))
                continue
            source_match = re.match(r"\s*SOURCE:\s*(.+)", line)
            if source_match:
                source = PersonalAssistantAgent._clean_source_label(source_match.group(1))
                continue
            stripped = line.strip()
            if not stripped.startswith("- "):
                continue
            body = stripped[2:].strip()
            url_match = re.search(r"(https?://\S+)\s*$", body)
            if not url_match:
                continue
            url = url_match.group(1).rstrip(".,);")
            title = body[:url_match.start()].strip(" -–—:;,.")
            if not title:
                continue
            clean_title = PersonalAssistantAgent._clean_search_fragment(title)
            snippet = f"{doc_title}: {clean_title}" if doc_title else clean_title
            row = {
                "title": clean_title,
                "snippet": snippet,
                "source": source or urlparse(url).netloc.replace("www.", ""),
                "url": url,
            }
            if not PersonalAssistantAgent._is_article_like_news_row(row):
                continue
            rows.append(row)
            if len(rows) >= max_items:
                break
        return rows

    @staticmethod
    def _news_topic_from_title(title: str) -> str:
        title = PersonalAssistantAgent._clean_search_fragment(title)
        title = re.sub(r"\([^)]*\d{1,2}/\d{1,2}[^)]*\)", "", title)
        topic = re.split(r"\s[-â€“â€”]\s|[:|]", title, maxsplit=1)[0]
        topic = re.sub(r"^(?:Tin\s+(?:tháº¿ giá»›i|tá»©c|má»›i)|Cáº­p nháº­t)\s*", "", topic, flags=re.IGNORECASE)
        words = topic.strip(" -â€“â€”:;,.").split()
        if len(words) > 10:
            topic = " ".join(words[:10])
        return topic.strip(" -â€“â€”:;,.") or title[:80].strip(" -â€“â€”:;,.")

    @staticmethod
    def _compact_article_evidence(text: str, limit: int = 1200) -> str:
        text = PersonalAssistantAgent._clean_search_fragment(text)
        if len(text) <= limit:
            return text
        clipped = text[:limit].strip()
        sentence_end = max(clipped.rfind(". "), clipped.rfind("? "), clipped.rfind("! "))
        if sentence_end >= limit // 2:
            clipped = clipped[: sentence_end + 1]
        return clipped.strip()

    @staticmethod
    def _news_article_evidence(
        rows: list[dict[str, str]],
        max_items: int,
        user_text: str = "",
    ) -> tuple[str, list[dict[str, str]]]:
        selected = rows[:max_items]
        blocks = [""] * len(selected)
        kept_rows: list[dict[str, str] | None] = [None] * len(selected)
        subject_tokens = []
        if user_text and not PersonalAssistantAgent._is_world_news_request(user_text):
            subject_tokens = PersonalAssistantAgent._news_subject_tokens(
                PersonalAssistantAgent._normalize_intent_text(user_text)
            )

        def fetch_row(idx: int, row: dict[str, str]) -> tuple[int, str, dict[str, str] | None]:
            text = ""
            url = row.get("url", "")
            if url:
                try:
                    fetched = _fetch_page_evidence(url, timeout_sec=5)
                    if isinstance(fetched, dict):
                        text = fetched.get("text", "") or ""
                except Exception:
                    text = ""
            evidence = PersonalAssistantAgent._compact_article_evidence(text)
            if len(evidence) < 120:
                evidence = PersonalAssistantAgent._compact_article_evidence(
                    row.get("summary") or row.get("title") or ""
                )
            evidence_norm = PersonalAssistantAgent._normalize_intent_text(evidence)
            title_norm = PersonalAssistantAgent._normalize_intent_text(row.get("title", ""))
            noisy_prefixes = ("doc nhieu", "tin moi nhat", "trang chu", "lien quan")
            if evidence_norm.startswith(noisy_prefixes) and title_norm[:40] not in evidence_norm:
                return idx, "", None
            if user_text and PersonalAssistantAgent._is_world_news_request(user_text):
                if not PersonalAssistantAgent._is_world_news_match(
                    f"{row.get('title', '')} {url} {evidence}"
                ):
                    return idx, "", None
            if subject_tokens:
                haystack = PersonalAssistantAgent._normalize_intent_text(
                    f"{row.get('title', '')} {url} {evidence}"
                )
                if not any(token in haystack for token in subject_tokens):
                    return idx, "", None
            kept_row = dict(row)
            kept_row["article_text"] = evidence
            block = (
                f"[ITEM {idx + 1}]\n"
                f"ORIGINAL_TITLE: {row.get('title') or row.get('topic')}\n"
                f"SEARCH_TOPIC: {row.get('topic')}\n"
                f"SOURCE: {row.get('source')}\n"
                f"URL: {url}\n"
                f"ARTICLE_TEXT: {evidence}"
            )
            return idx, block, kept_row

        if not selected:
            return "", []
        with ThreadPoolExecutor(max_workers=min(5, len(selected))) as executor:
            futures = [executor.submit(fetch_row, idx, row) for idx, row in enumerate(selected)]
            for future in as_completed(futures):
                idx, block, kept_row = future.result()
                blocks[idx] = block
                kept_rows[idx] = kept_row
        compact_rows = [row for row in kept_rows if row]
        return "\n\n".join(block for block in blocks if block), compact_rows

    def _format_news_rows_from_articles(
        self,
        *,
        user_text: str,
        search_query: str,
        rows: list[dict[str, str]],
        max_results: int,
    ) -> str:
        evidence, kept_rows = self._news_article_evidence(rows, max_results, user_text)
        if not evidence:
            return ""
        result_count = min(max_results, len(kept_rows))
        answer = self._answer_from_search(
            user_text=user_text,
            search_query=search_query,
            raw_data=evidence,
            tool_name="web_search",
            task_hint=(
                f"Format exactly {result_count} news items, numbered 1 to {result_count}. "
                "Use the same language as the user. If the user writes Vietnamese, write natural Vietnamese with accents. "
                "For each item, use exactly this shape: "
                "<number>. <short paraphrased title, not copied verbatim from ORIGINAL_TITLE>: "
                "**<main entity or key development>** <2-3 sentence summary based on ARTICLE_TEXT; the summary must be longer than the title>. "
                "(theo <SOURCE>) ; [<URL>](<URL>). "
                "Do not repeat ORIGINAL_TITLE as the summary. "
                "Do not use category, homepage, RSS, search, or video URLs. "
                "Only use URLs from the ITEM blocks. "
                "If ARTICLE_TEXT is thin, summarize the available evidence honestly, but still keep title and summary different."
            ),
            max_chars=max(settings.max_response_chars, max_results * 1000),
        )
        answer = re.sub(r"\n{2,}(\d+)\.\n{2,}(?=\(theo\s)", r" \1. ", answer)
        known_urls = [row.get("url", "") for row in kept_rows[:result_count] if row.get("url")]
        known_by_key = {url.split("?", 1)[0].rstrip("/").lower(): url for url in known_urls}

        def replace_link(match: re.Match[str]) -> str:
            label = match.group(1)
            url = match.group(2)
            url_key = url.split("?", 1)[0].rstrip("/").lower()
            if url_key in known_by_key:
                return match.group(0)
            domain = urlparse(url).netloc.lower().removeprefix("www.")
            candidates = [
                candidate for candidate in known_urls
                if urlparse(candidate).netloc.lower().removeprefix("www.") == domain
            ] or known_urls
            close = difflib.get_close_matches(url, candidates, n=1, cutoff=0.82)
            if not close:
                return match.group(0)
            fixed_url = close[0]
            return f"[{label}]({fixed_url})"

        return re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", replace_link, answer)

    @staticmethod
    def _vi_game_title(title: str) -> str:
        lower = title.lower()
        if "upcoming" in lower and "banner" in lower:
            return "Lá»‹ch banner sáº¯p tá»›i"
        if "headhunting" in lower and "banner" in lower:
            return "Danh sÃ¡ch banner headhunting"
        if "schedule" in lower and "banner" in lower:
            return "Lá»‹ch banner"
        if any(term in lower for term in ["summon", "rate up", "limited"]):
            return "ThÃ´ng tin banner"
        if "event" in lower:
            return "ThÃ´ng tin sá»± kiá»‡n"
        if "operator" in lower:
            return "ThÃ´ng tin operator/banner"
        return title.strip()

    @staticmethod
    def _game_row_kind(row: dict[str, str]) -> str:
        text = PersonalAssistantAgent._normalize_intent_text(f"{row.get('title', '')} {row.get('snippet', '')}")
        if any(term in text for term in ["banner", "headhunting", "summon", "rate up", "operator", "servant", "limited"]):
            return "banner"
        if any(term in text for term in ["event", "su kien", "side story", "anniversary", "campaign", "festival"]):
            return "event"
        return "other"

    @staticmethod
    def _game_rows_for_answer(raw_data: str, max_items: int) -> list[dict[str, str]]:
        rows = PersonalAssistantAgent._extract_result_rows(raw_data, max(max_items * 2, max_items))
        selected: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for row in rows:
            url = (row.get("url") or "").lower()
            if "youtube.com" in url or "youtu.be" in url:
                continue
            url_key = url.split("?", 1)[0].rstrip("/")
            if url_key and url_key in seen_urls:
                continue
            selected.append(row)
            if url_key:
                seen_urls.add(url_key)
            if len(selected) >= max_items:
                break
        return selected

    def _format_game_search_output_vi(self, raw_data: str, max_items: int = 5) -> str:
        rows = self._game_rows_for_answer(raw_data, max_items)
        if not rows:
            return self._normalize_answer(raw_data)
        banner_rows = [row for row in rows if self._game_row_kind(row) == "banner"]
        event_rows = [row for row in rows if self._game_row_kind(row) == "event"]
        other_rows = [row for row in rows if row not in banner_rows and row not in event_rows]
        lines = [
            "DÆ°á»›i Ä‘Ã¢y lÃ  thÃ´ng tin banner/sá»± kiá»‡n mÃ¬nh tháº¥y trong káº¿t quáº£ tÃ¬m kiáº¿m. Má»¥c nÃ o nguá»“n khÃ´ng nÃªu tÃªn cá»¥ thá»ƒ thÃ¬ mÃ¬nh ghi rÃµ lÃ  chÆ°a tháº¥y:",
            "",
        ]

        def add_group(title: str, group_rows: list[dict[str, str]], empty_text: str) -> None:
            lines.append(title)
            if not group_rows:
                lines.append(f"- {empty_text}")
                lines.append("")
                return
            for idx, row in enumerate(group_rows[:max_items], start=1):
                label = self._vi_game_title(row["title"])
                detail = row["snippet"] or row["title"]
                source = row["source"] or "nguá»“n tÃ¬m kiáº¿m"
                if row["url"]:
                    lines.append(f"{idx}. {label}: {detail} (theo {source}) ; [{row['url']}]({row['url']})")
                else:
                    lines.append(f"{idx}. {label}: {detail} (theo {source})")
            lines.append("")

        add_group(
            "Banner:",
            banner_rows,
            "Káº¿t quáº£ tÃ¬m kiáº¿m khÃ´ng nÃªu rÃµ banner/nhÃ¢n váº­t rate-up cá»¥ thá»ƒ.",
        )
        add_group(
            "Sá»± kiá»‡n:",
            event_rows,
            "Káº¿t quáº£ tÃ¬m kiáº¿m khÃ´ng nÃªu rÃµ tÃªn sá»± kiá»‡n Ä‘ang diá»…n ra.",
        )

        remaining = [row for row in other_rows if row not in banner_rows and row not in event_rows]
        if remaining:
            lines.append("Nguá»“n liÃªn quan khÃ¡c:")
            for idx, row in enumerate(remaining[:max_items], start=1):
                detail = row["snippet"] or row["title"]
                source = row["source"] or "nguá»“n tÃ¬m kiáº¿m"
                if row["url"]:
                    lines.append(f"{idx}. {row['title']}: {detail} (theo {source}) ; [{row['url']}]({row['url']})")
                else:
                    lines.append(f"{idx}. {row['title']}: {detail} (theo {source})")
            lines.append("")

        lines.append("Náº¿u nguá»“n khÃ´ng ghi tÃªn sá»± kiá»‡n hoáº·c danh sÃ¡ch nhÃ¢n váº­t/Servant, mÃ¬nh sáº½ nÃ³i rÃµ lÃ  chÆ°a tháº¥y thay vÃ¬ tá»± Ä‘oÃ¡n.")
        return "\n".join(lines).strip()

    def _detect_query_type(self, user_text: str) -> str:
        """PhÃ¢n loáº¡i query thÃ nh type Ä‘á»ƒ chá»n strategy phÃ¹ há»£p.

        Returns one of:
            "political"  â€“ ai lÃ  thá»§ tÆ°á»›ng, tá»•ng thá»‘ng hiá»‡n táº¡i...
            "fiction"    â€“ nhÃ¢n váº­t phim/game/truyá»‡n
            "price"      â€“ giÃ¡ xÄƒng, giÃ¡ vÃ ng, giÃ¡ sáº£n pháº©m
            "stock"      â€“ cá»• phiáº¿u, chá»©ng khoÃ¡n
            "factual"    â€“ kiáº¿n thá»©c: lÃ  gÃ¬, nghÄ©a lÃ  gÃ¬, cÃ´ng thá»©c
            "news"       â€“ tin tá»©c, sá»± kiá»‡n
            "general"    â€“ máº·c Ä‘á»‹nh
        """
        lower = user_text.lower()
        normalized = self._normalize_intent_text(user_text)

        # â”€â”€ Political: há»i vá» ngÆ°á»i giá»¯ chá»©c vá»¥ â”€â”€
        _POLITICAL_PATTERNS = [
            r'\b(thá»§ tÆ°á»›ng|tá»•ng thá»‘ng|phÃ³ tá»•ng thá»‘ng|chá»§ tá»‹ch nÆ°á»›c)\b',
            r'\b(tá»•ng bÃ­ thÆ°|bá»™ trÆ°á»Ÿng|thá»‘ng Ä‘á»‘c|chá»§ tá»‹ch quá»‘c há»™i)\b',
            r'\b(president|prime minister|chancellor|governor)\b',
            r'\bai lÃ \b.*\b(lÃ£nh Ä‘áº¡o|ngÆ°á»i Ä‘á»©ng Ä‘áº§u|leader)\b',
            r'\b(lÃ£nh Ä‘áº¡o|ngÆ°á»i Ä‘á»©ng Ä‘áº§u)\b.*\blÃ  ai\b',
        ]
        for pattern in _POLITICAL_PATTERNS:
            if re.search(pattern, lower):
                return "political"

        # â”€â”€ Fiction: nhÃ¢n váº­t phim/game/truyá»‡n â”€â”€
        for clue in _FICTION_CONTEXT_CLUES:
            if clue in lower:
                return "fiction"

        if (
            any(term in normalized for term in ("co phieu", "chung khoan", "ma chung khoan", "vnindex", "vn index"))
            or re.search(r"\b(?:nasdaq|nyse|hose|hnx|upcom|stock)\b", normalized)
        ):
            return "stock"

        # â”€â”€ Stock â”€â”€
        if any(kw.lower() in lower for kw in _STOCK_KEYWORDS_AGENT):
            return "stock"

        if (
            any(term in normalized for term in ("gia xang", "gia vang", "gia dau", "gia gas", "gia dien", "ty gia"))
            or re.search(r"\b(?:bitcoin|btc|eth|crypto)\b", normalized)
            or ("gia" in normalized.split() and "bao nhieu" in normalized)
            or "bao nhieu tien" in normalized
        ):
            return "price"

        # â”€â”€ Price â”€â”€
        _PRICE_PATTERNS = [
            r'\bgiÃ¡\s+(xÄƒng|vÃ ng|dáº§u|gas|Ä‘iá»‡n|bitcoin|btc|eth)\b',
            r'\bgiÃ¡\b.*\bbao nhiÃªu\b', r'\bbao nhiÃªu tiá»n\b',
            r'\btá»· giÃ¡\b',
        ]
        for pattern in _PRICE_PATTERNS:
            if re.search(pattern, lower):
                return "price"

        if any(term in normalized for term in ("tin tuc", "tin moi", "tin nong", "tin hom nay", "thoi su")):
            return "news"

        # â”€â”€ News â”€â”€
        if self._has_keyword_match(user_text, _DIRECT_NEWS_KEYWORDS):
            return "news"
        if any(kw in lower for kw in ["tÃ¬nh hÃ¬nh", "chiáº¿n sá»±", "chiáº¿n tranh"]):
            return "news"

        # â”€â”€ Game: banner, event, gacha, nhÃ¢n váº­t game â”€â”€
        game_keywords = [
            "banner", "gacha", "patch", "event", "character", "nhÃ¢n váº­t game",
            "arknights", "endfield", "enfield", "honkai", "genshin",
            "wuthering", "wuwa", "zenless", "zzz", "fgo",
        ]
        if any(kw in lower for kw in game_keywords):
            return "game"

        # â”€â”€ Factual: "lÃ  ai", "lÃ  gÃ¬" (khÃ´ng pháº£i political) â”€â”€
        for pattern in _FACTUAL_PATTERNS:
            if re.search(pattern, lower):
                return "factual"

        return "general"

    def _strip_search_prefix(self, user_text: str) -> tuple[str, str]:
        query = user_text.strip()
        lower = query.lower()
        for prefix in sorted(_SEARCH_PREFIX_TO_STRIP, key=len, reverse=True):
            if not lower.startswith(prefix):
                continue
            query = query[len(prefix):].strip(" ,.:!?")
            return query, query.lower()
        return query, lower

    @staticmethod
    def _political_role_template(lower: str) -> str:
        rules = [
            (KW_THU_TUONG, "current prime minister of {country} {year} wikipedia"),
            (KW_TONG_THONG, "current president of {country} {year} wikipedia"),
            ("phÃ³ tá»•ng thá»‘ng", "current vice president of {country} {year} wikipedia"),
            ("chá»§ tá»‹ch", "current chairman president of {country} {year} wikipedia"),
            ("tá»•ng bÃ­ thÆ°", "current general secretary of {country} {year} wikipedia"),
            ("bá»™ trÆ°á»Ÿng", "current minister of {country} {year}"),
            ("thá»‘ng Ä‘á»‘c", "current governor of {country} {year}"),
        ]
        for keyword, template in rules:
            if keyword in lower:
                return template
        return "current leader of {country} {year} wikipedia"

    def _rewrite_political_query(self, query: str, lower: str, year: str) -> str:
        country = next((en_name for vn_name, en_name in _COUNTRY_NAME_MAP.items() if vn_name in lower), None)
        if not country:
            return f"{query} {year} hiá»‡n táº¡i official source Reuters AP"

        template = self._political_role_template(lower).replace(" wikipedia", " official source Reuters AP wikipedia")
        return template.format(country=country, year=year)

    @staticmethod
    def _rewrite_fiction_query(query: str, lower: str) -> str:
        for clue_key, clue_val in _FICTION_CONTEXT_CLUES.items():
            if clue_key not in lower:
                continue
            clean_query = lower.replace(clue_key, "").strip()
            clean_query = re.sub(r'\b(lÃ  ai|lÃ  gÃ¬|who is|what is)\b', '', clean_query).strip()
            if clean_query:
                return f"{clean_query} {clue_val}"

        name_match = re.sub(r'\b(lÃ  ai|lÃ  gÃ¬|nhÃ¢n váº­t|who is|what is|character)\b', '', lower).strip()
        if name_match and len(name_match) >= 2:
            return f"{name_match} character wiki"
        return f"{query} wiki"

    @staticmethod
    def _rewrite_stock_query(query: str, lower: str) -> str:
        intl_markers = [
            "nasdaq", "nyse", "s&p", "dow jones", "apple", "google",
            "microsoft", "tesla", "amazon", "meta", "nvidia",
            "aapl", "msft", "googl", "tsla", "amzn", "nvda",
        ]
        is_international = any(marker in lower for marker in intl_markers)
        if is_international:
            return f"{query} Yahoo Finance stock price today latest"
        return f"{query} gia moi nhat hom nay DNSE VietStock CafeF"

    @staticmethod
    def _rewrite_game_query_to_english(query: str, lower: str, context: Dict[str, str] | None, year: str) -> str:
        game_aliases = [
            (("arknights endfield", "endfield", "enfield"), "Arknights Endfield"),
            (("honkai star rail", "star rail", "hsr", "honkai"), "Honkai Star Rail"),
            (("wuthering waves", "wuthering", "wuwa"), "Wuthering Waves"),
            (("zenless zone zero", "zenless", "zzz"), "Zenless Zone Zero"),
            (("fate grand order", "fgo", "fate/go"), "Fate Grand Order"),
            (("blue archive",), "Blue Archive"),
            (("uma musume", "umamusume"), "Uma Musume"),
            (("genshin impact", "genshin"), "Genshin Impact"),
            (("arknights",), "Arknights"),
            (("nikke",), "Nikke"),
        ]
        game_part = ""
        if "arknights" in lower and any(
            marker in lower
            for marker in [
                "khÃ´ng láº¥y arknights endfield", "khong lay arknights endfield",
                "khÃ´ng láº¥y endfield", "khong lay endfield",
                "not arknights endfield", "not endfield", "exclude endfield",
            ]
        ):
            game_part = "Arknights"
        for aliases, display_name in game_aliases:
            if game_part:
                break
            if any(alias in lower for alias in aliases):
                game_part = display_name
                break
        if not game_part and context and context.get("entity"):
            game_part = context["entity"]
        if not game_part:
            game_part = "gacha game"

        server_part = "Global"
        if re.search(r"\b(jp|japan|nhat|nháº­t)\b", lower):
            server_part = "JP"
        elif re.search(r"\b(cn|china|trung quoc|trung quá»‘c)\b", lower):
            server_part = "CN"
        elif re.search(r"\b(na|north america)\b", lower):
            server_part = "NA"

        if any(term in lower for term in ["upcoming", "sáº¯p tá»›i", "sap toi", "next"]):
            intent_part = "upcoming banner schedule"
        elif any(term in lower for term in ["event", "sá»± kiá»‡n", "su kien", "patch"]):
            intent_part = "current in-game event side story banner schedule"
        else:
            intent_part = "current banner schedule"
        if game_part == "Arknights" and server_part == "Global":
            return f"Arknights Global July {year} current banner rate up current in-game event side story lootbar oldwell wiki.gg -FES -offline -Shanghai"
        if game_part == "Zenless Zone Zero":
            return f"Zenless Zone Zero Global July {year} current banner current event current version schedule Game8 Prydwen"
        return f"{game_part} {server_part} {intent_part} {year}"

    def _rewrite_search_query(self, user_text: str, context: Dict[str, str] | None = None) -> str:
        """Rewrite query by detected type while preserving original intent."""
        query, lower = self._strip_search_prefix(user_text)
        query_type = self._detect_query_type(user_text)
        normalized = self._normalize_intent_text(user_text)
        year = datetime.now(_VN_TZ).strftime("%Y")

        if query_type == "political":
            return self._rewrite_political_query(query, lower, year)
        if query_type == "fiction":
            return self._rewrite_fiction_query(query, lower)
        if query_type == "price":
            if "xang" in normalized:
                today = datetime.now(_VN_TZ).strftime("%d/%m/%Y")
                return f"gia xang dau hom nay {today} TPHCM E5 RON95 diesel Petrolimex PVOIL Lien Bo Cong Thuong Tai chinh"
            if "moi nhat" not in normalized and "hom nay" not in normalized:
                return f"{query} moi nhat hom nay"
            return query
        if query_type == "stock":
            return self._rewrite_stock_query(query, lower)
        if query_type == "factual" and "wiki" not in lower and "wikipedia" not in lower:
            return f"{query} wiki"

        game_markers = ["honkai", "hsr", "star rail", "genshin", "wuthering", "wuwa", "zenless", "zzz", "arknights", "endfield", "enfield", "fgo", "fate grand order", "blue archive", "nikke", "uma musume", "umamusume", "banner", "patch", "event", "mobile game", "gacha"]
        if any(marker in lower for marker in game_markers):
            return self._rewrite_game_query_to_english(query, lower, context, year)
            # Chá»‰ giá»¯ tÃªn game, bá» pháº§n tiáº¿ng Viá»‡t Ä‘á»ƒ search English-centric
            known_games = ["arknights", "endfield", "enfield", "honkai", "genshin",
                           "wuthering", "wuwa", "zenless", "zzz", "fgo"]
            found_games = [g for g in known_games if g in lower]

            if "enfield" in lower and "endfield" not in lower:
                game_part = "Arknights Endfield"
            elif found_games:
                name_parts = []
                for g in found_games:
                    if g == "wuwa":
                        name_parts.append("Wuthering Waves")
                    elif g == "wuthering":
                        name_parts.append("Wuthering Waves")
                    elif g == "zzz":
                        name_parts.append("Zenless Zone Zero")
                    elif g == "zenless":
                        name_parts.append("Zenless Zone Zero")
                    elif g == "fgo":
                        name_parts.append("Fate Grand Order")
                    else:
                        name_parts.append(g.capitalize())
                game_part = " ".join(dict.fromkeys(name_parts))
            elif context and context.get("entity"):
                # USE RECENT CONTEXT â€” resolve vague follow-up like "banner trÆ°á»›c Ä‘Ã³"
                game_part = context["entity"]
            else:
                game_part = "gacha game"

            if "latest" not in lower and "má»›i nháº¥t" not in lower and KW_HOM_NAY not in lower:
                if "arknights" in found_games and not any(g in found_games for g in ["endfield", "enfield"]):
                    return "Arknights Global current headhunting banner rate up list"
                return f"{game_part} Global current event banner rate up character list Game8"
            return f"{game_part} Global event banner schedule Game8"

        return query

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # GROUNDING CONTEXT BUILDER
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _should_route_news_directly(self, user_text: str) -> bool:
        """Return True for simple news queries that should use web_search."""
        if self._detect_query_type(user_text) in {"price", "stock"}:
            return False
        normalized = self._normalize_intent_text(user_text)
        has_news_intent = self._has_keyword_match(user_text, _DIRECT_NEWS_KEYWORDS)
        if not has_news_intent:
            has_news_intent = any(
                marker in normalized
                for marker in ("tin tuc", "tin moi", "tin nong", "tin hom nay", "thoi su")
            )
        if not has_news_intent and ("tin" in normalized.split() or "tin t c" in normalized):
            has_news_intent = any(
                marker in normalized
                for marker in ["hom nay", "h m nay", "moi nhat", "nong", "quoc te", "the gioi", "th gi i", "the thao", "th thao", "trong nuoc"]
            )
        if not has_news_intent:
            return False
        if self._is_analytical_query(user_text):
            logger.debug("News keyword found but analytical query -> LLM path")
            return False
        return True

    def _should_route_weather_directly(self, user_text: str) -> bool:
        """Return True for direct weather queries."""
        if not settings.direct_weather_routing:
            return False
        query_type = self._detect_query_type(user_text)
        if query_type in {"price", "stock", "news"}:
            return False
        normalized = self._normalize_intent_text(user_text)
        weather_terms = ("thoi tiet", "weather", "nhiet do", "mua", "do am", "gio", "du bao")
        if not any(term in normalized for term in weather_terms):
            return False
        if self._has_keyword_match(user_text, _WEATHER_EXCLUDE_KEYWORDS):
            logger.debug("Weather keyword candidate but exclusion keyword found -> skip weather")
            return False
        if self._is_analytical_query(user_text):
            logger.debug("Weather keyword found but analytical query detected -> LLM path")
            return False
        return True

    def _should_route_realtime_directly(self, user_text: str) -> bool:
        """Return True for realtime info queries (prices, stocks, politics, characters)."""
        if self._is_exchange_rate_query(user_text):
            return False
        query_type = self._detect_query_type(user_text)
        if query_type in {"price", "stock"}:
            return True
        if self._has_keyword_match(user_text, REALTIME_KEYWORDS):
            if self._is_analytical_query(user_text):
                logger.debug("Real-time keyword found but analytical query detected -> LLM path")
                return False
            return True

        # â”€â”€ Game/live-service queries change frequently; avoid fragile general LLM routing â”€â”€
        normalized = self._normalize_intent_text(user_text)
        game_realtime_markers = [
            "honkai", "genshin", "wuthering", "wuwa", "zenless", "zzz",
            "arknights", "endfield", "enfield", "fgo",
            "banner", "patch", "event", "gacha", "game mobile", "mobile game", "esports", "esport",
        ]
        if any(marker in normalized for marker in game_realtime_markers):
            logger.info("Detected game realtime query -> routing to realtime direct")
            return True

        # â”€â”€ Check factual/political/fiction queries â”€â”€
        if query_type in ("political", "fiction"):
            logger.info("Detected %s query -> routing to realtime direct", query_type)
            return True

        # â”€â”€ "lÃ  ai" / "lÃ  gÃ¬" patterns â”€â”€
        lower = user_text.lower()
        for pattern in _WHO_WHAT_PATTERNS:
            if re.search(pattern, lower):
                is_basic = any(clue in lower for clue in _BASIC_KNOWLEDGE_CLUES)
                if not is_basic:
                    logger.info("Detected 'who/what is' query -> routing to realtime direct")
                    return True
                break

        return False

    @staticmethod
    def _is_exchange_rate_query(user_text: str) -> bool:
        """Detect currency-conversion wording without resolving currencies."""
        normalized = PersonalAssistantAgent._normalize_intent_text(user_text)
        if not normalized:
            return False
        markers = (
            "ty gia", "exchange rate", "currency exchange", "currency conversion",
            "quy doi", "doi tien", "doi sang", "doi ra",
            "bang bao nhieu", "bao nhieu tien",
        )
        return any(marker in normalized for marker in markers)

    def _should_route_music(self, user_text: str) -> bool:
        """Return True for any music command (play/pause/stop/resume/switch)."""
        all_kws = (
            MUSIC_KEYWORDS + MUSIC_STOP_KEYWORDS + MUSIC_PAUSE_KEYWORDS
            + MUSIC_RESUME_KEYWORDS + MUSIC_SWITCH_KEYWORDS
        )
        if self._has_keyword_match(user_text, all_kws):
            return True

        normalized = self._normalize_intent_text(user_text)
        music_terms = ("nhac", "bai", "bai hat", "music", "song")
        action_terms = (
            "mo", "phat", "bat", "nghe", "choi", "tat", "dong",
            "dung", "ngung", "tiep tuc", "chuyen", "doi",
            "play", "stop", "pause", "resume", "switch",
        )
        return (
            any(term in normalized for term in music_terms)
            and any(term in normalized for term in action_terms)
        )

    def _should_save_memory_directly(self, user_text: str) -> bool:
        """Return True if user is sharing personal info to remember."""
        if not settings.direct_memory_routing:
            return False
        normalized = self._normalize_intent_text(user_text)
        return any(
            re.search(self._normalize_intent_text(pattern), normalized)
            for pattern in MEMORY_PATTERNS
        )

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # EXTRACTION HELPERS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _extract_weather_location(self, user_text: str) -> str:
        """Extract location name from weather query."""
        lower = user_text.strip().lower()

        # â”€â”€ BÆ°á»›c 1: Loáº¡i Cá»¤M DÃ€I trÆ°á»›c (dÃ i â†’ ngáº¯n) â”€â”€
        _phrases = [
            "lÃ  bao nhiÃªu Ä‘á»™", "bao nhiÃªu Ä‘á»™", "Ä‘ang lÃ  bao nhiÃªu",
            "la bao nhieu do", "bao nhieu do", "dang la bao nhieu",
            "lÃ  bao nhiÃªu", "Ä‘ang tháº¿ nÃ o", "Ä‘ang ra sao",
            "la bao nhieu", "dang the nao", "dang ra sao",
            "nhÆ° tháº¿ nÃ o", "tháº¿ nÃ o", "ra sao", "nhu the nao", "the nao",
            "cÃ³ mÆ°a khÃ´ng", "cÃ³ náº¯ng khÃ´ng", "cÃ³ giÃ³ khÃ´ng",
            "co mua khong", "co nang khong", "co gio khong",
            "hiá»‡n táº¡i", "bÃ¢y giá»", "lÃºc nÃ y", "hien tai", "bay gio", "luc nay",
            KW_HOM_NAY, "hom nay", "ngÃ y mai", "ngay mai", "hiá»‡n nay", "hien nay",
            "cho mÃ¬nh", "cho tÃ´i", "giÃºp mÃ¬nh", "cho minh", "cho toi", "giup minh",
            "nhiá»‡t Ä‘á»™", "nhiá»‡t Ä‘Ã´", "nhiet do", "thá»i tiáº¿t", "thoi tiet", "dá»± bÃ¡o", "du bao",
            "Ä‘á»™ áº©m", "do am", "weather", "temperature",
            "bao nhiÃªu", "bao nhieu", "mÆ°a", "mua", "giÃ³", "gio", "náº¯ng", "nang",
        ]
        for phrase in sorted(_phrases, key=len, reverse=True):
            lower = lower.replace(phrase, " ")

        # â”€â”€ BÆ°á»›c 2: Loáº¡i filler ngáº¯n (word-boundary) â”€â”€
        lower = re.sub(r"\b(?:nhiet|nhiá»‡t|nhiÃªt)\s*(?:do|Ä‘á»™|Ä‘Ã´|dá»™)\b", " ", lower)
        for filler in ["á»Ÿ", "o", "táº¡i", "tai", "cá»§a", "cua", "cho", "xem", "lÃ ", "la", "cÃ³", "co"]:
            lower = re.sub(r'(?<!\w)' + re.escape(filler) + r'(?!\w)', ' ', lower)

        location = " ".join(lower.split()).strip(" ?!.,:")
        return location if len(location) >= 2 else "Ho Chi Minh City"

    def _extract_search_query(self, user_text: str, context: Dict[str, str] | None = None) -> str:
        """Extract a clean search query from user text for web_search.

        Delegates to _rewrite_search_query() for smart rewriting based on query type.
        """
        return self._rewrite_search_query(user_text, context)

    def _extract_music_song(self, user_text: str, keywords: list[str]) -> str:
        """Extract song name from music command by removing keyword prefixes."""
        text = user_text.strip()
        lower = text.lower()

        original_tokens = text.split()
        normalized_tokens = [self._normalize_intent_text(token) for token in original_tokens]
        normalized_tokens = [token for token in normalized_tokens if token]
        sorted_kws = sorted(keywords, key=len, reverse=True)
        for kw in sorted_kws:
            kw_tokens = self._normalize_intent_text(kw).split()
            if kw_tokens:
                for start in range(0, max(len(normalized_tokens) - len(kw_tokens) + 1, 0)):
                    if normalized_tokens[start:start + len(kw_tokens)] == kw_tokens:
                        after = " ".join(original_tokens[start + len(kw_tokens):]).strip(" ,.:;!?")
                        if len(after) >= 2:
                            return after

            idx = lower.find(kw.lower())
            if idx >= 0:
                after = text[idx + len(kw):].strip(" ,.:;!?")
                if len(after) >= 2:
                    return after

        return text

    @staticmethod
    def _extract_news_count(text: str) -> tuple[int, str]:
        """Extract requested article count from text and return (count, cleaned_text)."""
        lower = text
        # Match longer phrases first to avoid partial matches (e.g. "tin" inside "tin tá»©c")
        num_match = re.search(r'(\d+)\s*(?:tin tá»©c|tin bÃ i|báº£n tin|tin|bÃ i|cÃ¡i|má»¥c|máº©u|Ä‘iá»u)', lower, re.IGNORECASE)
        if not num_match:
            num_match = re.search(r'(\d+)', lower)

        if num_match:
            extracted_num = int(num_match.group(1))
            count = max(1, min(15, extracted_num))
            cleaned = lower[:num_match.start()] + lower[num_match.end():]
            return count, cleaned

        return 5, lower

    @staticmethod
    def _extract_search_query_from_news(original: str, cleaned: str) -> str:
        """Extract clean search query from news request by removing filler words."""
        today = datetime.now(_VN_TZ).strftime("%d/%m/%Y")
        normalized_original = PersonalAssistantAgent._normalize_intent_text(original)
        if PersonalAssistantAgent._is_world_news_request(original):
            return f"tin thoi su quoc te the gioi moi nhat hom nay {today} chinh tri ngoai giao an ninh"
        if PersonalAssistantAgent._is_sports_news_request(original):
            return f"tin the thao moi nhat hom nay {today}"
        if "bong da" in normalized_original or "football" in normalized_original or "soccer" in normalized_original:
            return f"tin bong da moi nhat hom nay {today}"

        query = PersonalAssistantAgent._normalize_intent_text(cleaned or original)
        fillers = {
            "lay cho toi", "cho toi", "cho minh", "lay", "tim", "xem",
            "liet ke", "dua cho", "gui cho", "ke cho", "tim cho",
            "get me", "give me", "show me", "find me", "tell me",
            "hay", "giup", "gium", "dum", "di", "nao", "nhe", "a",
            "vai", "mot so", "nhung", "cac", "cai", "ve", "tai", "o",
            "hom nay", "moi nhat", "hot nhat", "hot",
            "today", "latest", "top", "most", "hottest",
        }
        for filler in sorted(fillers, key=len, reverse=True):
            query = re.sub(r"\b" + re.escape(filler) + r"\b", " ", query)
        query = re.sub(r"(?<!tin )\btuc\b", " ", query)
        query = re.sub(r"\s+", " ", query).strip()

        if not query:
            query = PersonalAssistantAgent._normalize_intent_text(original)
        if PersonalAssistantAgent._user_prefers_vietnamese(original):
            if "tin tuc" not in query:
                query = f"tin tuc {query}".strip()
            return f"{query} moi nhat hom nay {today}"
        if "news" not in query:
            query = f"{query} news".strip()
        return f"{query} latest today {today}"

    @staticmethod
    def _display_news_topic(user_text: str, search_query: str) -> str:
        normalized = PersonalAssistantAgent._normalize_intent_text(user_text)
        if PersonalAssistantAgent._is_world_news_request(user_text):
            return "th\u1ebf gi\u1edbi"
        if PersonalAssistantAgent._is_sports_news_request(user_text):
            return "th\u1ec3 thao"
        if "bong da" in normalized or "football" in normalized or "soccer" in normalized:
            return "b\u00f3ng \u0111\u00e1"
        return search_query

    @staticmethod
    def _extract_requested_result_count(text: str, default: int = 5, cap: int = 15) -> int:
        """Extract an explicit requested result count without confusing 24h/date hints."""
        lower = (text or "").lower()
        patterns = [
            r'(?:láº¥y|cho|tÃ³m táº¯t|liá»‡t kÃª|top)\s+(\d+)\s*(?:tin|bÃ i|káº¿t quáº£|má»¥c|nguá»“n)',
            r'(\d+)\s*(?:tin|bÃ i|káº¿t quáº£|má»¥c|nguá»“n)',
            r'top\s+(\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, lower)
            if match:
                return max(1, min(cap, int(match.group(1))))
        return default

    @staticmethod
    def _extract_preferred_name_from_memory_input(user_text: str) -> str:
        """Extract user preferred name/nickname from memory-like statements."""
        text = user_text.strip()
        patterns = [
            r"(?:gọi tôi là|cứ gọi tôi|nickname của tôi là|biệt danh của tôi là)\s+([^\n,.;!?]{2,40})",
            r"(?:tôi tên là|mình tên là|tên mình là)\s+([^\n,.;!?]{2,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                candidate = match.group(1).strip(" .,:;!?\"")
                candidate = re.split(r"\s+(?:nhÃ©|nha|áº¡|Ä‘Ã³|nhÃ¡)$", candidate, maxsplit=1, flags=re.IGNORECASE)[0]
                if 2 <= len(candidate) <= 40:
                    return candidate
        return ""

    def _get_preferred_call_name(self) -> str:
        """Get preferred call name from memory profile if available."""
        profile = self.memory.global_store.profile
        for key in ("nickname", "preferred_name", "name"):
            val = (profile.get(key) or "").strip()
            if val:
                return val
        return ""

    @staticmethod
    def _news_topic_filter_hint(topic_lower: str) -> str:
        if any(kw in topic_lower for kw in _WORLD_TOPIC_MARKERS):
            return (
                "Chá»§ Ä‘á» lÃ  TIN QUá»C Táº¾ / THáº¾ GIá»šI: CHá»ˆ chá»n tin vá» cÃ¡c sá»± kiá»‡n QUá»C Táº¾ "
                "(chiáº¿n tranh, ngoáº¡i giao, kinh táº¿ tháº¿ giá»›i, cÃ´ng nghá»‡ toÃ n cáº§u...). "
                "TUYá»†T Äá»I KHÃ”NG láº¥y tin ná»™i bá»™ Viá»‡t Nam.\n"
            )
        if any(kw in topic_lower for kw in ["viá»‡t nam", "trong nÆ°á»›c", "ná»™i Ä‘á»‹a"]):
            return (
                "Chá»§ Ä‘á» lÃ  TIN TRONG NÆ¯á»šC: CHá»ˆ láº¥y tin vá» Viá»‡t Nam. "
                "Æ¯u tiÃªn cÃ¡c nhÃ³m chá»§ Ä‘á» quá»‘c káº¿ dÃ¢n sinh: chÃ­nh phá»§/chÃ­nh sÃ¡ch, "
                "kinh táº¿-vÄ© mÃ´/giÃ¡ cáº£, giao thÃ´ng-háº¡ táº§ng, Ä‘iá»‡n-nÄƒng lÆ°á»£ng, an sinh-xÃ£ há»™i. "
                "KHÃ”NG Æ°u tiÃªn tin thá»ƒ thao/giáº£i trÃ­/vÄƒn hÃ³a, trá»« khi thiáº¿u dá»¯ liá»‡u Ä‘Ã¡ng tin á»Ÿ nhÃ³m quá»‘c káº¿ dÃ¢n sinh. "
                "KHÃ”NG láº¥y tin dáº¡ng tá»•ng há»£p chung chung, khÃ´ng rÃµ sá»± kiá»‡n.\n"
            )
        return ""

    @staticmethod
    def _required_news_opening(
        preferred_name: str,
        max_results: int,
        topic: str,
        current_date: str,
        target_lang: str = "vi",
        *,
        freshness_hours: int | None = None,
        ranking_mode: str = "hot",
        expanded: bool = False,
    ) -> str:
        topic_lower = (topic or "").lower()
        window_text_en = f"within the last {freshness_hours} hours" if freshness_hours else f"for {current_date}"
        window_text_vi = f"trong {freshness_hours} giá» gáº§n Ä‘Ã¢y" if freshness_hours else f"trong ngÃ y {current_date}"
        if freshness_hours and freshness_hours >= 72:
            window_text_vi = "trong vÃ i ngÃ y gáº§n Ä‘Ã¢y"
            window_text_en = "from the last few days"
        if expanded:
            window_text_vi = f"mÃ¬nh Ä‘Ã£ má»Ÿ rá»™ng khung thá»i gian Ä‘áº¿n {window_text_vi} Ä‘á»ƒ Ä‘á»§ tin liÃªn quan nháº¥t"
            window_text_en = f"I expanded the window to {window_text_en} to find enough relevant source-backed items"
        mode_en = "latest" if ranking_mode == "latest" else ("breaking" if ranking_mode == "breaking" else "relevant and prominent")
        mode_vi = "má»›i nháº¥t" if ranking_mode == "latest" else ("vá»«a xáº£y ra" if ranking_mode == "breaking" else "liÃªn quan vÃ  ná»•i báº­t nháº¥t")
        display_topic_vi = {"sports": "thá»ƒ thao", "football": "bÃ³ng Ä‘Ã¡", "finance": "tÃ i chÃ­nh", "tech": "cÃ´ng nghá»‡", "fashion": "thá»i trang"}.get(topic_lower, topic)
        display_topic_en = {"sports": "sports", "football": "football", "finance": "finance", "tech": "technology", "fashion": "fashion"}.get(topic_lower, topic)
        if target_lang == "en":
            if any(marker in topic_lower for marker in _WORLD_TOPIC_MARKERS):
                base = f"here are the {max_results} {mode_en} world news highlights from sources with concrete URLs, updated {window_text_en}:"
            else:
                base = f"here are up to {max_results} {mode_en} {display_topic_en} news items from sources with concrete URLs, updated {window_text_en}:"
            return f"Hello {preferred_name}, {base}" if preferred_name else base[:1].upper() + base[1:]
        if any(marker in topic_lower for marker in _WORLD_TOPIC_MARKERS):
            base = f"dÆ°á»›i Ä‘Ã¢y lÃ  {max_results} Ä‘iá»ƒm tin tháº¿ giá»›i {mode_vi} tá»« cÃ¡c nguá»“n cÃ³ URL cá»¥ thá»ƒ, cáº­p nháº­t {window_text_vi}:"
        else:
            base = f"dÆ°á»›i Ä‘Ã¢y lÃ  tá»‘i Ä‘a {max_results} tin tá»©c {display_topic_vi} {mode_vi} tá»« cÃ¡c nguá»“n cÃ³ URL cá»¥ thá»ƒ, cáº­p nháº­t {window_text_vi}:"
        return f"ChÃ o {preferred_name}, {base}" if preferred_name else base[:1].upper() + base[1:]

    def _handle_direct_news(self, user_text: str) -> AssistantResponse:
        t0 = time.perf_counter()
        max_results, clean_text = self._extract_news_count(user_text)
        search_query = self._extract_search_query_from_news(user_text, clean_text)
        raw_news = web_search(search_query, max_results=max(max_results * 5, 20))
        initial_rows = self._news_rows_for_answer(raw_news, user_text, max_results) if raw_news else []
        initial_domains = {
            urlparse(row.get("url", "")).netloc.lower().removeprefix("www.")
            for row in initial_rows
            if row.get("url")
        }
        if raw_news and len(initial_rows) >= max(2, max_results // 2) and len(initial_domains) < min(2, max_results):
            dominant_domain = next(iter(initial_domains), "")
            if dominant_domain:
                extra_news = web_search(
                    f"{search_query} -site:{dominant_domain}",
                    max_results=max(max_results * 2, 8),
                )
                if extra_news:
                    raw_news = f"{raw_news}\n\n{extra_news}"
        if raw_news and self._is_world_news_request(user_text):
            preview_rows = self._news_rows_for_answer(raw_news, user_text, max_results)
            repeated_tokens = self._repeated_news_tokens(preview_rows)
            if repeated_tokens:
                exclusions = " ".join(f"-{token}" for token in repeated_tokens[:3])
                extra_news = web_search(
                    f"{search_query} {exclusions}",
                    max_results=max(max_results * 2, 8),
                )
                if extra_news:
                    raw_news = f"{raw_news}\n\n{extra_news}"
        if raw_news:
            topic_phrase = self._news_topic_phrase(user_text)
            today = datetime.now(_VN_TZ).strftime("%d/%m/%Y")
            for attempt in range(2):
                rows_now = self._news_rows_for_answer(raw_news, user_text, max_results)
                if not self._news_needs_more_sources(rows_now, max_results):
                    break
                domains = sorted(self._news_domains(rows_now))
                exclusions = " ".join(f"-site:{domain}" for domain in domains[:5])
                if self._is_world_news_request(user_text):
                    variants = [
                        f"tin quoc te moi nhat hom nay {today} chinh tri ngoai giao an ninh",
                        f"world news latest today {today} politics diplomacy security",
                    ]
                    extra_query = f"{variants[min(attempt, len(variants) - 1)]} {exclusions}".strip()
                elif topic_phrase:
                    if self._user_prefers_vietnamese(user_text):
                        variants = [
                            f"\"{topic_phrase}\" tin tuc moi nhat hom nay {today}",
                            f"\"{topic_phrase}\" tin moi cap nhat {today}",
                        ]
                    else:
                        variants = [
                            f"\"{topic_phrase}\" latest news today {today}",
                            f"\"{topic_phrase}\" breaking news {today}",
                        ]
                    extra_query = f"{variants[min(attempt, len(variants) - 1)]} {exclusions}".strip()
                else:
                    extra_query = f"{search_query} {exclusions}".strip()
                extra_news = web_search(extra_query, max_results=max(max_results * 5, 20))
                if not extra_news or extra_news in raw_news:
                    break
                raw_news = f"{raw_news}\n\n{extra_news}"

        if not raw_news or not raw_news.strip():
            answer = "Mình chưa tìm thấy tin tức phù hợp lúc này."
        else:
            rows = self._news_rows_for_answer(raw_news, user_text, max_results)
            if len(rows) >= max_results:
                answer = self._format_news_rows_from_articles(
                    user_text=user_text,
                    search_query=search_query,
                    rows=rows,
                    max_results=max_results,
                )
                if not answer:
                    answer = "Mình chưa tìm thấy bài báo cụ thể phù hợp để tóm tắt lúc này."
            elif rows:
                answer = self._format_news_rows_from_articles(
                    user_text=user_text,
                    search_query=search_query,
                    rows=rows,
                    max_results=len(rows),
                )
            else:
                answer = "Mình chưa tìm thấy bài báo cụ thể phù hợp để tóm tắt lúc này."

        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        self.memory.append_tool_ledger("web_search", "direct_news", answer[:200], provider="web_search")
        return AssistantResponse(text=answer, tool_events=["web_search"], latency_ms={"total": latency})

    @staticmethod
    def _repeated_news_tokens(rows: list[dict[str, str]]) -> list[str]:
        counts: dict[str, int] = {}
        for row in rows:
            for token in PersonalAssistantAgent._topic_token_set(
                f"{row.get('topic', '')} {row.get('summary', '')} {row.get('url', '')}"
            ):
                counts[token] = counts.get(token, 0) + 1
        ignored = {"news", "html", "ngay", "hom", "nay", "the", "gioi", "tin", "tuc"}
        return [
            token for token, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
            if count >= 2 and token not in ignored
        ]

    def _handle_direct_realtime(self, user_text: str) -> AssistantResponse:
        t0 = time.perf_counter()
        recent_context = self.memory.get_recent_context()
        query_type = self._detect_query_type(user_text)
        search_query = self._extract_search_query(user_text, recent_context)
        search_count = self._extract_requested_result_count(user_text, default=5)
        raw_data = web_search(search_query, max_results=search_count)

        if raw_data:
            task_hint = (
                "Tra loi dung trong tam cau hoi realtime. Neu du lieu co con so, ten banner, event, gia, "
                "lich hoac URL cu the thi trich ra trong cau tra loi. Khong bao nguoi dung tu mo link "
                "khi du lieu da co thong tin can tra loi."
            )
            arknights_query = "arknights" in f"{user_text} {search_query}".lower() and "endfield" not in f"{user_text} {search_query}".lower()
            if query_type == "game" and arknights_query:
                task_hint += (
                    " Rieng Arknights: phai tach ro Banner va Su kien in-game. "
                    "Voi Banner, neu du lieu co operator/rate-up thi neu ten operator tieu bieu. "
                    "Voi Su kien, uu tien current in-game event/side story dang dien ra va upcoming event; "
                    "neu thay '[Rerun] Such is the Joy of Our Reunion Rerun' hoac '[Festival] Ashes to Ashes, Ages on Ages' "
                    "trong du lieu thi phai neu dung ten do, khong chi noi chung chung."
                )
            answer = self._answer_from_search(
                user_text=user_text,
                search_query=search_query,
                raw_data=raw_data,
                tool_name="web_search",
                task_hint=task_hint,
                max_chars=max(settings.max_response_chars, search_count * 600 + 1000),
            )
        else:
            answer = "Mình chưa tìm thấy thông tin phù hợp lúc này."

        if query_type == "game":
            for game_name in [
                "Zenless Zone Zero", "Wuthering Waves", "Honkai Star Rail",
                "Genshin Impact", "Arknights", "Arknights Endfield", "Fate Grand Order",
            ]:
                if game_name.lower() in search_query.lower():
                    self.memory.set_recent_context(game_name, "game banner/event", user_text)
                    break

        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        self.memory.append_tool_ledger("web_search", f"direct_realtime_{query_type}", answer[:200], provider="web_search")
        return AssistantResponse(text=answer, tool_events=["web_search"], latency_ms={"total": latency})

    def _handle_direct_weather(self, user_text: str) -> AssistantResponse:
        t0 = time.perf_counter()
        location = self._extract_weather_location(user_text)
        try:
            answer = get_weather(location)
        except Exception:
            logger.exception("Weather tool failed")
            answer = f"Mình chưa lấy được dữ liệu thời tiết cho {location} lúc này."
        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        self.memory.append_tool_ledger("get_weather", "weather", answer[:200], provider="open-meteo")
        return AssistantResponse(text=answer, tool_events=["get_weather"], latency_ms={"total": latency})

    def _handle_direct_stock_price(self, user_text: str) -> AssistantResponse:
        t0 = time.perf_counter()
        ticker = extract_stock_ticker(user_text) or user_text.strip()
        answer = get_stock_price(ticker)
        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        self.memory.append_tool_ledger("get_stock_price", "stock_price", answer[:200], provider="yahoo_finance")
        return AssistantResponse(text=answer, tool_events=["get_stock_price"], latency_ms={"total": latency})

    def _handle_direct_memory(self, user_text: str) -> AssistantResponse:
        t0 = time.perf_counter()
        fact = user_text.strip()
        preferred = self._extract_preferred_name_from_memory_input(user_text)
        if preferred:
            self.memory.set_profile("name", preferred)
            fact = f"Tên người dùng là {preferred}"
        self.memory.remember_fact(fact)
        answer = save_memory(fact)
        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        self.memory.append_tool_ledger("save_memory", "memory", answer[:200])
        return AssistantResponse(text=answer, tool_events=["save_memory"], latency_ms={"total": latency})

    def _handle_direct_music(self, user_text: str) -> AssistantResponse:
        t0 = time.perf_counter()
        lower = user_text.lower()
        if self._has_keyword_match(user_text, MUSIC_STOP_KEYWORDS):
            song = self._extract_music_song(user_text, MUSIC_STOP_KEYWORDS)
            answer = stop_music(song_hint=song if song != user_text else None)
            events = ["stop_music"]
        elif self._has_keyword_match(user_text, MUSIC_PAUSE_KEYWORDS):
            answer = pause_music()
            events = ["pause_music"]
        elif self._has_keyword_match(user_text, MUSIC_RESUME_KEYWORDS):
            answer = resume_music()
            events = ["resume_music"]
        else:
            song = self._extract_music_song(user_text, MUSIC_KEYWORDS + MUSIC_SWITCH_KEYWORDS)
            answer = play_music(song)
            events = ["play_music"]
        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        self.memory.append_tool_ledger(events[0], "music", answer[:200])
        return AssistantResponse(text=answer, tool_events=events, latency_ms={"total": latency})

    def _handle_general(self, user_text: str) -> AssistantResponse:
        t0 = time.perf_counter()
        history = self._filter_history_for_api(self.memory.get_message_history())
        try:
            result = self._run_llm_with_retry(user_text, message_history=history)
            answer = self._normalize_answer(result.output)
            tool_events = [
                getattr(part, "tool_name", "")
                for msg in getattr(result, "new_messages", lambda: [])()
                for part in getattr(msg, "parts", [])
                if getattr(part, "part_kind", "") == "tool-call" and getattr(part, "tool_name", "")
            ]
        except Exception:
            logger.exception("General LLM route failed")
            answer = "Mình chưa xử lý được câu hỏi này lúc này."
            tool_events = []
        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        if tool_events:
            self.memory.append_tool_ledger(tool_events[-1], "general", answer[:200])
        return AssistantResponse(text=answer, tool_events=tool_events, latency_ms={"total": latency})

    def run(self, user_text: str) -> AssistantResponse:
        if not user_text or not user_text.strip():
            return AssistantResponse(text="Mình chưa nghe rõ, bạn nói lại giúp mình nhé.", latency_ms={"total": 0})

        user_text = user_text.strip()
        logger.info("User: %s", user_text[:120])

        if self._should_route_music(user_text):
            return self._handle_direct_music(user_text)
        if self._detect_query_type(user_text) == "stock" and not self._is_analytical_query(user_text):
            return self._handle_direct_stock_price(user_text)
        if self._should_route_news_directly(user_text):
            return self._handle_direct_news(user_text)
        if self._should_route_weather_directly(user_text):
            return self._handle_direct_weather(user_text)
        if self._should_save_memory_directly(user_text):
            return self._handle_direct_memory(user_text)
        if self._should_route_realtime_directly(user_text):
            return self._handle_direct_realtime(user_text)
        return self._handle_general(user_text)

