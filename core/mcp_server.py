"""MCP Server â€” expose all assistant tools via Model Context Protocol.

Run standalone:
    python -m core.mcp_server

Or import and attach to the Pydantic-AI agent for MCP-native tool routing.
"""

from __future__ import annotations

from fastmcp import FastMCP

from core.tools import (
    calculate,
    get_current_datetime,
    get_weather,
    save_memory,
    web_search,
    knowledge_search,
    translate_text,
    get_exchange_rate,
    play_music,
    stop_music,
    pause_music,
    resume_music,
    get_stock_price,
)

mcp = FastMCP("ROBOTS Personal Assistant")


# â”€â”€ Weather â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_get_weather(location: str) -> str:
    """Láº¥y thá»i tiáº¿t hiá»‡n táº¡i vÃ  dá»± bÃ¡o hÃ´m nay theo Ä‘á»‹a Ä‘iá»ƒm.

    Args:
        location: TÃªn thÃ nh phá»‘ hoáº·c Ä‘á»‹a Ä‘iá»ƒm cáº§n láº¥y thá»i tiáº¿t.
    """
    return get_weather(location)


# â”€â”€ Web Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_web_search(query: str) -> str:
    """TÃ¬m kiáº¿m thÃ´ng tin má»›i nháº¥t trÃªn web.

    Args:
        query: Truy váº¥n cáº§n tÃ¬m kiáº¿m trÃªn web.
    """
    return web_search(query)


# â”€â”€ Save Memory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_save_memory(fact: str) -> str:
    """LÆ°u má»™t sá»± tháº­t hoáº·c sá»Ÿ thÃ­ch quan trá»ng cá»§a ngÆ°á»i dÃ¹ng vÃ o bá»™ nhá»› dÃ i háº¡n.

    Args:
        fact: ThÃ´ng tin ngáº¯n gá»n cáº§n ghi nhá»› vá» ngÆ°á»i dÃ¹ng.
    """
    return save_memory(fact)


# â”€â”€ Current Datetime â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_get_current_datetime() -> str:
    """Láº¥y ngÃ y giá» hiá»‡n táº¡i theo mÃºi giá» Viá»‡t Nam (UTC+7)."""
    return get_current_datetime()


# â”€â”€ Calculate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_calculate(expression: str) -> str:
    """TÃ­nh toÃ¡n biá»ƒu thá»©c toÃ¡n há»c Ä‘Æ¡n giáº£n.

    Args:
        expression: Biá»ƒu thá»©c toÃ¡n há»c cáº§n tÃ­nh (vÃ­ dá»¥: '1+1', '15*3', '100/4').
    """
    return calculate(expression)


# â”€â”€ Translate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_translate_text(text: str, target_lang: str = "en") -> str:
    """Dá»‹ch vÄƒn báº£n sang ngÃ´n ngá»¯ khÃ¡c.

    Args:
        text: VÄƒn báº£n cáº§n dá»‹ch.
        target_lang: MÃ£ ngÃ´n ngá»¯ Ä‘Ã­ch (en, vi, ja, ko, zh, fr, de, ...).
    """
    return translate_text(text, target_lang)


# â”€â”€ Knowledge Search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_knowledge_search(query: str, topic: str = "general") -> str:
    """TÃ¬m kiáº¿m kiáº¿n thá»©c chuyÃªn sÃ¢u vá» má»™t chá»§ Ä‘á» cá»¥ thá»ƒ.

    Args:
        query: CÃ¢u há»i hoáº·c chá»§ Ä‘á» cáº§n tÃ¬m kiáº¿m.
        topic: LÄ©nh vá»±c (general, science, tech, history, geography).
    """
    return knowledge_search(query, topic)


# â”€â”€ Exchange Rate â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_get_exchange_rate(base_currency: str = "USD", target_currency: str = "VND") -> str:
    """Láº¥y tá»· giÃ¡ ngoáº¡i tá»‡ má»›i nháº¥t.

    Args:
        base_currency: MÃ£ tiá»n tá»‡ gá»‘c (vÃ­ dá»¥: USD, EUR, JPY, GBP).
        target_currency: MÃ£ tiá»n tá»‡ Ä‘Ã­ch (vÃ­ dá»¥: VND, USD, EUR).
    """
    return get_exchange_rate(base_currency, target_currency)


# â”€â”€ Music_tool / Play Music â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_play_music(url: str) -> str:
    """Music_tool.play_music â€” phÃ¡t nháº¡c tá»« URL (YouTube, SoundCloud, hoáº·c link MP3 trá»±c tiáº¿p).

    Args:
        url: ÄÆ°á»ng link bÃ i hÃ¡t (YouTube URL, SoundCloud URL, hoáº·c link MP3).
    """
    return play_music(url)


# â”€â”€ Music_tool / Stop Music â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_stop_music(song_hint: str = "") -> str:
    """Music_tool.stop_music â€” táº¯t nháº¡c vÃ  Ä‘Ã³ng cá»­a sá»• nháº¡c Ä‘ang phÃ¡t (Æ°u tiÃªn match theo tÃªn bÃ i náº¿u cÃ³).

    Args:
        song_hint: Tá»« khÃ³a tÃªn bÃ i hÃ¡t Ä‘á»ƒ Ä‘Ã³ng Ä‘Ãºng cá»­a sá»• tÆ°Æ¡ng á»©ng.
    """
    return stop_music(song_hint=song_hint or None)


# â”€â”€ Music_tool / Pause Music â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_pause_music() -> str:
    """Music_tool.pause_music â€” táº¡m dá»«ng nháº¡c Ä‘ang phÃ¡t (khÃ´ng Ä‘Ã³ng tab)."""
    return pause_music()


# â”€â”€ Music_tool / Resume Music â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_resume_music() -> str:
    """Music_tool.resume_music â€” tiáº¿p tá»¥c phÃ¡t nháº¡c Ä‘ang táº¡m dá»«ng (khÃ´ng má»Ÿ tab má»›i)."""
    return resume_music()


# â”€â”€ Stock Price â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.tool()
def mcp_get_stock_price(ticker: str) -> str:
    """Láº¥y giÃ¡ cá»• phiáº¿u realtime tá»« CafeF (VN) hoáº·c Yahoo Finance (quá»‘c táº¿).

    Args:
        ticker: MÃ£ cá»• phiáº¿u (VD: VIC, VNM, HPG, AAPL, NVDA, MSFT).
    """
    return get_stock_price(ticker)


# â”€â”€ MCP Resources (contextual info) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@mcp.resource("assistant://info")
def assistant_info() -> str:
    """ThÃ´ng tin vá» assistant hiá»‡n táº¡i."""
    from core.settings import settings
    return (
        f"ROBOTS Personal Assistant\n"
        f"Model: {settings.model}\n"
        f"Profile: {settings.runtime_profile}\n"
        f"Max tokens: {settings.llm_max_completion_tokens}"
    )


if __name__ == "__main__":
    mcp.run()
