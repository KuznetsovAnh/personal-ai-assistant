"""MCP Server — expose all assistant tools via Model Context Protocol.

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


# ── Weather ──────────────────────────────────────────────────────────────────
@mcp.tool()
def mcp_get_weather(location: str) -> str:
    """Lấy thời tiết hiện tại và dự báo hôm nay theo địa điểm.

    Args:
        location: Tên thành phố hoặc địa điểm cần lấy thời tiết.
    """
    return get_weather(location)


# ── Web Search ───────────────────────────────────────────────────────────────
@mcp.tool()
def mcp_web_search(query: str) -> str:
    """Tìm kiếm thông tin mới nhất trên web.

    Args:
        query: Truy vấn cần tìm kiếm trên web.
    """
    return web_search(query)


# ── Save Memory ──────────────────────────────────────────────────────────────
@mcp.tool()
def mcp_save_memory(fact: str) -> str:
    """Lưu một sự thật hoặc sở thích quan trọng của người dùng vào bộ nhớ dài hạn.

    Args:
        fact: Thông tin ngắn gọn cần ghi nhớ về người dùng.
    """
    return save_memory(fact)


# ── Current Datetime ─────────────────────────────────────────────────────────
@mcp.tool()
def mcp_get_current_datetime() -> str:
    """Lấy ngày giờ hiện tại theo múi giờ Việt Nam (UTC+7)."""
    return get_current_datetime()


# ── Calculate ────────────────────────────────────────────────────────────────
@mcp.tool()
def mcp_calculate(expression: str) -> str:
    """Tính toán biểu thức toán học đơn giản.

    Args:
        expression: Biểu thức toán học cần tính (ví dụ: '1+1', '15*3', '100/4').
    """
    return calculate(expression)


# ── Translate ────────────────────────────────────────────────────────────────
@mcp.tool()
def mcp_translate_text(text: str, target_lang: str = "en") -> str:
    """Dịch văn bản sang ngôn ngữ khác.

    Args:
        text: Văn bản cần dịch.
        target_lang: Mã ngôn ngữ đích (en, vi, ja, ko, zh, fr, de, ...).
    """
    return translate_text(text, target_lang)


# ── Knowledge Search ─────────────────────────────────────────────────────────
@mcp.tool()
def mcp_knowledge_search(query: str, topic: str = "general") -> str:
    """Tìm kiếm kiến thức chuyên sâu về một chủ đề cụ thể.

    Args:
        query: Câu hỏi hoặc chủ đề cần tìm kiếm.
        topic: Lĩnh vực (general, science, tech, history, geography).
    """
    return knowledge_search(query, topic)


# ── Exchange Rate ─────────────────────────────────────────────────────────────
@mcp.tool()
def mcp_get_exchange_rate(base_currency: str = "USD", target_currency: str = "VND") -> str:
    """Lấy tỷ giá ngoại tệ mới nhất.

    Args:
        base_currency: Mã tiền tệ gốc (ví dụ: USD, EUR, JPY, GBP).
        target_currency: Mã tiền tệ đích (ví dụ: VND, USD, EUR).
    """
    return get_exchange_rate(base_currency, target_currency)


# ── Music_tool / Play Music ───────────────────────────────────────────────────
@mcp.tool()
def mcp_play_music(url: str) -> str:
    """Music_tool.play_music — phát nhạc từ URL (YouTube, SoundCloud, hoặc link MP3 trực tiếp).

    Args:
        url: Đường link bài hát (YouTube URL, SoundCloud URL, hoặc link MP3).
    """
    return play_music(url)


# ── Music_tool / Stop Music ───────────────────────────────────────────────────
@mcp.tool()
def mcp_stop_music(song_hint: str = "") -> str:
    """Music_tool.stop_music — tắt nhạc và đóng cửa sổ nhạc đang phát (ưu tiên match theo tên bài nếu có).

    Args:
        song_hint: Từ khóa tên bài hát để đóng đúng cửa sổ tương ứng.
    """
    return stop_music(song_hint=song_hint or None)


# ── Music_tool / Pause Music ──────────────────────────────────────────────────
@mcp.tool()
def mcp_pause_music() -> str:
    """Music_tool.pause_music — tạm dừng nhạc đang phát (không đóng tab)."""
    return pause_music()


# ── Music_tool / Resume Music ─────────────────────────────────────────────────
@mcp.tool()
def mcp_resume_music() -> str:
    """Music_tool.resume_music — tiếp tục phát nhạc đang tạm dừng (không mở tab mới)."""
    return resume_music()


# ── Stock Price ───────────────────────────────────────────────────────────────
@mcp.tool()
def mcp_get_stock_price(ticker: str) -> str:
    """Lấy giá cổ phiếu realtime từ CafeF (VN) hoặc Yahoo Finance (quốc tế).

    Args:
        ticker: Mã cổ phiếu (VD: VIC, VNM, HPG, AAPL, NVDA, MSFT).
    """
    return get_stock_price(ticker)


# ── MCP Resources (contextual info) ─────────────────────────────────────────
@mcp.resource("assistant://info")
def assistant_info() -> str:
    """Thông tin về assistant hiện tại."""
    from core.settings import settings
    return (
        f"ROBOTS Personal Assistant\n"
        f"Model: {settings.model}\n"
        f"Profile: {settings.runtime_profile}\n"
        f"Max tokens: {settings.llm_max_completion_tokens}"
    )


if __name__ == "__main__":
    mcp.run()
