"""Personal Assistant Agent powered by Pydantic-AI — LLM-first (true agent).

Architecture:
- The LLM is the single router: it reads the user's question, chooses the right
  tool(s) via function calling, then synthesizes the final answer from the tool
  evidence.
- No hardcoded keyword routing. The system prompt + tool docstrings are the
  single source of truth for when/why a tool is used.
- Deterministic, heavy work (news search, web search, weather, stock, music,
  exchange rates) lives inside tools (core.tools / core.search), not here.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from core.settings import settings
from core.memory import AssistantMemory
from core.models import AssistantResponse
from core.tools import (
    get_weather,
    save_memory,
    web_search,
    get_current_datetime,
    calculate,
    translate_text,
    knowledge_search,
    get_exchange_rate,
    play_music,
    stop_music,
    pause_music,
    resume_music,
    get_stock_price,
)

logger = logging.getLogger(__name__)

_VN_TZ = timezone(timedelta(hours=7))
_VN_WEEKDAYS = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

REGEX_NEWS_NUMBERED_ITEM = r'(?m)^\s*(?:[1-9]|1[0-5])\.\s+\S+'


def _get_current_datetime_str() -> str:
    """Get current datetime string in Vietnamese for system prompt injection."""
    now = datetime.now(_VN_TZ)
    weekday = _VN_WEEKDAYS[now.weekday()]
    return (
        f"Thời điểm hiện tại: {now.strftime('%H:%M:%S')}, "
        f"{weekday}, ngày {now.strftime('%d')} tháng {now.strftime('%m')} năm {now.strftime('%Y')} "
        f"(giờ Việt Nam, UTC+7). "
        f"ĐÂY LÀ THỜI GIAN THỰC, KHÔNG PHẢI TƯƠNG LAI."
    )


# NOTE: {current_datetime} is intentionally the LAST token. Keeping it here makes
# the whole prompt prefix above stable so the provider can reuse its prompt/KV cache.
SYSTEM_PROMPT_TEMPLATE = (
    "You are an intelligent personal AI assistant acting as a tool-using agent. "
    "Always follow this workflow: (1) read the user's message and identify the intent; "
    "(2) call the appropriate tool(s) and read their output; "
    "(3) answer using only that evidence plus stable built-in knowledge. "
    "TOOL SELECTION GUIDE (call the right tool for the intent): "
    "web search and news (realtime facts, prices like xăng/vàng/crypto, current events, 'tin tức', politics, sports, world or Vietnam headlines, people, definitions not in the knowledge base) → web_search(query); "
    "weather or forecast → get_weather(location); "
    "stock quotes, 'cổ phiếu', 'chứng khoán', market indexes → get_stock_price(ticker); "
    "music: play/nghe a song → play_music(url_or_query); 'tắt nhạc'/'đóng nhạc' (turn music OFF, close the window entirely) → stop_music(song_hint); 'tạm dừng'/'dừng nhạc'/'ngừng phát giữa chừng' (pause mid-song, window stays open) → pause_music(); 'tiếp tục'/'phát tiếp' → resume_music(); "
    "save a fact about the user ('nhớ rằng', name, preferences) → save_memory(fact); "
    "current date/time → get_current_datetime(); "
    "math → calculate(expression); "
    "currency exchange rate → get_exchange_rate(currency_from, currency_to); "
    "translation → translate_text(text, target_lang); "
    "local knowledge-base questions → knowledge_search(query). "
    "If no tool fits, answer from your own knowledge. "
    "ANSWER RULES: "
    "Priorities: 1) correctness, 2) relevance, 3) clarity, 4) brevity. "
    "Respond in the user's language and never switch unless asked. "
    "Summarize tool results into a clean, natural answer; never paste raw tool output "
    "and never mention tool names or internal mechanics. "
    "Do not invent numbers, dates, names, prices, or sources; when evidence is weak or "
    "sources conflict, state the uncertainty instead of guessing. "
    "Always cite factual claims with an inline markdown link [Title/Domain](URL) when a URL is available. "
    "When the user asks for a specific number of items (e.g. '5 tin', '3 giá'), return exactly that many "
    "distinct items when enough evidence exists; otherwise say how many you found and list them all. "
    "Ignore results that are category pages, RSS feeds, videos, spam, or unrelated. "
    "For 'hôm nay'/'hiện tại'/'mới nhất' requests, do not present stale data as current; "
    "if only stale data exists, say clearly that today's data was not found. "
    "For news, prefer article-like results with concrete URLs. "
    "NEWS FORMAT (STRICT): If the user asks for N news items and gives no topic ('chủ đề bạn muốn' / 'tùy bạn'), pick ONE topic and announce it in one line. "
    "Then write exactly N items, each on its own line, numbered 1), 2), 3) ... with NO bullets and NO '- ' prefix, in exactly this shape: "
    "**\"[Short topic]\"** : [Summary]. [Nguồn : (Source name) - (Article title as hyperlink)]. "
    "Short topic = 3-6 Vietnamese words capturing the heart of the article, evocative and NOT a verbatim copy of the original headline. "
    "Summary = EXACTLY 2 sentences (~40-55 words total), linked naturally: sentence 1 states the subject + its most notable new action/event; sentence 2 states the impact, notable change, or reactions from stakeholders. "
    "Write full subject-verb sentences with crisp rhythm, never terse or formulaic; do NOT insert trivial numbers/details just to bait a click. "
    "Source name = the outlet's display name (e.g. VnExpress, Tuổi Trẻ, Dân Trí); Article title = the original headline, rendered as a markdown hyperlink to its URL. "
    "Concrete example: 1) **\"Triều Tiên thử tên lửa dồn dập\"** : Triều Tiên phóng liên tiếp hai tên lửa đạn đạo ra vùng biển phía đông chỉ trong 3 giờ nhằm phô diễn năng lực hạt nhân. Động thái hiếm thấy này lập tức vấp phải sự cảnh giác và lên án gay gắt từ Hàn Quốc cùng Nhật Bản. [Nguồn : VnExpress - Triều Tiên phóng tên lửa liên tiếp trong vòng 3 giờ](https://example.com/bai-viet). "
    "Rules: never reuse the same URL for two items; never copy a generic aggregator headline like 'Tin nóng thế giới ngày X' or 'Tổng hợp tin tức...'; never use '(via Source)' or '([Source](url))'; do not add a closing question. {current_datetime}"
)


@dataclass
class AgentDeps:
    """Dependencies injected into the Pydantic-AI agent at runtime."""
    memory: AssistantMemory


def _build_openai_compatible_model() -> OpenAIChatModel:
    provider = OpenAIProvider(
        base_url=settings.base_url.rstrip("/"),
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
            "max_tokens": settings.llm_max_completion_tokens,
            "timeout": settings.request_timeout,
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


class PersonalAssistantAgent:
    def __init__(self) -> None:
        self.memory = AssistantMemory()
        self._agent = _build_pydantic_agent()
        self.memory.sync_music_state_to_tools()

    @staticmethod
    def _is_news_response(text: str) -> bool:
        """Detect if text is a numbered news list (e.g. '1. ...\n\n2. ...')."""
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

    def run(self, user_text: str) -> AssistantResponse:
        """Handle one user message end-to-end through the LLM agent."""
        if not user_text or not user_text.strip():
            return AssistantResponse(
                text="Mình chưa nghe rõ, bạn nói lại giúp mình nhé.",
                latency_ms={"total": 0},
            )

        user_text = user_text.strip()
        logger.info("User: %s", user_text[:120])

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
            logger.exception("LLM agent failed")
            answer = "Mình chưa xử lý được câu hỏi này lúc này."
            tool_events = []

        latency = int((time.perf_counter() - t0) * 1000)
        self.memory.add_action_to_history(user_text, answer)
        if tool_events:
            self.memory.append_tool_ledger(tool_events[-1], "agent", answer[:200])
        return AssistantResponse(text=answer, tool_events=tool_events, latency_ms={"total": latency})


