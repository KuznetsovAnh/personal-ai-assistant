"""Shared dataclasses for assistant responses and chat history.

This module contains lightweight transport models used across the assistant
runtime. The classes here intentionally have no business logic so they remain
safe to import from UI, agent, memory, and test modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class AssistantResponse:
    text: str
    tool_events: List[str] = field(default_factory=list)
    latency_ms: Dict[str, int] = field(default_factory=dict)
