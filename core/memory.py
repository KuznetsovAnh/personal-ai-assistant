"""Memory manager for the personal assistant.

MULTI-CONVERSATION ARCHITECTURE:
├── data/memory.json              # GLOBAL: profile, facts, action_states (music state!)
└── data/conversations/
    ├── index.json                # List of all conversations
    ├── <uuid>.json               # Each conversation: history + recent_context
    └── ...

Key principles:
- Action states (music playing/paused) are GLOBAL — shared across ALL conversations
- Conversation history + recent_context are PER-CONVERSATION — isolated per chat
- When Ctrl+C / terminal closes → current chat saved, music state persists globally
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from core.settings import settings

_VN_TZ = timezone(timedelta(hours=7))


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class ConversationInfo:
    """Metadata for a single conversation."""
    id: str
    name: str
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass(slots=True)
class ConversationStore:
    """Per-conversation memory: conversation history + recent context."""
    conversation: List[ModelMessage] = field(default_factory=list)
    recent_context: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GlobalMemoryStore:
    """Global memory shared across ALL conversations."""
    memory_version: int = 4
    profile: Dict[str, str] = field(default_factory=dict)
    facts: List[str] = field(default_factory=list)
    action_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_ledger: List[Dict[str, Any]] = field(default_factory=list)
    active_conversation_id: str = ""  # Last used conversation


# ══════════════════════════════════════════════════════════════════════════════
# AssistantMemory — Multi-conversation manager
# ══════════════════════════════════════════════════════════════════════════════

class AssistantMemory:
    def __init__(
        self,
        file_path: Path | None = None,
        max_messages: int | None = None,
    ) -> None:
        self.file_path = file_path or settings.memory_file
        self.max_messages = max_messages or settings.memory_max_messages
        self._lock = threading.Lock()

        # Conversation directory
        self.conv_dir = self.file_path.parent / "conversations"
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.conv_dir / "index.json"

        # Load global memory
        self.global_store = self._load_global()

        # Load or create conversation
        self.current_conversation_id = self.global_store.active_conversation_id or ""
        self.conv_store = self._load_conversation(self.current_conversation_id)

        # Sync music state to tools on startup
        self.sync_music_state_to_tools()

    # ── Global Memory ────────────────────────────────────────────────────────

    def _load_global(self) -> GlobalMemoryStore:
        if not self.file_path.exists():
            return GlobalMemoryStore()

        data = json.loads(self.file_path.read_text(encoding="utf-8"))
        return GlobalMemoryStore(
            memory_version=4 if data.get("memory_version", 1) < 4 else data["memory_version"],
            profile=data.get("profile", {}),
            facts=data.get("facts", []),
            action_states=data.get("action_states", {}),
            tool_ledger=data.get("tool_ledger", [])[:300],
            active_conversation_id=data.get("active_conversation_id", ""),
        )

    def _save_global(self) -> None:
        from pydantic_core import to_jsonable_python

        payload = {
            "memory_version": self.global_store.memory_version,
            "profile": self.global_store.profile,
            "facts": self.global_store.facts,
            "action_states": self.global_store.action_states,
            "tool_ledger": self.global_store.tool_ledger[-300:],
            "active_conversation_id": self.global_store.active_conversation_id,
        }
        self.file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Conversation Management ──────────────────────────────────────────────

    def _load_conversation(self, conv_id: str) -> ConversationStore:
        if not conv_id:
            return ConversationStore()

        conv_file = self.conv_dir / f"{conv_id}.json"
        if not conv_file.exists():
            return ConversationStore()

        data = json.loads(conv_file.read_text(encoding="utf-8"))
        conversation: List[ModelMessage] = []
        raw_conv = data.get("conversation", [])
        if raw_conv:
            try:
                conversation = ModelMessagesTypeAdapter.validate_python(raw_conv)
                conversation = self._clean_tool_messages(conversation)
            except Exception:
                conversation = []

        return ConversationStore(
            conversation=conversation,
            recent_context=data.get("recent_context", {}),
        )

    def _save_conversation(self) -> None:
        if not self.current_conversation_id:
            return

        conv_file = self.conv_dir / f"{self.current_conversation_id}.json"
        from pydantic_core import to_jsonable_python

        payload = {
            "conversation": to_jsonable_python(self.conv_store.conversation),
            "recent_context": self.conv_store.recent_context,
        }
        conv_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self.index_file.exists():
            return []
        data = json.loads(self.index_file.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _save_index(self, index: List[Dict[str, Any]]) -> None:
        self.index_file.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Public API: Conversation CRUD ────────────────────────────────────────

    def get_active_conversation_id(self) -> str:
        """Get current conversation ID."""
        return self.current_conversation_id

    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations (sorted by updated_at, newest first)."""
        index = self._load_index()
        index.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return index

    def create_conversation(self, name: str = "Cuộc trò chuyện mới") -> str:
        """Create a new conversation and switch to it."""
        conv_id = str(uuid.uuid4())[:8]
        now = datetime.now(_VN_TZ).isoformat()

        # Add to index
        index = self._load_index()
        index.append({
            "id": conv_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        })
        self._save_index(index)

        # Switch to new conversation
        self.current_conversation_id = conv_id
        self.conv_store = ConversationStore()
        self.global_store.active_conversation_id = conv_id
        self._save_global()

        return conv_id

    def switch_conversation(self, conv_id: str) -> bool:
        """Switch to an existing conversation. Returns True if successful."""
        # Save current conversation first
        self._save_conversation()

        # Load target conversation
        conv_file = self.conv_dir / f"{conv_id}.json"
        if not conv_file.exists():
            return False

        self.current_conversation_id = conv_id
        self.conv_store = self._load_conversation(conv_id)
        self.global_store.active_conversation_id = conv_id
        self._save_global()
        return True

    def rename_conversation(self, conv_id: str, new_name: str) -> bool:
        """Rename a conversation."""
        index = self._load_index()
        for conv in index:
            if conv["id"] == conv_id:
                conv["name"] = new_name
                self._save_index(index)
                return True
        return False

    def delete_conversation(self, conv_id: str) -> bool:
        """Delete a conversation. If it's the active one, create a new one."""
        index = self._load_index()
        index = [c for c in index if c["id"] != conv_id]
        self._save_index(index)

        # Delete conversation file
        conv_file = self.conv_dir / f"{conv_id}.json"
        if conv_file.exists():
            conv_file.unlink()

        # If deleted active conversation, create new one
        if self.current_conversation_id == conv_id:
            self.current_conversation_id = ""
            self.conv_store = ConversationStore()
            self.create_conversation()
            return True

        return True

    def auto_rename_first_message(self, conv_id: str, user_message: str) -> None:
        """Auto-rename conversation from first user message (if still default name)."""
        index = self._load_index()
        for conv in index:
            if conv["id"] == conv_id and conv["name"] == "Cuộc trò chuyện mới":
                # Use first 50 chars of message as name
                name = user_message.strip()[:50]
                if len(user_message) > 50:
                    name += "..."
                conv["name"] = name
                self._save_index(index)
                break

    def update_conversation_timestamp(self) -> None:
        """Update the updated_at timestamp for current conversation."""
        if not self.current_conversation_id:
            return
        index = self._load_index()
        now = datetime.now(_VN_TZ).isoformat()
        for conv in index:
            if conv["id"] == self.current_conversation_id:
                conv["updated_at"] = now
                conv["message_count"] = conv.get("message_count", 0) + 1
                self._save_index(index)
                break

    # ── Conversation Content ─────────────────────────────────────────────────

    def get_message_history(self) -> List[ModelMessage]:
        """Return current conversation history."""
        return list(self.conv_store.conversation)

    def set_message_history(self, messages: List[ModelMessage]) -> None:
        """Replace conversation history."""
        self.conv_store.conversation = messages[-self.max_messages:]
        self._save_conversation()

    def get_recent_context(self) -> Dict[str, str]:
        """Get recent context for follow-up resolution."""
        return dict(self.conv_store.recent_context)

    def set_recent_context(self, entity: str, topic: str, query: str) -> None:
        """Store context for follow-up query resolution."""
        self.conv_store.recent_context = {
            "entity": entity.strip(),
            "topic": topic.strip(),
            "query": query.strip(),
        }
        self._save_conversation()

    def clear_recent_context(self) -> None:
        """Clear recent context."""
        self.conv_store.recent_context = {}
        self._save_conversation()

    def add_action_to_history(self, user_text: str, assistant_text: str) -> None:
        """Add user/assistant pair to conversation history."""
        now = datetime.now(_VN_TZ)

        user_msg = ModelRequest(
            parts=[UserPromptPart(content=user_text, timestamp=now)],
        )
        assistant_msg = ModelResponse(
            parts=[TextPart(content=assistant_text)],
        )

        self.conv_store.conversation.append(user_msg)
        self.conv_store.conversation.append(assistant_msg)

        if len(self.conv_store.conversation) > self.max_messages:
            self.conv_store.conversation = self.conv_store.conversation[-self.max_messages:]

        self._save_conversation()
        self.update_conversation_timestamp()

    # ── Global Memory (profile, facts, action_states) ────────────────────────

    def remember_fact(self, fact: str) -> None:
        normalized = fact.strip()
        if normalized and normalized not in self.global_store.facts:
            self.global_store.facts.append(normalized)
            self._save_global()

    def set_profile(self, key: str, value: str) -> None:
        self.global_store.profile[key] = value.strip()
        self._save_global()

    def get_action_state(self, action_type: str) -> Dict[str, Any]:
        return dict(self.global_store.action_states.get(action_type, {}))

    def set_action_state(self, action_type: str, state: Dict[str, Any]) -> None:
        with self._lock:
            self.global_store.action_states[action_type] = state
            self._save_global()

    def clear_action_state(self, action_type: str) -> None:
        with self._lock:
            self.global_store.action_states.pop(action_type, None)
            self._save_global()

    def append_tool_ledger(self, tool_name: str, intent: str, summary: str, provider: str = "", status: str = "ok") -> None:
        now = datetime.now(_VN_TZ).isoformat()
        entry = {
            "ts": now,
            "tool": tool_name,
            "intent": intent,
            "summary": summary[:400],
            "provider": provider,
            "status": status,
        }
        with self._lock:
            self.global_store.tool_ledger.append(entry)
            if len(self.global_store.tool_ledger) > 300:
                self.global_store.tool_ledger = self.global_store.tool_ledger[-300:]
            self._save_global()

    # ── Music State Sync ─────────────────────────────────────────────────────

    def sync_music_state_to_tools(self) -> None:
        """Restore tools.py music globals from persisted action state."""
        import core.tools as tools
        state = self.get_action_state("music")
        if not state:
            return
        tools._music_is_active = state.get("active", False)
        tools._music_is_paused = state.get("paused", False)
        tools._music_song_name = state.get("song", None)

    def sync_music_state_from_tools(self) -> None:
        """Save tools.py music globals to persisted action state."""
        import core.tools as tools
        if tools._music_is_active:
            self.set_action_state("music", {
                "active": tools._music_is_active,
                "paused": tools._music_is_paused,
                "song": tools._music_song_name,
            })
        else:
            self.clear_action_state("music")

    # ── System Prompt Context ────────────────────────────────────────────────

    def get_context_system_prompt(self) -> str:
        parts: List[str] = []
        if self.global_store.profile:
            profile_text = "\n".join(f"- {key}: {value}" for key, value in self.global_store.profile.items())
            parts.append(f"Thông tin hồ sơ người dùng đã biết:\n{profile_text}")
        if self.global_store.facts:
            fact_text = "\n".join(f"- {fact}" for fact in self.global_store.facts[-20:])
            parts.append(f"Các facts dài hạn cần ghi nhớ:\n{fact_text}")
        action_context = self._build_action_state_prompt()
        if action_context:
            parts.append(action_context)
        return "\n\n".join(parts)

    def _build_action_state_prompt(self) -> str:
        lines: List[str] = []
        music_state = self.get_action_state("music")
        if music_state and music_state.get("active"):
            song = music_state.get("song", "nhạc")
            if music_state.get("paused"):
                lines.append(
                    f"Nhạc: bài \"{song}\" đang tạm dừng. "
                    "User có thể nói 'tiếp tục phát' để nghe tiếp hoặc 'tắt nhạc' để đóng."
                )
            else:
                lines.append(
                    f"Nhạc: bài \"{song}\" đang phát. "
                    "User có thể nói 'dừng nhạc' để tạm dừng, 'tắt nhạc' để đóng, "
                    "hoặc 'chuyển bài' để đổi sang bài khác."
                )
        if not lines:
            return ""
        return "TRẠNG THÁI HÀNH ĐỘNG HIỆN TẠI:\n" + "\n".join(lines)

    @staticmethod
    def _clean_tool_messages(messages: List[ModelMessage]) -> List[ModelMessage]:
        return [
            msg for msg in messages
            if not any(
                getattr(part, "part_kind", "") in ("tool-call", "tool-return")
                for part in msg.parts
            )
        ]
