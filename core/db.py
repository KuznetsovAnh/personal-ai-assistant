"""PostgreSQL + Redis persistence layer for the assistant.

Storage split (single source of truth, no duplication):
- PostgreSQL (durable): profile, facts, tool ledger, conversations + messages.
- Redis (transient): action states (music) + the active-conversation pointer.

All values are JSON-serialisable. ``Database`` owns the connections and exposes
small, single-purpose methods so ``core.memory`` stays a thin orchestrator and
each write is atomic (Postgres ACID / Redis single-command SET).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import psycopg
import redis

from core.settings import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS conversations (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL,
    message_count  INTEGER NOT NULL DEFAULT 0,
    messages       JSONB NOT NULL DEFAULT '[]'::jsonb,
    recent_context JSONB NOT NULL DEFAULT '{}'::jsonb
);
"""

# PostgreSQL kv keys (durable).
PG_KEY_PROFILE = "profile"
PG_KEY_FACTS = "facts"
PG_KEY_TOOL_LEDGER = "tool_ledger"

# Redis keys (transient).
REDIS_KEY_ACTION_STATES = "assistant:action_states"
REDIS_KEY_ACTIVE_CONV = "assistant:active_conversation_id"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


class Database:
    """Owns a single PostgreSQL connection and a Redis client (created lazily)."""

    def __init__(self, postgres_db: Optional[str] = None, redis_db: int = 0) -> None:
        self._postgres_db = postgres_db
        self._redis_db = redis_db
        self._pg: Optional[psycopg.Connection] = None
        self._redis: Optional[redis.Redis] = None

    def connect(self) -> None:
        """Open connections (idempotent). Raises if the services are unreachable."""
        if self._pg is None:
            self._pg = psycopg.connect(
                host=settings.postgres_host,
                port=settings.postgres_port,
                user=settings.postgres_user,
                password=settings.postgres_password,
                dbname=self._postgres_db or settings.postgres_db,
                autocommit=True,
            )
        if self._redis is None:
            self._redis = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=self._redis_db,
                decode_responses=True,
            )

    def close(self) -> None:
        if self._pg is not None:
            self._pg.close()
            self._pg = None
        if self._redis is not None:
            self._redis.close()
            self._redis = None

    def init_schema(self) -> None:
        self.connect()
        with self._pg.transaction():
            self._pg.execute(_SCHEMA)

    # ── Durable kv (PostgreSQL) ───────────────────────────────────────────────

    def kv_get(self, key: str) -> Any:
        self.connect()
        row = self._pg.execute(
            "SELECT value FROM kv WHERE key = %s", (key,)
        ).fetchone()
        if row is None:
            return None
        value = row[0]
        return json.loads(value) if isinstance(value, str) else value

    def kv_set(self, key: str, value: Any) -> None:
        self.connect()
        self._pg.execute(
            "INSERT INTO kv (key, value) VALUES (%s, %s::jsonb) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            (key, _dumps(value)),
        )

    # ── Transient state (Redis) ───────────────────────────────────────────────

    def redis_get_json(self, key: str) -> Any:
        self.connect()
        raw = self._redis.get(key)
        return json.loads(raw) if raw else None

    def redis_set_json(self, key: str, value: Any) -> None:
        self.connect()
        self._redis.set(key, _dumps(value))

    # ── Conversations (PostgreSQL) ────────────────────────────────────────────

    def list_conversations(self) -> List[Dict[str, Any]]:
        self.connect()
        rows = self._pg.execute(
            "SELECT id, name, created_at, updated_at, message_count "
            "FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
        return [
            {
                "id": r[0],
                "name": r[1],
                "created_at": r[2],
                "updated_at": r[3],
                "message_count": r[4],
            }
            for r in rows
        ]

    def get_conversation(self, conv_id: str) -> Optional[Dict[str, Any]]:
        self.connect()
        row = self._pg.execute(
            "SELECT messages, recent_context FROM conversations WHERE id = %s",
            (conv_id,),
        ).fetchone()
        if row is None:
            return None
        return {"messages": row[0], "recent_context": row[1]}

    def sync_index(self, index: List[Dict[str, Any]]) -> None:
        """Make the conversations table match *index* (upsert + delete missing).

        Only metadata columns are touched; each conversation's messages and
        recent_context are preserved.
        """
        self.connect()
        with self._pg.transaction():
            ids = [c["id"] for c in index]
            if ids:
                self._pg.execute(
                    "DELETE FROM conversations WHERE id <> ALL(%s)", (ids,)
                )
            else:
                self._pg.execute("DELETE FROM conversations")
            for c in index:
                self._pg.execute(
                    "INSERT INTO conversations "
                    "(id, name, created_at, updated_at, message_count) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, "
                    "updated_at = EXCLUDED.updated_at, message_count = EXCLUDED.message_count",
                    (
                        c["id"],
                        c["name"],
                        c["created_at"],
                        c["updated_at"],
                        c.get("message_count", 0),
                    ),
                )

    def save_conversation_content(
        self,
        conv_id: str,
        messages: List[Any],
        recent_context: Dict[str, str],
    ) -> None:
        self.connect()
        self._pg.execute(
            "UPDATE conversations SET messages = %s::jsonb, recent_context = %s::jsonb "
            "WHERE id = %s",
            (_dumps(messages), _dumps(recent_context), conv_id),
        )

