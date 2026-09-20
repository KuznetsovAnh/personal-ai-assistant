"""One-off migration from the legacy JSON files into PostgreSQL + Redis.

Reads ``data/memory.json`` and ``data/conversations/*.json`` and writes the
contents into the DB (Postgres durable state + Redis transient state). Existing
rows for the migrated keys are overwritten. Run once after ``docker compose up``.

Usage:  python -m scripts.migrate_json_to_db   (run from the project root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from core.db import (
    Database,
    PG_KEY_FACTS,
    PG_KEY_PROFILE,
    PG_KEY_TOOL_LEDGER,
    REDIS_KEY_ACTIVE_CONV,
    REDIS_KEY_ACTION_STATES,
)
from core.settings import settings

_ROOT = Path(__file__).resolve().parent.parent


def _data_file(path: Path) -> Path:
    return path if path.is_absolute() else _ROOT / path


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def migrate() -> int:
    db = Database()
    db.init_schema()

    memory_file = _data_file(settings.memory_file)
    conv_dir = memory_file.parent / "conversations"
    global_data = _read_json(memory_file) or {}
    index = _read_json(conv_dir / "index.json") or []

    # Durable global state -> PostgreSQL
    db.kv_set(PG_KEY_PROFILE, global_data.get("profile", {}))
    db.kv_set(PG_KEY_FACTS, global_data.get("facts", []))
    db.kv_set(PG_KEY_TOOL_LEDGER, global_data.get("tool_ledger", [])[-300:])

    # Transient global state -> Redis
    db.redis_set_json(REDIS_KEY_ACTION_STATES, global_data.get("action_states", {}))
    db.redis_set_json(REDIS_KEY_ACTIVE_CONV, global_data.get("active_conversation_id", ""))

    # Conversations -> PostgreSQL
    db.sync_index(index)
    migrated = 0
    for conv in index:
        conv_id = conv.get("id", "")
        if not conv_id:
            continue
        data = _read_json(conv_dir / f"{conv_id}.json") or {}
        db.save_conversation_content(
            conv_id,
            data.get("conversation", []),
            data.get("recent_context", {}),
        )
        migrated += 1

    db.close()
    print(f"Migrated global state + {len(index)} conversation(s) ({migrated} with content).")
    return 0


if __name__ == "__main__":
    sys.exit(migrate())
