"""Integration tests for the PostgreSQL + Redis persistence layer.

These require the Docker services (Postgres on 5433, Redis on 6379) to be
running. They use a dedicated ``assistant_test`` database and Redis db index 15
so the real dev data is never touched. If PostgreSQL is unreachable the tests
are skipped (so ``python -m unittest discover`` still passes the JSON tests).
"""

from __future__ import annotations

import unittest

import psycopg

from core.db import (
    Database,
    PG_KEY_PROFILE,
    REDIS_KEY_ACTION_STATES,
)
from core.memory import AssistantMemory
from core.settings import settings

TEST_DB = "assistant_test"
TEST_REDIS_DB = 15


def _ensure_test_database() -> None:
    """Create ``assistant_test`` if it does not exist (idempotent)."""
    admin = psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
        autocommit=True,
    )
    try:
        exists = admin.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
        ).fetchone()
        if not exists:
            admin.execute(f'CREATE DATABASE "{TEST_DB}"')
    finally:
        admin.close()


class _DbTestCase(unittest.TestCase):
    """Shared setup: a fresh Database against the isolated test database."""

    def setUp(self) -> None:
        try:
            _ensure_test_database()
        except psycopg.OperationalError as exc:
            self.skipTest(f"PostgreSQL not reachable: {exc}")
        self.db = Database(postgres_db=TEST_DB, redis_db=TEST_REDIS_DB)
        self.db.connect()
        self.db.init_schema()
        self.db._pg.execute("TRUNCATE kv, conversations")
        self.db._redis.flushdb()

    def tearDown(self) -> None:
        self.db.close()


class DatabaseTest(_DbTestCase):
    def test_kv_roundtrip_and_overwrite(self) -> None:
        self.db.kv_set(PG_KEY_PROFILE, {"name": "Anh"})
        self.assertEqual(self.db.kv_get(PG_KEY_PROFILE), {"name": "Anh"})

        self.db.kv_set(PG_KEY_PROFILE, {"name": "Anh", "age": 30})
        self.assertEqual(self.db.kv_get(PG_KEY_PROFILE), {"name": "Anh", "age": 30})
        self.assertIsNone(self.db.kv_get("missing"))

    def test_redis_roundtrip(self) -> None:
        self.db.redis_set_json(REDIS_KEY_ACTION_STATES, {"music": {"active": True}})
        self.assertEqual(
            self.db.redis_get_json(REDIS_KEY_ACTION_STATES),
            {"music": {"active": True}},
        )
        self.assertIsNone(self.db.redis_get_json("missing"))

    def test_conversation_index_and_content_roundtrip(self) -> None:
        index = [
            {
                "id": "c1",
                "name": "one",
                "created_at": "2026-01-01T00:00:00+07:00",
                "updated_at": "2026-01-01T00:00:00+07:00",
                "message_count": 0,
            }
        ]
        self.db.sync_index(index)
        self.assertEqual([c["id"] for c in self.db.list_conversations()], ["c1"])

        self.db.save_conversation_content("c1", [{"role": "user", "content": "hi"}], {"entity": "x"})
        got = self.db.get_conversation("c1")
        self.assertEqual(got["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(got["recent_context"], {"entity": "x"})

    def test_sync_index_deletes_missing_and_preserves_content(self) -> None:
        a = {"id": "a", "name": "A", "created_at": "t", "updated_at": "t", "message_count": 1}
        b = {"id": "b", "name": "B", "created_at": "t", "updated_at": "t", "message_count": 0}
        self.db.sync_index([a, b])
        self.db.save_conversation_content("a", [{"role": "user", "content": "keep"}], {})
        self.assertEqual(len(self.db.list_conversations()), 2)

        # Removing "b" must delete its row and keep "a"'s messages intact.
        self.db.sync_index([a])
        self.assertEqual([c["id"] for c in self.db.list_conversations()], ["a"])
        self.assertEqual(
            self.db.get_conversation("a")["messages"],
            [{"role": "user", "content": "keep"}],
        )


class AssistantMemoryDbTest(_DbTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._orig_sync_to = AssistantMemory.sync_music_state_to_tools
        self._orig_sync_from = AssistantMemory.sync_music_state_from_tools
        AssistantMemory.sync_music_state_to_tools = lambda self: None
        AssistantMemory.sync_music_state_from_tools = lambda self: None

    def tearDown(self) -> None:
        AssistantMemory.sync_music_state_to_tools = self._orig_sync_to
        AssistantMemory.sync_music_state_from_tools = self._orig_sync_from
        super().tearDown()

    def _new_memory(self) -> AssistantMemory:
        return AssistantMemory(db=self.db, max_messages=20)

    def test_global_state_roundtrip(self) -> None:
        mem = self._new_memory()
        mem.set_profile("name", "Anh")
        mem.remember_fact("thích cà phê")
        mem.set_action_state("music", {"active": True, "song": "X"})

        # A fresh instance (new Database) must read the same state back.
        mem2 = AssistantMemory(
            db=Database(postgres_db=TEST_DB, redis_db=TEST_REDIS_DB),
            max_messages=20,
        )
        self.assertEqual(mem2.global_store.profile.get("name"), "Anh")
        self.assertIn("thích cà phê", mem2.global_store.facts)
        self.assertEqual(mem2.get_action_state("music"), {"active": True, "song": "X"})
        mem2._db.close()

    def test_conversation_crud_roundtrip(self) -> None:
        mem = self._new_memory()
        cid = mem.create_conversation("chat")
        self.assertEqual(mem.get_active_conversation_id(), cid)

        mem.add_action_to_history("chào", "chào bạn")
        self.assertEqual(len(mem.get_message_history()), 2)

        # Idempotency holds in DB mode too.
        mem.add_action_to_history("chào", "chào bạn")
        self.assertEqual(len(mem.get_message_history()), 2)

        self.assertTrue(mem.rename_conversation(cid, "đã đổi tên"))
        names = {c["name"] for c in mem.list_conversations()}
        self.assertIn("đã đổi tên", names)

        mem.delete_conversation(cid)
        self.assertNotIn(cid, [c["id"] for c in mem.list_conversations()])

    def test_active_pointer_persists_across_instances(self) -> None:
        mem = self._new_memory()
        cid = mem.create_conversation()

        mem2 = self._new_memory()
        self.assertEqual(mem2.get_active_conversation_id(), cid)


if __name__ == "__main__":
    unittest.main()

