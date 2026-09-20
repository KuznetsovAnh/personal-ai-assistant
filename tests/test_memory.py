"""Unit tests for AssistantMemory JSON persistence.

Covers the two bugs fixed in ``core/memory.py``:
1. Torn/corrupt writes  -> fixed by atomic write (temp file + os.replace).
2. Lost updates under   -> fixed by a cross-process lock + reload-before-save.
   concurrent access
Plus turn idempotency / pair integrity for conversation history.
"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from core.memory import AssistantMemory


class AssistantMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        # Skip the heavy `import core.tools` (music sync) during tests.
        self._orig_sync_to = AssistantMemory.sync_music_state_to_tools
        self._orig_sync_from = AssistantMemory.sync_music_state_from_tools
        AssistantMemory.sync_music_state_to_tools = lambda self: None
        AssistantMemory.sync_music_state_from_tools = lambda self: None

        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.mem_file = self.data_dir / "memory.json"

    def tearDown(self) -> None:
        AssistantMemory.sync_music_state_to_tools = self._orig_sync_to
        AssistantMemory.sync_music_state_from_tools = self._orig_sync_from
        self._tmp.cleanup()

    def _new_memory(self) -> AssistantMemory:
        return AssistantMemory(file_path=self.mem_file, max_messages=20, use_db=False)

    # ── Atomic write ────────────────────────────────────────────────────────
    def test_atomic_write_roundtrip_and_no_tmp_leftover(self) -> None:
        mem = self._new_memory()
        mem.set_profile("name", "Anh")
        mem.remember_fact("thích cà phê")

        # No .tmp file should remain after writing.
        leftovers = [p.name for p in self.data_dir.iterdir() if p.name.endswith(".tmp")]
        self.assertEqual(leftovers, [])

        # Reload from disk and verify content persisted.
        mem2 = self._new_memory()
        self.assertEqual(mem2.global_store.profile.get("name"), "Anh")
        self.assertIn("thích cà phê", mem2.global_store.facts)

    # ── Lost-update fix (concurrent writes) ─────────────────────────────────
    def test_concurrent_global_writes_do_not_lose_updates(self) -> None:
        n_threads = 4
        per_thread = 25

        def worker(tid: int) -> None:
            mem = self._new_memory()
            for i in range(per_thread):
                mem.remember_fact(f"fact-t{tid}-{i}")
                mem.append_tool_ledger("test", "intent", f"summary-t{tid}-{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        mem = self._new_memory()
        facts = set(mem.global_store.facts)
        expected_facts = {
            f"fact-t{t}-{i}" for t in range(n_threads) for i in range(per_thread)
        }
        self.assertTrue(expected_facts <= facts, f"missing facts: {expected_facts - facts}")
        self.assertEqual(len(mem.global_store.tool_ledger), n_threads * per_thread)

    # ── Conversation pair integrity + idempotency ────────────────────────────
    def test_add_action_to_history_pairs_and_is_idempotent(self) -> None:
        mem = self._new_memory()
        mem.create_conversation()

        mem.add_action_to_history("chào bạn", "chào bạn, tôi khỏe")
        self.assertEqual(len(mem.get_message_history()), 2)

        # Re-adding the exact same pair (e.g. LLM retry) must be a no-op.
        mem.add_action_to_history("chào bạn", "chào bạn, tôi khỏe")
        self.assertEqual(len(mem.get_message_history()), 2)

        # A different turn appends two more messages.
        mem.add_action_to_history("mấy giờ rồi?", "bây giờ là 10h")
        self.assertEqual(len(mem.get_message_history()), 4)

    # ── Re-entrant lock (delete active conversation) ─────────────────────────
    def test_delete_active_conversation_creates_new_without_deadlock(self) -> None:
        mem = self._new_memory()
        cid = mem.create_conversation()
        self.assertEqual(mem.get_active_conversation_id(), cid)

        mem.delete_conversation(cid)
        self.assertNotEqual(mem.get_active_conversation_id(), cid)
        self.assertTrue(mem.get_active_conversation_id())
        ids = [c["id"] for c in mem.list_conversations()]
        self.assertNotIn(cid, ids)


if __name__ == "__main__":
    unittest.main()
