import os
import sys
import unittest

# Ensure flowlang package is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.memory import HermesMemoryStore
from flowlang.persona import SoulManager, DEFAULT_SOULS
from flowlang.ai_providers import _system_prompt


class TestHermesPhase1(unittest.TestCase):

    def setUp(self):
        self.test_db_path = f"./.flowlang_state/test_hermes_memory_{self._testMethodName}.db"
        self.test_souls_dir = f"./.flowlang/test_souls_{self._testMethodName}"
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_memory_store_fts5_search(self):
        """Test persistent memory storage and FTS5 full-text search indexing."""
        store = HermesMemoryStore(db_path=self.test_db_path)
        store.start_session("sess_001", "enterprise_erp_flow")

        m1 = store.add_memory(
            category="procedural",
            title="Double-Entry Ledger Fix",
            content="When balancing ledger transactions, ensure credit and debit entries match exactly to 2 decimal places.",
            flow_name="enterprise_erp_flow",
            checkpoint_name="impl_code",
            team_name="code_engineers",
            session_id="sess_001",
            tags=["ledger", "accounting", "bugfix"]
        )
        self.assertGreater(m1, 0)

        m2 = store.add_memory(
            category="report",
            title="Market Discovery Gap Analysis",
            content="Enterprise clients require multi-currency VAT tax engines and GAAP compliant audit logging.",
            flow_name="enterprise_erp_flow",
            checkpoint_name="market_discovery",
            team_name="market_researcher",
            session_id="sess_001",
            tags=["market", "erp", "spec"]
        )
        self.assertGreater(m2, 0)

        # Search for ledger memory
        results = store.search_memories("ledger transactions", flow_name="enterprise_erp_flow")
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("Double-Entry Ledger Fix", results[0]["title"])

        # Search for VAT tax engine
        results_tax = store.search_memories("VAT tax", category="report")
        self.assertGreaterEqual(len(results_tax), 1)
        self.assertEqual(results_tax[0]["team_name"], "market_researcher")

        # Test prompt context formatting
        prompt_ctx = store.format_memory_prompt_context("ledger", "enterprise_erp_flow")
        self.assertIn("HERMES PERSISTENT MEMORY", prompt_ctx)
        self.assertIn("Double-Entry Ledger Fix", prompt_ctx)

    def test_soul_manager(self):
        """Test SOUL.md persona loading and governance identity injection."""
        sm = SoulManager(souls_dir=self.test_souls_dir)
        
        # Verify default persona retrieval
        arch_soul = sm.get_soul("system_architect")
        self.assertIn("System Architect Agent", arch_soul)
        self.assertIn("microservice topologies", arch_soul)

        # Custom SOUL file creation & load test
        custom_soul_path = os.path.join(self.test_souls_dir, "custom_team_SOUL.md")
        with open(custom_soul_path, "w", encoding="utf-8") as f:
            f.write("# Custom Security Soul\nEnforce Zero-Trust Auth.")

        custom_soul = sm.get_soul("custom_team", custom_path=custom_soul_path)
        self.assertIn("Zero-Trust Auth", custom_soul)

        # Test prompt header formatting
        header = sm.format_soul_prompt_header("system_architect")
        self.assertIn("GOVERNANCE IDENTITY & PERSONA", header)

    def test_system_prompt_integration(self):
        """Test system prompt generation with persona identity and memory context."""
        prompt = _system_prompt(
            verb="try",
            team="code_engineers",
            flow_name="enterprise_erp_flow",
            query_text="ledger balancing"
        )
        self.assertIn("GOVERNANCE IDENTITY & PERSONA", prompt)
        self.assertIn("Code Engineers Agent", prompt)
        self.assertIn("Execute a task and report results as JSON", prompt)


if __name__ == "__main__":
    unittest.main()
