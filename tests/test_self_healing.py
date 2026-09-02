import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.self_heal import ReflectiveSelfHealer, RepairPlan
from flowlang.memory import HermesMemoryStore
from flowlang.skills import SkillManager


class TestSelfHealing(unittest.TestCase):

    def setUp(self):
        self.test_db = f"./.flowlang_state/test_heal_{self._testMethodName}.db"
        self.test_skills = f"./.flowlang/test_heal_skills_{self._testMethodName}"

    def tearDown(self):
        if os.path.exists(self.test_db):
            try:
                os.remove(self.test_db)
            except Exception:
                pass
        if os.path.exists(self.test_skills):
            try:
                for f in os.listdir(self.test_skills):
                    os.remove(os.path.join(self.test_skills, f))
                os.rmdir(self.test_skills)
            except Exception:
                pass

    def test_reflective_self_healer(self):
        """Test ReflectiveSelfHealer diagnosis, repair plan synthesis, memory logging, and skill auto-learning."""
        memory = HermesMemoryStore(db_path=self.test_db)
        skills = SkillManager(skills_dir=self.test_skills)
        healer = ReflectiveSelfHealer(memory_store=memory, skill_manager=skills)

        failing_code = "const balance = ledger.debits - ledger.credits; // Bug: missing abs check"
        error_msg = "Micro-checkpoint 'balance_audit' failed threshold: 0.00% < required 100.00%"

        plan = healer.diagnose_and_heal(
            error_message=error_msg,
            failing_code=failing_code,
            flow_name="erp_audit_flow",
            checkpoint_name="balance_audit",
            assigned_team="qa_engineers",
            use_mock=True
        )

        self.assertIsInstance(plan, RepairPlan)
        self.assertIsNotNone(plan.proposed_fix)

        # Verify resolution logged to persistent memory
        memories = memory.search_memories(query="balance_audit", category="error_resolution")
        self.assertGreaterEqual(len(memories), 1)

        # Verify auto-learned skill
        matching_skills = skills.find_matching_skills(team_name="qa_engineers", query_text="balance_audit")
        self.assertGreaterEqual(len(matching_skills), 1)


if __name__ == "__main__":
    unittest.main()
