import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.skills import SkillManager
from flowlang.ai_providers import _system_prompt


class TestHermesPhase3(unittest.TestCase):

    def setUp(self):
        self.test_skills_dir = f"./.flowlang/test_skills_{self._testMethodName}"

    def tearDown(self):
        if os.path.exists(self.test_skills_dir):
            try:
                for f in os.listdir(self.test_skills_dir):
                    os.remove(os.path.join(self.test_skills_dir, f))
                os.rmdir(self.test_skills_dir)
            except Exception:
                pass

    def test_skill_manager_creation_and_matching(self):
        """Test skill creation, persistence, matching, and context generation."""
        sm = SkillManager(skills_dir=self.test_skills_dir)

        # Create custom skill
        skill_path = sm.create_skill(
            name="security_audit_hardening",
            description="Audit HTTP headers, OAuth2 tokens, and CORS origin policies.",
            assigned_team="qa_engineers",
            procedural_steps=[
                "Check for Content-Security-Policy (CSP) headers.",
                "Ensure SameSite=Strict on session cookies.",
                "Validate JWT signature with RS256 key."
            ],
            triggers=["security", "audit", "jwt", "cors"],
            success_rate=0.99
        )
        self.assertTrue(os.path.exists(skill_path))

        # Find matching skill
        matches = sm.find_matching_skills(team_name="qa_engineers", query_text="jwt token security")
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0]["name"], "security_audit_hardening")

        # Test prompt context formatting
        ctx = sm.format_skill_prompt_context(team_name="qa_engineers", query_text="security")
        self.assertIn("HERMES PROCEDURAL SKILLS", ctx)
        self.assertIn("SECURITY_AUDIT_HARDENING", ctx)
        self.assertIn("Content-Security-Policy", ctx)

    def test_system_prompt_skill_integration(self):
        """Test _system_prompt injection of procedural skills."""
        prompt = _system_prompt(
            verb="try",
            team="code_engineers",
            query_text="tsx react component"
        )
        self.assertIn("HERMES PROCEDURAL SKILLS", prompt)
        self.assertIn("ZERO_CIRCULAR_IMPORTS_TSX", prompt)


if __name__ == "__main__":
    unittest.main()
