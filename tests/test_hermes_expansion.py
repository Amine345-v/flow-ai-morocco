import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.subagent import SubAgentOrchestrator, SubAgent
from flowlang.browser_agent import HermesBrowserAgent
from flowlang.vision_agent import VisionInspector


class TestHermesExpansion(unittest.TestCase):

    def test_subagent_orchestration(self):
        """Test hierarchical sub-agent spawning, task delegation, and swarm execution."""
        orchestrator = SubAgentOrchestrator()

        sa1 = orchestrator.spawn(
            role_name="market_researcher",
            task_description="Research top 5 CRM competitors pricing"
        )
        sa2 = orchestrator.spawn(
            role_name="system_architect",
            task_description="Draft microservices topology for Auth service"
        )

        self.assertEqual(len(orchestrator.subagents), 2)

        # Execute sub-agent manually
        res1 = sa1.execute(flow_name="test_flow", use_mock=True)
        self.assertEqual(res1["subagent_id"], sa1.subagent_id)
        self.assertEqual(sa1.status, "COMPLETED")

    def test_browser_agent(self):
        """Test HermesBrowserAgent search and URL fetching."""
        browser = HermesBrowserAgent(timeout_s=3)
        res = browser.search_web("FlowLang AI agent DSL", num_results=2)

        self.assertEqual(res["query"], "FlowLang AI agent DSL")
        self.assertGreaterEqual(len(res["results"]), 1)

    def test_vision_inspector(self):
        """Test VisionInspector image analysis."""
        test_img = "./tests/sample_ui.png"
        with open(test_img, "wb") as f:
            f.write(b"PNG_MOCK_BYTES_DATA")

        inspector = VisionInspector()
        report = inspector.analyze_image(test_img, audit_prompt="Check UI contrast")

        self.assertEqual(report["format"], "PNG")
        self.assertGreater(report["visual_score"], 0.8)

        if os.path.exists(test_img):
            os.remove(test_img)


if __name__ == "__main__":
    unittest.main()
