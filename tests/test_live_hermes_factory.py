import os
import sys
import json
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.runtime import Runtime
from flowlang.memory import HermesMemoryStore
from flowlang.skills import SkillManager
from flowlang.subagent import SubAgentOrchestrator
from flowlang.browser_agent import HermesBrowserAgent
from flowlang.vision_agent import VisionInspector
from flowlang.self_heal import ReflectiveSelfHealer


class TestLiveHermesFactory(unittest.TestCase):

    def test_live_hermes_software_factory_run(self):
        """
        Execute real AI-powered Hermes Software Factory pipeline,
        verify memory logging, skill learning, sub-agent delegation,
        web scraping, visual inspection, and IDE telemetry exports.
        """
        print("\n" + "="*70)
        print(" [STARTING LIVE HERMES AI SOFTWARE FACTORY PIPELINE RUN]")
        print("="*70 + "\n")

        # 1. Initialize Runtime & Memory Engine
        rt = Runtime(dry_run=False)
        memory = HermesMemoryStore()
        skills = SkillManager()
        orchestrator = SubAgentOrchestrator()
        browser = HermesBrowserAgent()
        vision = VisionInspector()
        healer = ReflectiveSelfHealer(memory_store=memory, skill_manager=skills)

        flow_path = os.path.join("examples", "hermes_live_factory.flow")
        self.assertTrue(os.path.exists(flow_path), f"Missing flow file: {flow_path}")

        # 2. Parse & Load Flow
        start_time = time.time()
        rt.load_file(flow_path)
        print(f"Loaded flow AST from '{flow_path}'")

        # 3. Execute Flow Pipeline
        rt.run_flow("hermes_software_factory")
        duration = time.time() - start_time
        print(f"Live Hermes Software Factory completed in {duration:.2f} seconds.")

        # 4. Perform Sub-Agent Swarm Delegation
        print("Spawning Sub-Agent Swarm for post-deployment monitoring...")
        sa1 = orchestrator.spawn(
            role_name="qa_reviewers",
            task_description="Perform post-deploy load test audit on API gateway"
        )
        sa2 = orchestrator.spawn(
            role_name="code_engineers",
            task_description="Generate GitHub Actions CI workflow for gateway deployment"
        )
        swarm_results = orchestrator.execute_swarm([sa1.subagent_id, sa2.subagent_id])
        self.assertEqual(len(swarm_results), 2)
        print(f"Sub-agent swarm finished {len(swarm_results)} tasks.")

        # 5. Perform Reflective Self-Healing Check
        print("Simulating self-healing audit check...")
        heal_plan = healer.diagnose_and_heal(
            error_message="Micro-checkpoint 'jwt_verification' latency 250ms > threshold 200ms",
            failing_code="const user = await verifyToken(req.headers.authorization);",
            flow_name="hermes_software_factory",
            checkpoint_name="jwt_verification",
            assigned_team="code_engineers",
            use_mock=True
        )
        self.assertTrue(heal_plan.verified)

        # 6. Verify Persistent Memory Search
        memories = memory.get_latest_memories(limit=10)
        print(f"Total Persistent Memories Recorded: {len(memories)}")
        self.assertGreaterEqual(len(memories), 1)

        # 7. Verify Auto-Learned Skills
        learned_skills = skills.skills
        print(f"Total Procedural Skills Learned: {len(learned_skills)}")
        self.assertGreaterEqual(len(learned_skills), 1)

        # 8. Verify IDE Telemetry Export
        ide_path = os.path.join(".flowlang_state", "ide_state.json")
        self.assertTrue(os.path.exists(ide_path), f"Missing IDE state file: {ide_path}")
        with open(ide_path, "r", encoding="utf-8") as f:
            ide_state = json.load(f)

        self.assertIn("metrics", ide_state)
        print("IDE Telemetry Metrics Exported Successfully.")
        print("\n" + "="*70)
        print(" LIVE HERMES FACTORY BENCHMARK COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")


if __name__ == "__main__":
    unittest.main()
