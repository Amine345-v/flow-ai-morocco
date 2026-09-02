import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.scheduler import FlowScheduler
from flowlang.sandbox import SandboxDriver


class TestHermesPhase4(unittest.TestCase):

    def test_flow_scheduler_daemon(self):
        """Test FlowScheduler background job registration, triggering, and status export."""
        scheduler = FlowScheduler(poll_interval_s=0.2)
        execution_count = 0

        def sample_flow_task():
            nonlocal execution_count
            execution_count += 1

        scheduler.add_job(
            job_id="test_job_1",
            flow_name="qa_security_audit",
            cron_expr="@every_5s",
            callback=sample_flow_task
        )

        scheduler.start()
        time.sleep(1.0)  # Wait for initial execution

        statuses = scheduler.get_job_statuses()
        self.assertEqual(len(statuses), 1)
        self.assertEqual(statuses[0]["flow_name"], "qa_security_audit")
        self.assertGreaterEqual(execution_count, 1)

        scheduler.stop()

    def test_sandbox_driver_execution(self):
        """Test SandboxDriver isolated command execution."""
        driver = SandboxDriver(mode="local")
        res = driver.run_command("echo FlowLang Sandbox Test", timeout_s=5)

        self.assertEqual(res["exit_code"], 0)
        self.assertIn("FlowLang Sandbox Test", res["stdout"])
        self.assertEqual(res["sandbox_type"], "local_process")


if __name__ == "__main__":
    unittest.main()
