import unittest
import os
import shutil
from flowlang.runtime import Runtime
from flowlang.errors import RuntimeFlowError

class TestMicroCheckpoints(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(dry_run=True)
        self.state_dir = "./.test_micro_flowlang_state"
        if os.path.exists(self.state_dir):
            shutil.rmtree(self.state_dir)
        self.runtime.persistence.base_path = self.state_dir

    def tearDown(self):
        if os.path.exists(self.state_dir):
            shutil.rmtree(self.state_dir)

    def test_single_micro_checkpoint(self):
        code = """
        team QATeam : Command<Judge> [size=3, distribution=round_robin];

        flow audit_flow(using: QATeam) {
            checkpoint "audit_stage" (report: final_report) {
                micro_checkpoint "auth_check" (using: QATeam) {
                    pass_flag = true;
                }
            }
        }
        """
        self.runtime.load(code)
        self.runtime.run_flow("audit_flow")

        self.assertIn("micro_checkpoints", self.runtime.metrics)
        self.assertEqual(self.runtime.metrics["micro_checkpoints"], 1)
        self.assertEqual(self.runtime.metrics["micro_checks"], 1)

    def test_batch_team_powered_micro_checkpoints(self):
        code = """
        team AuditorTeam : Command<Judge> [size=5, distribution=round_robin];

        flow batch_flow(using: AuditorTeam) {
            items = ["rule_1", "rule_2", "rule_3", "rule_4", "rule_5", "rule_6", "rule_7", "rule_8", "rule_9", "rule_10"];
            
            checkpoint "deep_audit" (report: summary) {
                micro_check "security_suite" (using: AuditorTeam, batch: items, strategy: round_robin) {
                    processed = true;
                }
            }
        }
        """
        self.runtime.load(code)
        self.runtime.run_flow("batch_flow")

        self.assertEqual(self.runtime.metrics["micro_checkpoints"], 1)
        self.assertEqual(self.runtime.metrics["micro_checks"], 10)

        # Check DAG nodes in SystemTreeEngine
        if self.runtime.system_tree:
            self.assertTrue(self.runtime.system_tree.node_count >= 10)
            self.assertTrue(self.runtime.system_tree.is_valid_dag)

    def test_large_scale_micro_checks_dag(self):
        """Verify that checkpoints with a large number of micro-checks maintain DAG invariants and run fast."""
        code = """
        team MassiveWorkers : Command<Try> [size=10, distribution=round_robin];

        flow scale_flow(using: MassiveWorkers) {
            tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50];

            checkpoint "mega_stage" {
                micro_checkpoint "lint_all" (using: MassiveWorkers, batch: tasks) {
                    checked = true;
                }
            }
        }
        """
        self.runtime.load(code)
        self.runtime.run_flow("scale_flow")

        self.assertEqual(self.runtime.metrics["micro_checks"], 50)
        if self.runtime.system_tree:
            self.assertTrue(self.runtime.system_tree.is_valid_dag)

    def test_micro_checkpoint_threshold_pass(self):
        code = """
        team Inspector : Command<Judge> [size=2];

        flow threshold_flow(using: Inspector) {
            check_list = ["a", "b", "c"];

            checkpoint "stage" {
                micro_check "quality" (using: Inspector, items: check_list, threshold: 0.5) {
                    ok = true;
                }
            }
        }
        """
        self.runtime.load(code)
        self.runtime.run_flow("threshold_flow")

        self.assertEqual(self.runtime.metrics["micro_checkpoints"], 1)

if __name__ == "__main__":
    unittest.main()
