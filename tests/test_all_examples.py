import unittest
import os
from pathlib import Path
from flowlang.parser import parse
from flowlang.runtime import Runtime

class TestAllExampleFiles(unittest.TestCase):
    def test_parse_and_load_all_examples(self):
        examples_dir = Path(__file__).resolve().parents[1] / "examples"
        flow_files = list(examples_dir.rglob("*.flow"))
        self.assertGreater(len(flow_files), 0, "No .flow example files found!")

        for flow_path in flow_files:
            with self.subTest(file=flow_path.name):
                # 1. Test parsing
                tree = parse(flow_path)
                self.assertIsNotNone(tree, f"Parsing failed for {flow_path}")

                # 2. Test runtime loading (parse + semantic + struct build)
                runtime = Runtime(dry_run=True)
                code = flow_path.read_text(encoding="utf-8")
                runtime.load(code)

                # Verify flows exist in the parsed tree
                flow_names = [str(f.children[0]) for f in runtime.tree.find_data("flow_decl")]
                self.assertGreater(len(flow_names), 0, f"No flows found in {flow_path}")

if __name__ == "__main__":
    unittest.main()
