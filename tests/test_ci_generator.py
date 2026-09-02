import os
import unittest
import shutil
from flowlang.ci_generator import CIGenerator


class TestCIGenerator(unittest.TestCase):

    def setUp(self):
        self.test_dir = "./dist/test_ci_cd"
        self.generator = CIGenerator(output_dir=self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_generate_dockerfile(self):
        path = self.generator.generate_dockerfile("test-app")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("FROM node:20-alpine", content)
        self.assertIn("test-app", content)

    def test_generate_k8s_manifests(self):
        path = self.generator.generate_k8s_manifests("test-app")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("kind: Deployment", content)
        self.assertIn("kind: Service", content)

    def test_generate_github_workflow(self):
        path = self.generator.generate_github_workflow("test-app")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Hermes Autonomous CI/CD Pipeline", content)

    def test_generate_all(self):
        results = self.generator.generate_all("test-app")
        self.assertIn("dockerfile", results)
        self.assertIn("k8s", results)
        self.assertIn("github_actions", results)


if __name__ == "__main__":
    unittest.main()
