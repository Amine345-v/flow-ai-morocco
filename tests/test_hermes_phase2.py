import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.mcp_config import MCPConfigManager
from flowlang.mcp_client import MCPClientHost


class TestHermesPhase2(unittest.TestCase):

    def setUp(self):
        self.test_cfg_path = f"./.flowlang/test_mcp_config_{self._testMethodName}.json"
        if os.path.exists(self.test_cfg_path):
            try:
                os.remove(self.test_cfg_path)
            except Exception:
                pass

    def tearDown(self):
        if os.path.exists(self.test_cfg_path):
            try:
                os.remove(self.test_cfg_path)
            except Exception:
                pass

    def test_mcp_config_manager_filtering(self):
        """Test MCP config loading, server registration, and role tool filtering policies."""
        cfg_mgr = MCPConfigManager(config_path=self.test_cfg_path)
        
        cfg_mgr.register_server("test_server", {
            "transport": "stdio",
            "command": "python",
            "roles": {
                "system_architect": {"allow_tools": ["git_status"], "deny_tools": ["run_cli"]},
                "code_engineers": {"allow_tools": ["*"]}
            }
        })

        # Test tool permissions
        self.assertTrue(cfg_mgr.is_tool_allowed_for_team("test_server", "git_status", "system_architect"))
        self.assertFalse(cfg_mgr.is_tool_allowed_for_team("test_server", "run_cli", "system_architect"))
        self.assertTrue(cfg_mgr.is_tool_allowed_for_team("test_server", "any_tool", "code_engineers"))

        tools = [
            {"name": "git_status"},
            {"name": "run_cli"},
            {"name": "build_app"}
        ]
        filtered_arch = cfg_mgr.filter_tools_for_team("test_server", tools, "system_architect")
        self.assertEqual(len(filtered_arch), 1)
        self.assertEqual(filtered_arch[0]["name"], "git_status")

    def test_mcp_client_host_connect(self):
        """Test MCP client host connection, stdio tool discovery, and tool call execution."""
        host = MCPClientHost(config_path=self.test_cfg_path)
        host.config_mgr.register_server("devops", {
            "command": sys.executable,
            "args": ["-m", "flowlang.mcp_server"],
            "transport": "stdio",
            "roles": {
                "code_engineers": {"allow_tools": ["*"]},
                "qa_engineers": {"allow_tools": ["git_status"]}
            }
        })

        host.connect_all()

        # Check discovered tools
        all_tools = host.list_available_tools(team_name="code_engineers")
        self.assertGreaterEqual(len(all_tools), 1)
        tool_names = [t["name"] for t in all_tools]
        self.assertIn("git_status", tool_names)

        # Check QA team filter (only git_status allowed)
        qa_tools = host.list_available_tools(team_name="qa_engineers")
        qa_tool_names = [t["name"] for t in qa_tools]
        self.assertIn("git_status", qa_tool_names)

        # Execute git_status tool call
        res = host.call_tool("git_status", {}, team_name="code_engineers")
        self.assertIsNotNone(res)
        self.assertNotIn("error", res)

        # Clean up client processes
        host.close()


if __name__ == "__main__":
    unittest.main()
