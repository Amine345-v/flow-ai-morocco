import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flowlang.tui import HermesTUIShell


class TestHermesPhase5(unittest.TestCase):

    def test_tui_shell_commands(self):
        """Test Hermes TUI shell command execution, help menu, soul view, skills list, and exit behavior."""
        shell = HermesTUIShell()

        # Help command
        self.assertTrue(shell.execute_command("/help"))

        # Soul command
        self.assertTrue(shell.execute_command("/soul system_architect"))

        # Skills list
        self.assertTrue(shell.execute_command("/skills list"))

        # Memory list
        self.assertTrue(shell.execute_command("/memory list"))

        # MCP status
        self.assertTrue(shell.execute_command("/mcp status"))

        # Scheduler status
        self.assertTrue(shell.execute_command("/scheduler status"))

        # Exit command
        self.assertFalse(shell.execute_command("/exit"))


if __name__ == "__main__":
    unittest.main()
