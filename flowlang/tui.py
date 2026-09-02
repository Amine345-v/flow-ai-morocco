import os
import sys
import json
from typing import List, Optional
from loguru import logger

from .memory import HermesMemoryStore
from .persona import SoulManager
from .skills import SkillManager
from .mcp_client import MCPClientHost
from .scheduler import FlowScheduler


def _safe_print(*args, **kwargs):
    """Safely print text on Windows terminals without UnicodeEncodeError."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = text.encode(encoding, errors="replace").decode(encoding)
        print(safe_text, **kwargs)


class HermesTUIShell:
    """
    Rich Interactive Terminal User Interface (TUI) Shell for FlowLang / JOL Studio.
    Inspired by Nous Research's Hermes Agent CLI / TUI experience.
    Provides slash commands for real-time memory search, skill inspection, MCP tools,
    SOUL persona configuration, and background cron job monitoring.
    """

    def __init__(self):
        self.memory = HermesMemoryStore()
        self.persona = SoulManager()
        self.skills = SkillManager()
        self.mcp = MCPClientHost()
        self.scheduler = FlowScheduler()

    def print_banner(self):
        _safe_print("\n" + "="*70)
        _safe_print(" [HERMES AGENT INTERACTIVE TUI SHELL v1.0.0]")
        _safe_print(" Persistent Memory | Self-Improving Skills | MCP Tools | SOUL Governance")
        _safe_print(" Type /help to view available slash commands or /exit to quit.")
        _safe_print("="*70 + "\n")

    def execute_command(self, cmd_line: str) -> bool:
        """Process a single CLI command string. Returns False to exit shell."""
        line = cmd_line.strip()
        if not line:
            return True

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("/exit", "exit", "/quit", "quit"):
            _safe_print("Exiting Hermes TUI Shell. Goodbye!")
            self.mcp.close()
            return False

        elif cmd == "/help":
            _safe_print("\nAvailable Hermes Slash Commands:")
            _safe_print("  /help                     - Display this help message")
            _safe_print("  /memory search <query>   - Full-text search FTS5 persistent memory store")
            _safe_print("  /memory list             - View recent procedural memories")
            _safe_print("  /skills list              - List all auto-learned procedural skills")
            _safe_print("  /skills add <name> <team> - Learn a new procedural skill")
            _safe_print("  /soul view <team>        - Inspect SOUL.md identity for an agent team role")
            _safe_print("  /mcp status              - View connected MCP servers and available tools")
            _safe_print("  /mcp call <tool> <json>  - Manually dispatch an MCP tool call")
            _safe_print("  /scheduler status        - View background cron flow job telemetry")
            _safe_print("  /flow run <file>         - Compile and execute a FlowLang flow AST")
            _safe_print("  /exit                    - Exit interactive shell\n")

        elif cmd == "/memory":
            sub = args[0].lower() if args else "list"
            if sub == "search":
                query = " ".join(args[1:]) if len(args) > 1 else ""
                results = self.memory.search_memories(query=query, limit=5)
                _safe_print(f"\nSearch Results for '{query}' ({len(results)} found):")
                for r in results:
                    _safe_print(f"  • #{r['id']} [{r['category'].upper()}] {r['title']} (Team: {r.get('team_name', 'N/A')})")
                    _safe_print(f"    {r['content'][:150]}...\n")
            else:
                memories = self.memory.get_latest_memories(limit=5)
                _safe_print(f"\nRecent Persistent Memories ({len(memories)} total):")
                for r in memories:
                    _safe_print(f"  • #{r['id']} [{r['category'].upper()}] {r['title']} ({r['timestamp'][:19]})")
                _safe_print("")

        elif cmd == "/skills":
            sub = args[0].lower() if args else "list"
            if sub == "list":
                _safe_print(f"\nAuto-Learned Procedural Skills ({len(self.skills.skills)} total):")
                for s in self.skills.skills.values():
                    _safe_print(f"  • {s['name'].upper()} [Team: {s.get('assigned_team', 'N/A')}] - {s['description']}")
                    _safe_print(f"    Triggers: {', '.join(s.get('triggers', []))}\n")
            elif sub == "add" and len(args) >= 3:
                name = args[1]
                team = args[2]
                desc = " ".join(args[3:]) if len(args) > 3 else f"Custom procedural skill for {team}"
                self.skills.create_skill(name=name, description=desc, assigned_team=team, procedural_steps=[desc])
                _safe_print(f"Created skill '{name}' for team '{team}'.")

        elif cmd == "/soul":
            team = args[0] if args else "code_engineers"
            soul_text = self.persona.get_soul(team)
            _safe_print(f"\nSOUL Governance Identity for '{team}':\n")
            _safe_print(soul_text + "\n")

        elif cmd == "/mcp":
            sub = args[0].lower() if args else "status"
            if sub == "status":
                telemetry = self.mcp.export_ide_telemetry()
                _safe_print("\nConnected MCP Tool Servers & Hub Telemetry:")
                for s in telemetry:
                    _safe_print(f"  • Server: '{s['id']}' | Transport: {s['transport']} | Status: {s['status'].upper()}")
                    _safe_print(f"    Tools ({s['toolCount']}): {', '.join(s['tools'])}\n")
            elif sub == "call" and len(args) >= 2:
                tool_name = args[1]
                params_str = " ".join(args[2:]) if len(args) > 2 else "{}"
                try:
                    params = json.loads(params_str)
                    res = self.mcp.call_tool(tool_name, params)
                    _safe_print(f"\nResult of '{tool_name}':")
                    _safe_print(json.dumps(res, indent=2) + "\n")
                except Exception as ex:
                    _safe_print(f"Tool execution error: {ex}")

        elif cmd == "/scheduler":
            statuses = self.scheduler.get_job_statuses()
            _safe_print(f"\nBackground Cron Scheduler Telemetry ({len(statuses)} registered jobs):")
            for j in statuses:
                _safe_print(f"  • Job '{j['job_id']}' | Flow: '{j['flow_name']}' | Schedule: {j['cron_expr']} | Status: {j['status']}")
            _safe_print("")

        elif cmd == "/flow":
            if len(args) >= 2 and args[0].lower() == "run":
                flow_file = args[1]
                _safe_print(f"Executing FlowLang file '{flow_file}'...")
                try:
                    from .parser import parse
                    from .runtime import Runtime
                    if os.path.exists(flow_file):
                        with open(flow_file, "r", encoding="utf-8") as f:
                            code = f.read()
                        tree = parse(code)
                        rt = Runtime()
                        rt.load(tree)
                        first_flow = tree.children[0].children[0]
                        rt.run_flow(str(first_flow))
                        _safe_print("Flow execution finished successfully.")
                    else:
                        _safe_print(f"File not found: {flow_file}")
                except Exception as ex:
                    _safe_print(f"Flow execution error: {ex}")

        else:
            _safe_print(f"Unknown slash command '{cmd}'. Type /help for options.")

        return True

    def run_interactive_loop(self):
        """Run interactive CLI input loop."""
        self.print_banner()
        self.mcp.connect_all()
        try:
            while True:
                user_input = input("hermes@flowlang> ")
                if not self.execute_command(user_input):
                    break
        except (KeyboardInterrupt, EOFError):
            _safe_print("\nExiting Hermes TUI Shell. Goodbye!")
            self.mcp.close()


def main():
    shell = HermesTUIShell()
    if len(sys.argv) > 1:
        shell.execute_command(" ".join(sys.argv[1:]))
    else:
        shell.run_interactive_loop()


if __name__ == "__main__":
    main()
