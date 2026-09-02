import os
import json
from typing import Any, Dict, List, Optional
from loguru import logger


DEFAULT_MCP_SERVERS = {
    "devops_system": {
        "command": "python",
        "args": ["-m", "flowlang.mcp_server"],
        "transport": "stdio",
        "description": "Native FlowLang DevOps & Multi-Domain System MCP Server",
        "roles": {
            "code_engineers": {"allow_tools": ["*"]},
            "system_architect": {"allow_tools": ["git_status", "list_files", "build_app"]},
            "qa_engineers": {"allow_tools": ["git_status", "run_cli", "list_files"]}
        }
    },
    "local_gateway": {
        "url": "http://localhost:8088",
        "transport": "http",
        "description": "Real MCP Software Gateway HTTP Server",
        "roles": {
            "market_researcher": {"allow_tools": ["fetch_quote", "anonymize_patient"]},
            "qa_engineers": {"allow_tools": ["nmap_scan", "audit_headers", "emit_ocsf"]}
        }
    }
}


class MCPConfigManager:
    """
    Manages MCP server registrations and team tool access policies
    for FlowLang / JOL Studio inspired by Hermes Agent config.yaml.
    """

    def __init__(self, config_path: str = "./.flowlang/mcp_servers.json"):
        self.config_path = config_path
        self.ensure_config_dir()
        self.servers: Dict[str, Any] = self.load_config()

    def ensure_config_dir(self):
        config_dir = os.path.dirname(self.config_path)
        if config_dir:
            os.makedirs(config_dir, exist_ok=True)
        if not os.path.exists(self.config_path):
            self.save_config(DEFAULT_MCP_SERVERS)

    def load_config(self) -> Dict[str, Any]:
        """Load MCP server config from JSON file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read MCP config at '{self.config_path}': {e}. Using defaults.")
        return dict(DEFAULT_MCP_SERVERS)

    def save_config(self, servers: Dict[str, Any]):
        """Save MCP server config to disk."""
        self.servers = servers
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(servers, f, indent=2)

    def register_server(self, server_id: str, server_config: Dict[str, Any]):
        """Dynamically add or update an MCP server configuration."""
        self.servers[server_id] = server_config
        self.save_config(self.servers)

    def is_tool_allowed_for_team(self, server_id: str, tool_name: str, team_name: Optional[str] = None) -> bool:
        """
        Check if a given tool is permitted for a team role according to security policies.
        """
        if not team_name:
            return True

        server_cfg = self.servers.get(server_id, {})
        roles_cfg = server_cfg.get("roles", {})
        team_cfg = roles_cfg.get(team_name, {})

        allow_list = team_cfg.get("allow_tools", ["*"])
        deny_list = team_cfg.get("deny_tools", [])

        if tool_name in deny_list or "*" in deny_list:
            return False

        if "*" in allow_list or tool_name in allow_list:
            return True

        return False

    def filter_tools_for_team(self, server_id: str, tools: List[Dict[str, Any]], team_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter a list of tools returned by an MCP server based on team permissions."""
        if not team_name:
            return tools

        return [
            tool for tool in tools
            if self.is_tool_allowed_for_team(server_id, tool.get("name", ""), team_name)
        ]
