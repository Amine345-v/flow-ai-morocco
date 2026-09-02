import os
import sys
import json
import subprocess
import requests
from typing import Any, Dict, List, Optional
from loguru import logger
from .mcp_config import MCPConfigManager


class MCPClientHost:
    """
    Universal Model Context Protocol (MCP) Client Host for FlowLang / JOL Studio.
    Supports stdio (subprocess) and HTTP transports to connect external tool servers,
    discover available tools, enforce team role access filters, and dispatch tool calls.
    """

    def __init__(self, config_path: str = "./.flowlang/mcp_servers.json"):
        self.config_mgr = MCPConfigManager(config_path)
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.discovered_tools: Dict[str, List[Dict[str, Any]]] = {}
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def connect_all(self):
        """Initialize connections to all configured MCP servers."""
        for server_id, cfg in self.config_mgr.servers.items():
            try:
                transport = cfg.get("transport", "stdio")
                if transport == "stdio":
                    self._connect_stdio_server(server_id, cfg)
                elif transport == "http":
                    self._connect_http_server(server_id, cfg)
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server '{server_id}': {e}")

    def _connect_stdio_server(self, server_id: str, cfg: Dict[str, Any]):
        """Spawn subprocess and perform MCP initialize + tools/list handshake."""
        cmd = cfg.get("command", "python")
        args = cfg.get("args", [])
        full_cmd = [cmd] + args

        env = dict(os.environ)
        # Ensure PYTHONPATH includes workspace root
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        env["PYTHONPATH"] = workspace_root + os.pathsep + env.get("PYTHONPATH", "")

        proc = subprocess.Popen(
            full_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env
        )
        self.active_processes[server_id] = proc

        # Perform JSON-RPC initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "FlowLangMCPClient", "version": "1.0.0"}
            }
        }
        resp = self._send_stdio_rpc(server_id, init_req)

        # List tools
        tools_req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        tools_resp = self._send_stdio_rpc(server_id, tools_req)
        tools = tools_resp.get("result", {}).get("tools", []) if tools_resp else []
        self.discovered_tools[server_id] = tools
        logger.info(f"🔗 [MCPClient] Connected stdio server '{server_id}' (Found {len(tools)} tools)")

    def _send_stdio_rpc(self, server_id: str, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        proc = self.active_processes.get(server_id)
        if not proc or proc.poll() is not None:
            return None

        line = json.dumps(request) + "\n"
        proc.stdin.write(line)
        proc.stdin.flush()

        resp_line = proc.stdout.readline()
        if resp_line:
            try:
                return json.loads(resp_line.strip())
            except Exception:
                pass
        return None

    def _connect_http_server(self, server_id: str, cfg: Dict[str, Any]):
        """Probe HTTP MCP Gateway server status."""
        url = cfg.get("url", "http://localhost:8088").rstrip("/")
        try:
            r = requests.get(f"{url}/status", timeout=2)
            if r.status_code == 200:
                # Builtin HTTP tool set
                tools = [
                    {"name": "git_status", "description": "Get git workspace status", "domain": "digital"},
                    {"name": "build_app", "description": "Build dynamic software module", "domain": "digital"},
                    {"name": "fetch_quote", "description": "Fetch live financial quotes", "domain": "economic"},
                    {"name": "nmap_scan", "description": "Perform cyber port audit", "domain": "cyber"},
                    {"name": "generate_stl", "description": "Generate 3D CAD mesh", "domain": "mechanical"},
                    {"name": "anonymize_patient", "description": "Anonymize HIPAA records", "domain": "clinical"}
                ]
                self.discovered_tools[server_id] = tools
                logger.info(f"🌐 [MCPClient] Connected HTTP Gateway server '{server_id}'")
        except Exception:
            logger.debug(f"HTTP MCP Gateway '{server_id}' not online.")

    def list_available_tools(self, team_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all discovered MCP tools across active servers, filtered by team permission policies."""
        all_tools = []
        for server_id, tools in self.discovered_tools.items():
            filtered = self.config_mgr.filter_tools_for_team(server_id, tools, team_name)
            for t in filtered:
                t_copy = dict(t)
                t_copy["server_id"] = server_id
                all_tools.append(t_copy)
        return all_tools

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], team_name: Optional[str] = None) -> Dict[str, Any]:
        """Dispatch a tool call to the matching MCP server."""
        # Find which server owns this tool
        target_server = None
        for server_id, tools in self.discovered_tools.items():
            if any(t.get("name") == tool_name for t in tools):
                if self.config_mgr.is_tool_allowed_for_team(server_id, tool_name, team_name):
                    target_server = server_id
                    break
                else:
                    return {"error": f"Tool '{tool_name}' is denied for team role '{team_name}' by security policy."}

        if not target_server:
            return {"error": f"MCP Tool '{tool_name}' not found or not permitted."}

        cfg = self.config_mgr.servers.get(target_server, {})
        transport = cfg.get("transport", "stdio")

        if transport == "stdio":
            call_req = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            resp = self._send_stdio_rpc(target_server, call_req)
            if resp and "result" in resp:
                return resp["result"]
            return {"error": f"Failed stdio execution for tool '{tool_name}'", "response": resp}

        elif transport == "http":
            url = cfg.get("url", "http://localhost:8088").rstrip("/")
            try:
                r = requests.post(url, json={"action": tool_name, "params": arguments}, timeout=10)
                if r.status_code == 200:
                    return r.json()
            except Exception as ex:
                return {"error": f"HTTP dispatch error for tool '{tool_name}': {ex}"}

        return {"error": f"Unsupported transport '{transport}'"}

    def export_ide_telemetry(self) -> List[Dict[str, Any]]:
        """Export MCP server hub and tool status payload for JOL Studio IDE visualization."""
        summary = []
        for server_id, cfg in self.config_mgr.servers.items():
            tools = self.discovered_tools.get(server_id, [])
            summary.append({
                "id": server_id,
                "transport": cfg.get("transport", "stdio"),
                "description": cfg.get("description", ""),
                "status": "connected" if server_id in self.discovered_tools else "disconnected",
                "toolCount": len(tools),
                "tools": [t.get("name") for t in tools]
            })
        return summary

    def close(self):
        """Terminate all active subprocesses and close IO handles cleanly."""
        for server_id, proc in self.active_processes.items():
            try:
                if proc.stdin:
                    proc.stdin.close()
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=1)
            except Exception:
                pass
        self.active_processes.clear()
