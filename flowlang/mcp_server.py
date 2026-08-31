"""
Standard Model Context Protocol (MCP) Server for FlowLang & JOL Studio IDE.

Implements standard MCP JSON-RPC 2.0 protocol over stdio and HTTP, routing tool calls
directly to real domain software (Git, Docker, CoinGecko, Socket Scanner, 3D CAD STL, SHA-256 HIPAA).
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Add workspace root to sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from flowlang.runtime import Runtime
from flowlang.domains import get_all_domains, DOMAIN_REGISTRY
from flowlang.mcp_gateway import MCPGatewayEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [FlowLang-MCP] %(message)s")


class FlowLangMCPServer:
    """MCP Server implementing standard JSON-RPC 2.0 protocol for FlowLang & Real Domain Tools."""

    def __init__(self):
        self.engine = MCPGatewayEngine()
        self.telemetry_path = Path(__file__).parent.parent / "jol-ide" / "public" / "ide_state.json"

    def get_server_info(self) -> Dict[str, Any]:
        return {
            "name": "FlowLang-RealSoftware-MCP-Server",
            "version": "1.0.0",
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": True,
                "resources": True,
                "prompts": True,
                "logging": True
            }
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "git_status",
                "description": "DevOps: Run real git status on workspace repository.",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "fetch_quote",
                "description": "Economic: Fetch real live stock/crypto price quote via public API.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Ticker symbol (e.g. 'bitcoin', 'ethereum', 'solana')"}
                    }
                }
            },
            {
                "name": "nmap_scan",
                "description": "Cyber: Perform real TCP socket port scan on host.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "description": "Target IP or domain (e.g. '127.0.0.1')"}
                    }
                }
            },
            {
                "name": "generate_stl",
                "description": "Mechanical: Generate real 3D ASCII STL geometry solid file on disk.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string", "description": "STL file name"},
                        "size": {"type": "number", "description": "Cube dimension size"}
                    }
                }
            },
            {
                "name": "anonymize_patient",
                "description": "Clinical: Perform real SHA-256 cryptographic HIPAA PII anonymization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ssn": {"type": "string"},
                        "dob": {"type": "string"}
                    },
                    "required": ["name"]
                }
            }
        ]

    def list_resources(self) -> List[Dict[str, Any]]:
        return [
            {
                "uri": "flowlang://telemetry/ide_state",
                "name": "Live IDE Telemetry State",
                "mimeType": "application/json"
            },
            {
                "uri": "flowlang://connectors/status",
                "name": "Real Domain Connectors Status",
                "mimeType": "application/json"
            }
        ]

    def read_resource(self, uri: str) -> Dict[str, Any]:
        if uri == "flowlang://telemetry/ide_state":
            if self.telemetry_path.exists():
                return {"contents": [{"uri": uri, "mimeType": "application/json", "text": self.telemetry_path.read_text(encoding="utf-8")}]}
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps({"status": "idle"})}]}
        elif uri == "flowlang://connectors/status":
            return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(self.engine.get_all_statuses(), indent=2)}]}
        else:
            raise ValueError(f"Unknown resource URI: {uri}")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "git_status":
            res = self.engine.dispatch_mcp_action("digital", "git_status", arguments)
            return {"content": [{"type": "text", "text": str(res.get("output"))}]}
        elif tool_name == "fetch_quote":
            res = self.engine.dispatch_mcp_action("economic", "fetch_quote", arguments)
            return {"content": [{"type": "text", "text": json.dumps(res.get("output"), indent=2)}]}
        elif tool_name == "nmap_scan":
            res = self.engine.dispatch_mcp_action("cyber", "nmap_scan", arguments)
            return {"content": [{"type": "text", "text": json.dumps(res.get("output"), indent=2)}]}
        elif tool_name == "generate_stl":
            res = self.engine.dispatch_mcp_action("mechanical", "generate_stl", arguments)
            return {"content": [{"type": "text", "text": json.dumps(res.get("output"), indent=2)}]}
        elif tool_name == "anonymize_patient":
            res = self.engine.dispatch_mcp_action("clinical", "anonymize_patient", arguments)
            return {"content": [{"type": "text", "text": json.dumps(res.get("output"), indent=2)}]}
        else:
            return {"content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "isError": True}

    def handle_json_rpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        try:
            if method == "initialize":
                return {"jsonrpc": "2.0", "id": req_id, "result": self.get_server_info()}
            elif method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": self.list_tools()}}
            elif method == "tools/call":
                res = self.call_tool(params.get("name"), params.get("arguments", {}))
                return {"jsonrpc": "2.0", "id": req_id, "result": res}
            elif method == "resources/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": self.list_resources()}}
            elif method == "resources/read":
                return {"jsonrpc": "2.0", "id": req_id, "result": self.read_resource(params.get("uri"))}
            else:
                return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method '{method}' not found"}}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32603, "message": str(e)}}

    def run_stdio(self):
        logging.info("Starting FlowLang Standard Real Software MCP Server on stdio...")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                response = self.handle_json_rpc(req)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except Exception as ex:
                err_resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {ex}"}}
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    server = FlowLangMCPServer()
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("MCP Server Info:", json.dumps(server.get_server_info(), indent=2))
        print("\nTesting Real Git Status Tool Call:")
        print(json.dumps(server.call_tool("git_status", {}), indent=2))
        print("\nTesting Real Live Quote Fetch Tool Call:")
        print(json.dumps(server.call_tool("fetch_quote", {"symbol": "bitcoin"}), indent=2))
    else:
        server.run_stdio()
