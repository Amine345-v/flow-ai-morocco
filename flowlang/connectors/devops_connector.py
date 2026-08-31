"""
Real DevOps & Coding Software MCP Connector.
Performs real Git operations, CLI command execution, and file system inspection.
"""

import os
import sys
import subprocess
import shutil
from typing import Dict, Any, List


class DevOpsConnector:
    """Real DevOps & Coding MCP Connector interacting with live system tools."""

    def __init__(self, workspace_path: str = "."):
        self.workspace_path = os.path.abspath(workspace_path)

    def get_status(self) -> Dict[str, Any]:
        git_path = shutil.which("git")
        python_path = sys.executable
        return {
            "name": "DevOps Real Software Connector",
            "domain": "digital",
            "gitPath": git_path or "Not Found",
            "pythonPath": python_path,
            "osPlatform": sys.platform,
            "workspace": self.workspace_path,
            "status": "connected"
        }

    def list_files(self, max_files: int = 50) -> List[Dict[str, Any]]:
        file_list = []
        for root, _, files in os.walk(self.workspace_path):
            if any(ignore in root for ignore in [".git", "node_modules", "__pycache__", ".venv", "dist"]):
                continue
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, self.workspace_path)
                try:
                    stat = os.stat(full_p)
                    file_list.append({
                        "path": rel_p,
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime
                    })
                except Exception:
                    file_list.append({"path": rel_p, "size_bytes": 0})
                if len(file_list) >= max_files:
                    break
            if len(file_list) >= max_files:
                break
        return file_list

    def git_status(self) -> str:
        try:
            res = subprocess.run(["git", "status"], cwd=self.workspace_path, capture_output=True, text=True, timeout=5)
            return res.stdout if res.returncode == 0 else f"Git error: {res.stderr}"
        except Exception as e:
            return f"Git command failed: {e}"

    def git_log(self, max_count: int = 5) -> str:
        try:
            res = subprocess.run(["git", "log", f"-n{max_count}", "--oneline"], cwd=self.workspace_path, capture_output=True, text=True, timeout=5)
            return res.stdout if res.returncode == 0 else f"Git log error: {res.stderr}"
        except Exception as e:
            return f"Git log failed: {e}"

    def open_vscode(self, target_path: str = ".") -> str:
        """Launch VSCode editor on specified path or current workspace."""
        try:
            code_bin = shutil.which("code") or "code"
            target = os.path.abspath(target_path) if target_path and target_path != "." else self.workspace_path
            
            # Launch VSCode process asynchronously
            if sys.platform.startswith("win"):
                subprocess.Popen(f'code "{target}"', shell=True)
            else:
                subprocess.Popen([code_bin, target])

            return f"VSCode process launched successfully for path: {target}"
        except Exception as e:
            return f"Failed to launch VSCode: {e}"

    def build_app(self, app_name: str = "accountant_erp", prompt_description: str = "") -> Dict[str, Any]:
        """Synthesize a complete professional web application using Gemini AI over MCP."""
        try:
            from pathlib import Path
            import google.generativeai as genai
            
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "AIzaSyBh4AFxbzLHpZCGzMYXlkSKS79svs40iHE"
            genai.configure(api_key=key)
            model_name = os.getenv("FLOWLANG_GEMINI_MODEL", "gemini-3.6-flash")
            model = genai.GenerativeModel(model_name)
            
            # Format Component Name (e.g. accountant_erp -> AccountantERP)
            pascal_name = "".join([part.capitalize() for part in app_name.replace("-", "_").split("_")])
            if not pascal_name.endswith("App") and not pascal_name.endswith("ERP") and not pascal_name.endswith("System"):
                pascal_name += "App"

            desc = prompt_description or f"Enterprise Software Application for {app_name.replace('_', ' ').title()}"

            prompt = f"""
Write a complete, state-of-the-art React TypeScript component named '{pascal_name}'.
This component is a full-featured web application: {desc}.

Requirements:
- Imports: React, useState, Lucide icons from 'lucide-react' (e.g. DollarSign, FileText, Calculator, TrendingUp, PieChart, Layers, Plus, CheckCircle, Shield, Building, CreditCard, ArrowUpRight, ArrowDownRight, RefreshCw, BarChart2, Search, Filter, Settings, Activity, Database).
- High visual aesthetics: Dark mode background (#0b1121), vibrant status badges, responsive layout, glassmorphism cards, interactive tabs, real-time metric indicators.
- Production UI capabilities: Includes interactive CRUD operations, data tables with search/filter, dashboard KPIs, live state management, and export actions.
- Code Quality: Clean TypeScript code, no placeholder stubs, valid syntax.

OUTPUT ONLY VALID TSX CODE. DO NOT INCLUDE MARKDOWN CODE FENCES.
"""
            res = model.generate_content(prompt)
            code_text = res.text.strip()
            if code_text.startswith("```"):
                lines = code_text.splitlines()
                if lines[0].startswith("```"): lines = lines[1:]
                if lines[-1].startswith("```"): lines = lines[:-1]
                code_text = "\n".join(lines)

            target = Path(self.workspace_path) / ".." / "jol-ide" / "components" / "apps" / f"{pascal_name}.tsx"
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(code_text)

            return {
                "status": "success",
                "app": pascal_name,
                "file_path": str(target),
                "bytes_generated": len(code_text),
                "message": f"Successfully generated {pascal_name} via Gemini 3.6 Flash ({len(code_text)} bytes)."
            }
        except Exception as e:
            return {"status": "error", "message": f"App building failed: {e}"}

    def run_cli_command(self, command: str) -> str:
        try:
            res = subprocess.run(command, shell=True, cwd=self.workspace_path, capture_output=True, text=True, timeout=10)
            out = res.stdout if res.stdout else res.stderr
            return out.strip() if out else "Command completed with code 0 (no output)."
        except Exception as e:
            return f"Execution failed: {e}"
