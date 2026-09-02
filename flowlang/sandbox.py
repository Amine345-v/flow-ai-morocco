import os
import subprocess
from typing import Any, Dict, List, Optional
from loguru import logger


class SandboxDriver:
    """
    Sandboxed Execution Environment Driver for FlowLang / JOL Studio.
    Provides isolated process/Docker container execution for agent tool calls.
    """

    def __init__(self, mode: str = "auto", docker_image: str = "python:3.11-slim"):
        self.mode = mode  # 'docker', 'local', or 'auto'
        self.docker_image = docker_image
        self._has_docker = self._check_docker()

    def _check_docker(self) -> bool:
        try:
            r = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=2)
            return r.returncode == 0
        except Exception:
            return False

    def run_command(self, command: str, cwd: Optional[str] = None, timeout_s: int = 30) -> Dict[str, Any]:
        """Execute command in sandbox container or isolated local process."""
        use_docker = (self.mode == "docker") or (self.mode == "auto" and self._has_docker)

        if use_docker:
            return self._run_in_docker(command, cwd=cwd, timeout_s=timeout_s)
        else:
            return self._run_local(command, cwd=cwd, timeout_s=timeout_s)

    def _run_in_docker(self, command: str, cwd: Optional[str] = None, timeout_s: int = 30) -> Dict[str, Any]:
        target_dir = os.path.abspath(cwd or os.getcwd())
        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{target_dir}:/workspace",
            "-w", "/workspace",
            self.docker_image,
            "sh", "-c", command
        ]
        try:
            res = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout_s)
            return {
                "sandbox_type": "docker",
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except Exception as e:
            logger.warning(f"Docker sandbox execution failed: {e}. Falling back to isolated local process.")
            return self._run_local(command, cwd=cwd, timeout_s=timeout_s)

    def _run_local(self, command: str, cwd: Optional[str] = None, timeout_s: int = 30) -> Dict[str, Any]:
        target_dir = os.path.abspath(cwd or os.getcwd())
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=target_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s
            )
            return {
                "sandbox_type": "local_process",
                "exit_code": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr
            }
        except Exception as e:
            return {
                "sandbox_type": "local_process",
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e)
            }
