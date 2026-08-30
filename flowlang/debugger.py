# ============================================================================
# FlowLang — Interactive Debugger & Execution Inspector
# ============================================================================

from typing import Any, Dict, List, Optional, Callable
import time
import json


class FlowDebugger:
    """Interactive step-through debugger and inspector for FlowLang execution."""

    def __init__(self, step_mode: bool = False):
        self.step_mode = step_mode
        self.breakpoints: set[str] = set()
        self.trace_history: List[Dict[str, Any]] = []
        self.on_step_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def add_breakpoint(self, checkpoint_name: str):
        """Set a breakpoint at a specific checkpoint name."""
        self.breakpoints.add(checkpoint_name)

    def remove_breakpoint(self, checkpoint_name: str):
        """Remove a breakpoint."""
        self.breakpoints.discard(checkpoint_name)

    def on_checkpoint_enter(self, name: str, variables: Dict[str, Any]):
        """Triggered before executing a checkpoint."""
        entry = {
            "timestamp": time.time(),
            "event": "checkpoint_enter",
            "checkpoint": name,
            "variables": {k: str(v) for k, v in variables.items() if not k.startswith("__")}
        }
        self.trace_history.append(entry)

        if name in self.breakpoints or self.step_mode:
            self._pause_and_inspect(f"Breakpoint hit at checkpoint '{name}'", entry)

    def on_checkpoint_exit(self, name: str, variables: Dict[str, Any], report: Any = None):
        """Triggered after completing a checkpoint."""
        entry = {
            "timestamp": time.time(),
            "event": "checkpoint_exit",
            "checkpoint": name,
            "report": str(report) if report else None,
            "variables": {k: str(v) for k, v in variables.items() if not k.startswith("__")}
        }
        self.trace_history.append(entry)

    def _pause_and_inspect(self, message: str, current_state: Dict[str, Any]):
        print(f"\n[DEBUGGER] {message}")
        print(f"State: {json.dumps(current_state, indent=2)}")
        if self.on_step_callback:
            self.on_step_callback(message, current_state)

    def dump_trace(self) -> List[Dict[str, Any]]:
        return self.trace_history
