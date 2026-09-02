import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from .parser import parse
from .semantic import SemanticAnalyzer
from .runtime import Runtime
from .debugger import FlowDebugger

__all__ = ["parse", "SemanticAnalyzer", "Runtime", "FlowDebugger"]

