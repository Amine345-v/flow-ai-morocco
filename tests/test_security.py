"""
Security tests for FlowLang.
"""
import os
import pytest
import tempfile
from pathlib import Path

from flowlang.parser import Parser
from flowlang.runtime import Runtime, RuntimeError


def test_safe_path_handling():
    """Test that path handling is secure and prevents directory traversal."""
    runtime = Runtime()
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Try to access files outside the allowed directory
        with pytest.raises(RuntimeError, match="Access denied"):
            runtime._resolve_path("../../etc/passwd", base_dir=tmpdir)
        
        # Try to create a file in a non-existent subdirectory
        with pytest.raises(RuntimeError, match="Invalid path"):
            runtime._resolve_path("nonexistent/../file.txt", base_dir=tmpdir)
        
        # Valid path should work
        valid_path = runtime._resolve_path("subdir/file.txt", base_dir=tmpdir)
        assert str(valid_path).startswith(tmpdir)
        assert valid_path.name == "file.txt"


def test_sandboxed_execution():
    """Test that potentially dangerous operations are sandboxed."""
    runtime = Runtime()
    
    # Try to execute shell commands
    with pytest.raises(RuntimeError, match="not allowed"):
        runtime.execute("flow test { checkpoint \"x\" { x: system(\"rm -rf /\") } }")
    
    # Try to access filesystem
    with pytest.raises(RuntimeError, match="not allowed"):
        runtime.execute("flow test { checkpoint \"x\" { x: open(\"/etc/passwd\") } }")


def test_memory_safety():
    """Test that the runtime is protected against memory exhaustion."""
    runtime = Runtime()
    
    # Test with very large input
    large_input = "flow test { checkpoint \"x\" { x: process(\"" + "x" * 10_000_000 + "\") } }"
    with pytest.raises(RuntimeError, match="Input too large"):
        runtime.execute(large_input)
    
    # Test with deep recursion
    deep_recursion = """
    chain deep {
        nodes = ["a"]
        a -> a
    }
    
    process p {
        chain = "deep"
    }
    
    flow test {
        checkpoint "x" {
            p.touch("a")
        }
    }
    """
    with pytest.raises(RuntimeError, match="Recursion limit"):
        runtime.execute(deep_recursion)


def test_secure_deserialization():
    """Test that deserialization is secure."""
    runtime = Runtime()
    
    # Test with malicious pickle data
    malicious_pickle = b"cos\nsystem\n(S'echo vulnerable'\ntR."
    with pytest.raises(RuntimeError, match="Unsafe deserialization"):
        runtime._deserialize(malicious_pickle)
    
    # Test with safe JSON
    safe_json = '{"key": "value"}'.encode()
    result = runtime._deserialize(safe_json)
    assert result == {"key": "value"}


def test_rate_limiting():
    """Test that rate limiting is enforced."""
    runtime = Runtime()
    
    # Configure rate limiting
    runtime.configure(rate_limit={"calls": 5, "per_seconds": 1})
    
    # Make calls up to the limit
    for i in range(5):
        runtime.execute(f"flow test {{ checkpoint \"x\" {{ x: process({i}) }} }}")
    
    # Next call should be rate limited
    with pytest.raises(RuntimeError, match="Rate limit exceeded"):
        runtime.execute("flow test { checkpoint \"x\" { x: process(6) } }")
    
    # After waiting, should work again
    import time
    time.sleep(1.1)  # Slightly more than the rate limit window
    runtime.execute("flow test { checkpoint \"x\" { x: process(7) } }")


def test_secure_logging():
    """Test that sensitive data is not logged."""
    runtime = Runtime()
    
    # Configure logging to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as log_file:
        runtime.configure(log_file=log_file.name)
        
        # Execute flow with sensitive data
        runtime.execute("""
        flow test {
            checkpoint "x" {
                team: process(api_key="secret123")
            }
        }
        """)
        
        # Read the log file
        log_file.seek(0)
        log_content = log_file.read().decode()
        
        # Sensitive data should be redacted
        assert "secret123" not in log_content
        assert "***REDACTED***" in log_content
    
    # Clean up
    os.unlink(log_file.name)
