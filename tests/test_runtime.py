"""
Unit tests for the FlowLang runtime.
"""
import pytest
from pathlib import Path

from flowlang.runtime import Runtime, RuntimeError


def test_runtime_initialization(runtime):
    """Test that runtime initializes with default state."""
    assert runtime.teams == {}
    assert runtime.chains == {}
    assert runtime.processes == {}
    assert runtime.current_flow is None


def test_team_registration(runtime):
    """Test registering teams with the runtime."""
    runtime.register_team("devs", "DEVELOPMENT")
    runtime.register_team("ops", "OPERATIONS")
    
    assert "devs" in runtime.teams
    assert "ops" in runtime.teams
    assert runtime.teams["devs"].kind == "DEVELOPMENT"
    assert runtime.teams["ops"].kind == "OPERATIONS"


def test_chain_operations(runtime):
    """Test chain creation and touch operations."""
    # Create a chain
    runtime.create_chain("deploy_flow", ["build", "test", "deploy", "verify"])
    
    # Test chain exists
    assert "deploy_flow" in runtime.chains
    assert len(runtime.chains["deploy_flow"].nodes) == 4
    
    # Test touching a node
    runtime.touch_chain("deploy_flow", "test")
    test_node = runtime.chains["deploy_flow"].get_node("test")
    assert test_node is not None
    assert test_node.last_touched is not None


def test_process_operations(runtime):
    """Test process creation and operations."""
    # Create a process
    runtime.create_process("deployment", "deploy_flow")
    
    # Test process exists
    assert "deployment" in runtime.processes
    process = runtime.processes["deployment"]
    assert process.chain_name == "deploy_flow"
    
    # Test process operations
    runtime.mark_process("deployment", "in_progress", "Starting deployment")
    assert process.status == "in_progress"
    assert len(process.history) == 1
    
    # Test process policies
    with pytest.raises(RuntimeError):
        # Should fail if require_reason is True and no reason provided
        runtime.mark_process("deployment", "completed", "")


def test_flow_execution(runtime):
    """Test basic flow execution."""
    flow = """
    flow test_flow(dev_team, ops_team) {
        checkpoint "start" {
            dev_team: write_code()
            ops_team: setup_infrastructure()
        }
        
        checkpoint "deploy" {
            ops_team: deploy()
        }
    }
    """
    
    # Register teams
    runtime.register_team("dev_team", "DEVELOPMENT")
    runtime.register_team("ops_team", "OPERATIONS")
    
    # Execute flow
    runtime.execute(flow)
    
    # Verify execution
    assert runtime.current_flow is not None
    assert len(runtime.current_flow.checkpoints) == 2


def test_error_handling(runtime):
    """Test error handling in the runtime."""
    # Test non-existent team
    with pytest.raises(RuntimeError, match="Team 'nonexistent' not found"):
        runtime.register_team("nonexistent", "INVALID_TYPE")
    
    # Test duplicate team registration
    runtime.register_team("test_team", "DEVELOPMENT")
    with pytest.raises(RuntimeError, match="Team 'test_team' already exists"):
        runtime.register_team("test_team", "DEVELOPMENT")
    
    # Test invalid process operation
    with pytest.raises(RuntimeError, match="Process 'nonexistent' not found"):
        runtime.mark_process("nonexistent", "started", "test")