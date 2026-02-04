"""
Unit tests for the FlowLang parser.
"""
import pytest

from flowlang.parser import Parser, ParseError
from flowlang.ast import Flow, Checkpoint, TeamAction


def test_parse_simple_flow(parser):
    """Test parsing a simple flow definition."""
    flow_src = """
    flow simple_flow(dev_team) {
        checkpoint "start" {
            dev_team: write_code()
        }
    }
    """
    
    flow = parser.parse(flow_src)
    
    assert isinstance(flow, Flow)
    assert flow.name == "simple_flow"
    assert "dev_team" in flow.parameters
    assert len(flow.checkpoints) == 1
    
    checkpoint = flow.checkpoints[0]
    assert checkpoint.name == "start"
    assert len(checkpoint.actions) == 1
    
    action = checkpoint.actions[0]
    assert action.team == "dev_team"
    assert action.command == "write_code"
    assert not action.args


def test_parse_complex_flow(parser):
    """Test parsing a more complex flow with multiple checkpoints and actions."""
    flow_src = """
    flow complex_flow(dev, qa, ops) {
        checkpoint "develop" {
            dev: write_code()
            dev: run_tests()
            qa: review_code()
        }
        
        checkpoint "deploy" {
            ops: deploy(env="staging")
            qa: test_in_staging()
        }
        
        checkpoint "release" {
            ops: deploy(env="production")
            ops: notify_team(message="Deployment complete")
        }
    }
    """
    
    flow = parser.parse(flow_src)
    
    assert flow.name == "complex_flow"
    assert set(flow.parameters) == {"dev", "qa", "ops"}
    assert len(flow.checkpoints) == 3
    
    # Check first checkpoint
    cp1 = flow.checkpoints[0]
    assert cp1.name == "develop"
    assert len(cp1.actions) == 3
    assert cp1.actions[0].command == "write_code"
    assert cp1.actions[1].command == "run_tests"
    assert cp1.actions[2].command == "review_code"
    
    # Check action with arguments
    cp2 = flow.checkpoints[1]
    assert cp2.actions[0].args == [("env", "staging")]
    
    # Check multiple arguments
    cp3 = flow.checkpoints[2]
    assert cp3.actions[1].args == [("message", "Deployment complete")]


def test_parse_errors(parser):
    """Test parsing invalid flow definitions."""
    # Missing flow name
    with pytest.raises(ParseError):
        parser.parse("flow {}")
    
    # Missing opening brace
    with pytest.raises(ParseError):
        parser.parse("flow test_flow }")
    
    # Invalid team name
    with pytest.raises(ParseError):
        parser.parse("flow test_flow { checkpoint test { 123: test() } }")


def test_parse_chain_definition(parser):
    """Test parsing chain definitions."""
    src = """
    chain deployment_flow {
        nodes = ["build", "test", "deploy"]
    }
    """
    
    chain = parser.parse_chain(src)
    assert chain.name == "deployment_flow"
    assert chain.nodes == ["build", "test", "deploy"]


def test_parse_process_definition(parser):
    """Test parsing process definitions with policies."""
    src = """
    process deployment {
        chain = "deployment_flow"
        
        policy {
            require_reason = true
            allowed_status = ["pending", "in_progress", "completed", "failed"]
            max_children = 5
            protected_nodes = ["deploy"]
            audit_required = true
        }
    }
    """
    
    process = parser.parse_process(src)
    assert process.name == "deployment"
    assert process.chain_name == "deployment_flow"
    assert process.policy.require_reason is True
    assert "deploy" in process.policy.protected_nodes
    assert process.policy.max_children == 5
