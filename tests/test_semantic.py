"""
Semantic analysis tests for FlowLang.
"""
import pytest
from pathlib import Path

from flowlang.parser import Parser
from flowlang.semantic import SemanticAnalyzer, SemanticError
from flowlang.ast import CommandKind


def test_team_declaration_validation():
    """Test validation of team declarations."""
    src = """
    type Command<Dev, Test>;
    
    team devs: Command<Dev>;
    team testers: Command<Test>;
    """
    
    parser = Parser()
    tree = parser.parse(src)
    analyzer = SemanticAnalyzer()
    analyzer.analyze(tree)
    
    # Check teams were registered
    assert "devs" in analyzer.teams
    assert analyzer.teams["devs"].kind == CommandKind("Dev")
    assert "testers" in analyzer.teams
    assert analyzer.teams["testers"].kind == CommandKind("Test")


def test_duplicate_team_declaration():
    """Test that duplicate team declarations are caught."""
    src = """
    type Command<Dev>;
    
    team devs: Command<Dev>;
    team devs: Command<Dev>;  # Duplicate
    """
    
    parser = Parser()
    tree = parser.parse(src)
    analyzer = SemanticAnalyzer()
    
    with pytest.raises(SemanticError, match="Duplicate team declaration: 'devs'"):
        analyzer.analyze(tree)


def test_undefined_team_reference():
    """Test that references to undefined teams are caught."""
    src = """
    type Command<Dev>;
    
    flow test_flow(dev_team, non_existent_team) {
        checkpoint "start" {
            dev_team: do_something()
            non_existent_team: do_another_thing()
        }
    }
    """
    
    parser = Parser()
    tree = parser.parse(src)
    analyzer = SemanticAnalyzer()
    
    with pytest.raises(SemanticError, match="Undefined team: 'non_existent_team'"):
        analyzer.analyze(tree)


def test_chain_validation():
    """Test validation of chain declarations."""
    src = """
    chain deployment {
        nodes = ["build", "test", "deploy"]
    }
    """
    
    parser = Parser()
    tree = parser.parse_chain(src)
    analyzer = SemanticAnalyzer()
    analyzer.analyze_chain(tree)
    
    assert "deployment" in analyzer.chains
    assert analyzer.chains["deployment"].nodes == {"build", "test", "deploy"}


def test_process_validation():
    """Test validation of process declarations."""
    src = """
    type Command<Dev>;
    
    team devs: Command<Dev>;
    
    chain deployment_flow {
        nodes = ["build", "test"]
    }
    
    process deployment {
        chain = "deployment_flow"
        
        policy {
            require_reason = true
            allowed_status = ["pending", "in_progress"]
        }
    }
    """
    
    parser = Parser()
    
    # Parse and analyze chain first
    chain_src = """
    chain deployment_flow {
        nodes = ["build", "test"]
    }
    """
    chain_tree = parser.parse_chain(chain_src)
    analyzer = SemanticAnalyzer()
    analyzer.analyze_chain(chain_tree)
    
    # Then parse and analyze the process
    process_src = """
    process deployment {
        chain = "deployment_flow"
        
        policy {
            require_reason = true
            allowed_status = ["pending", "in_progress"]
        }
    }
    """
    process_tree = parser.parse_process(process_src)
    analyzer.analyze_process(process_tree)
    
    assert "deployment" in analyzer.processes
    assert analyzer.processes["deployment"].chain_name == "deployment_flow"


def test_flow_validation():
    """Test validation of flow declarations."""
    src = """
    type Command<Dev, Test>;
    
    team devs: Command<Dev>;
    team testers: Command<Test>;
    
    flow test_flow(dev_team, test_team) {
        checkpoint "develop" {
            dev_team: write_code()
            test_team: review_code()
        }
        
        checkpoint "deploy" {
            dev_team: deploy()
            test_team: test()
        }
    }
    """
    
    parser = Parser()
    tree = parser.parse(src)
    analyzer = SemanticAnalyzer()
    
    # Register teams first
    team_src = """
    type Command<Dev, Test>;
    
    team devs: Command<Dev>;
    team testers: Command<Test>;
    """
    team_tree = parser.parse(team_src)
    analyzer.analyze(team_tree)
    
    # Then analyze the flow
    analyzer.analyze_flow(tree)
    
    assert "test_flow" in analyzer.flows
    assert len(analyzer.flows["test_flow"].checkpoints) == 2


def test_undefined_chain_reference():
    """Test that references to undefined chains are caught."""
    src = """
    process deployment {
        chain = "non_existent_chain"
    }
    """
    
    parser = Parser()
    tree = parser.parse_process(src)
    analyzer = SemanticAnalyzer()
    
    with pytest.raises(SemanticError, match="Undefined chain: 'non_existent_chain'"):
        analyzer.analyze_process(tree)


def test_semantic_ok_example():
    p = Path(__file__).resolve().parents[1] / "examples" / "example1.flow"
    tree = parse(p)
    # should not raise
    SemanticAnalyzer(tree).analyze()


def test_semantic_field_error():
    # wrong field on JudgeResult
    src = (
        'result JudgeResult { confidence: number; score: number; pass: boolean; };\n'
        'type Command<Judge>;\n'
        'team J: Command<Judge> [size=1];\n'
        'flow F(using: J) {\n'
        '  checkpoint "C" {\n'
        '    X = J.judge("t", "crit");\n'
        '    bad = X.no_such_field;\n'
        '  }\n'
        '}\n'
    )
    tree = parse(src)
    with pytest.raises(SemanticError):
        SemanticAnalyzer(tree).analyze()
