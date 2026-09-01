"""
Test suite for Jol Studio Course Implementation (jol_studio_course.md)

Verifies:
1. CriticalFeature parsing & Constitutional Lock (Pydantic validation)
2. Sonar .echo check on Order, Chain, and ProcessTree
3. Triple-Check Protocol & Structural Gap Report (SGR) generation
"""

import pytest
from flowlang.types import (
    CriticalFeature,
    parse_critical_feature,
    Order,
    CommandKind,
    EchoSignature,
    ImpactKind,
    DriftResult,
    StructuralGapReport,
)
from flowlang.runtime import Runtime, EvalContext


def test_critical_feature_constitutional_lock():
    """Module 3 & 4: Test CriticalFeature immutability and schema coercion."""
    feat_dict = {
        "name": "MarketAnalysis",
        "value": {"target_audience": "enterprise"},
        "echo_signature": "high",
        "impact": "constraint",
        "ancestry_link": "root_node"
    }
    feat = parse_critical_feature(feat_dict, fallback_origin="root_node")
    assert feat is not None
    assert feat.name == "MarketAnalysis"
    assert feat.echo_signature == EchoSignature.HIGH
    assert feat.impact == ImpactKind.CONSTRAINT
    assert feat.ancestry_link == "root_node"


def test_order_echo_sonar_check():
    """Module 8: Test Order.echo sonar drift detection."""
    order = Order(id="order_1", payload="test", kind=CommandKind.Try)
    feat = parse_critical_feature({"name": "DatabaseUrl", "value": "postgres://localhost:5432"})
    order.critical_features.append(feat)

    # 1. Verification without drift
    res_ok = order.echo("DatabaseUrl", "postgres://localhost:5432")
    assert isinstance(res_ok, DriftResult)
    assert res_ok.drift_detected is False

    # 2. Drift detection (value mismatch)
    res_drift = order.echo("DatabaseUrl", "postgres://remote:5432")
    assert res_drift.drift_detected is True
    assert "Drift detected" in res_drift.message

    # 3. Missing feature drift
    res_missing = order.echo("NonExistent", "val")
    assert res_missing.drift_detected is True
    assert "missing" in res_missing.message.lower()


def test_runtime_chain_and_process_echo():
    """Module 9 & 11: Test chain.echo and process.echo in runtime."""
    rt = Runtime(dry_run=True)
    rt.chains["sys_chain"] = {
        "nodes": {"Auth": {}, "Database": {}},
        "order": ["Auth", "Database"],
        "effects": {"Auth": "satisfied", "Database": "pending"}
    }
    rt.processes["main_proc"] = {
        "nodes": {"Step1": {}},
        "marks": {"Step1": "completed"}
    }

    ctx = EvalContext(variables={}, checkpoints=[])

    # Chain .echo test
    res_chain_ok = rt._chain_call("sys_chain", "echo", ["Auth", "satisfied"], {}, ctx)
    assert res_chain_ok.drift_detected is False

    res_chain_drift = rt._chain_call("sys_chain", "echo", ["Auth", "failed"], {}, ctx)
    assert res_chain_drift.drift_detected is True

    # Process .echo test
    res_proc_ok = rt._process_call("main_proc", "echo", ["Step1", "completed"], {}, ctx)
    assert res_proc_ok.drift_detected is False


def test_triple_check_protocol_and_sgr():
    """Module 14: Test Triple-Check Protocol and StructuralGapReport (SGR)."""
    rt = Runtime(dry_run=True)
    ctx = EvalContext(variables={}, checkpoints=[])

    # Simulated Judge result with failure
    failed_judge_result = {"pass": False, "score": 0.4, "reason": "Security vulnerability detected"}
    checked_res = rt._perform_triple_check(failed_judge_result, [], {}, ctx)

    assert "sgr" in checked_res
    sgr_dict = checked_res["sgr"]
    assert sgr_dict["feasibility_check"] is False
    assert sgr_dict["passed"] is False
    assert ctx.last_structural_gap is not None
    assert "Structural Gap Report" in ctx.last_structural_gap
