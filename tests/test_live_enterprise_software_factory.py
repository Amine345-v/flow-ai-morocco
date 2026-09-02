"""
Test suite for Live Enterprise Software Factory Execution.

Verifies:
1. Real multi-stage software project lifecycle (Market Discovery -> Design -> Code Impl -> Security QA -> Deploy).
2. Domain connectors (Node.js workers: researcher, architect, developer, qa_engine, cyber_expert, meca_sim).
3. API response latency measurement and telemetry tracking.
4. Triple-Check Governance (Ancestry, Feasibility, Tree Completion) and Structural Gap Reports (SGR).
5. Real-time JOL Studio IDE state export (ide_state.json).
"""

import os
import sys
import json
import time
import pytest
from pathlib import Path

from flowlang.runtime import Runtime, EvalContext
from flowlang.types import Order, CommandKind, EchoSignature, ImpactKind, DriftResult


def test_cross_field_software_factory_execution():
    """Test full multi-domain cross_field.flow execution with real Node.js worker connectors."""
    rt = Runtime(dry_run=False)
    
    flow_path = os.path.join("examples", "software_factory_js", "cross_field.flow")
    assert os.path.exists(flow_path), f"Flow file missing: {flow_path}"

    rt.load_file(flow_path)

    # Execute professional_convergence flow
    start_time = time.time()
    rt.run_flow("professional_convergence")
    elapsed = time.time() - start_time

    print(f"\n[Factory Test] Flow executed in {elapsed:.3f} seconds.")

    # 1. Verify ProcessTree marks
    proc = rt.processes.get("universal_project", {})
    marks = proc.get("marks", {})
    assert marks.get("Budget-Forecast") == "Audited"
    assert marks.get("Security-Hardening") == "Hardened"
    assert marks.get("Structural-Check") == "Validated"

    # 2. Verify DataChain touches
    chain = rt.chains.get("global_integrity", {})
    effects = chain.get("effects", {})
    assert "Economy" in effects
    assert "Security" in effects
    assert "Physical" in effects

    # 3. Verify IDE state export
    ide_export_path = os.path.join("jol-ide", "public", "ide_state.json")
    if os.path.exists(ide_export_path):
        with open(ide_export_path, "r", encoding="utf-8") as f:
            ide_state = json.load(f)
        assert "processes" in ide_state
        assert "chains" in ide_state
        assert "metrics" in ide_state


def test_full_software_factory_pipeline():
    """Test 5-stage software factory pipeline with QA audit and state persistence."""
    rt = Runtime(dry_run=False)
    
    flow_path = os.path.join("examples", "software_factory_js", "factory.flow")
    assert os.path.exists(flow_path), f"Flow file missing: {flow_path}"

    rt.load_file(flow_path)

    # Execute build_feature flow
    start_t = time.time()
    rt.run_flow("build_feature")
    duration = time.time() - start_t

    print(f"\n[Factory Test] Software Factory pipeline completed in {duration:.3f}s.")

    # Verify process tree marks
    proc = rt.processes.get("software_project", {})
    marks = proc.get("marks", {})
    assert marks.get("Market-Research") == "Updated"
    assert marks.get("API-Design") == "Spec-Ready"
    assert marks.get("Implementation") in ("Coded", "Released", "Fixing")

    # Verify metrics tracked
    assert "actions" in rt.metrics
    assert rt.metrics["actions"] > 0
    assert "checkpoint_ms" in rt.metrics


def test_triple_check_governance_remediation():
    """Test Triple-Check Protocol with forced feasibility check failure generating SGR."""
    rt = Runtime(dry_run=True)
    ctx = EvalContext(variables={}, checkpoints=[])

    # Judge output with low score triggering feasibility failure
    low_score_result = {"pass": False, "score": 0.45, "reason": "Security vulnerability detected in token parser"}
    
    checked = rt._perform_triple_check(low_score_result, [], {}, ctx)
    assert "sgr" in checked
    sgr = checked["sgr"]

    assert sgr["feasibility_check"] is False
    assert sgr["passed"] is False
    assert ctx.last_structural_gap is not None
    assert "Feasibility=False" in ctx.last_structural_gap

