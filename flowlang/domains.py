"""
Professional Domains Registry for FlowLang & JOL Studio IDE.

Provides domain specifications, agent role presets, telemetry thresholds,
and DSL starter templates for all 6 core professional domains:
1. Digital / Software Engineering
2. Economic / Quantitative Fintech
3. Cyber / Security Operations
4. Mechanical / Robotics & Automation
5. Electro / IoT & Embedded Systems
6. Clinical / Healthcare & Bio-Governance
"""

import json
from typing import Dict, Any, List

DOMAIN_REGISTRY: Dict[str, Dict[str, Any]] = {
    "digital": {
        "id": "digital",
        "name": "Digital / DevOps Software Factory",
        "icon": "Settings",
        "color": "#22d3ee",  # cyan-400
        "bgClass": "bg-cyan-400/10",
        "borderClass": "border-cyan-500/50",
        "teams": ["market_researchers", "system_architects", "code_engineers", "qa_reviewers", "product_thinker"],
        "verbs": ["search", "try", "judge", "ask"],
        "chains": ["development_pipeline", "deployment_gate"],
        "processes": ["CRM_SaaS_Factory"],
        "metrics": {
            "test_coverage_threshold": 0.90,
            "max_rollback_count": 3,
            "security_scan_policy": "Strict"
        },
        "description": "Autonomous software development, CI/CD pipelines, code synthesis, and automated QA gates."
    },
    "economic": {
        "id": "economic",
        "name": "Economic / Quantitative Fintech",
        "icon": "BarChart3",
        "color": "#4ade80",  # green-400
        "bgClass": "bg-green-400/10",
        "borderClass": "border-green-500/50",
        "teams": ["quant_analysts", "risk_officers", "execution_traders", "audit_committee"],
        "verbs": ["judge", "search", "try", "ask"],
        "chains": ["market_liquidity", "portfolio_rebalance", "var_compliance"],
        "processes": ["Algo_Trading_Pipeline"],
        "metrics": {
            "var_confidence_level": 0.99,
            "sharpe_ratio_min": 1.8,
            "max_drawdown_percent": 5.0
        },
        "description": "Algorithmic trading, Value-at-Risk calculation, liquidity monitoring, and automated portfolio rebalancing."
    },
    "cyber": {
        "id": "cyber",
        "name": "Cyber / SecOps & Zero-Trust",
        "icon": "Shield",
        "color": "#f87171",  # red-400
        "bgClass": "bg-red-400/10",
        "borderClass": "border-red-500/50",
        "teams": ["threat_hunters", "red_team_operators", "blue_team_defenders", "compliance_auditor"],
        "verbs": ["search", "judge", "try", "ask"],
        "chains": ["threat_matrix", "incident_response", "patch_verification"],
        "processes": ["Zero_Trust_Audit"],
        "metrics": {
            "mitre_attack_coverage": "98%",
            "cvss_critical_threshold": 8.0,
            "zero_trust_compliance": "Enforced"
        },
        "description": "Automated penetration testing, MITRE ATT&CK mapping, threat hunting, and OCSF telemetry analysis."
    },
    "mechanical": {
        "id": "mechanical",
        "name": "Mechanical / Robotics & Industrial Kinematics",
        "icon": "Briefcase",
        "color": "#fb923c",  # orange-400
        "bgClass": "bg-orange-400/10",
        "borderClass": "border-orange-500/50",
        "teams": ["kinematic_solvers", "cad_engineers", "stress_analysts", "safety_controller"],
        "verbs": ["try", "judge", "search", "ask"],
        "chains": ["actuator_feedback", "thermal_dissipation", "structural_load"],
        "processes": ["Robotic_Arm_Assembly"],
        "metrics": {
            "positional_accuracy_mm": 0.01,
            "max_torque_nm": 450,
            "safety_stop_latency_ms": 2.5
        },
        "description": "Multi-axis motion planning, inverse kinematics validation, stress analysis, and industrial automation."
    },
    "electro": {
        "id": "electro",
        "name": "Electro / IoT & Microcontroller Embedded Fleet",
        "icon": "Cpu",
        "color": "#c084fc",  # purple-400
        "bgClass": "bg-purple-400/10",
        "borderClass": "border-purple-500/50",
        "teams": ["firmware_devs", "signal_analysts", "power_managers", "ota_distributor"],
        "verbs": ["try", "judge", "search", "ask"],
        "chains": ["sensor_bus", "power_mode", "ota_update_ring"],
        "processes": ["Smart_Grid_Node"],
        "metrics": {
            "battery_life_years": 10,
            "latency_p99_ms": 15,
            "crypto_handshake": "AES-256-GCM"
        },
        "description": "Embedded C/Rust firmware synthesis, sensor telemetry processing, power optimization, and OTA rollouts."
    },
    "clinical": {
        "id": "clinical",
        "name": "Clinical / Healthcare & Bio-Governance",
        "icon": "Activity",
        "color": "#ec4899",  # pink-500
        "bgClass": "bg-pink-500/10",
        "borderClass": "border-pink-500/50",
        "teams": ["clinical_trial_managers", "diagnostic_ai", "hipaa_auditors", "patient_advocates"],
        "verbs": ["judge", "search", "ask", "try"],
        "chains": ["patient_consent", "biomarker_pipeline", "fda_submission"],
        "processes": ["Phase3_Clinical_Trial"],
        "metrics": {
            "hipaa_anonymization": "100%",
            "p_value_significance": 0.001,
            "adverse_event_monitoring": "Real-time"
        },
        "description": "Clinical trial data pipelines, HIPAA-compliant patient privacy enforcement, diagnostic AI validation, and FDA submission governance."
    }
}


def get_domain_spec(domain_id: str) -> Dict[str, Any]:
    """Retrieve details for a specific professional domain."""
    return DOMAIN_REGISTRY.get(domain_id, DOMAIN_REGISTRY["digital"])


def get_all_domains() -> List[Dict[str, Any]]:
    """Return all registered professional domain specifications."""
    return list(DOMAIN_REGISTRY.values())
