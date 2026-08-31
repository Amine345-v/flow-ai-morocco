import { useState, useEffect } from 'react';
import { Flow, SystemChainNode, ProcessTreeNode, CommandKind } from '../types';
import { getStoredAIConfig } from '../components/AIModelSettingsModal';

export interface SimulationState {
    flow: Flow | null;
    chain: SystemChainNode[];
    tree: ProcessTreeNode | null;
    resources: Record<string, any>;
    files: Record<string, string>;
    lastUpdate: string;
    isSimulating: boolean;
    refreshState: () => Promise<void>;
    executeFlowPrompt: (prompt: string, domain?: string) => Promise<void>;
}

function buildSynthesizedFiles(flowName: string, promptText: string, domain: string) {
    const cleanFlowName = flowName.endsWith('.flow') ? flowName : `${flowName}.flow`;
    const baseName = cleanFlowName.replace(/\.flow$/i, '');
    const cleanPrompt = (promptText || "Execute software pipeline").replace(/"/g, '\\"');
    
    const flowLangDSL = `// ============================================================================
// FlowLang DSL — Initial Order Variable Execution
// ============================================================================

order initial_human_order = "${cleanPrompt}";

process ${baseName}_roadmap "${baseName.toUpperCase()} Process Architecture Tree" {
    root: "${baseName}";
    branch "${baseName}" -> ["CoreEngine", "LogicHandlers", "SecurityGate", "APIExporter"];

    node "CoreEngine" { priority: "critical"; status: "implemented"; };
    node "LogicHandlers" { priority: "high"; status: "implemented"; };
    node "SecurityGate" { priority: "high"; status: "implemented"; };
    node "APIExporter" { priority: "medium"; status: "implemented"; };
}

chain ${baseName}_execution_chain {
    nodes: [RequirementDiscovery, LogicSynthesis, VerificationGate, ReleaseDeploy];
    propagation: causal(decay=0.85, forward=true);
}

team ${domain}_architects : Command<Search>      [size=3];
team logic_engineers    : Command<Try>         [size=4];
team qa_auditors       : Command<Judge>       [size=2];
team release_thinker    : Command<Communicate> [size=1];

flow ${baseName}_execution(using: ${domain}_architects, logic_engineers, qa_auditors, release_thinker) {
    context retention: checkpoint;
    merge_policy: deep_merge;

    checkpoint "requirement_discovery" (report: req_brief) {
        req_brief = ${domain}_architects.search(initial_human_order);
        ${baseName}_execution_chain.touch("RequirementDiscovery", effect=1.0);
        ${baseName}_roadmap.mark("CoreEngine", "in_progress", reason="Order variable parsed");
    }

    checkpoint "logic_synthesis" (report: synthesized_code) {
        synthesized_code = logic_engineers.try(initial_human_order);
        ${baseName}_execution_chain.touch("LogicSynthesis", effect=0.95);
        ${baseName}_roadmap.mark("LogicHandlers", "implemented", reason="Handlers compiled");
    }

    checkpoint "quality_gate" (report: qa_verdict) {
        qa_verdict = qa_auditors.judge(synthesized_code, "Zero-warning static analysis & zero-trust audit");
        ${baseName}_execution_chain.touch("VerificationGate", effect=0.9);
        ${baseName}_roadmap.mark("SecurityGate", "tested", reason="Security gate approved");
    }

    checkpoint "production_release" (report: live_status) {
        live_status = release_thinker.ask(initial_human_order);
        ${baseName}_execution_chain.touch("ReleaseDeploy", effect=1.0);
        ${baseName}_roadmap.mark("${baseName}", "deployed", reason="FlowLang pipeline live");
    }
}
`;

    return {
        [cleanFlowName]: flowLangDSL,
        [`${baseName}_controller.ts`]: `// TypeScript Logic Controller for ${baseName}\n// Executing initial order: "${cleanPrompt}"\n\nexport class ${baseName.toUpperCase()}_Controller {\n  public initialOrder = "${cleanPrompt}";\n\n  public async executePipeline(payload: any) {\n    console.log("JOLWork executing ${baseName} controller with order:", this.initialOrder);\n    return {\n      status: "SUCCESS",\n      order: this.initialOrder,\n      timestamp: Date.now()\n    };\n  }\n}`,
        [`${baseName}_view.tsx`]: `// React View Viewport Component for ${baseName}\nimport React from 'react';\n\nexport const ${baseName.toUpperCase()}_View: React.FC = () => {\n  return (\n    <div className="p-6 bg-slate-900 text-white rounded-xl border border-purple-500/30">\n      <h2 className="text-xl font-bold">${baseName} Stage Viewport</h2>\n      <p className="text-xs text-purple-300 font-mono mt-1">Initial Order: "${cleanPrompt}"</p>\n    </div>\n  );\n};`,
        [`${baseName}_schema.json`]: JSON.stringify({
            flowName: cleanFlowName,
            domain,
            initialOrder: cleanPrompt,
            status: "ACTIVE_SYNTHESIS"
        }, null, 2)
    };
}

// Generate tailored dynamic state matching any prompt / project domain / flow file
function generateDomainState(prompt: string, domain: string, targetFlow?: string) {
    const p = (prompt || "").toLowerCase();
    const effectiveFlowName = targetFlow || (p.includes("ecom") ? "ecom_erp.flow" : p.includes("secops") ? "security_audit.flow" : "software_factory.flow");

    if (p.includes("ecom") || p.includes("ecommerce") || p.includes("shop") || p.includes("store") || p.includes("cart")) {
        return {
            flow: {
                id: 'flow_ecom',
                name: `E-Commerce ERP Storefront & Core (${effectiveFlowName})`,
                usingTeams: ['ecom_architects', 'payment_engineers', 'inventory_auditors'],
                teams: {},
                checkpoints: [
                    { id: 'cp1', name: '1. Catalog & Product Inventory Indexing', report: 'Indexed 10,000 SKU items with real-time stock levels' },
                    { id: 'cp2', name: '2. Shopping Cart & Inventory Reservation', report: 'Validated cart state & locked 15-min item hold window' },
                    { id: 'cp3', name: '3. Stripe / PayPal Payment Gateway Integration', report: 'Processed 3D-Secure payment transaction & tokenization' },
                    { id: 'cp4', name: '4. Order Fulfillment & Warehouse Dispatch', report: 'Generated shipping label & dispatched tracking webhooks' }
                ],
                currentCheckpointIndex: 3,
                mergePolicy: 'deep_merge' as const,
                historyLog: []
            },
            chain: [
                { id: 'c1', name: 'Product Catalog Scanner', order: { id: 'o1', type: CommandKind.SEARCH, content: 'Scan product catalog SKUs & inventory balances', status: 'completed' }, impactLevel: 0.2 },
                { id: 'c2', name: 'Inventory Lock Engine', order: { id: 'o2', type: CommandKind.TRY, content: 'Reserve cart items & apply promo discount coupons', status: 'completed' }, impactLevel: 0.5 },
                { id: 'c3', name: 'Payment Security Gate', order: { id: 'o3', type: CommandKind.JUDGE, content: 'Verify Stripe / PayPal 3D-Secure transaction signature', status: 'completed' }, impactLevel: 0.8 },
                { id: 'c4', name: 'Fulfillment Dispatcher', order: { id: 'o4', type: CommandKind.COMMUNICATE, content: 'Emit Order Fulfills Webhook to logistics ERP', status: 'completed' }, impactLevel: 0.9 }
            ] as SystemChainNode[],
            tree: {
                id: 'root_ecom',
                name: 'E-Commerce ERP Storefront & Core',
                geneticCode: '50',
                type: 'root' as const,
                status: 'healthy' as const,
                children: [
                    {
                        id: 'node_ecom_catalog',
                        name: 'Product Catalog & Inventory Ledger',
                        geneticCode: '51',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_cat_1', name: 'SKU Inventory Balances', geneticCode: '5101', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_cat_2', name: 'Pricing & Discount Rules Engine', geneticCode: '5102', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_ecom_pay',
                        name: 'Payment Gateway & Checkout Gate',
                        geneticCode: '52',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_pay_1', name: 'Stripe 3D-Secure Tokenizer', geneticCode: '5201', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    }
                ]
            } as ProcessTreeNode,
            files: buildSynthesizedFiles(effectiveFlowName, prompt, domain)
        };
    }

    if (p.includes("security_audit") || p.includes("secops") || p.includes("security") || p.includes("nmap") || p.includes("port") || domain === "cyber") {
        return {
            flow: {
                id: 'flow_cyber',
                name: `Zero-Trust SecOps Security Engine (${effectiveFlowName})`,
                usingTeams: ['secops_scanners', 'threat_analysts', 'zero_trust_auditors'],
                teams: {},
                checkpoints: [
                    { id: 'cp1', name: '1. Socket Port & Network Reconnaissance', report: 'Scanned ports 22, 80, 443, 3000, 8088 on target host' },
                    { id: 'cp2', name: '2. HTTP Security Header Audit', report: 'Verified HSTS, CSP, and X-Frame-Options policies' },
                    { id: 'cp3', name: '3. OCSF v1.4 SIEM Telemetry Logging', report: 'Emitted standardized OCSF security event telemetry' }
                ],
                currentCheckpointIndex: 2,
                mergePolicy: 'last_wins' as const,
                historyLog: []
            },
            chain: [
                { id: 'c1', name: 'TCP Port Reconnaissance Probe', order: { id: 'o1', type: CommandKind.SEARCH, content: 'Scan TCP ports 22, 80, 443, 8088 for open services', status: 'completed' }, impactLevel: 0.3 },
                { id: 'c2', name: 'HTTP Security Headers Evaluator', order: { id: 'o2', type: CommandKind.TRY, content: 'Audit CSP, HSTS, and X-Content-Type-Options headers', status: 'completed' }, impactLevel: 0.5 },
                { id: 'c3', name: 'Zero-Trust Microsegmentation Gate', order: { id: 'o3', type: CommandKind.JUDGE, content: 'Verify zero-trust token signature & socket access permissions', status: 'completed' }, impactLevel: 0.8 },
                { id: 'c4', name: 'OCSF Telemetry Emitter', order: { id: 'o4', type: CommandKind.COMMUNICATE, content: 'Publish OCSF v1.4 Security Event Log to SIEM cluster', status: 'completed' }, impactLevel: 0.9 }
            ] as SystemChainNode[],
            tree: {
                id: 'root_cyber',
                name: 'Zero-Trust SecOps Security Engine',
                geneticCode: '10',
                type: 'root' as const,
                status: 'healthy' as const,
                children: [
                    {
                        id: 'node_nmap',
                        name: 'Network Reconnaissance Probe',
                        geneticCode: '11',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_nm_1', name: 'TCP Port Scanner (22, 80, 443)', geneticCode: '1101', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_nm_2', name: 'Service Fingerprint Evaluator', geneticCode: '1102', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_sec_hdr',
                        name: 'HTTP Security Headers Auditor',
                        geneticCode: '12',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_hdr_1', name: 'Strict-Transport-Security (HSTS)', geneticCode: '1201', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_hdr_2', name: 'Content-Security-Policy (CSP)', geneticCode: '1202', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_ocsf',
                        name: 'OCSF v1.4 SIEM Telemetry Engine',
                        geneticCode: '13',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_ocsf_1', name: 'Standardized Security Event Schema', geneticCode: '1301', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    }
                ]
            } as ProcessTreeNode,
            files: buildSynthesizedFiles(effectiveFlowName, prompt, domain)
        };
    } else if (p.includes("hospital") || p.includes("clinical") || p.includes("hipaa") || p.includes("fhir") || p.includes("patient") || domain === "clinical") {
        return {
            flow: {
                id: 'flow_clinical',
                name: `HIPAA Clinical Bio-Governance Bus (${effectiveFlowName})`,
                usingTeams: ['bio_analysts', 'crypto_guardians', 'fhir_architects'],
                teams: {},
                checkpoints: [
                    { id: 'cp1', name: '1. SHA-256 PII Patient Redaction', report: 'Anonymized patient record: DOB redacted, SHA-256 hash salted' },
                    { id: 'cp2', name: '2. HL7 / FHIR R4 Condition Resource Synthesis', report: 'Generated valid FDA compliant FHIR R4 condition bundle JSON' },
                    { id: 'cp3', name: '3. Clinical Double-Blind Statistical Sign', report: 'Verified p-value < 0.001 double-blind statistical significance' }
                ],
                currentCheckpointIndex: 2,
                mergePolicy: 'deep_merge' as const,
                historyLog: []
            },
            chain: [
                { id: 'c1', name: 'Clinical EHR Data Reader', order: { id: 'o1', type: CommandKind.SEARCH, content: 'Extract raw patient record & diagnostic codes', status: 'completed' }, impactLevel: 0.2 },
                { id: 'c2', name: 'SHA-256 Cryptographic Anonymizer', order: { id: 'o2', type: CommandKind.TRY, content: 'Salt and hash SSN, name, and DOB for HIPAA compliance', status: 'completed' }, impactLevel: 0.5 },
                { id: 'c3', name: 'FDA Governance & Verification Gate', order: { id: 'o3', type: CommandKind.JUDGE, content: 'Verify p-value statistical significance & double-blind trial parameters', status: 'completed' }, impactLevel: 0.7 },
                { id: 'c4', name: 'FHIR R4 Resource Publisher', order: { id: 'o4', type: CommandKind.COMMUNICATE, content: 'Emit HL7 / FHIR R4 JSON clinical trial bundle to registry', status: 'completed' }, impactLevel: 0.9 }
            ] as SystemChainNode[],
            tree: {
                id: 'root_clinical',
                name: 'HIPAA Clinical Bio-Governance Engine',
                geneticCode: '20',
                type: 'root' as const,
                status: 'healthy' as const,
                children: [
                    {
                        id: 'node_hipaa',
                        name: 'Cryptographic SHA-256 Redaction',
                        geneticCode: '21',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_hip_1', name: 'Patient SSN & Name Salted Hash', geneticCode: '2101', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_hip_2', name: 'DOB Age Group Masker', geneticCode: '2102', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_fhir',
                        name: 'HL7 / FHIR R4 Bundle Synthesizer',
                        geneticCode: '22',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_fhir_1', name: 'FHIR R4 Condition Resource Schema', geneticCode: '2201', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    }
                ]
            } as ProcessTreeNode,
            files: buildSynthesizedFiles(effectiveFlowName, prompt, domain)
        };
    } else if (p.includes("bridge_engineering") || p.includes("cad") || p.includes("robotic") || p.includes("kinematic") || p.includes("stl") || domain === "mechanical") {
        return {
            flow: {
                id: 'flow_mechanical',
                name: `3D Robotics CAD & Kinematics Engine (${effectiveFlowName})`,
                usingTeams: ['cad_engineers', 'kinematics_solvers', 'stress_analysts'],
                teams: {},
                checkpoints: [
                    { id: 'cp1', name: '1. 3D ASCII STL Solid Mesh Export', report: 'Synthesized 3D ASCII STL bracket geometry file on disk' },
                    { id: 'cp2', name: '2. 3-DOF Robot Kinematics Solver', report: 'Solved inverse kinematics joint angle transformations [45°, 30°, -10°]' },
                    { id: 'cp3', name: '3. Structural Load & Torque Verification', report: 'Verified 0.01mm tolerance and 450 Nm torque bounds' }
                ],
                currentCheckpointIndex: 2,
                mergePolicy: 'last_wins' as const,
                historyLog: []
            },
            chain: [
                { id: 'c1', name: '3D Geometry Mesh Parameter Reader', order: { id: 'o1', type: CommandKind.SEARCH, content: 'Read CAD bracket dimensions & mesh density', status: 'completed' }, impactLevel: 0.3 },
                { id: 'c2', name: 'ASCII STL Solid Generator', order: { id: 'o2', type: CommandKind.TRY, content: 'Generate ASCII STL facet normal mesh file on disk', status: 'completed' }, impactLevel: 0.5 },
                { id: 'c3', name: 'Forward Kinematics Matrix Solver', order: { id: 'o3', type: CommandKind.JUDGE, content: 'Solve 3-DOF robot joint matrices & verify 450 Nm torque limit', status: 'completed' }, impactLevel: 0.8 },
                { id: 'c4', name: 'CAD Engineering Visualizer', order: { id: 'o4', type: CommandKind.COMMUNICATE, content: 'Render 3D STL mesh in browser CAD viewport', status: 'completed' }, impactLevel: 0.9 }
            ] as SystemChainNode[],
            tree: {
                id: 'root_mechanical',
                name: '3D Robotics CAD & Kinematics Engine',
                geneticCode: '30',
                type: 'root' as const,
                status: 'healthy' as const,
                children: [
                    {
                        id: 'node_stl',
                        name: '3D ASCII STL Mesh Generator',
                        geneticCode: '31',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_stl_1', name: 'Facet Normal Solid Mesh Synthesizer', geneticCode: '3101', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_kin',
                        name: '3-DOF Kinematics Matrix Solver',
                        geneticCode: '32',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_kin_1', name: 'Joint Transformation Matrices [45°, 30°]', geneticCode: '3201', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    }
                ]
            } as ProcessTreeNode,
            files: buildSynthesizedFiles(effectiveFlowName, prompt, domain)
        };
    } else if (p.includes("accounting_erp") || p.includes("accountant") || p.includes("ledger") || p.includes("invoice") || (domain === "economic" && !p.includes("ecom"))) {
        return {
            flow: {
                id: 'flow_economic',
                name: `Accountant ERP Enterprise System (${effectiveFlowName})`,
                usingTeams: ['financial_architects', 'ledger_engineers', 'compliance_auditors', 'ui_designers'],
                teams: {},
                checkpoints: [
                    { id: 'cp1', name: '1. Chart of Accounts & GAAP Setup', report: 'Defined 5-level COA Assets (1000), Liabilities (2000), Equity (3000), Revenue (4000)' },
                    { id: 'cp2', name: '2. Double-Entry Ledger & Invoicing Engine', report: 'Validated mathematical debit == credit balance & VAT 20% calculation' },
                    { id: 'cp3', name: '3. Financial Statements & Audit Engine', report: 'Synthesized Income Statement Net Profit & Balance Sheet equality' },
                    { id: 'cp4', name: '4. Production App Export & Deployment', report: 'Exported React TSX Accountant ERP dashboard application' }
                ],
                currentCheckpointIndex: 2,
                mergePolicy: 'deep_merge' as const,
                historyLog: []
            },
            chain: [
                { id: 'c1', name: 'GAAP COA Hierarchy Scanner', order: { id: 'o1', type: CommandKind.SEARCH, content: 'Analyze GAAP Chart of Accounts & General Ledger Table Schema', status: 'completed' }, impactLevel: 0.2 },
                { id: 'c2', name: 'Double-Entry Transaction Engine', order: { id: 'o2', type: CommandKind.TRY, content: 'Synthesize VAT 20% Invoicing and Debit/Credit Equivalence Engine', status: 'completed' }, impactLevel: 0.4 },
                { id: 'c3', name: 'Sub-Second Audit Trail Signer', order: { id: 'o3', type: CommandKind.JUDGE, content: 'Verify SHA-256 Ledger Audit Log & Financial Statement Rules', status: 'completed' }, impactLevel: 0.7 },
                { id: 'c4', name: 'P&L Statement Renderer', order: { id: 'o4', type: CommandKind.COMMUNICATE, content: 'Render Live Accountant ERP Application Dashboard Stage', status: 'completed' }, impactLevel: 0.9 }
            ] as SystemChainNode[],
            tree: {
                id: 'root_economic',
                name: 'Accountant ERP Enterprise System',
                geneticCode: '00',
                type: 'root' as const,
                status: 'healthy' as const,
                children: [
                    {
                        id: 'node_gl',
                        name: 'General Ledger Core (Debits == Credits)',
                        geneticCode: '01',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_gl_1', name: 'Journal Entry SHA-256 Signer', geneticCode: '0101', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_gl_2', name: 'Sub-second Audit Logger', geneticCode: '0102', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_coa',
                        name: 'Chart of Accounts (1000-5000 GAAP)',
                        geneticCode: '02',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_coa_1', name: 'Assets & Liabilities Tree', geneticCode: '0201', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_coa_2', name: 'SaaS Revenue & COGS Rules', geneticCode: '0202', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_tax',
                        name: 'VAT 20% Tax & Billing Engine',
                        geneticCode: '03',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_tax_1', name: 'Invoice Net & Gross Calculator', geneticCode: '0301', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_statements',
                        name: 'Financial Statements Engine',
                        geneticCode: '04',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_st_1', name: 'P&L Income Statement Generator', geneticCode: '0401', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_st_2', name: 'Balance Sheet Verification', geneticCode: '0402', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    }
                ]
            } as ProcessTreeNode,
            files: buildSynthesizedFiles(effectiveFlowName, prompt, domain)
        };
    } else {
        // Dynamic Custom Prompt Synthesis
        const title = prompt.length > 30 ? prompt.substring(0, 30) + "..." : prompt;
        return {
            flow: {
                id: 'flow_custom',
                name: `${title} (${effectiveFlowName})`,
                usingTeams: ['software_architects', 'ast_compilers', 'qa_auditors'],
                teams: {},
                checkpoints: [
                    { id: 'cp1', name: '1. Architecture & Specification Analysis', report: `Inspected prompt parameters for ${title}` },
                    { id: 'cp2', name: '2. Microservice & Logic Synthesis', report: 'Synthesized domain microservices and component interfaces' },
                    { id: 'cp3', name: '3. Diagnostic Verification & Quality Gate', report: 'Passed zero-warning unit tests & validated telemetry' }
                ],
                currentCheckpointIndex: 2,
                mergePolicy: 'deep_merge' as const,
                historyLog: []
            },
            chain: [
                { id: 'c1', name: 'Domain Specification Scanner', order: { id: 'o1', type: CommandKind.SEARCH, content: `Inspect requirement parameters for ${title}`, status: 'completed' }, impactLevel: 0.2 },
                { id: 'c2', name: 'Microservice Logic Synthesizer', order: { id: 'o2', type: CommandKind.TRY, content: 'Synthesize FlowLang AST microservices & TypeScript handlers', status: 'completed' }, impactLevel: 0.5 },
                { id: 'c3', name: 'Quality Gate Auditor', order: { id: 'o3', type: CommandKind.JUDGE, content: 'Evaluate test coverage & structural rules', status: 'completed' }, impactLevel: 0.7 },
                { id: 'c4', name: 'Web App Staging Deployer', order: { id: 'o4', type: CommandKind.COMMUNICATE, content: 'Deploy synthesized application stage to viewport', status: 'completed' }, impactLevel: 0.9 }
            ] as SystemChainNode[],
            tree: {
                id: 'root_custom',
                name: `${title}`,
                geneticCode: '40',
                type: 'root' as const,
                status: 'healthy' as const,
                children: [
                    {
                        id: 'node_custom_1',
                        name: 'Dynamic Microservice Core',
                        geneticCode: '41',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_cust_1_1', name: 'Logic Handler Engine', geneticCode: '4101', type: 'leaf' as const, status: 'healthy' as const },
                            { id: 'node_cust_1_2', name: 'Telemetry Evaluator', geneticCode: '4102', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    },
                    {
                        id: 'node_custom_2',
                        name: 'CI/CD Quality Gate',
                        geneticCode: '42',
                        type: 'branch' as const,
                        status: 'healthy' as const,
                        children: [
                            { id: 'node_cust_2_1', name: 'Automated Unit Test Suite', geneticCode: '4201', type: 'leaf' as const, status: 'healthy' as const }
                        ]
                    }
                ]
            } as ProcessTreeNode,
            files: buildSynthesizedFiles(effectiveFlowName, title, domain)
        };
    }
}

export const useSimulation = () => {
    const [state, setState] = useState<Omit<SimulationState, 'refreshState' | 'executeFlowPrompt'>>({
        flow: null,
        chain: [],
        tree: null,
        resources: {},
        files: {},
        lastUpdate: '',
        isSimulating: false
    });

    const fetchState = async () => {
        try {
            const response = await fetch('/ide_state.json?t=' + Date.now());
            if (!response.ok) throw new Error('State file not found');

            const data = await response.json();

            setState(prev => {
                // If a domain-specific tree is currently active in memory, do not allow stale root_economic / root_system from disk to overwrite it
                const isCurrentTreeCustom = prev.tree && prev.tree.id && prev.tree.id !== 'root_economic' && prev.tree.id !== 'root_system';
                const isFetchedTreeDefault = data.tree && (data.tree.id === 'root_economic' || data.tree.id === 'root_system');

                if (isCurrentTreeCustom && isFetchedTreeDefault) {
                    return prev;
                }

                return {
                    flow: data.flow || prev.flow,
                    chain: data.chain || prev.chain,
                    tree: data.tree || prev.tree,
                    resources: data.resources || prev.resources,
                    files: data.files || prev.files,
                    lastUpdate: new Date().toLocaleTimeString(),
                    isSimulating: false
                };
            });
        } catch (err) {
            console.debug("Simulation state load fallback:", err);
        }
    };

    const executeFlowPrompt = async (prompt: string, domain: string = 'digital') => {
        // Extract target flow file if specified in prompt
        const match = prompt.match(/\[Target Flow:\s*([^\]]+)\]/i);
        const targetFlow = match ? match[1].trim() : undefined;

        // 1. Immediately apply synthesized domain visualization state matching target flow
        const domainState = generateDomainState(prompt, domain, targetFlow);

        setState(prev => ({
            ...prev,
            flow: domainState.flow,
            chain: domainState.chain,
            tree: domainState.tree,
            files: domainState.files || buildSynthesizedFiles(targetFlow || 'custom_domain_pipeline.flow', prompt, domain),
            lastUpdate: new Date().toLocaleTimeString(),
            isSimulating: true
        }));

        try {
            const config = getStoredAIConfig();
            const response = await fetch('http://localhost:8088/cowork', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain,
                    prompt,
                    model: config.model,
                    apiKey: config.apiKey
                })
            });

            if (response.ok) {
                // Keep the active domainState synthesized from prompt
                console.log("MCP gateway prompt execution success for domain", domain);
            }
        } catch (err) {
            console.error("Execute flow prompt error:", err);
        } finally {
            setState(prev => ({ ...prev, isSimulating: false }));
        }
    };

    useEffect(() => {
        fetchState();
    }, []);

    return {
        ...state,
        refreshState: fetchState,
        executeFlowPrompt
    };
};
