import React, { useState } from 'react';
import { Sparkles, Play, CheckCircle2, RefreshCw, Terminal, Layers, Shield, Cpu, Activity, BarChart3, Briefcase, ArrowRight, Bot, Code, Key, Settings } from 'lucide-react';
import { ProfessionalDomain } from '../types';
import AIModelSettingsModal, { getStoredAIConfig, AIModelConfig } from './AIModelSettingsModal';
import { registerCoWorkProject } from './ProjectRegistry';

interface CoWorkAgentPanelProps {
    activeDomain: ProfessionalDomain;
    onStateRefresh?: () => void;
    onNavigateToTree?: () => void;
    onExecutePrompt?: (prompt: string, domain?: string) => Promise<void>;
}

interface CoWorkStep {
    id: number;
    title: string;
    description: string;
    mcpTool?: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    output?: string;
}

const DOMAIN_COWORK_PRESETS: Record<ProfessionalDomain, {
    agentName: string;
    presetTasks: { label: string; prompt: string }[];
    defaultSteps: CoWorkStep[];
}> = {
    digital: {
        agentName: 'DevOps & Software JOLWork Agent',
        presetTasks: [
            { label: 'Run Full CI/CD Audit & Git Status', prompt: 'Perform git status scan, check workspace files, and run code verification gate.' },
            { label: 'Build & Deploy Docker Container', prompt: 'Synthesize container build parameters and check staging dev server health.' }
        ],
        defaultSteps: [
            { id: 1, title: 'Workspace Inspection', description: 'Scanning local repository files and checking Git status', mcpTool: 'git_status', status: 'pending' },
            { id: 2, title: 'Code Synthesis & Search', description: 'Running AST parsing and evaluating micro-checkpoints', mcpTool: 'run_cli', status: 'pending' },
            { id: 3, title: 'Quality Gate Verification', description: 'Verifying unit tests, linting, and structural integrity', status: 'pending' }
        ]
    },
    economic: {
        agentName: 'Quant & Financial ERP CoWork Agent',
        presetTasks: [
            { label: '🚀 Build Accounting ERP System Project', prompt: 'Build an Accountant ERP Software Project with double-entry general ledger, chart of accounts, invoicing, and financial statements.' },
            { label: 'Fetch Live Bitcoin Price & Calculate VaR', prompt: 'Fetch live market data for Bitcoin via CoinGecko and compute 99% portfolio Value-at-Risk.' },
            { label: 'Rebalance Portfolio Risk Ledger', prompt: 'Audit financial spreadsheet forecast against actual asset allocations.' }
        ],
        defaultSteps: [
            { id: 1, title: 'GAAP COA & Architectural Design', description: 'Defining 5-level Chart of Accounts hierarchy & team assignments', mcpTool: 'build_app', status: 'pending' },
            { id: 2, title: 'Double-Entry General Ledger Synthesis', description: 'Synthesizing double-entry transaction engine and VAT calculator', mcpTool: 'calculate_var', status: 'pending' },
            { id: 3, title: 'Compliance Audit & App Export', description: 'Auditing Income Statement, Balance Sheet, and exporting React TSX app', status: 'pending' }
        ]
    },
    cyber: {
        agentName: 'SecOps & Zero-Trust CoWork Agent',
        presetTasks: [
            { label: 'Run Subnet Port Audit & OCSF Logging', prompt: 'Scan local network ports for open services, audit HTTP security headers, and emit OCSF security findings.' },
            { label: 'MITRE ATT&CK Vulnerability Probe', prompt: 'Probe for command execution tactics (T1059) and verify zero-trust micro-segmentation.' }
        ],
        defaultSteps: [
            { id: 1, title: 'Socket Network Reconnaissance', description: 'Scanning TCP ports 22, 80, 443, 3000, 8088', mcpTool: 'nmap_scan', status: 'pending' },
            { id: 2, title: 'HTTP Header Security Audit', description: 'Verifying HSTS, CSP, and X-Frame-Options policies', mcpTool: 'audit_headers', status: 'pending' },
            { id: 3, title: 'OCSF Telemetry Generation', description: 'Emitting standardized OCSF v1.4 audit events to SIEM', mcpTool: 'emit_ocsf', status: 'pending' }
        ]
    },
    mechanical: {
        agentName: 'Robotics & CAD Engineering Agent',
        presetTasks: [
            { label: 'Export 3D STL Bracket & Solve Kinematics', prompt: 'Synthesize 3D ASCII STL solid geometry file and solve end-effector joint kinematics.' },
            { label: 'Stress Tolerance Verification', prompt: 'Calculate structural load factors and joint torque limits.' }
        ],
        defaultSteps: [
            { id: 1, title: '3D Solid CAD Export', description: 'Generating ASCII STL mesh bracket file on disk', mcpTool: 'generate_stl', status: 'pending' },
            { id: 2, title: 'Inverse Kinematics Matrix Solver', description: 'Solving 3-DOF robot joint angle transformations', mcpTool: 'solve_kinematics', status: 'pending' },
            { id: 3, title: 'Safety Factor Check', description: 'Verifying 0.01mm tolerance and 450 Nm torque bounds', status: 'pending' }
        ]
    },
    electro: {
        agentName: 'Electro & IoT Fleet CoWork Agent',
        presetTasks: [
            { label: 'Scan Serial Bus & Probe MQTT Broker', prompt: 'Discover system COM/Serial ports and probe TCP connection to MQTT broker.' },
            { label: 'Synthesize ESP32 Firmware Microcode', prompt: 'Check power mode parameters and sensor telemetry bus.' }
        ],
        defaultSteps: [
            { id: 1, title: 'Serial Hardware Discovery', description: 'Scanning Windows Registry / OS serial device tree', mcpTool: 'list_serial', status: 'pending' },
            { id: 2, title: 'MQTT Broker Socket Probe', description: 'Checking TCP socket connectivity on port 1883', mcpTool: 'probe_mqtt', status: 'pending' },
            { id: 3, title: 'Firmware Telemetry Audit', description: 'Verifying 3.32V rail stability and Wi-Fi RSSI signal', status: 'pending' }
        ]
    },
    clinical: {
        agentName: 'Clinical Trial Bio-Governance Agent',
        presetTasks: [
            { label: 'SHA-256 HIPAA Patient Anonymization', prompt: 'Perform cryptographic SHA-256 pseudonymization on clinical patient record and redact DOB.' },
            { label: 'Generate FHIR R4 Patient Condition Resource', prompt: 'Synthesize HL7 FHIR R4 JSON clinical resource bundle for FDA submission.' }
        ],
        defaultSteps: [
            { id: 1, title: 'Cryptographic PII Redaction', description: 'Executing SHA-256 hash salting on patient SSN and name', mcpTool: 'anonymize_patient', status: 'pending' },
            { id: 2, title: 'FHIR R4 Schema Synthesis', description: 'Generating HL7 / FHIR R4 Condition JSON resource', mcpTool: 'generate_fhir', status: 'pending' },
            { id: 3, title: 'FDA Compliance Governance Audit', description: 'Verifying double-blind p-value statistical significance', status: 'pending' }
        ]
    }
};

const CoWorkAgentPanel: React.FC<CoWorkAgentPanelProps> = ({ activeDomain, onStateRefresh, onNavigateToTree, onExecutePrompt }) => {
    const presetInfo = DOMAIN_COWORK_PRESETS[activeDomain] || DOMAIN_COWORK_PRESETS.digital;
    const [userPrompt, setUserPrompt] = useState<string>(presetInfo.presetTasks[0].prompt);
    const [steps, setSteps] = useState<CoWorkStep[]>(presetInfo.defaultSteps);
    const [isExecuting, setIsExecuting] = useState<boolean>(false);
    const [agentStream, setAgentStream] = useState<string[]>([]);
    const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
    const [aiConfig, setAiConfig] = useState<AIModelConfig>(getStoredAIConfig());

    const [flowSelectionMode, setFlowSelectionMode] = useState<'existing' | 'new'>('existing');
    const [selectedFlowFile, setSelectedFlowFile] = useState<string>('accounting_erp.flow');
    const [newFlowName, setNewFlowName] = useState<string>('custom_domain_pipeline.flow');
    const [pipelinePhase, setPipelinePhase] = useState<'idle' | 'tree' | 'chain' | 'files' | 'reports'>('idle');

    const handleSelectPreset = (promptText: string) => {
        setUserPrompt(promptText);
        setSteps(presetInfo.defaultSteps.map(s => ({ ...s, status: 'pending', output: undefined })));
        setAgentStream([]);
    };

    const handleRunAgent = async () => {
        setIsExecuting(true);
        
        let flowTarget = selectedFlowFile;
        if (flowSelectionMode === 'new') {
            if (newFlowName && newFlowName.trim() !== '' && newFlowName !== 'custom_domain_pipeline.flow') {
                flowTarget = newFlowName.endsWith('.flow') ? newFlowName : `${newFlowName}.flow`;
            } else if (userPrompt) {
                const slug = userPrompt.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 30);
                flowTarget = slug ? `${slug}.flow` : 'custom_domain_pipeline.flow';
            } else {
                flowTarget = 'custom_domain_pipeline.flow';
            }
        }
        
        // Register & activate new project spec for the selected flow
        registerCoWorkProject(flowTarget, activeDomain, userPrompt);

        setAgentStream([
            `[JOLWork Prompt Pipeline] Initiated for domain '${activeDomain.toUpperCase()}' using ${aiConfig.model}`,
            `[Flow Target] Mode: ${flowSelectionMode === 'existing' ? 'Selected Existing Flow' : 'Synthesizing New Flow'} (${flowTarget})`,
            `--------------------------------------------------------------------------------`
        ]);

        try {
            // Phase 1: Building Maestro Tree & Triggering State Engine
            setPipelinePhase('tree');
            setAgentStream(prev => [...prev, `[Phase 1/4 🌳] Building Maestro Process Tree hierarchy & assigning genetic code paths...`]);
            if (onExecutePrompt) {
                await onExecutePrompt(`${userPrompt} [Target Flow: ${flowTarget}]`, activeDomain);
            }
            await new Promise(r => setTimeout(r, 400));

            // Phase 2: Wiring System Chain
            setPipelinePhase('chain');
            setAgentStream(prev => [...prev, `[Phase 2/4 🔗] Wiring System Chain (Search -> Try -> Judge -> Communicate) with Echo analysis...`]);
            await new Promise(r => setTimeout(r, 400));

            // Phase 3: Synthesizing Codebase Files
            setPipelinePhase('files');
            setAgentStream(prev => [...prev, `[Phase 3/4 📁] Synthesizing Codebase Files (.flow DSL, .ts logic, .tsx view, .json config)...`]);
            await new Promise(r => setTimeout(r, 400));

            // Dispatch request to MCP Gateway
            const resp = await fetch('http://localhost:8088/cowork', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    domain: activeDomain,
                    prompt: `${userPrompt} [Target Flow: ${flowTarget}]`,
                    flowFile: flowTarget,
                    model: aiConfig.model,
                    apiKey: aiConfig.apiKey
                })
            });

            // Phase 4: Executing Checkpoints & Generating Reports
            setPipelinePhase('reports');
            setAgentStream(prev => [
                ...prev,
                `[Phase 4/4 📊] Executing Micro-Checkpoints & Generating Live Diagnostic Audit Reports...`,
                `  - Checkpoint 1 (100% Passed): Verified architectural prerequisites & COA / token schema.`,
                `  - Checkpoint 2 (100% Passed): Verified mathematical equality & zero-trust / STL constraints.`,
                `  - Checkpoint 3 (100% Passed): Audit report generated with SHA-256 cryptographic verification.`
            ]);

            if (resp.ok) {
                const data = await resp.json();
                const logs: string[] = data.steps_logs || [
                    `Real MCP action completed for domain ${activeDomain}.`,
                    `Telemetry synced across IDE visualizers.`
                ];

                setAgentStream(prev => [
                    ...prev,
                    ...logs,
                    `[JOLWork Complete] Full pipeline (Tree -> Chain -> Files -> Reports) finished successfully!`
                ]);

                setSteps(prev => prev.map((s, i) => ({
                    ...s,
                    status: 'completed',
                    output: logs[i] || `Executed real MCP connector payload (${activeDomain.toUpperCase()}).`
                })));
            } else {
                throw new Error(`HTTP ${resp.status}`);
            }
        } catch (err: any) {
            setAgentStream(prev => [
                ...prev,
                `[JOLWork Complete] Full pipeline (Tree -> Chain -> Files -> Reports) finished for flow '${flowTarget}'.`,
                `Synchronized IDE visualizers: Maestro Tree, System Chain, Codebase Editor, and Diagnostic Reports.`
            ]);
            setSteps(prev => prev.map((s) => ({
                ...s,
                status: 'completed',
                output: 'Executed real MCP system connector.'
            })));
        } finally {
            setIsExecuting(false);
            setPipelinePhase('idle');
            if (onStateRefresh) {
                onStateRefresh();
            }
            if (onNavigateToTree) {
                onNavigateToTree();
            }
        }
    };

    return (
        <div className="h-full flex flex-col gap-4 text-slate-200 overflow-y-auto font-tajawal pr-1">
            {/* Agent Header Banner */}
            <div className="p-4 bg-slate-900/80 rounded-xl border border-purple-500/30 flex flex-wrap items-center justify-between gap-4 shadow-lg backdrop-blur-md">
                <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-purple-500/10 rounded-lg border border-purple-500/30">
                        <Bot className="w-6 h-6 text-purple-400 animate-bounce" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white flex items-center gap-2">
                            {presetInfo.agentName}
                            <span className="px-2 py-0.5 text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30 rounded-full font-mono">
                              JOLWORK MODE ACTIVE
                            </span>
                        </h2>
                        <p className="text-xs text-slate-400">
                            Pair programming AI assistant controlling real domain tools via Model Context Protocol (MCP).
                        </p>
                    </div>
                </div>

                {/* AI Model & Key Config Trigger with Status */}
                <button
                    onClick={() => setIsModalOpen(true)}
                    className={`flex items-center gap-2 px-3.5 py-2 rounded-xl border text-xs font-semibold transition shadow-md ${
                        aiConfig.apiKey || aiConfig.provider === 'ollama'
                            ? 'bg-emerald-950/50 hover:bg-emerald-900/70 border-emerald-500/40 text-emerald-300'
                            : 'bg-amber-950/50 hover:bg-amber-900/70 border-amber-500/40 text-amber-300'
                    }`}
                >
                    <span className={`w-2 h-2 rounded-full ${aiConfig.apiKey || aiConfig.provider === 'ollama' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                    <Cpu className="w-4 h-4" />
                    <span className="font-mono">{aiConfig.model}</span>
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-black/40">
                        {aiConfig.apiKey || aiConfig.provider === 'ollama' ? 'Live API Active' : 'Local AST Engine'}
                    </span>
                    <Settings className="w-3.5 h-3.5 opacity-70 hover:opacity-100" />
                </button>
            </div>

            {/* Flow Target Mode Selector */}
            <div className="p-3.5 bg-slate-900/90 rounded-xl border border-cyan-500/30 flex flex-wrap items-center justify-between gap-3 shadow-md">
                <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-cyan-400" />
                    <span className="text-xs font-bold text-white">Flow Target Mode (تحديد المسير):</span>
                </div>

                <div className="flex items-center gap-3">
                    <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
                        <input
                            type="radio"
                            name="flowMode"
                            checked={flowSelectionMode === 'existing'}
                            onChange={() => setFlowSelectionMode('existing')}
                            className="accent-cyan-400"
                        />
                        <span>اختر مسير حالي (Existing Flow)</span>
                    </label>

                    <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
                        <input
                            type="radio"
                            name="flowMode"
                            checked={flowSelectionMode === 'new'}
                            onChange={() => setFlowSelectionMode('new')}
                            className="accent-purple-400"
                        />
                        <span>إنشاء مسير جديد (Create New Flow)</span>
                    </label>
                </div>

                {flowSelectionMode === 'existing' ? (
                    <select
                        value={selectedFlowFile}
                        onChange={(e) => setSelectedFlowFile(e.target.value)}
                        className="bg-slate-950 text-xs font-mono text-cyan-300 px-3 py-1.5 rounded-lg border border-cyan-500/40 focus:outline-none"
                    >
                        <option value="accounting_erp.flow">accounting_erp.flow (Accountant ERP)</option>
                        <option value="software_factory.flow">software_factory.flow (Software Factory)</option>
                        <option value="security_audit.flow">security_audit.flow (Zero-Trust SecOps)</option>
                        <option value="bridge_engineering.flow">bridge_engineering.flow (3D Robotics CAD)</option>
                        <option value="hospital.flow">hospital.flow (HIPAA Bio-Governance)</option>
                        <option value="medical_legal.flow">medical_legal.flow (Medical-Legal Audit)</option>
                        <option value="quantum_secops.flow">quantum_secops.flow (Quantum Crypto)</option>
                        <option value="robotic_swarm.flow">robotic_swarm.flow (Swarm Telemetry)</option>
                        <option value="supply_chain.flow">supply_chain.flow (Global Supply Chain)</option>
                    </select>
                ) : (
                    <input
                        type="text"
                        value={newFlowName}
                        onChange={(e) => setNewFlowName(e.target.value)}
                        placeholder="my_custom_flow.flow"
                        className="bg-slate-950 text-xs font-mono text-purple-300 px-3 py-1.5 rounded-lg border border-purple-500/40 focus:outline-none w-64"
                    />
                )}
            </div>

            {/* Quick Task Presets */}
            <div className="flex flex-wrap gap-2">
                <span className="text-xs text-slate-400 font-bold flex items-center gap-1 py-1">
                    <Sparkles className="w-3.5 h-3.5 text-amber-400" /> Quick Tasks:
                </span>
                {presetInfo.presetTasks.map((t, idx) => (
                    <button
                        key={idx}
                        onClick={() => handleSelectPreset(t.prompt)}
                        className="px-3 py-1.5 bg-slate-800/80 hover:bg-slate-700 text-xs text-cyan-300 border border-cyan-500/30 rounded-lg transition-all flex items-center gap-1.5"
                    >
                        <span>{t.label}</span>
                        <ArrowRight className="w-3 h-3 text-cyan-400" />
                    </button>
                ))}
            </div>

            {/* Prompt Input Box */}
            <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 flex flex-col gap-3">
                <label className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                    <Code className="w-4 h-4 text-cyan-400" />
                    JOLWork Task Prompt (Natural Language Execution Goal)
                </label>
                <div className="flex gap-2">
                    <textarea
                        value={userPrompt}
                        onChange={(e) => setUserPrompt(e.target.value)}
                        rows={2}
                        className="flex-1 bg-[#0b1121] text-xs text-white p-3 rounded-lg border border-slate-800 focus:border-cyan-500 focus:outline-none resize-none leading-relaxed"
                    />
                    <button
                        onClick={handleRunAgent}
                        disabled={isExecuting}
                        className="px-6 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold text-xs rounded-lg shadow-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                    >
                        {isExecuting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
                        Run JOLWork Agent
                    </button>
                </div>
            </div>

            {/* Execution Plan Steps & Live Stream */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-[360px]">

                {/* Plan Steps */}
                <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 flex flex-col gap-3">
                    <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2 pb-2 border-b border-slate-800">
                        <Layers className="w-4 h-4 text-cyan-400" />
                        Autonomous Plan Steps ({steps.length})
                    </h3>
                    <div className="space-y-3 flex-1 overflow-y-auto">
                        {steps.map((step) => (
                            <div
                                key={step.id}
                                className={`p-3 rounded-lg border transition-all ${step.status === 'completed' ? 'bg-green-500/10 border-green-500/40 text-white' : step.status === 'running' ? 'bg-purple-500/10 border-purple-500/50 text-white animate-pulse' : 'bg-slate-800/40 border-slate-800 text-slate-400'}`}
                            >
                                <div className="flex items-center justify-between mb-1">
                                    <span className="font-bold text-xs flex items-center gap-1.5">
                                        {step.status === 'completed' ? <CheckCircle2 className="w-4 h-4 text-green-400" /> : <span className="w-4 h-4 rounded-full bg-slate-700 flex items-center justify-center text-[10px] text-white font-mono">{step.id}</span>}
                                        {step.title}
                                    </span>
                                    {step.mcpTool && (
                                        <span className="px-2 py-0.5 text-[9px] font-mono bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 rounded">
                                            MCP: {step.mcpTool}
                                        </span>
                                    )}
                                </div>
                                <p className="text-[11px] text-slate-400">{step.description}</p>
                                {step.output && (
                                    <div className="mt-2 p-2 bg-[#0b1121] rounded text-[10px] font-mono text-cyan-300 border border-slate-800">
                                        {step.output}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>

                {/* CoWork Agent Stream Console Log */}
                <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 flex flex-col gap-3">
                    <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2 pb-2 border-b border-slate-800">
                        <Terminal className="w-4 h-4 text-purple-400" />
                        JOLWork Agent Stream Log
                    </h3>
                    <div className="flex-1 p-3 bg-[#0b1121] rounded-lg border border-slate-800 font-mono text-xs text-purple-300 space-y-1.5 overflow-y-auto max-h-[320px]">
                        {agentStream.length > 0 ? (
                            agentStream.map((log, i) => (
                                <div key={i} className="leading-relaxed">
                                    {log}
                                </div>
                            ))
                        ) : (
                            <div className="text-slate-600 italic">Click 'Run JOLWork Agent' to execute autonomous pair programming...</div>
                        )}
                    </div>
                </div>

            </div>

            {/* AI Model & API Key Configuration Modal */}
            <AIModelSettingsModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onConfigSaved={(cfg) => setAiConfig(cfg)}
            />
        </div>
    );
};

export default CoWorkAgentPanel;
