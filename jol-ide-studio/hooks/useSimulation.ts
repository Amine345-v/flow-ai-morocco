import { useState, useEffect } from 'react';
import { Flow, SystemChainNode, ProcessTreeNode, CommandKind } from '../types';
import { getStoredAIConfig } from '../components/AIModelSettingsModal';
import { synthesizeFlowArchitectureWithAI } from '../services/geminiService';

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

// ============================================================================
// FlowLang DSL Parser — Extracts structure from .flow file content
// ============================================================================

interface ParsedFlow {
    flowName: string;
    teams: { name: string; kind: string }[];
    checkpoints: { id: string; name: string; report: string }[];
    processTree: { root: string; branches: Record<string, string[]>; nodes: Record<string, { priority: string; status: string }> } | null;
    chainNodes: string[];
}

function parseFlowDSL(content: string): ParsedFlow {
    const result: ParsedFlow = {
        flowName: 'unknown',
        teams: [],
        checkpoints: [],
        processTree: null,
        chainNodes: []
    };

    // Extract flow name: flow "name" { or flow name(using:...) {
    const flowNameMatch = content.match(/flow\s+"([^"]+)"/i) || content.match(/flow\s+(\w+)\s*\(/i);
    if (flowNameMatch) result.flowName = flowNameMatch[1];

    // Extract process tree
    const processMatch = content.match(/process\s+(\w+)\s+"([^"]+)"\s*\{([\s\S]*?)\n\}/);
    if (processMatch) {
        const processBody = processMatch[3];
        const rootMatch = processBody.match(/root:\s*"([^"]+)"/);
        const branches: Record<string, string[]> = {};
        const nodes: Record<string, { priority: string; status: string }> = {};

        const branchRegex = /branch\s+"([^"]+)"\s*->\s*\[([^\]]+)\]/g;
        let bm;
        while ((bm = branchRegex.exec(processBody)) !== null) {
            branches[bm[1]] = bm[2].split(',').map(s => s.trim().replace(/"/g, ''));
        }

        const nodeRegex = /node\s+"([^"]+)"\s*\{\s*priority:\s*"([^"]+)";\s*status:\s*"([^"]+)"/g;
        let nm;
        while ((nm = nodeRegex.exec(processBody)) !== null) {
            nodes[nm[1]] = { priority: nm[2], status: nm[3] };
        }

        result.processTree = { root: rootMatch ? rootMatch[1] : processMatch[2], branches, nodes };
    }

    // Extract chain nodes
    const chainMatch = content.match(/chain\s+\w+\s*\{[^}]*nodes:\s*\[([^\]]+)\]/);
    if (chainMatch) {
        result.chainNodes = chainMatch[1].split(',').map(s => s.trim());
    }

    // Extract teams — both syntax forms
    const teamRegex1 = /team\s+(\w+)\s*:\s*Command<(\w+)>/g;
    let tm1;
    while ((tm1 = teamRegex1.exec(content)) !== null) {
        result.teams.push({ name: tm1[1], kind: tm1[2] });
    }
    const teamRegex2 = /team\s+"([^"]+)"\s*\{[^}]*kind\s+(\w+)/g;
    let tm2;
    while ((tm2 = teamRegex2.exec(content)) !== null) {
        result.teams.push({ name: tm2[1], kind: tm2[2] });
    }

    // Extract checkpoints
    const cpRegex = /checkpoint\s+"([^"]+)"/g;
    let cpIdx = 0;
    let cpMatch;
    while ((cpMatch = cpRegex.exec(content)) !== null) {
        cpIdx++;
        result.checkpoints.push({
            id: `cp${cpIdx}`,
            name: `${cpIdx}. ${cpMatch[1].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}`,
            report: `Executed checkpoint "${cpMatch[1]}" successfully`
        });
    }

    return result;
}

// ============================================================================
// Build IDE visualization state from parsed flow + order (prompt)
// ============================================================================

const COMMAND_KIND_MAP: Record<string, CommandKind> = {
    'Search': CommandKind.SEARCH,
    'Try': CommandKind.TRY,
    'Judge': CommandKind.JUDGE,
    'Communicate': CommandKind.COMMUNICATE
};

function buildTreeFromParsed(parsed: ParsedFlow, order: string): ProcessTreeNode {
    const pt = parsed.processTree;
    if (!pt) {
        return {
            id: 'root_flow', name: parsed.flowName, geneticCode: '00',
            type: 'root', status: 'healthy',
            children: parsed.checkpoints.map((cp, i) => ({
                id: `node_cp_${i}`, name: cp.name, geneticCode: `0${i + 1}`,
                type: 'leaf' as const, status: 'healthy' as const
            }))
        };
    }

    function buildChildren(parentName: string, depth: number): ProcessTreeNode[] {
        const kids = pt!.branches[parentName] || [];
        return kids.map((childName, idx) => {
            const hasSubKids = pt!.branches[childName] && pt!.branches[childName].length > 0;
            const nodeInfo = pt!.nodes[childName];
            return {
                id: `node_${childName.toLowerCase()}`,
                name: childName.replace(/_/g, ' '),
                geneticCode: `${depth}${idx + 1}`,
                type: hasSubKids ? 'branch' as const : 'leaf' as const,
                status: (nodeInfo?.status === 'implemented' || nodeInfo?.status === 'deployed') ? 'healthy' as const : 'healthy' as const,
                children: hasSubKids ? buildChildren(childName, depth * 10 + idx + 1) : undefined
            };
        });
    }

    return {
        id: `root_${parsed.flowName}`,
        name: `${pt.root} — order: "${order.length > 40 ? order.slice(0, 40) + '...' : order}"`,
        geneticCode: '00',
        type: 'root',
        status: 'healthy',
        children: buildChildren(pt.root, 0)
    };
}

function buildChainFromParsed(parsed: ParsedFlow, order: string): SystemChainNode[] {
    if (parsed.chainNodes.length > 0) {
        const kindCycle = [CommandKind.SEARCH, CommandKind.TRY, CommandKind.JUDGE, CommandKind.COMMUNICATE];
        return parsed.chainNodes.map((nodeName, i) => ({
            id: `c${i + 1}`,
            name: nodeName.replace(/([A-Z])/g, ' $1').trim(),
            order: {
                id: `o${i + 1}`,
                type: kindCycle[i % kindCycle.length],
                content: `${nodeName} — executing order: "${order.length > 50 ? order.slice(0, 50) + '...' : order}"`,
                status: 'completed'
            },
            impactLevel: 0.2 + (0.8 * (i / Math.max(parsed.chainNodes.length - 1, 1)))
        }));
    }

    // Fallback: build chain from teams
    return parsed.teams.map((team, i) => ({
        id: `c${i + 1}`,
        name: team.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
        order: {
            id: `o${i + 1}`,
            type: COMMAND_KIND_MAP[team.kind] || CommandKind.TRY,
            content: `${team.kind} execution for order: "${order}"`,
            status: 'completed'
        },
        impactLevel: 0.2 + (0.8 * (i / Math.max(parsed.teams.length - 1, 1)))
    }));
}

function buildFlowFromParsed(parsed: ParsedFlow, order: string, flowFileName: string): Flow {
    return {
        id: `flow_${parsed.flowName}`,
        name: `${parsed.flowName} (${flowFileName})`,
        usingTeams: parsed.teams.map(t => t.name),
        teams: {},
        checkpoints: parsed.checkpoints.length > 0
            ? parsed.checkpoints
            : [{ id: 'cp1', name: '1. Order Execution', report: `Executed order: "${order}"` }],
        currentCheckpointIndex: Math.max(0, parsed.checkpoints.length - 1),
        mergePolicy: 'deep_merge' as const,
        historyLog: []
    };
}

// ============================================================================
// Known flow file contents registry (fetched or bundled)
// ============================================================================

const KNOWN_FLOW_FILES: Record<string, string> = {};

async function loadFlowFileContent(flowFileName: string): Promise<string | null> {
    if (KNOWN_FLOW_FILES[flowFileName]) return KNOWN_FLOW_FILES[flowFileName];

    // Try fetching from examples directory served by vite
    try {
        const resp = await fetch(`/examples/${flowFileName}?t=${Date.now()}`);
        if (resp.ok) {
            const text = await resp.text();
            KNOWN_FLOW_FILES[flowFileName] = text;
            return text;
        }
    } catch {}

    // Try fetching from public root
    try {
        const resp = await fetch(`/${flowFileName}?t=${Date.now()}`);
        if (resp.ok) {
            const text = await resp.text();
            KNOWN_FLOW_FILES[flowFileName] = text;
            return text;
        }
    } catch {}

    return null;
}

// ============================================================================
// Build synthesized codebase files with initial_human_order
// ============================================================================

function buildSynthesizedFiles(flowFileName: string, order: string, domain: string, flowContent?: string): Record<string, string> {
    const cleanFlowName = flowFileName.endsWith('.flow') ? flowFileName : `${flowFileName}.flow`;
    const baseName = cleanFlowName.replace(/\.flow$/i, '');
    const cleanOrder = (order || 'Execute pipeline').replace(/"/g, '\\"');

    // If we have real flow content, inject the order variable at top
    const dslContent = flowContent
        ? `// initial_human_order injected by JOLWork CoWork Agent\norder initial_human_order = "${cleanOrder}";\n\n${flowContent}`
        : `// ============================================================================
// FlowLang DSL — Generated for order execution
// ============================================================================

order initial_human_order = "${cleanOrder}";

process ${baseName}_roadmap "${baseName} Architecture" {
    root: "${baseName}";
    branch "${baseName}" -> ["CoreEngine", "LogicHandlers", "SecurityGate", "Exporter"];
    node "CoreEngine" { priority: "critical"; status: "implemented"; };
    node "LogicHandlers" { priority: "high"; status: "implemented"; };
    node "SecurityGate" { priority: "high"; status: "implemented"; };
    node "Exporter" { priority: "medium"; status: "implemented"; };
}

chain ${baseName}_chain {
    nodes: [Discovery, Synthesis, Verification, Deploy];
    propagation: causal(decay=0.85, forward=true);
}

team ${domain}_architects : Command<Search> [size=3];
team logic_engineers : Command<Try> [size=4];
team qa_auditors : Command<Judge> [size=2];
team deployer : Command<Communicate> [size=1];

flow ${baseName}_exec(using: ${domain}_architects, logic_engineers, qa_auditors, deployer) {
    context retention: checkpoint;
    merge_policy: deep_merge;

    checkpoint "discovery" (report: brief) {
        brief = ${domain}_architects.search(initial_human_order);
    }
    checkpoint "synthesis" (report: code) {
        code = logic_engineers.try(initial_human_order);
    }
    checkpoint "verification" (report: verdict) {
        verdict = qa_auditors.judge(code, "zero-warning audit");
    }
    checkpoint "deploy" (report: status) {
        status = deployer.ask(initial_human_order);
    }
}
`;

    return {
        [cleanFlowName]: dslContent,
        [`${baseName}_controller.ts`]: `// Controller for ${baseName}\n// initial_human_order = "${cleanOrder}"\n\nexport class ${baseName.replace(/[^a-zA-Z0-9]/g, '_')}_Controller {\n  public initialOrder = "${cleanOrder}";\n  public async execute(payload: any) {\n    console.log("Executing order:", this.initialOrder);\n    return { status: "SUCCESS", order: this.initialOrder, timestamp: Date.now() };\n  }\n}`,
        [`${baseName}_view.tsx`]: `import React from 'react';\n\nexport const ${baseName.replace(/[^a-zA-Z0-9]/g, '_')}_View: React.FC = () => (\n  <div className="p-6 bg-slate-900 text-white rounded-xl border border-purple-500/30">\n    <h2 className="text-xl font-bold">${baseName} Viewport</h2>\n    <p className="text-xs text-purple-300 font-mono mt-1">Order: "${cleanOrder}"</p>\n  </div>\n);`,
        [`${baseName}_schema.json`]: JSON.stringify({ flowName: cleanFlowName, domain, initialOrder: cleanOrder, status: "ACTIVE" }, null, 2)
    };
}

// ============================================================================
// Main hook
// ============================================================================

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
                const isCurrentTreeCustom = prev.tree && prev.tree.id && !prev.tree.id.startsWith('root_economic') && !prev.tree.id.startsWith('root_system');
                const isFetchedTreeDefault = data.tree && (data.tree.id === 'root_economic' || data.tree.id === 'root_system');
                if (isCurrentTreeCustom && isFetchedTreeDefault) return prev;
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
        // Extract target flow file from prompt: [Target Flow: xxx.flow]
        const match = prompt.match(/\[Target Flow:\s*([^\]]+)\]/i);
        const targetFlow = match ? match[1].trim() : undefined;
        const cleanPrompt = prompt.replace(/\[Target Flow:[^\]]*\]/i, '').trim();

        setState(prev => ({ ...prev, isSimulating: true }));

        // 1. Try to load the actual .flow file content
        let flowContent: string | null = null;
        if (targetFlow) {
            flowContent = await loadFlowFileContent(targetFlow);
        }

        // 2. Parse the flow file DSL to extract its structure
        let parsed: ParsedFlow;
        const flowFileName = targetFlow || 'custom_pipeline.flow';

        if (flowContent) {
            parsed = parseFlowDSL(flowContent);
            console.log(`[FlowLang] Parsed "${flowFileName}": ${parsed.checkpoints.length} checkpoints, ${parsed.teams.length} teams, ${parsed.chainNodes.length} chain nodes`);
        } else {
            // Synthesize FlowLang architecture directly via AI Provider Engine
            const aiArch = await synthesizeFlowArchitectureWithAI(cleanPrompt, domain);
            flowContent = aiArch.dslContent;
            parsed = parseFlowDSL(flowContent);
            console.log(`[AI Provider] Synthesized FlowLang architecture for prompt "${cleanPrompt}": ${parsed.checkpoints.length} checkpoints, ${parsed.chainNodes.length} chain nodes`);
        }

        // 3. Build visualization state FROM the parsed flow structure
        const tree = buildTreeFromParsed(parsed, cleanPrompt);
        const chain = buildChainFromParsed(parsed, cleanPrompt);
        const flow = buildFlowFromParsed(parsed, cleanPrompt, flowFileName);
        const files = buildSynthesizedFiles(flowFileName, cleanPrompt, domain, flowContent || undefined);

        // 4. Apply to state immediately
        setState(prev => ({
            ...prev,
            flow,
            chain,
            tree,
            files,
            lastUpdate: new Date().toLocaleTimeString(),
            isSimulating: true
        }));

        // 5. Also hit the MCP gateway
        try {
            const config = getStoredAIConfig();
            await fetch('http://localhost:8088/cowork', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ domain, prompt: cleanPrompt, flowFile: flowFileName, model: config.model, apiKey: config.apiKey })
            });
        } catch (err) {
            console.debug("MCP gateway fallback:", err);
        } finally {
            setState(prev => ({ ...prev, isSimulating: false }));
        }
    };

    useEffect(() => { fetchState(); }, []);

    return { ...state, refreshState: fetchState, executeFlowPrompt };
};
