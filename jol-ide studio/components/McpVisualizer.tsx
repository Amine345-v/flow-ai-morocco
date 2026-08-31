import React, { useState } from 'react';
import { Server, Play, Terminal, CheckCircle2, AlertCircle, RefreshCw, Layers, Shield, Cpu, Activity, BarChart3, Briefcase, Zap } from 'lucide-react';
import { ProfessionalDomain, McpToolCall } from '../types';

interface McpVisualizerProps {
    activeDomain: ProfessionalDomain;
    onExecuteDomainFlow?: (domain: ProfessionalDomain) => void;
}

const MCP_TOOLS = [
    {
        name: 'flowlang_run',
        description: 'Execute FlowLang DSL across professional domains in live/simulation mode.',
        params: { flow_file: 'examples/software_factory.flow', target_flow: 'build_crm_saas', dry_run: true }
    },
    {
        name: 'flowlang_inspect',
        description: 'Inspect real-time telemetry, checkpoints, and process tree state.',
        params: { domain: 'digital' }
    },
    {
        name: 'flowlang_get_domains',
        description: 'List specifications, teams, and metrics for all 6 professional domains.',
        params: {}
    },
    {
        name: 'flowlang_touch_chain',
        description: 'Trigger sensitivity damping effect on system chain nodes.',
        params: { chain_name: 'development_pipeline', node_name: 'Testing', effect: 1.0 }
    },
    {
        name: 'flowlang_eval_checkpoint',
        description: 'Evaluate micro-checkpoint governance batch checks against passing threshold.',
        params: { checkpoint_name: 'quality_gate', items: ['unit_test', 'security_scan', 'load_test'], threshold: 0.9 }
    }
];

const McpVisualizer: React.FC<McpVisualizerProps> = ({ activeDomain, onExecuteDomainFlow }) => {
    const [selectedTool, setSelectedTool] = useState<string>('flowlang_run');
    const [isExecuting, setIsExecuting] = useState<boolean>(false);
    const [callLogs, setCallLogs] = useState<McpToolCall[]>([
        {
            id: 'mcp-101',
            tool: 'flowlang_get_domains',
            arguments: {},
            timestamp: new Date(Date.now() - 120000).toLocaleTimeString(),
            status: 'success',
            result: { status: 'OK', domains: ['digital', 'economic', 'cyber', 'mechanical', 'electro', 'clinical'] }
        },
        {
            id: 'mcp-102',
            tool: 'flowlang_inspect',
            arguments: { domain: activeDomain },
            timestamp: new Date(Date.now() - 45000).toLocaleTimeString(),
            status: 'success',
            result: { activeDomain, telemetrySynced: true, impactLevel: 1.0 }
        }
    ]);
    const [paramInput, setParamInput] = useState<string>(
        JSON.stringify(MCP_TOOLS[0].params, null, 2)
    );

    const handleSelectTool = (toolName: string) => {
        setSelectedTool(toolName);
        const toolObj = MCP_TOOLS.find(t => t.name === toolName);
        if (toolObj) {
            setParamInput(JSON.stringify(toolObj.params, null, 2));
        }
    };

    const handleRunTool = () => {
        setIsExecuting(true);
        let parsedArgs = {};
        try {
            parsedArgs = JSON.parse(paramInput);
        } catch (e) {
            parsedArgs = { raw: paramInput };
        }

        setTimeout(() => {
            const newCall: McpToolCall = {
                id: `mcp-${Date.now().toString().slice(-4)}`,
                tool: selectedTool,
                arguments: parsedArgs,
                timestamp: new Date().toLocaleTimeString(),
                status: 'success',
                result: {
                    jsonrpc: '2.0',
                    status: 'EXECUTED_OK',
                    domain: activeDomain,
                    message: `MCP Tool '${selectedTool}' executed successfully via stdio protocol.`,
                    payload: parsedArgs
                }
            };
            setCallLogs(prev => [newCall, ...prev]);
            setIsExecuting(false);
            if (onExecuteDomainFlow && selectedTool === 'flowlang_run') {
                onExecuteDomainFlow(activeDomain);
            }
        }, 600);
    };

    return (
        <div className="h-full flex flex-col gap-4 text-slate-200 overflow-y-auto pr-1 font-tajawal">
            {/* Server Status Header */}
            <div className="p-4 bg-slate-900/80 rounded-xl border border-cyan-500/30 flex flex-wrap items-center justify-between gap-4 shadow-lg backdrop-blur-md">
                <div className="flex items-center gap-3">
                    <div className="p-2.5 bg-cyan-500/10 rounded-lg border border-cyan-500/30">
                        <Server className="w-6 h-6 text-cyan-400 animate-pulse" />
                    </div>
                    <div>
                        <h2 className="text-lg font-bold text-white flex items-center gap-2">
                            Model Context Protocol (MCP) Server
                            <span className="px-2 py-0.5 text-[10px] bg-green-500/20 text-green-400 border border-green-500/30 rounded-full font-mono">
                                JSON-RPC 2.0 ACTIVE
                            </span>
                        </h2>
                        <p className="text-xs text-slate-400">
                            Protocol Version: 2024-11-05 | Transport: Stdio / HTTP Bridge | Active Domain: <span className="text-cyan-400 uppercase font-semibold">{activeDomain}</span>
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={() => handleSelectTool('flowlang_inspect')}
                        className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs font-semibold rounded-lg text-cyan-300 border border-cyan-500/30 flex items-center gap-1.5 transition-all"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 ${isExecuting ? 'animate-spin' : ''}`} />
                        Sync Telemetry
                    </button>
                </div>
            </div>

            {/* Main MCP Interactive Studio Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1 min-h-[480px]">
                
                {/* Panel 1: MCP Tools Explorer */}
                <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 flex flex-col gap-3">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                        <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                            <Zap className="w-4 h-4 text-amber-400" />
                            Registered MCP Tools ({MCP_TOOLS.length})
                        </h3>
                    </div>

                    <div className="space-y-2 flex-1 overflow-y-auto">
                        {MCP_TOOLS.map(t => {
                            const isSelected = selectedTool === t.name;
                            return (
                                <button
                                    key={t.name}
                                    onClick={() => handleSelectTool(t.name)}
                                    className={`w-full text-left p-3 rounded-lg border transition-all ${isSelected ? 'bg-cyan-500/10 border-cyan-500/50 text-white shadow-md' : 'bg-slate-800/40 border-slate-800 text-slate-400 hover:bg-slate-800/80 hover:text-slate-200'}`}
                                >
                                    <div className="flex items-center justify-between">
                                        <span className="font-mono text-xs font-bold text-cyan-300">{t.name}</span>
                                        {isSelected && <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_cyan]" />}
                                    </div>
                                    <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">{t.description}</p>
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Panel 2: Tool Execution & Parameters Form */}
                <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 flex flex-col gap-3">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                        <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                            <Terminal className="w-4 h-4 text-purple-400" />
                            Tool Invocation & JSON Parameters
                        </h3>
                        <span className="text-xs font-mono text-cyan-400">{selectedTool}</span>
                    </div>

                    <div className="flex-1 flex flex-col gap-2">
                        <label className="text-[10px] text-slate-500 uppercase tracking-widest font-mono">Input JSON Schema Payload</label>
                        <textarea
                            value={paramInput}
                            onChange={(e) => setParamInput(e.target.value)}
                            rows={10}
                            className="w-full bg-[#0b1121] text-cyan-300 font-mono text-xs p-3 rounded-lg border border-slate-800 focus:border-cyan-500 focus:outline-none resize-none leading-relaxed"
                        />
                    </div>

                    <button
                        onClick={handleRunTool}
                        disabled={isExecuting}
                        className="w-full py-2.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold text-xs rounded-lg shadow-lg flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                    >
                        {isExecuting ? (
                            <>
                                <RefreshCw className="w-4 h-4 animate-spin" />
                                Executing JSON-RPC Command...
                            </>
                        ) : (
                            <>
                                <Play className="w-4 h-4 fill-white" />
                                Invoke MCP Tool ({selectedTool})
                            </>
                        )}
                    </button>
                </div>

                {/* Panel 3: Live MCP Audit & JSON-RPC Stream */}
                <div className="p-4 bg-slate-900/60 rounded-xl border border-slate-800 flex flex-col gap-3">
                    <div className="flex items-center justify-between pb-2 border-b border-slate-800">
                        <h3 className="text-sm font-bold text-slate-300 flex items-center gap-2">
                            <Layers className="w-4 h-4 text-green-400" />
                            MCP JSON-RPC Audit Stream ({callLogs.length})
                        </h3>
                    </div>

                    <div className="space-y-3 flex-1 overflow-y-auto pr-1">
                        {callLogs.map((log) => (
                            <div key={log.id} className="p-3 bg-[#0b1121] rounded-lg border border-slate-800/80 text-xs font-mono">
                                <div className="flex items-center justify-between text-[11px] mb-1.5">
                                    <span className="text-cyan-400 font-bold flex items-center gap-1">
                                        <CheckCircle2 className="w-3 h-3 text-green-400" />
                                        {log.tool}
                                    </span>
                                    <span className="text-slate-500">{log.timestamp}</span>
                                </div>
                                <div className="text-[10px] text-slate-400 bg-black/30 p-2 rounded border border-slate-800/50 overflow-x-auto">
                                    <pre>{JSON.stringify(log.result, null, 2)}</pre>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default McpVisualizer;
