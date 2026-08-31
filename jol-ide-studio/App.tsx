import React, { useState } from 'react';
import CodeEditor from './components/CodeEditor';
import FlowVisualizer from './components/FlowVisualizer';
import ChainVisualizer from './components/ChainVisualizer';
import TreeVisualizer from './components/TreeVisualizer';
import MonolithLog from './components/MonolithLog';
import ProfessionalRegistry from './components/ProfessionalRegistry';
import ResourceView from './components/ResourceView';
import McpVisualizer from './components/McpVisualizer';
import DomainSoftwareConnector from './components/DomainSoftwareConnector';
import CoWorkAgentPanel from './components/CoWorkAgentPanel';
import DomainDashboard from './components/DomainDashboard';
import ProjectRegistry from './components/ProjectRegistry';
import AIModelSettingsModal, { getStoredAIConfig, AIModelConfig } from './components/AIModelSettingsModal';
import { useSimulation } from './hooks/useSimulation';
import { Flow, Order, SystemChainNode, ProcessTreeNode, ProfessionalDomain } from './types';
import { Activity, GitBranch, Layers, FileCode, Hexagon, Cpu, ShieldCheck, Database, Server, Zap, Sparkles, LayoutDashboard, Briefcase, FolderGit2, Key, Settings } from 'lucide-react';

// Initial Fallback Flow
const INITIAL_FLOW: Flow = {
    id: 'f1',
    name: 'مسير استكشافي (Init)',
    usingTeams: ['software_engineers'],
    teams: {},
    checkpoints: [
        { id: 'cp1', name: 'البداية' }
    ],
    currentCheckpointIndex: 0,
    mergePolicy: 'deep_merge',
    historyLog: []
};

const INITIAL_TREE: ProcessTreeNode = {
    id: 'root',
    name: 'المنتج الرقمي',
    geneticCode: '0',
    type: 'root',
    status: 'healthy',
    children: []
};

const App: React.FC = () => {
    const [activeTab, setActiveTab] = useState<'cowork' | 'dashboard' | 'flow' | 'chain' | 'tree' | 'code' | 'resources' | 'projects' | 'mcp'>('projects');
    const [activeDomain, setActiveDomain] = useState<ProfessionalDomain>('digital');
    const [isAISettingsOpen, setIsAISettingsOpen] = useState<boolean>(false);
    const [aiConfig, setAiConfig] = useState<AIModelConfig>(getStoredAIConfig());
    const sim = useSimulation();

    const handleAddOrder = (order: Order) => {
        console.log("New Human Order:", order);
    };

    return (
        <div className="min-h-screen bg-[#0b1121] text-slate-200 flex flex-col md:flex-row font-tajawal selection:bg-cyan-500/30">

            {/* Sidebar: Command & Domain Registry */}
            <aside className="w-full md:w-1/3 lg:w-1/4 p-4 flex flex-col gap-4 border-l border-slate-800 bg-[#0f172a] shadow-2xl z-10">
                <div className="mb-4 text-center md:text-right">
                    <h1 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-500 mb-1 flex items-center gap-2 justify-center md:justify-start">
                        <Hexagon className="w-6 h-6 text-cyan-500 fill-cyan-500/20 animate-spin-slow" />
                        JOL Studio IDE
                    </h1>
                    <div className="flex items-center gap-2 justify-center md:justify-end">
                        <span className={`w-2 h-2 rounded-full ${sim.isSimulating ? 'bg-green-500 animate-pulse' : 'bg-purple-400'}`} />
                        <p className="text-[10px] text-slate-400 font-mono tracking-wide uppercase">
                            {sim.isSimulating ? 'Live Governance Active' : 'Real MCP Software Hub Active'}
                        </p>
                    </div>
                </div>

                <div className="flex-1 flex flex-col gap-4 min-h-0">
                    <ProfessionalRegistry
                        activeDomain={activeDomain}
                        onDomainChange={setActiveDomain}
                    />

                    <div className="flex-1 min-h-0">
                        <MonolithLog logs={sim.flow?.teams ? Object.keys(sim.flow.teams).map(t => ({ id: t, type: 'TRY' as any, content: t, status: 'completed' })) : []} />
                    </div>
                </div>

                <div className="p-3 bg-slate-900/50 rounded-lg border border-slate-800">
                    <div className="flex items-center justify-between text-[10px] text-slate-500 mb-2">
                        <span className="flex items-center gap-1">
                            <Cpu className="w-3 h-3 text-cyan-400" />
                            RUNTIME METRICS
                        </span>
                        <span className="text-purple-400 font-mono font-bold">MCP GATEWAY :8088</span>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-[10px]">
                        <div className="bg-black/20 p-2 rounded">
                            <p className="text-slate-600">Update</p>
                            <p className="text-cyan-400 font-mono">{sim.lastUpdate || '--:--'}</p>
                        </div>
                        <div className="bg-black/20 p-2 rounded">
                            <p className="text-slate-600">Nodes</p>
                            <p className="text-cyan-400 font-mono">{sim.chain.length}</p>
                        </div>
                    </div>
                </div>
            </aside>

            {/* Main Studio Area */}
            <main className="flex-1 p-6 overflow-hidden flex flex-col relative gap-4">
                <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-cyan-900/10 via-[#0b1121] to-[#0b1121] pointer-events-none"></div>

                {/* Software Connector Header */}
                <DomainSoftwareConnector activeDomain={activeDomain} />

                {/* Navigation: Tab Controller */}
                <div className="flex flex-wrap items-center justify-between gap-2 mb-2 border-b border-slate-800 pb-2 relative z-10">
                    <div className="flex flex-wrap gap-2">
                        <button
                            onClick={() => setActiveTab('cowork')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'cowork' ? 'bg-purple-900/40 text-purple-300 border-b-2 border-purple-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <Sparkles className="w-4 h-4 text-amber-400 animate-pulse" />
                            <span>CoWork Agent (AI Team)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('dashboard')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'dashboard' ? 'bg-slate-800/80 text-cyan-400 border-b-2 border-cyan-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <LayoutDashboard className="w-4 h-4" />
                            <span>لوحة التحكم (Dashboard)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('flow')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'flow' ? 'bg-slate-800/80 text-cyan-400 border-b-2 border-cyan-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <Activity className="w-4 h-4" />
                            <span>المسير (Flow)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('chain')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'chain' ? 'bg-slate-800/80 text-purple-400 border-b-2 border-purple-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <Layers className="w-4 h-4" />
                            <span>السلسلة (Chain)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('tree')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'tree' ? 'bg-slate-800/80 text-green-400 border-b-2 border-green-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <GitBranch className="w-4 h-4" />
                            <span>الشجرة (Maestro)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('code')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'code' ? 'bg-slate-800/80 text-orange-400 border-b-2 border-orange-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <FileCode className="w-4 h-4" />
                            <span>البرمجة (Code)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('resources')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'resources' ? 'bg-slate-800/80 text-yellow-400 border-b-2 border-yellow-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <Database className="w-4 h-4" />
                            <span>الموارد (Domain)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('projects')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'projects' ? 'bg-cyan-900/40 text-cyan-300 border-b-2 border-cyan-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <FolderGit2 className="w-4 h-4 text-cyan-400" />
                            <span>المشاريع (Projects)</span>
                        </button>
                        <button
                            onClick={() => setActiveTab('mcp')}
                            className={`flex items-center gap-2 px-4 py-2.5 rounded-t-lg transition-all text-xs font-bold ${activeTab === 'mcp' ? 'bg-slate-800/80 text-pink-400 border-b-2 border-pink-400 shadow-lg' : 'text-slate-500 hover:text-slate-300 hover:bg-slate-800/30'}`}
                        >
                            <Server className="w-4 h-4 text-pink-400 animate-pulse" />
                            <span>MCP Studio</span>
                        </button>
                    </div>

                    {/* AI Model & Key Configuration Quick Trigger */}
                    <button
                        onClick={() => setIsAISettingsOpen(true)}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gradient-to-r from-purple-900/60 to-cyan-900/60 hover:from-purple-800/80 hover:to-cyan-800/80 text-cyan-300 border border-cyan-500/30 text-xs font-bold transition shadow-md"
                        title="Configure AI Model and API Key"
                    >
                        <Cpu className="w-3.5 h-3.5 text-cyan-400" />
                        <span className="font-mono">{aiConfig.model}</span>
                        <Key className="w-3 h-3 text-amber-400" />
                        <Settings className="w-3.5 h-3.5 text-slate-400" />
                    </button>
                </div>

                {/* Dynamic Content: The Studio Floor */}
                <div className="flex-1 overflow-hidden relative z-10">
                    {activeTab === 'projects' && (
                        <div className="h-full animate-fade-in overflow-y-auto">
                            <ProjectRegistry onStateRefresh={sim.refreshState} onNavigateToTree={() => setActiveTab('tree')} />
                        </div>
                    )}
                    {activeTab === 'cowork' && (
                        <div className="h-full animate-fade-in">
                            <CoWorkAgentPanel activeDomain={activeDomain} onStateRefresh={sim.refreshState} onNavigateToTree={() => setActiveTab('tree')} />
                        </div>
                    )}
                    {activeTab === 'dashboard' && (
                        <div className="h-full animate-fade-in overflow-y-auto">
                            <DomainDashboard activeDomain={activeDomain} liveData={sim.resources} />
                        </div>
                    )}
                    {activeTab === 'flow' && (
                        <div className="h-full overflow-auto animate-fade-in">
                            <FlowVisualizer flow={sim.flow || INITIAL_FLOW} onUpdateFlow={() => { }} />
                        </div>
                    )}
                    {activeTab === 'chain' && (
                        <div className="h-full overflow-auto animate-fade-in shadow-inner bg-black/10 rounded-xl p-4">
                            <ChainVisualizer chain={sim.chain} />
                        </div>
                    )}
                    {activeTab === 'tree' && (
                        <div className="h-full overflow-auto animate-fade-in">
                            <TreeVisualizer data={sim.tree || INITIAL_TREE} onStateRefresh={sim.refreshState} />
                        </div>
                    )}
                    {activeTab === 'code' && (
                        <div className="h-full animate-fade-in">
                            <CodeEditor />
                        </div>
                    )}
                    {activeTab === 'resources' && (
                        <div className="h-full animate-fade-in">
                            <ResourceView domain={activeDomain} data={sim.resources || {}} />
                        </div>
                    )}
                    {activeTab === 'mcp' && (
                        <div className="h-full animate-fade-in">
                            <McpVisualizer activeDomain={activeDomain} />
                        </div>
                    )}
                </div>

                {/* Footer Logic Status */}
                <div className="mt-2 pt-3 border-t border-slate-800/50 flex items-center justify-between relative z-10 px-2">
                    <div className="flex items-center gap-4 text-[9px] text-slate-600 font-mono">
                        <div className="flex items-center gap-1">
                            <span className="text-cyan-500">PROJECT:</span> {sim.flow?.name || 'IDLE'}
                        </div>
                        <div className="flex items-center gap-1">
                            <span className="text-purple-400 uppercase tracking-tighter">DOMAIN:</span> {activeDomain.toUpperCase()}
                        </div>
                    </div>
                    <div className="flex items-center gap-2 text-[9px] text-slate-500 font-mono">
                        <Sparkles className="w-3 h-3 text-amber-400" />
                        <span>REAL MCP SOFTWARE HUB READY</span>
                        <ShieldCheck className="w-3 h-3 text-green-500 ml-2" />
                        <span>GOVERNANCE ACTIVE</span>
                    </div>
                </div>

                {/* AI Model & API Key Configuration Modal */}
                <AIModelSettingsModal
                    isOpen={isAISettingsOpen}
                    onClose={() => setIsAISettingsOpen(false)}
                    onConfigSaved={(cfg) => setAiConfig(cfg)}
                />
            </main>
        </div>
    );
};

export default App;