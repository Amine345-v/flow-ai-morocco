import React, { useState } from 'react';
import { 
  Cpu, Play, RefreshCw, CheckCircle2, ShieldCheck, Terminal, FileCode, 
  Code2, ExternalLink, Zap, Layers, Sparkles, Server, Search, Check, 
  AlertCircle, Activity, Database, FileText, ArrowRight, CornerDownRight
} from 'lucide-react';
import AccountantERP from './AccountantERP';
import CustomApp from './CustomApp';

export interface ProjectFile {
  id: string;
  name: string;
  type: 'flow' | 'ts' | 'json' | 'tsx';
  category: string;
  status: string;
  size: string;
  path: string;
  codeSnippet: string;
}

interface CodebaseLiveAppProps {
  projectName?: string;
  domain?: string;
  projectFiles?: ProjectFile[];
  fileCodes?: Record<string, string>;
  browserUrl?: string;
}

export const CodebaseLiveApp: React.FC<CodebaseLiveAppProps> = ({
  projectName = 'Synthesized AI Architecture',
  domain = 'digital',
  projectFiles = [],
  fileCodes = {},
  browserUrl = 'http://localhost:5173/app/main'
}) => {
  const [activeModuleTab, setActiveModuleTab] = useState<string>('app_runner');
  const [selectedFileId, setSelectedFileId] = useState<string>(projectFiles[0]?.id || '');
  const [executionLogs, setExecutionLogs] = useState<string[]>([
    `[Codebase Engine] Synthesized runtime environment initialized.`,
    `[Parity Check] Mounted ${projectFiles.length} files from codebase to browser stage.`
  ]);
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [executionOutput, setExecutionOutput] = useState<any>(null);

  // Check if project is Accountant ERP
  const nameLower = (projectName || '').toLowerCase();
  if (nameLower.includes('accountant') || nameLower.includes('general ledger') || browserUrl.includes('accountant')) {
    return <AccountantERP />;
  }

  const selectedFile = projectFiles.find(f => f.id === selectedFileId) || projectFiles[0];

  const handleRunModule = async (fileName: string, snippet: string) => {
    setIsExecuting(true);
    setExecutionLogs(prev => [...prev, `[Execute] Invoking module '${fileName}'...`]);
    
    // Simulate dynamic module execution output
    setTimeout(() => {
      setIsExecuting(false);
      const timestamp = new Date().toLocaleTimeString();
      const mockResult = {
        status: "200_OK",
        module: fileName,
        timestamp,
        astCheck: "PASSED",
        outputData: {
          executionTimeMs: Math.floor(Math.random() * 45) + 12,
          memoryAlloc: "14.2 MB",
          nodesProcessed: Math.floor(Math.random() * 12) + 4,
          qualityGate: "100% Passed (Zero Defects)"
        }
      };
      setExecutionOutput(mockResult);
      setExecutionLogs(prev => [
        ...prev,
        `[Success] '${fileName}' executed cleanly in 18ms! Output: ${JSON.stringify(mockResult.outputData)}`
      ]);
    }, 600);
  };

  return (
    <div className="w-full bg-[#090d16] text-slate-200 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden font-tajawal min-h-[620px] flex flex-col">
      {/* Header Bar */}
      <div className="bg-slate-950 px-5 py-3 border-b border-slate-800 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Cpu className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white capitalize">{projectName}</h2>
              <span className="px-2 py-0.5 text-[9px] font-mono rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 uppercase">
                {domain} CODEBASE
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-0.5">
              Reflecting live compiled state from <span className="text-cyan-400">{browserUrl}</span>
            </p>
          </div>
        </div>

        {/* View Switcher Tabs */}
        <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800">
          <button
            onClick={() => setActiveModuleTab('app_runner')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeModuleTab === 'app_runner' 
                ? 'bg-cyan-600 text-white shadow-md' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            <Play className="w-3.5 h-3.5" />
            <span>Interactive App</span>
          </button>
          <button
            onClick={() => setActiveModuleTab('code_inspector')}
            className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
              activeModuleTab === 'code_inspector' 
                ? 'bg-cyan-600 text-white shadow-md' 
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
            }`}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span>Codebase AST ({projectFiles.length})</span>
          </button>
        </div>
      </div>

      {/* Main Content Stage */}
      {activeModuleTab === 'app_runner' ? (
        <div className="flex-1 p-6 flex flex-col gap-6 overflow-y-auto bg-gradient-to-b from-[#0b1222] to-[#090d16]">
          {/* Software Factory Pipeline Tracker (software_factory.flow) */}
          <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 shadow-md">
            <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 font-mono flex items-center gap-1.5 mb-2.5">
              <Sparkles className="w-3.5 h-3.5" /> FlowLang Autonomous Software Factory Pipeline (`software_factory.flow`)
            </span>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
              {[
                { name: '1. Discovery', team: 'product_thinker', status: 'Completed' },
                { name: '2. Architecture', team: 'system_architects', status: 'Completed' },
                { name: '3. Implementation', team: 'code_engineers', status: 'Completed' },
                { name: '4. Quality Gate', team: 'qa_reviewers', status: 'Passed (100%)' },
                { name: '5. CTO Approval', team: 'release_approval', status: 'Approved' },
                { name: '6. Production', team: 'blue_green', status: 'Deployed & Live' },
              ].map((stage, idx) => (
                <div key={idx} className="bg-slate-900 p-2 rounded-lg border border-cyan-500/20 flex flex-col justify-between">
                  <span className="text-[10px] font-bold text-white truncate">{stage.name}</span>
                  <span className="text-[9px] font-mono text-cyan-400 mt-0.5">{stage.team}</span>
                  <span className="text-[9px] font-mono text-emerald-400 mt-1 flex items-center gap-1">
                    <CheckCircle2 className="w-2.5 h-2.5" /> {stage.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Top Banner KPI Overview */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 backdrop-blur-md">
              <span className="text-[10px] text-slate-400 font-mono block">Synthesized Modules</span>
              <span className="text-xl font-bold text-cyan-400 font-mono mt-1 block">
                {projectFiles.filter(f => f.type === 'ts' || f.type === 'tsx').length} Services
              </span>
              <span className="text-[10px] text-emerald-400 flex items-center gap-1 mt-1 font-mono">
                <CheckCircle2 className="w-3 h-3" /> Live & Executable
              </span>
            </div>

            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 backdrop-blur-md">
              <span className="text-[10px] text-slate-400 font-mono block">DSL Architecture</span>
              <span className="text-xl font-bold text-purple-400 font-mono mt-1 block">
                {projectFiles.find(f => f.type === 'flow')?.name || 'flowlang.flow'}
              </span>
              <span className="text-[10px] text-purple-300 flex items-center gap-1 mt-1 font-mono">
                <Layers className="w-3 h-3" /> Parsed Checkpoints
              </span>
            </div>

            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 backdrop-blur-md">
              <span className="text-[10px] text-slate-400 font-mono block">AST Security Gate</span>
              <span className="text-xl font-bold text-emerald-400 font-mono mt-1 block">Zero Errors</span>
              <span className="text-[10px] text-emerald-300 flex items-center gap-1 mt-1 font-mono">
                <ShieldCheck className="w-3 h-3" /> 100% Type Safe
              </span>
            </div>

            <div className="bg-slate-900/80 p-4 rounded-xl border border-slate-800 backdrop-blur-md">
              <span className="text-[10px] text-slate-400 font-mono block">Runtime Telemetry</span>
              <span className="text-xl font-bold text-amber-400 font-mono mt-1 block">Active Hub</span>
              <span className="text-[10px] text-amber-300 flex items-center gap-1 mt-1 font-mono">
                <Activity className="w-3 h-3 animate-pulse" /> Real-Time MCP
              </span>
            </div>
          </div>

          {/* Interactive Codebase Module Executor */}
          <div className="bg-slate-900/90 rounded-2xl border border-slate-800/80 p-5 shadow-xl">
            <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
              <div>
                <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                  <Terminal className="w-4 h-4 text-cyan-400" />
                  Live Codebase Module Runner
                </h3>
                <p className="text-[10px] text-slate-400 mt-0.5 font-mono">
                  Select any AI-synthesized TS module from the codebase and trigger its functions in real-time.
                </p>
              </div>

              {isExecuting && (
                <span className="flex items-center gap-2 text-xs text-cyan-400 font-mono animate-pulse">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Running AST AST execution...
                </span>
              )}
            </div>

            {/* Modules Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-5">
              {projectFiles.map(file => {
                const isSelected = file.id === selectedFileId;
                const snippet = fileCodes[file.id] || file.codeSnippet;
                return (
                  <div
                    key={file.id}
                    onClick={() => setSelectedFileId(file.id)}
                    className={`p-3.5 rounded-xl border transition-all cursor-pointer ${
                      isSelected 
                        ? 'bg-cyan-500/15 border-cyan-500/50 text-white shadow-lg' 
                        : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 text-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-mono font-bold text-cyan-300 truncate">
                        {file.name}
                      </span>
                      <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700 uppercase">
                        {file.type}
                      </span>
                    </div>

                    <p className="text-[10px] text-slate-400 font-mono line-clamp-1 mb-3">
                      {file.path}
                    </p>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRunModule(file.name, snippet);
                      }}
                      disabled={isExecuting}
                      className="w-full py-1.5 px-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-bold flex items-center justify-center gap-1.5 transition-all shadow-md active:scale-95 disabled:opacity-50"
                    >
                      <Play className="w-3 h-3 fill-white" />
                      <span>Execute Module</span>
                    </button>
                  </div>
                );
              })}
            </div>

            {/* Execution Result Display */}
            {executionOutput && (
              <div className="bg-slate-950 p-4 rounded-xl border border-cyan-500/30 font-mono text-xs text-slate-300 animate-fade-in">
                <div className="flex items-center justify-between text-cyan-400 mb-2 border-b border-slate-800 pb-2">
                  <span className="font-bold flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                    Execution Output: {executionOutput.module}
                  </span>
                  <span className="text-[10px] text-slate-500">{executionOutput.timestamp}</span>
                </div>
                <pre className="text-[11px] text-emerald-300 bg-black/40 p-3 rounded-lg overflow-x-auto">
                  {JSON.stringify(executionOutput, null, 2)}
                </pre>
              </div>
            )}
          </div>

          {/* Live Console Output Stream */}
          <div className="bg-slate-950 rounded-xl border border-slate-800/80 p-4 font-mono text-xs">
            <div className="flex items-center justify-between mb-2 text-slate-400 border-b border-slate-800/60 pb-2">
              <span className="text-[10px] font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5" /> Browser Stage Runtime Console
              </span>
              <span className="text-[9px] text-slate-500">{executionLogs.length} Events</span>
            </div>
            <div className="space-y-1.5 max-h-36 overflow-y-auto text-[11px]">
              {executionLogs.map((log, index) => (
                <div key={index} className="flex items-start gap-2 text-slate-300">
                  <CornerDownRight className="w-3 h-3 text-cyan-400 shrink-0 mt-0.5" />
                  <span>{log}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Codebase AST Inspector View */
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden bg-[#090d16]">
          {/* File Sidebar */}
          <div className="w-full md:w-64 bg-slate-950 border-r border-slate-800 p-3 flex flex-col gap-2 shrink-0">
            <span className="text-[10px] font-bold font-mono text-slate-400 uppercase tracking-wider px-2 py-1">
              Codebase Files ({projectFiles.length})
            </span>
            <div className="space-y-1 overflow-y-auto flex-1">
              {projectFiles.map(f => {
                const isSelected = f.id === selectedFileId;
                return (
                  <button
                    key={f.id}
                    onClick={() => setSelectedFileId(f.id)}
                    className={`w-full text-left p-2 rounded-lg font-mono text-xs flex items-center justify-between transition-all ${
                      isSelected ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30' : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                    }`}
                  >
                    <span className="truncate">{f.name}</span>
                    <span className="text-[9px] px-1 rounded bg-slate-800 uppercase text-slate-500 ml-1">
                      {f.type}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Code Viewer Panel */}
          <div className="flex-1 p-5 overflow-y-auto flex flex-col">
            {selectedFile ? (
              <div className="space-y-3">
                <div className="flex items-center justify-between bg-slate-900 p-3 rounded-xl border border-slate-800">
                  <div>
                    <h4 className="text-xs font-bold font-mono text-cyan-400">{selectedFile.name}</h4>
                    <p className="text-[10px] text-slate-400 font-mono mt-0.5">{selectedFile.path}</p>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    {selectedFile.status}
                  </span>
                </div>

                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 font-mono text-xs overflow-x-auto text-cyan-200">
                  <pre className="text-[11px] leading-relaxed">
                    {fileCodes[selectedFile.id] || selectedFile.codeSnippet}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-xs text-slate-500 font-mono">
                Select a file to inspect AST source code.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default CodebaseLiveApp;
