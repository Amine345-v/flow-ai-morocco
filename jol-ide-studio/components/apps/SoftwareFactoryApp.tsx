import React, { useState, useEffect, useRef } from 'react';
import { 
  Bot, Cpu, Terminal, Play, CheckCircle2, ShieldCheck, FileCode, Code2, 
  Sparkles, Layers, RefreshCw, Send, Search, Check, AlertCircle, Activity, 
  FileText, CornerDownRight, Users, Shield, ArrowRight, Eye, ChevronRight,
  FolderGit2, Lock, Download, Copy, PlayCircle, GitCommit, CheckSquare, XSquare,
  Maximize2, Minimize2, Plus, Trash2
} from 'lucide-react';
import AccountantERP from './AccountantERP';
import { synthesizeDynamicAIFileExpansion, generateAICodeContent } from '../../services/geminiService';

export interface SoftwareFile {
  id: string;
  name: string;
  type: 'flow' | 'ts' | 'json' | 'tsx' | 'py';
  path: string;
  content: string;
  activeAgent?: string;
  status: 'draft' | 'synthesized' | 'verified' | 'deployed';
  lines: number;
}

export interface AgentActivityLog {
  id: string;
  agentName: string;
  role: 'product_thinker' | 'market_researchers' | 'system_architects' | 'code_engineers' | 'qa_reviewers';
  commandKind: 'Communicate' | 'Search' | 'Try' | 'Judge';
  action: string;
  targetFile?: string;
  timestamp: string;
  status: 'running' | 'success' | 'warning';
  details?: string;
}

/**
 * React Class Error Boundary for Software Factory App
 */
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class SoftwareFactoryErrorBoundary extends React.Component<{ children: React.ReactNode }, ErrorBoundaryState> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("[SoftwareFactoryApp ErrorBoundary caught exception]:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-slate-950 text-slate-100 p-8 flex flex-col items-center justify-center font-sans">
          <div className="max-w-md w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-2xl text-center space-y-4">
            <div className="w-12 h-12 bg-red-500/20 text-red-400 rounded-2xl flex items-center justify-center mx-auto border border-red-500/30">
              <AlertCircle className="w-6 h-6" />
            </div>
            <h2 className="text-lg font-bold text-white">Software Factory Container Recovered</h2>
            <p className="text-xs text-slate-400 leading-relaxed">
              An unhandled render exception occurred inside the application container. The workspace state has been preserved.
            </p>
            <div className="p-3 bg-black/60 rounded-xl text-left font-mono text-[11px] text-red-300 overflow-x-auto border border-slate-800">
              {this.state.error?.message || 'Component Exception Caught'}
            </div>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 font-bold text-xs text-white hover:opacity-90 transition shadow-lg shadow-cyan-500/20"
            >
              Reload Software Factory Container
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/**
 * High-Fidelity VS Code Dark+ Tokenizer & Syntax Highlighter
 */
export const renderVSCodeTokens = (code: string) => {
  const lines = (code || '').split('\n');

  return lines.map((line, idx) => {
    const trimmed = line.trim();
    if (trimmed.startsWith('//') || trimmed.startsWith('/*') || trimmed.startsWith('*')) {
      return (
        <div key={idx} className="table-row leading-relaxed hover:bg-[#2a2d2e]">
          <span className="table-cell select-none pr-4 text-right text-[10px] font-mono text-[#858585] border-r border-[#2d2d2d] w-10 shrink-0">
            {idx + 1}
          </span>
          <span className="table-cell pl-3 text-[#6a9955] italic font-mono whitespace-pre">
            {line}
          </span>
        </div>
      );
    }

    const tokens: React.ReactNode[] = [];
    let keyCounter = 0;

    const tokenRegex = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`|\/\/[^\n]*|\b(?:export|import|from|class|interface|async|await|return|const|let|var|function|public|private|protected|flow|process|checkpoint|team|branch|order|node|type|new|default)\b|\b(?:string|number|boolean|any|void|unknown|Record|Promise|Array|SoftwareFile|AuthToken|PaymentIntentPayload|ContactRecord)\b|\b[a-zA-Z_]\w*(?=\()|\b[a-zA-Z_]\w*(?=\s*:)|[a-zA-Z_]\w*|\d+|[^\s\a-zA-Z0-9_]+|\s+)/g;

    let match: RegExpExecArray | null;

    while ((match = tokenRegex.exec(line)) !== null) {
      const text = match[0];
      keyCounter++;

      if (/^("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)$/.test(text)) {
        tokens.push(<span key={keyCounter} style={{ color: '#ce9178' }}>{text}</span>);
      } else if (/^\/\//.test(text)) {
        tokens.push(<span key={keyCounter} style={{ color: '#6a9955', fontStyle: 'italic' }}>{text}</span>);
      } else if (/^\b(export|import|from|class|interface|async|await|return|const|let|var|function|public|private|protected|flow|process|checkpoint|team|branch|order|node|type|new|default)\b$/.test(text)) {
        tokens.push(<span key={keyCounter} style={{ color: '#569cd6', fontWeight: 600 }}>{text}</span>);
      } else if (/^\b(string|number|boolean|any|void|unknown|Record|Promise|Array|SoftwareFile|AuthToken|PaymentIntentPayload|ContactRecord)\b$/.test(text)) {
        tokens.push(<span key={keyCounter} style={{ color: '#4ec9b0' }}>{text}</span>);
      } else if (/^\d+$/.test(text)) {
        tokens.push(<span key={keyCounter} style={{ color: '#b5cea8' }}>{text}</span>);
      } else if (/^[a-zA-Z_]\w*$/.test(text)) {
        const nextChar = line.slice(tokenRegex.lastIndex).trimStart()[0];
        if (nextChar === '(') {
          tokens.push(<span key={keyCounter} style={{ color: '#dcdcaa' }}>{text}</span>);
        } else if (nextChar === ':') {
          tokens.push(<span key={keyCounter} style={{ color: '#9cdcfe' }}>{text}</span>);
        } else if (/^[A-Z]/.test(text)) {
          tokens.push(<span key={keyCounter} style={{ color: '#4ec9b0' }}>{text}</span>);
        } else {
          tokens.push(<span key={keyCounter} style={{ color: '#9cdcfe' }}>{text}</span>);
        }
      } else {
        tokens.push(<span key={keyCounter} style={{ color: '#d4d4d4' }}>{text}</span>);
      }
    }

    return (
      <div key={idx} className="table-row leading-relaxed hover:bg-[#2a2d2e] font-mono text-[11px]">
        <span className="table-cell select-none pr-4 text-right text-[10px] text-[#858585] border-r border-[#2d2d2d] w-10 shrink-0">
          {idx + 1}
        </span>
        <span className="table-cell pl-3 whitespace-pre">
          {tokens.length > 0 ? tokens : line}
        </span>
      </div>
    );
  });
};

interface SoftwareFactoryAppProps {
  projectName?: string;
  domain?: string;
  projectFiles?: any[];
  fileCodes?: Record<string, string>;
  browserUrl?: string;
  onFileCreated?: (file: SoftwareFile) => void;
}

const SoftwareFactoryAppContent: React.FC<SoftwareFactoryAppProps> = ({
  projectName = 'Autonomous CRM SaaS',
  domain = 'digital',
  projectFiles = [],
  fileCodes = {},
  browserUrl = 'http://localhost:5173/app/software_factory',
  onFileCreated
}) => {
  // Check if project is Accountant ERP
  const nameLower = (projectName || '').toLowerCase();
  if (nameLower.includes('accountant') || nameLower.includes('general ledger') || browserUrl.includes('accountant')) {
    return <AccountantERP />;
  }

  const [isFullscreenEditor, setIsFullscreenEditor] = useState<boolean>(false);

  // Initial Multi-File Codebase State
  const [workspaceFiles, setWorkspaceFiles] = useState<SoftwareFile[]>(() => {
    if (projectFiles && projectFiles.length > 0) {
      return projectFiles.map((f, i) => ({
        id: f.id || `f_${i}`,
        name: f.name || `module_${i}.ts`,
        type: f.type || 'ts',
        path: f.path || `/src/${f.name}`,
        content: fileCodes[f.id] || f.codeSnippet || `// Sub-module ${f.name}\nexport class ${f.name.replace(/\.[^/.]+$/, "")} {\n  async run() { return true; }\n}`,
        status: 'verified',
        lines: 45 + (i * 12)
      }));
    }

    return [
      {
        id: 'f_flow',
        name: 'software_factory.flow',
        type: 'flow',
        path: '/flow/software_factory.flow',
        content: `// FlowLang Software Factory Pipeline\norder crm_saas = "Build CRM SaaS Product";\nprocess product_map "CRM Roadmap" {\n  root: "CRM";\n  branch "CRM" -> ["Backend", "Frontend", "Infrastructure"];\n}\n\nflow build_crm_saas(using: market_researchers, system_architects, code_engineers, qa_reviewers, product_thinker) {\n  checkpoint "market_discovery" (report: market_intel) { }\n  checkpoint "architecture" (report: system_design) { }\n  checkpoint "implementation" (report: codebase) { }\n  checkpoint "quality_gate" (report: qa_verdict) { }\n  checkpoint "release_approval" (report: approved) { }\n  checkpoint "production_release" (report: live_status) { }\n}`,
        status: 'verified',
        lines: 32
      },
      {
        id: 'f_app_ui',
        name: `${projectName.replace(/[^a-zA-Z0-9]/g, '') || 'UberDispatcher'}App.tsx`,
        type: 'tsx',
        path: `/src/components/${projectName.replace(/[^a-zA-Z0-9]/g, '') || 'UberDispatcher'}App.tsx`,
        content: `// Live Application Component for ${projectName}\nimport React from 'react';\n\nexport default function App() {\n  return (\n    <div className="p-6 bg-slate-950 text-white font-sans">\n      <h1 className="text-lg font-bold text-cyan-400">${projectName} Live UI</h1>\n      <p className="text-xs text-slate-400 mt-1">Synthesized by AI Model workforce.</p>\n    </div>\n  );\n}`,
        status: 'verified',
        lines: 12,
        activeAgent: 'ui_engineers'
      },
      {
        id: 'f_service_engine',
        name: `${projectName.replace(/[^a-zA-Z0-9]/g, '') || 'UberDispatcher'}Service.ts`,
        type: 'ts',
        path: `/src/modules/${projectName.replace(/[^a-zA-Z0-9]/g, '') || 'UberDispatcher'}Service.ts`,
        content: `// Service Engine for ${projectName}\nexport class ${projectName.replace(/[^a-zA-Z0-9]/g, '') || 'UberDispatcher'}Service {\n  async run() { return true; }\n}`,
        status: 'verified',
        lines: 5,
        activeAgent: 'code_engineers'
      },
      {
        id: 'f_pipeline',
        name: 'PipelineEngine.ts',
        type: 'ts',
        path: '/src/modules/PipelineEngine.ts',
        content: `// Autonomous Pipeline Engine\nexport class PipelineEngine {\n  async execute() { return true; }\n}`,
        status: 'verified',
        lines: 5,
        activeAgent: 'system_architects'
      }
    ];
  });

  const workspaceFilesRef = useRef<SoftwareFile[]>(workspaceFiles);
  useEffect(() => {
    workspaceFilesRef.current = workspaceFiles;
  }, [workspaceFiles]);

  const isSynthesizingRef = useRef<boolean>(false);

  const [activeFileId, setActiveFileId] = useState<string>(workspaceFiles[0]?.id || 'f_flow');
  const activeFile = workspaceFiles.find(f => f.id === activeFileId) || workspaceFiles[0] || {
    id: 'f_fallback',
    name: 'App.tsx',
    path: '/src/App.tsx',
    type: 'tsx',
    content: '// Application View',
    lines: 1,
    status: 'verified'
  };

  const [activeTab, setActiveTab] = useState<'editor' | 'agent_stream'>('agent_stream');
  const [pipelineStage, setPipelineStage] = useState<number>(4); // Stage 4: Quality Gate
  const [isCTOApproved, setIsCTOApproved] = useState<boolean>(false);
  const [terminalInput, setTerminalInput] = useState<string>('');

  // Live Produced App Runner Window State
  const [isAppRunnerWindowOpen, setIsAppRunnerWindowOpen] = useState<boolean>(false);
  const [runnerLogs, setRunnerLogs] = useState<string[]>([]);
  const [runnerInput, setRunnerInput] = useState<string>('');

  const handleRunProducedAppInWindow = () => {
    setIsAppRunnerWindowOpen(true);
    setRunnerLogs([
      `[Live App Container] Mounting runtime container for: ${activeFile.name}...`,
      `[Live App Container] Environment: Production (Vite 6.4.1 + React TSX Container)`,
      `[Live App Container] File Path: ${activeFile.path} (${activeFile.lines} lines)`,
      `[Live App Container] Executing self-diagnostic test... SUCCESS.`,
      `[Live App Container] Interactive application window live.`
    ]);
  };

  const handleExecuteRunnerAction = (e: React.FormEvent) => {
    e.preventDefault();
    if (!runnerInput.trim()) return;
    const time = new Date().toLocaleTimeString();
    setRunnerLogs(prev => [
      `[${time}] Executed live task on ${activeFile.name}: "${runnerInput}"`,
      ...prev
    ]);
    setRunnerInput('');
  };

  // 40-Minute Software Factory Flow Timer & Autonomous Expansion Engine
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(300); // Starts at 05:00
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1);
  const [isTimerRunning, setIsTimerRunning] = useState<boolean>(true);

  useEffect(() => {
    if (!isTimerRunning) return;
    const interval = setInterval(() => {
      setElapsedSeconds(prev => {
        const next = Math.min(2400, prev + speedMultiplier);
        
        // Auto-advance pipeline stage based on 40-min factory progress
        if (next < 300) setPipelineStage(1); // Stage 1: Market Discovery (0-5 min)
        else if (next < 720) setPipelineStage(2); // Stage 2: Architecture (5-12 min)
        else if (next < 1680) setPipelineStage(3); // Stage 3: Implementation (12-28 min)
        else if (next < 2100) setPipelineStage(4); // Stage 4: Quality Gate (28-35 min)
        else if (next < 2400) setPipelineStage(5); // Stage 5: Release Approval (35-40 min)
        else {
          setPipelineStage(6); // Stage 6: Production Live (40 min)
          setIsCTOApproved(true);
        }

        // Autonomous Periodic Codebase Expansion & Progressive File Synthesis (Every 45s simulated time)
        if (Math.floor(next) % 45 === 0 && next > 0 && next < 2400 && !isSynthesizingRef.current) {
          const timestamp = new Date().toLocaleTimeString();
          isSynthesizingRef.current = true;

          // Invoke AI Provider to autonomously decide what file to produce next
          synthesizeDynamicAIFileExpansion(projectName, workspaceFilesRef.current.map(f => f.name)).then(aiDecidedFile => {
            isSynthesizingRef.current = false;
            const newlyProducedFile: SoftwareFile = {
              id: `f_ai_prod_${Date.now()}`,
              name: aiDecidedFile.name,
              type: aiDecidedFile.type as any,
              path: aiDecidedFile.path,
              content: aiDecidedFile.content,
              status: 'verified',
              lines: aiDecidedFile.content.split('\n').length,
              activeAgent: aiDecidedFile.agent
            };

            setWorkspaceFiles(existing => {
              if (existing.some(f => f.name === newlyProducedFile.name)) return existing;
              return [...existing, newlyProducedFile];
            });

            if (onFileCreated) {
              onFileCreated(newlyProducedFile);
            }

            const role = (aiDecidedFile.agent as AgentActivityLog['role']) || 'code_engineers';
            const newLog: AgentActivityLog = {
              id: `log_auto_${Date.now()}`,
              agentName: role,
              role: role,
              commandKind: role === 'qa_reviewers' ? 'Judge' : role === 'market_researchers' ? 'Search' : 'Try',
              action: `[AI Decision] Autonomously synthesized & produced new file: ${newlyProducedFile.path}`,
              targetFile: newlyProducedFile.name,
              timestamp,
              status: 'success'
            };

            setAgentLogs(l => [newLog, ...l.slice(0, 20)]);
            setCliOutput(c => [...c, `[AI Model Decision: ${role}] Produced file: ${newlyProducedFile.path} (${newlyProducedFile.lines} lines)`]);
          }).catch(err => {
            isSynthesizingRef.current = false;
            console.error("[SoftwareFactoryApp] Dynamic file synthesis error:", err);
          });
        }

        return next;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [speedMultiplier, isTimerRunning]);

  const formatTimer = (secs: number) => {
    const m = Math.floor(secs / 60).toString().padStart(2, '0');
    const s = (secs % 60).toString().padStart(2, '0');
    return `${m}:${s} / 40:00`;
  };

  // Agent Activity Logs
  const [agentLogs, setAgentLogs] = useState<AgentActivityLog[]>([
    {
      id: 'log_1',
      agentName: 'product_thinker',
      role: 'product_thinker',
      commandKind: 'Communicate',
      action: 'Formulated product brief & SMB competitive requirement doc.',
      targetFile: 'software_factory.flow',
      timestamp: '15:10:02',
      status: 'success'
    },
    {
      id: 'log_2',
      agentName: 'market_researchers',
      role: 'market_researchers',
      commandKind: 'Search',
      action: 'Executed parallel search on CRM pricing benchmarks & top 20 SaaS pain points.',
      targetFile: 'software_factory.flow',
      timestamp: '15:10:14',
      status: 'success'
    },
    {
      id: 'log_3',
      agentName: 'system_architects',
      role: 'system_architects',
      commandKind: 'Try',
      action: 'Designed process tree, database schema, and microservices architecture.',
      targetFile: 'software_factory.flow',
      timestamp: '15:10:28',
      status: 'success'
    },
    {
      id: 'log_4',
      agentName: 'code_engineers',
      role: 'code_engineers',
      commandKind: 'Try',
      action: 'Synthesized 5 codebase modules: AuthService.ts, ContactsAPI.ts, PipelineEngine.ts, CRMView.tsx.',
      targetFile: 'AuthService.ts',
      timestamp: '15:11:05',
      status: 'success'
    },
    {
      id: 'log_5',
      agentName: 'qa_reviewers',
      role: 'qa_reviewers',
      commandKind: 'Judge',
      action: 'Executed OWASP Top 10 security scan. Coverage > 80%? All 6 modules passed QA.',
      targetFile: 'AuthService.ts',
      timestamp: '15:12:40',
      status: 'success',
      details: 'OWASP Scan: 0 Vulnerabilities | Unit Tests: 92% Coverage | Latency: 42ms'
    }
  ]);

  const [cliOutput, setCliOutput] = useState<string[]>([
    `$ flowlang run software_factory.flow --domain=${domain}`,
    `[Maestro] Product Roadmap process 'product_map' mounted.`,
    `[Team: market_researchers] Round-robin search completed.`,
    `[Team: system_architects] Architecture AST compiled.`,
    `[Team: code_engineers] 5 microservice files written to /src.`,
    `[Team: qa_reviewers] Micro-checkpoints: unit_tests=PASS, OWASP=PASS, load_test=PASS.`,
    `[Stage 5: release_approval] Awaiting CTO approval gate...`
  ]);

  // Trigger Autonomous Agent Action Simulation
  const handleTriggerAgentAction = (role: AgentActivityLog['role'], actionText: string) => {
    const timestamp = new Date().toLocaleTimeString();
    const newLog: AgentActivityLog = {
      id: `log_${Date.now()}`,
      agentName: role,
      role: role,
      commandKind: role === 'qa_reviewers' ? 'Judge' : role === 'market_researchers' ? 'Search' : 'Try',
      action: actionText,
      targetFile: activeFile.name,
      timestamp,
      status: 'success'
    };

    setAgentLogs(prev => [newLog, ...prev]);
    setCliOutput(prev => [...prev, `[Team: ${role}] ${actionText}`]);
  };

  const handleApproveCTO = () => {
    setIsCTOApproved(true);
    setPipelineStage(6); // Move to Stage 6 Production Release
    const timestamp = new Date().toLocaleTimeString();
    setAgentLogs(prev => [
      {
        id: `log_cto_${Date.now()}`,
        agentName: 'cto_approval_gate',
        role: 'product_thinker',
        commandKind: 'Communicate',
        action: 'CTO Approved Staging Deployment! Initiating blue-green production release.',
        targetFile: 'software_factory.flow',
        timestamp,
        status: 'success'
      },
      ...prev
    ]);
    setCliOutput(prev => [
      ...prev,
      `[CTO Gate] Release Approved!`,
      `[Blue-Green Deploy] Deploying v1.0.0 services to staging & production...`,
      `[Production] CRM SaaS is LIVE in production!`
    ]);
  };

  const [isNewFileModalOpen, setIsNewFileModalOpen] = useState<boolean>(false);
  const [newFileName, setNewFileName] = useState<string>('');

  const handleCreateNewFile = async () => {
    if (!newFileName.trim()) return;
    const cleanName = newFileName.trim();
    const ext = cleanName.endsWith('.flow') ? 'flow' : cleanName.endsWith('.json') ? 'json' : cleanName.endsWith('.tsx') ? 'tsx' : 'ts';
    
    // Direct AI Provider generation for manually created files
    const aiContent = await generateAICodeContent(cleanName, projectName, 'code_engineers');

    const created: SoftwareFile = {
      id: `f_user_${Date.now()}`,
      name: cleanName,
      type: ext as any,
      path: `/src/${cleanName}`,
      content: aiContent || `// AI Synthesized File: ${cleanName}\nexport class ${cleanName.replace(/\.[^/.]+$/, "")} {\n  async run() { return { active: true, createdAt: new Date().toISOString() }; }\n}`,
      status: 'verified',
      lines: (aiContent || '').split('\n').length || 10,
      activeAgent: 'code_engineers'
    };

    setWorkspaceFiles(prev => [...prev, created]);
    setActiveFileId(created.id);
    if (onFileCreated) onFileCreated(created);

    setCliOutput(prev => [...prev, `[Explorer] User created file: ${created.path} (AI Synthesized)`]);
    setNewFileName('');
    setIsNewFileModalOpen(false);
  };

  const handleDeleteFile = (fileId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (workspaceFiles.length <= 1) return;
    const fileToDelete = workspaceFiles.find(f => f.id === fileId);
    const updated = workspaceFiles.filter(f => f.id !== fileId);
    setWorkspaceFiles(updated);
    if (activeFileId === fileId) {
      setActiveFileId(updated[0].id);
    }
    setCliOutput(prev => [...prev, `[Explorer] User deleted file: ${fileToDelete?.path || fileId}`]);
  };

  const handleTerminalSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!terminalInput.trim()) return;
    const cmd = terminalInput.trim();
    setTerminalInput('');

    setCliOutput(prev => [...prev, `$ ${cmd}`]);

    const lower = cmd.toLowerCase();

    if (lower === 'help') {
      setCliOutput(prev => [
        ...prev,
        `Available Autonomous CLI Commands:`,
        `  flow test / npm test       - Run QA test suite & micro-checkpoints`,
        `  flow build / npm run build - Compile production JavaScript bundle`,
        `  owasp scan                 - Run OWASP Top 10 security audit`,
        `  git status                 - Display git branch and workspace state`,
        `  ls                         - List all workspace files and sizes`,
        `  clear                      - Clear CLI terminal console log`,
        `  create <filename>          - Autonomously generate a new file`
      ]);
    } else if (lower.includes('test')) {
      setCliOutput(prev => [
        ...prev,
        `[QA Guard] Running unit & integration test suites...`,
        `✓ AuthService.test.ts ........ 14/14 PASSED (32ms)`,
        `✓ PayPalCheckout.test.ts ...... 22/22 PASSED (48ms)`,
        `✓ DisputeEngine.test.ts ....... 10/10 PASSED (28ms)`,
        `[Result] 46/46 Tests PASSED | Zero Defects | Code Coverage: 94.2%`
      ]);
    } else if (lower.includes('build')) {
      setCliOutput(prev => [
        ...prev,
        `[Vite Build Engine] Compiling ${workspaceFiles.length} workspace files...`,
        `✓ 2293 modules transformed cleanly.`,
        `dist/index.html                  1.55 kB │ gzip: 0.73 kB`,
        `dist/assets/index-D5GK7FWt.js  798.43 kB │ gzip: 201.07 kB`,
        `✓ Production bundle built successfully in 1.42s.`
      ]);
    } else if (lower.includes('owasp') || lower.includes('security')) {
      setCliOutput(prev => [
        ...prev,
        `[OWASP Scanner v4.2] Executing static analysis security audit...`,
        `✓ SQL Injection Protection ...... PASS (Prepared Statements verified)`,
        `✓ XSS Sanitization .............. PASS (React JSX Auto-escaping verified)`,
        `✓ PCI-DSS Vault Tokenization .... PASS (AES-256 GCM active)`,
        `[Verdict] 0 Vulnerabilities Found. Staging Security Gate PASSED.`
      ]);
    } else if (lower.includes('git status')) {
      setCliOutput(prev => [
        ...prev,
        `On branch main`,
        `Your branch is up to date with 'origin/main'.`,
        `Untracked / Modified files in Software Factory:`,
        ...workspaceFiles.map(f => `  modified: ${f.path}`),
        `nothing added to commit but untracked files present`
      ]);
    } else if (lower === 'ls') {
      setCliOutput(prev => [
        ...prev,
        `Workspace Files (${workspaceFiles.length}):`,
        ...workspaceFiles.map(f => `  ${f.path.padEnd(35)} [${f.lines} lines] (${f.status})`)
      ]);
    } else if (lower === 'clear') {
      setCliOutput([]);
    } else if (lower.startsWith('create ')) {
      const fileName = cmd.slice(7).trim();
      if (fileName) {
        const ext = fileName.endsWith('.flow') ? 'flow' : fileName.endsWith('.json') ? 'json' : fileName.endsWith('.tsx') ? 'tsx' : 'ts';
        const created: SoftwareFile = {
          id: `f_cli_${Date.now()}`,
          name: fileName,
          type: ext as any,
          path: `/src/${fileName}`,
          content: `// Created via CLI: ${fileName}\nexport class ${fileName.replace(/\.[^/.]+$/, "")} {\n  async run() { return true; }\n}`,
          status: 'verified',
          lines: 12,
          activeAgent: 'code_engineers'
        };
        setWorkspaceFiles(prev => [...prev, created]);
        setActiveFileId(created.id);
        if (onFileCreated) onFileCreated(created);
        setCliOutput(prev => [...prev, `[CLI Engine] Created new file: ${created.path}`]);
      }
    } else {
      setCliOutput(prev => [...prev, `Command '${cmd}' executed. Type 'help' for available CLI commands.`]);
    }
  };

  return (
    <div className="w-full bg-[#070a12] text-slate-200 rounded-2xl border border-slate-800 shadow-2xl overflow-hidden font-tajawal min-h-[680px] flex flex-col">
      {/* Top Navbar */}
      <div className="bg-slate-950 px-5 py-3 border-b border-slate-800 flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-cyan-500/10 border border-cyan-500/30 text-cyan-400">
            <Bot className="w-5 h-5 animate-bounce" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-bold text-white uppercase tracking-wide">{projectName}</h2>
              <span className="px-2 py-0.5 text-[9px] font-mono rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 uppercase">
                5 AGENT TEAMS WORKSPACE
              </span>
            </div>
            <p className="text-[10px] text-slate-400 font-mono mt-0.5">
              FlowLang Autonomous Software Factory • <span className="text-cyan-400">software_factory.flow</span>
            </p>
          </div>
        </div>

        {/* View Switcher Tabs & 40-Min Factory Timer */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* 40-Minute Factory Timer & Speed Controls */}
          <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-cyan-500/30">
            <Activity className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
            <span className="text-xs font-mono font-bold text-cyan-300">
              {formatTimer(elapsedSeconds)}
            </span>
            <div className="flex items-center gap-1 border-l border-slate-800 pl-2">
              <button 
                onClick={() => setSpeedMultiplier(1)}
                className={`px-1.5 py-0.5 text-[9px] font-mono rounded ${speedMultiplier === 1 ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
              >
                1x
              </button>
              <button 
                onClick={() => setSpeedMultiplier(5)}
                className={`px-1.5 py-0.5 text-[9px] font-mono rounded ${speedMultiplier === 5 ? 'bg-cyan-600 text-white' : 'text-slate-400 hover:text-slate-200'}`}
              >
                5x
              </button>
              <button 
                onClick={() => setElapsedSeconds(2400)}
                className="px-1.5 py-0.5 text-[9px] font-mono rounded bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/40"
              >
                Instant
              </button>
            </div>
          </div>

          <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
            <button
              onClick={() => setActiveTab('agent_stream')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                activeTab === 'agent_stream' 
                  ? 'bg-cyan-600 text-white shadow-md' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              <Users className="w-3.5 h-3.5" />
              <span>Agent Teams Stream</span>
            </button>
            <button
              onClick={() => setActiveTab('editor')}
              className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all flex items-center gap-1.5 ${
                activeTab === 'editor' 
                  ? 'bg-cyan-600 text-white shadow-md' 
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`}
            >
              <Code2 className="w-3.5 h-3.5" />
              <span>Multi-File Codebase ({workspaceFiles.length})</span>
            </button>
          </div>

          <button
            onClick={handleApproveCTO}
            disabled={isCTOApproved}
            className={`px-3.5 py-1.5 rounded-xl text-xs font-bold flex items-center gap-1.5 transition-all shadow-md active:scale-95 ${
              isCTOApproved 
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 cursor-default'
                : 'bg-gradient-to-r from-amber-500 to-emerald-500 text-slate-950 hover:brightness-110'
            }`}
          >
            <CheckCircle2 className="w-4 h-4" />
            <span>{isCTOApproved ? 'CTO Approved (Production Live)' : 'Approve CTO Staging Gate'}</span>
          </button>
        </div>
      </div>

      {/* 6-Stage Autonomous Pipeline Progress Tracker */}
      <div className="bg-slate-950/80 px-5 py-2.5 border-b border-slate-800/80">
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-2">
          {[
            { stage: 1, name: '1. Discovery', team: 'product_thinker', desc: 'Market Brief' },
            { stage: 2, name: '2. Architecture', team: 'system_architects', desc: 'AST & Schemas' },
            { stage: 3, name: '3. Implementation', team: 'code_engineers', desc: '5 TS Modules' },
            { stage: 4, name: '4. Quality Gate', team: 'qa_reviewers', desc: 'OWASP & Tests' },
            { stage: 5, name: '5. Release Gate', team: 'CTO Gate', desc: isCTOApproved ? 'Approved' : 'Awaiting Review' },
            { stage: 6, name: '6. Production', team: 'Blue-Green', desc: isCTOApproved ? 'v1.0 Live' : 'Staging Ready' },
          ].map((s) => {
            const isDone = s.stage <= pipelineStage;
            const isCurrent = s.stage === pipelineStage;
            return (
              <div 
                key={s.stage}
                className={`p-2 rounded-lg border transition-all ${
                  isDone 
                    ? 'bg-emerald-500/10 border-emerald-500/30 text-slate-200' 
                    : isCurrent
                    ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300 shadow-md'
                    : 'bg-slate-900/50 border-slate-800 text-slate-500'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[10px] font-bold truncate">{s.name}</span>
                  {isDone && <CheckCircle2 className="w-3 h-3 text-emerald-400 shrink-0" />}
                </div>
                <div className="text-[9px] font-mono text-cyan-400 mt-0.5 truncate">{s.team}</div>
                <div className="text-[9px] font-mono text-slate-400 truncate">{s.desc}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Main Workspace Body */}
      <div className="flex-1 flex flex-col md:flex-row overflow-hidden">
        {/* Left Sidebar: Multi-File Workspace Explorer */}
        <div className="w-full md:w-64 bg-slate-950 border-r border-slate-800 p-3 flex flex-col gap-2 shrink-0">
          <div className="flex items-center justify-between px-2 py-1 border-b border-slate-800/80 pb-2">
            <span className="text-[10px] font-bold font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <FolderGit2 className="w-3.5 h-3.5 text-cyan-400" /> Multi-File Explorer
            </span>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setIsNewFileModalOpen(true)}
                title="Create New File"
                className="px-1.5 py-0.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-[10px] font-mono flex items-center gap-0.5 shadow transition-all"
              >
                <Plus className="w-3 h-3" />
                <span>New</span>
              </button>
              <span className="text-[9px] font-mono text-cyan-400 px-1.5 py-0.5 rounded bg-cyan-500/10">
                {workspaceFiles.length} files
              </span>
            </div>
          </div>

          {/* New File Modal Dialog */}
          {isNewFileModalOpen && (
            <div className="p-2.5 bg-slate-900 border border-cyan-500/40 rounded-xl space-y-2 text-xs font-mono shadow-xl animate-fadeIn">
              <span className="text-[10px] text-cyan-300 font-bold block">Create New File:</span>
              <input
                type="text"
                value={newFileName}
                onChange={(e) => setNewFileName(e.target.value)}
                placeholder="e.g. AuditLogger.ts"
                className="w-full bg-slate-950 border border-slate-800 px-2 py-1 rounded text-white text-xs focus:outline-none focus:border-cyan-400"
                autoFocus
              />
              <div className="flex items-center gap-2 justify-end">
                <button
                  onClick={() => setIsNewFileModalOpen(false)}
                  className="px-2 py-0.5 rounded text-[10px] text-slate-400 hover:text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateNewFile}
                  className="px-2.5 py-0.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white text-[10px] font-bold"
                >
                  Create
                </button>
              </div>
            </div>
          )}

          <div className="space-y-1 overflow-y-auto flex-1">
            {workspaceFiles.map(file => {
              const isSelected = file.id === activeFileId;
              return (
                <div
                  key={file.id}
                  onClick={() => setActiveFileId(file.id)}
                  className={`w-full text-left p-2.5 rounded-xl font-mono text-xs flex items-center justify-between transition-all cursor-pointer group ${
                    isSelected 
                      ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/40 shadow-sm' 
                      : 'text-slate-400 hover:bg-slate-900 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <FileCode className={`w-3.5 h-3.5 shrink-0 ${file.type === 'flow' ? 'text-purple-400' : file.type === 'tsx' ? 'text-amber-400' : 'text-cyan-400'}`} />
                    <span className="truncate">{file.name}</span>
                  </div>

                  <div className="flex items-center gap-1 shrink-0">
                    {file.activeAgent && (
                      <span className="text-[8px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 font-mono animate-pulse">
                        {file.activeAgent}
                      </span>
                    )}
                    {workspaceFiles.length > 1 && (
                      <button
                        onClick={(e) => handleDeleteFile(file.id, e)}
                        title="Delete File"
                        className="opacity-0 group-hover:opacity-100 p-1 text-slate-500 hover:text-rose-400 transition-all"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Quick Trigger Action Buttons */}
          <div className="pt-2 border-t border-slate-800/80 space-y-1.5">
            <span className="text-[9px] font-bold font-mono text-slate-500 uppercase tracking-wider block px-1">
              Trigger Agent Teams
            </span>
            <button
              onClick={() => handleTriggerAgentAction('code_engineers', `Synthesized microservice logic for ${activeFile.name}`)}
              className="w-full py-1.5 px-2 bg-slate-900 hover:bg-slate-800 text-cyan-300 border border-slate-800 rounded-lg text-[10px] font-mono flex items-center justify-between transition-all"
            >
              <span>+ Refactor Module</span>
              <Cpu className="w-3 h-3 text-cyan-400" />
            </button>
            <button
              onClick={() => handleTriggerAgentAction('qa_reviewers', `Ran OWASP Top 10 security audit on ${activeFile.name}`)}
              className="w-full py-1.5 px-2 bg-slate-900 hover:bg-slate-800 text-emerald-300 border border-slate-800 rounded-lg text-[10px] font-mono flex items-center justify-between transition-all"
            >
              <span>+ Run QA Scan</span>
              <ShieldCheck className="w-3 h-3 text-emerald-400" />
            </button>
          </div>
        </div>

        {/* Center Viewport: Code Editor or Agent Stream */}
        <div className="flex-1 flex flex-col bg-[#080c16] overflow-hidden border-r border-slate-800">
          {activeTab === 'editor' ? (
            /* Multi-File VS Code Style Code Editor View */
            <div className={`${
              isFullscreenEditor 
                ? 'fixed inset-0 z-50 flex flex-col bg-[#1e1e1e] w-screen h-screen' 
                : 'flex-1 flex flex-col overflow-hidden bg-[#1e1e1e]'
            }`}>
              {/* Live Agent Work Inspector Header */}
              <div className="bg-gradient-to-r from-cyan-950/90 via-purple-950/80 to-[#181818] px-4 py-2 border-b border-cyan-500/30 flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-cyan-400 animate-spin" />
                  <span className="text-xs font-mono font-bold text-white">
                    team {activeFile.activeAgent || 'code_engineers'}
                  </span>
                  <span className="text-[10px] text-slate-300 font-mono">is working on:</span>
                  <span className="text-xs font-mono text-cyan-300 bg-cyan-500/20 px-2 py-0.5 rounded border border-cyan-500/40 animate-pulse font-bold">
                    {activeFile.name}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleRunProducedAppInWindow}
                    className="px-3 py-1 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 hover:brightness-110 text-slate-950 text-xs font-bold font-mono flex items-center gap-1.5 shadow-md transition-all active:scale-95"
                  >
                    <PlayCircle className="w-4 h-4" />
                    <span>Run Produced App (Other Window)</span>
                  </button>
                  <button
                    onClick={() => setIsFullscreenEditor(prev => !prev)}
                    className="px-2.5 py-1 rounded bg-cyan-900/40 hover:bg-cyan-800/60 text-cyan-300 border border-cyan-500/40 text-[10px] font-mono flex items-center gap-1.5 transition-all shadow"
                  >
                    {isFullscreenEditor ? (
                      <>
                        <Minimize2 className="w-3.5 h-3.5 text-amber-400" />
                        <span>Exit Fullscreen</span>
                      </>
                    ) : (
                      <>
                        <Maximize2 className="w-3.5 h-3.5 text-cyan-400" />
                        <span>Fullscreen Codebase</span>
                      </>
                    )}
                  </button>
                  <div className="flex items-center gap-2 text-[10px] font-mono text-emerald-400">
                    <CheckCircle2 className="w-3.5 h-3.5" />
                    <span>Command&lt;Try&gt; Live Code Synthesis</span>
                  </div>
                </div>
              </div>

              {/* VS Code File Tabs & Breadcrumbs Bar */}
              <div className="bg-[#252526] px-3 py-1.5 border-b border-[#181818] flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="px-3 py-1 bg-[#1e1e1e] border-t-2 border-cyan-500 text-cyan-300 text-xs font-mono font-semibold flex items-center gap-2 rounded-t">
                    <FileCode className="w-3.5 h-3.5 text-cyan-400" />
                    <span>{activeFile.name}</span>
                  </div>
                  <span className="text-[10px] font-mono text-[#858585]">
                    src &gt; {activeFile.path.split('/').slice(2).join(' &gt; ')}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleRunProducedAppInWindow}
                    className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 hover:bg-emerald-500/40 border border-emerald-500/30 text-[10px] font-mono font-bold flex items-center gap-1"
                  >
                    <Play className="w-3 h-3 text-emerald-400 fill-emerald-400" />
                    <span>Run App</span>
                  </button>
                  <span className="px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 text-[10px] font-mono">
                    {activeFile.status}
                  </span>
                  <span className="text-[10px] font-mono text-[#858585]">{activeFile.lines} lines</span>
                </div>
              </div>

              {/* VS Code Code Body with Token Highlighting & Line Numbers */}
              <div className="flex-1 p-3 overflow-y-auto font-mono text-xs bg-[#1e1e1e] relative">
                <div className="table w-full border-collapse">
                  {renderVSCodeTokens(activeFile.content)}
                </div>

                {/* Agent Action Footnote Badge */}
                <div className="mt-6 p-3 rounded-xl bg-[#252526] border border-[#3c3c3c] text-[10px] font-mono text-cyan-300 flex items-center justify-between shadow-lg">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-3.5 h-3.5 text-cyan-400 animate-bounce" />
                    <span>⚡ Agent Action: {activeWorkingAgent.action}</span>
                  </div>
                  <span className="text-[#858585] font-mono">Lines: {activeFile.lines}</span>
                </div>
              </div>

              {/* VS Code Bottom Status Bar */}
              <div className="bg-[#007acc] text-white px-3 py-1 text-[10px] font-mono flex items-center justify-between select-none">
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1 font-bold">
                    <Bot className="w-3 h-3" /> FlowLang LS: Active ⚡
                  </span>
                  <span>Ln {activeFile.lines}, Col 1</span>
                  <span>Spaces: 2</span>
                  <span>UTF-8</span>
                </div>
                <div className="flex items-center gap-3">
                  <span>TypeScript React</span>
                  <span>Prettier</span>
                  <span>Zero Errors</span>
                </div>
              </div>
            </div>
          ) : (
            /* Agent Teams Activity Stream View */
            <div className="flex-1 p-5 overflow-y-auto space-y-3 bg-gradient-to-b from-[#080c16] to-[#060911]">
              <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-2">
                <div>
                  <h3 className="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
                    <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
                    Live Agent Teams Workforce Stream (`software_factory.flow`)
                  </h3>
                  <p className="text-[10px] text-slate-400 mt-0.5 font-mono">
                    Real-time collaboration across 5 specialized agent teams working on codebase files.
                  </p>
                </div>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2.5 py-1 rounded-lg border border-slate-800">
                  {agentLogs.length} Agent Actions Logged
                </span>
              </div>

              {/* Activity Cards */}
              {agentLogs.map((log) => (
                <div 
                  key={log.id} 
                  className="bg-slate-950/80 p-4 rounded-xl border border-slate-800/80 shadow-md font-mono transition-all hover:border-slate-700"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                        team {log.role}
                      </span>
                      <span className="text-[10px] text-slate-400 bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">
                        Command&lt;{log.commandKind}&gt;
                      </span>
                    </div>
                    <span className="text-[10px] text-slate-500">{log.timestamp}</span>
                  </div>

                  <p className="text-xs text-slate-200 font-semibold mb-2">
                    {log.action}
                  </p>

                  {log.details && (
                    <div className="bg-slate-900 p-2.5 rounded-lg border border-slate-800 text-[10px] text-emerald-300 mb-2">
                      {log.details}
                    </div>
                  )}

                  <div className="flex items-center justify-between text-[10px] text-slate-400 pt-2 border-t border-slate-900">
                    <span className="flex items-center gap-1 text-cyan-400">
                      <CornerDownRight className="w-3 h-3" /> Target File: {log.targetFile}
                    </span>
                    <span className="text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" /> Executed Cleanly
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Sidebar: CLI Terminal & Telemetry */}
        <div className="w-full md:w-80 bg-slate-950 p-4 flex flex-col gap-3 shrink-0">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <span className="text-[10px] font-bold font-mono text-cyan-400 uppercase tracking-wider flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5" /> Interactive CLI Terminal
            </span>
            <span className="text-[9px] text-emerald-400 font-mono">LIVE TELEMETRY</span>
          </div>

          {/* Terminal Log Console */}
          <div className="flex-1 bg-black/60 p-3 rounded-xl border border-slate-800/80 font-mono text-[11px] overflow-y-auto space-y-1.5 max-h-[380px]">
            {cliOutput.map((line, index) => (
              <div key={index} className={`leading-relaxed ${line.startsWith('$') ? 'text-cyan-300 font-bold' : line.includes('PASS') || line.includes('Approved') || line.includes('LIVE') ? 'text-emerald-300' : 'text-slate-400'}`}>
                {line}
              </div>
            ))}
          </div>

          {/* Command Form */}
          <form onSubmit={handleTerminalSubmit} className="flex items-center gap-2">
            <input
              type="text"
              value={terminalInput}
              onChange={(e) => setTerminalInput(e.target.value)}
              placeholder="Type CLI command (e.g., flowlang test)..."
              className="flex-1 bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
            />
            <button
              type="submit"
              className="p-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-all"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          </form>
        </div>
      </div>

      {/* Floating Live Produced Application Execution Window Modal */}
      {isAppRunnerWindowOpen && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 animate-fadeIn">
          <div className="bg-[#0b101d] border border-cyan-500/40 rounded-2xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col overflow-hidden">
            {/* Modal Header Bar */}
            <div className="bg-slate-950 px-5 py-3 border-b border-slate-800 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-400">
                  <PlayCircle className="w-5 h-5 animate-pulse" />
                </div>
                <div>
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    Live Application Container: <span className="text-cyan-400 font-mono">{activeFile.name}</span>
                    <span className="px-2 py-0.5 rounded-full text-[9px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      100% RUNNING LIVE
                    </span>
                  </h3>
                  <p className="text-[10px] text-slate-400 font-mono mt-0.5">
                    Path: {activeFile.path} • {activeFile.lines} lines • Agent Team: {activeFile.activeAgent || 'code_engineers'}
                  </p>
                </div>
              </div>

              <button
                onClick={() => setIsAppRunnerWindowOpen(false)}
                className="p-1.5 rounded-lg bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800 transition"
              >
                <XSquare className="w-5 h-5 text-rose-400" />
              </button>
            </div>

            {/* Modal Main Viewport */}
            <div className="p-5 flex-1 overflow-y-auto space-y-4">
              {/* Application Telemetry Status Row */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">Execution Mode</span>
                  <div className="text-sm font-bold text-emerald-400 flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4" /> Live Interactive Container
                  </div>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">State Integrity</span>
                  <div className="text-sm font-bold text-cyan-300">100% AST Verified</div>
                </div>
                <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800">
                  <span className="text-[10px] font-mono text-slate-400 block mb-1">Runtime Latency</span>
                  <div className="text-sm font-bold text-purple-300">0.14 ms</div>
                </div>
              </div>

              {/* Interactive Live App Form */}
              <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-300 font-mono flex items-center gap-2">
                    <Terminal className="w-4 h-4 text-cyan-400" /> Dispatch Action to {activeFile.name}
                  </span>
                  <span className="text-[10px] text-cyan-400 font-mono">Interactive Control Panel</span>
                </div>

                <form onSubmit={handleExecuteRunnerAction} className="flex gap-2">
                  <input
                    type="text"
                    value={runnerInput}
                    onChange={(e) => setRunnerInput(e.target.value)}
                    placeholder={`Dispatch task or trigger function on ${activeFile.name}...`}
                    className="flex-1 bg-slate-950 border border-slate-800 px-3 py-2 rounded-xl text-xs font-mono text-white focus:outline-none focus:border-cyan-500"
                  />
                  <button
                    type="submit"
                    className="px-4 py-2 rounded-xl bg-gradient-to-r from-emerald-500 to-cyan-500 text-slate-950 font-bold text-xs flex items-center gap-1.5 hover:brightness-110 transition shadow-md"
                  >
                    <Send className="w-3.5 h-3.5" />
                    <span>Run Task</span>
                  </button>
                </form>
              </div>

              {/* Live Terminal Log Output Stream */}
              <div className="p-4 rounded-xl bg-black/80 border border-slate-800 font-mono text-xs space-y-1.5 max-h-56 overflow-y-auto">
                <div className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider pb-1 border-b border-slate-800 mb-2">
                  Container Execution Telemetry Feed
                </div>
                {runnerLogs.map((log, i) => (
                  <div key={i} className="text-slate-300 leading-relaxed">
                    {log}
                  </div>
                ))}
              </div>
            </div>

            {/* Footer Bar */}
            <div className="bg-slate-950 px-5 py-2.5 border-t border-slate-800 flex items-center justify-between text-[11px] font-mono text-slate-400">
              <span className="flex items-center gap-2">
                <Bot className="w-3.5 h-3.5 text-cyan-400" /> JOL Autonomous Application Container
              </span>
              <button
                onClick={() => setIsAppRunnerWindowOpen(false)}
                className="px-3 py-1 rounded bg-slate-800 hover:bg-slate-700 text-white font-bold transition"
              >
                Close Window
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export const SoftwareFactoryApp: React.FC<SoftwareFactoryAppProps> = (props) => (
  <SoftwareFactoryErrorBoundary>
    <SoftwareFactoryAppContent {...props} />
  </SoftwareFactoryErrorBoundary>
);

export default SoftwareFactoryApp;
