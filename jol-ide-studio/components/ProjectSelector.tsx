import React, { useState, useEffect } from 'react';
import { Briefcase, ChevronDown, Check, Sparkles, FolderGit2, Play, RefreshCw, Cpu, Shield, Activity, Layers, Database, Trash2 } from 'lucide-react';
import { ProfessionalDomain } from '../types';

export interface StudioProject {
  id: string;
  name: string;
  domain: ProfessionalDomain;
  flowFile: string;
  icon: string;
  description: string;
  prompt: string;
}

export const ALL_STUDIO_PROJECTS: StudioProject[] = [
  {
    id: 'accountant_erp',
    name: 'Accountant ERP Enterprise System',
    domain: 'economic',
    flowFile: 'accounting_erp.flow',
    icon: '📊',
    description: 'Double-entry general ledger, 5-level COA, invoicing VAT 20%, P&L & Balance Sheet.',
    prompt: 'Build an Accountant ERP Software Project with double-entry general ledger, chart of accounts, invoicing, and financial statements.'
  },
  {
    id: 'software_factory',
    name: 'Autonomous Multi-Agent Software Factory',
    domain: 'digital',
    flowFile: 'software_factory.flow',
    icon: '🚀',
    description: 'Multi-agent AI software synthesizer with Lark AST parser and Git deployment gate.',
    prompt: 'Execute Autonomous Multi-Agent Software Factory project with Lark AST parser and Git quality gate.'
  },
  {
    id: 'security_audit',
    name: 'Zero-Trust SecOps Security Engine',
    domain: 'cyber',
    flowFile: 'security_audit.flow',
    icon: '🛡️',
    description: 'TCP port scanner, HTTP header security auditor, and OCSF v1.4 event logging.',
    prompt: 'Run Zero-Trust SecOps Security Engine project with Nmap port audit and OCSF event logging.'
  },
  {
    id: 'cad_kinematics',
    name: '3D Robotics CAD & Kinematics Engine',
    domain: 'mechanical',
    flowFile: 'bridge_engineering.flow',
    icon: '⚙️',
    description: '3D ASCII STL mesh generator, 3-DOF robot forward kinematics solver & load analysis.',
    prompt: 'Build 3D Robotics CAD & Kinematics Engine with ASCII STL mesh and forward kinematics solver.'
  },
  {
    id: 'clinical_governance',
    name: 'HIPAA Clinical Bio-Governance Bus',
    domain: 'clinical',
    flowFile: 'hospital.flow',
    icon: '🧬',
    description: 'Cryptographic SHA-256 patient PII anonymization and HL7 / FHIR R4 clinical bundle generator.',
    prompt: 'Execute HIPAA Clinical Bio-Governance Bus project with SHA-256 PII redaction and FHIR R4 resource generation.'
  },
  {
    id: 'medical_legal',
    name: 'Medical-Legal Practice Governance System',
    domain: 'clinical',
    flowFile: 'medical_legal.flow',
    icon: '⚖️',
    description: 'Medico-legal compliance audit engine with clinical records verification & malpractice risk score.',
    prompt: 'Build Medical-Legal Practice Governance System with clinical record audit and malpractice risk assessment.'
  },
  {
    id: 'quantum_secops',
    name: 'Quantum Cryptographic SecOps Scanner',
    domain: 'cyber',
    flowFile: 'quantum_secops.flow',
    icon: '🔐',
    description: 'Post-quantum encryption audit pipeline with zero-knowledge proof verification & socket probing.',
    prompt: 'Execute Quantum Cryptographic SecOps Scanner project with post-quantum encryption audit.'
  },
  {
    id: 'robotic_swarm',
    name: 'Swarm Robotics Kinematics Controller',
    domain: 'electro',
    flowFile: 'robotic_swarm.flow',
    icon: '📡',
    description: 'Multi-node serial telemetry collector and MQTT broker topic listener for autonomous robotics swarms.',
    prompt: 'Execute Swarm Robotics Kinematics Controller project with serial discovery and MQTT topic telemetry.'
  },
  {
    id: 'supply_chain',
    name: 'Global Logistics & Supply Chain Ledger',
    domain: 'economic',
    flowFile: 'supply_chain.flow',
    icon: '🌐',
    description: 'Decentralized trade ledger with real-time commodity pricing feed and multi-currency VaR calculation.',
    prompt: 'Build Global Logistics & Supply Chain Ledger with real-time market pricing and 99% portfolio VaR.'
  }
];

interface ProjectSelectorProps {
  currentProjectId?: string;
  onSelectProject: (project: StudioProject) => void;
  className?: string;
}

const getCustomProjects = (): StudioProject[] => {
  try {
    const stored = localStorage.getItem('jol_custom_projects');
    if (!stored) return [];
    const customList = JSON.parse(stored);
    return customList.map((p: any) => ({
      id: p.id,
      name: p.name,
      domain: p.domain || 'digital',
      flowFile: p.flowFile || `${p.id}.flow`,
      icon: '⚡',
      description: p.description || `AI Synthesized Project`,
      prompt: p.name || p.id
    }));
  } catch {
    return [];
  }
};

export const ProjectSelector: React.FC<ProjectSelectorProps> = ({
  currentProjectId = 'accountant_erp',
  onSelectProject,
  className = ''
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedId, setSelectedId] = useState(currentProjectId);
  const [isSwitching, setIsSwitching] = useState(false);
  const [customProjects, setCustomProjects] = useState<StudioProject[]>(getCustomProjects);
  const [deletedIds, setDeletedIds] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('jol_deleted_projects');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    const handleUpdate = () => {
      setCustomProjects(getCustomProjects());
    };
    window.addEventListener('jol_project_changed', handleUpdate);
    window.addEventListener('jol_projects_updated', handleUpdate);
    return () => {
      window.removeEventListener('jol_project_changed', handleUpdate);
      window.removeEventListener('jol_projects_updated', handleUpdate);
    };
  }, []);

  const allMap = new Map<string, StudioProject>();
  ALL_STUDIO_PROJECTS.forEach(p => allMap.set(p.id, p));
  customProjects.forEach(p => allMap.set(p.id, p));
  const combinedProjects = Array.from(allMap.values());
  const availableProjects = combinedProjects.filter(p => !deletedIds.includes(p.id));
  const activeProject = availableProjects.find(p => p.id === selectedId) || availableProjects[0] || ALL_STUDIO_PROJECTS[0];

  const handleSelect = async (project: StudioProject) => {
    setSelectedId(project.id);
    setIsOpen(false);
    setIsSwitching(true);
    try {
      await onSelectProject(project);
    } finally {
      setIsSwitching(false);
    }
  };

  const handleDelete = async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (availableProjects.length <= 1) {
      alert("Cannot delete the last remaining project in workspace.");
      return;
    }
    if (!window.confirm("Do you want to delete this project from the workspace?")) return;

    const newDeleted = [...deletedIds, projectId];
    setDeletedIds(newDeleted);
    try {
      localStorage.setItem('jol_deleted_projects', JSON.stringify(newDeleted));
    } catch (err) {
      console.error("Save deleted projects err:", err);
    }

    if (selectedId === projectId) {
      const remaining = availableProjects.filter(p => p.id !== projectId);
      if (remaining.length > 0) {
        handleSelect(remaining[0]);
      }
    }
  };

  const getDomainBadge = (domain: ProfessionalDomain) => {
    switch (domain) {
      case 'economic': return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
      case 'cyber': return 'bg-rose-500/20 text-rose-300 border-rose-500/30';
      case 'mechanical': return 'bg-amber-500/20 text-amber-300 border-amber-500/30';
      case 'electro': return 'bg-purple-500/20 text-purple-300 border-purple-500/30';
      case 'clinical': return 'bg-pink-500/20 text-pink-300 border-pink-500/30';
      default: return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30';
    }
  };

  return (
    <div className={`relative ${className}`}>
      {/* Dropdown Trigger Bar */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        disabled={isSwitching}
        className="w-full flex items-center justify-between gap-3 px-3.5 py-2 rounded-xl bg-slate-950/90 hover:bg-slate-900 border border-cyan-500/30 hover:border-cyan-400 text-xs text-slate-200 transition-all shadow-lg backdrop-blur-md group"
      >
        <div className="flex items-center gap-2.5 overflow-hidden">
          <span className="text-base">{activeProject.icon}</span>
          <div className="flex flex-col text-left truncate">
            <div className="flex items-center gap-1.5">
              <span className="font-bold text-white group-hover:text-cyan-300 transition-colors truncate">
                {activeProject.name}
              </span>
              <span className={`px-1.5 py-0.2 text-[9px] font-mono rounded border uppercase ${getDomainBadge(activeProject.domain)}`}>
                {activeProject.domain}
              </span>
            </div>
            <span className="text-[10px] text-slate-400 font-mono truncate">
              {activeProject.flowFile}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-1 shrink-0">
          {isSwitching ? (
            <RefreshCw className="w-3.5 h-3.5 text-cyan-400 animate-spin" />
          ) : (
            <ChevronDown className={`w-4 h-4 text-cyan-400 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} />
          )}
        </div>
      </button>

      {/* Floating Menu overlay */}
      {isOpen && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setIsOpen(false)} />
          <div className="absolute top-full left-0 mt-2 w-full min-w-[320px] max-w-md bg-slate-900/95 border border-cyan-500/40 rounded-2xl p-2 shadow-2xl z-40 backdrop-blur-xl animate-fade-in max-h-96 overflow-y-auto space-y-1">
            <div className="px-3 py-2 border-b border-slate-800 flex items-center justify-between">
              <span className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider flex items-center gap-1">
                <FolderGit2 className="w-3.5 h-3.5" /> اختر مشروع FlowLang ({availableProjects.length})
              </span>
              <span className="text-[9px] font-mono text-slate-500">Live Project Switcher</span>
            </div>

            {availableProjects.map((proj) => {
              const isSelected = proj.id === selectedId;
              return (
                <div
                  key={proj.id}
                  onClick={() => handleSelect(proj)}
                  className={`w-full flex items-start gap-2.5 p-2.5 rounded-xl text-left transition-all cursor-pointer group/item ${isSelected ? 'bg-cyan-500/15 border border-cyan-500/40 text-white' : 'hover:bg-slate-800/60 text-slate-300 hover:text-white border border-transparent'}`}
                >
                  <span className="text-lg mt-0.5">{proj.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-0.5">
                      <span className={`text-xs font-bold truncate ${isSelected ? 'text-cyan-300' : 'text-slate-200'}`}>
                        {proj.name}
                      </span>
                      <div className="flex items-center gap-1 shrink-0 ml-1">
                        {isSelected && <Check className="w-4 h-4 text-cyan-400" />}
                        <button
                          onClick={(e) => handleDelete(proj.id, e)}
                          title="حذف المشروع (Delete Project)"
                          className="p-1 rounded hover:bg-rose-500/20 text-slate-500 hover:text-rose-400 transition-colors"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                    <p className="text-[10px] text-slate-400 line-clamp-1 leading-snug">
                      {proj.description}
                    </p>
                    <div className="flex items-center justify-between text-[9px] font-mono text-slate-500 mt-1">
                      <span>{proj.flowFile}</span>
                      <span className={`px-1 rounded border uppercase ${getDomainBadge(proj.domain)}`}>
                        {proj.domain}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
};

export default ProjectSelector;
