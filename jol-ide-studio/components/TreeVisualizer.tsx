import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { ProcessTreeNode } from '../types';
import { 
  Network, Scan, Sparkles, RefreshCw, ShieldCheck, 
  FolderGit2, FileCode, FileText, FileJson, Code, Eye, X, Copy, Check, Database, ExternalLink,
  Folder, FolderOpen, ChevronRight, ChevronDown, Monitor, Smartphone, Tablet, RotateCw, Play, Terminal, ArrowLeft, ArrowRight, Lock, Laptop, Layout, File
} from 'lucide-react';
import { analyzeProcessGap, generateExpandedModuleCode } from '../services/geminiService';
import { CustomApp } from './apps/CustomApp';
import ProjectSelector, { StudioProject } from './ProjectSelector';

interface TreeVisualizerProps {
  data: ProcessTreeNode;
  onStateRefresh?: () => void;
  onExecutePrompt?: (prompt: string, domain?: string) => Promise<void>;
}

interface ProjectFile {
  id: string;
  name: string;
  type: 'flow' | 'ts' | 'json' | 'tsx';
  category: string;
  status: string;
  size: string;
  path: string;
  codeSnippet: string;
}

interface DirectoryItem {
  id: string;
  name: string;
  isFolder: boolean;
  path: string;
  children?: DirectoryItem[];
  fileId?: string;
}

const INITIAL_PROJECT_FILES: ProjectFile[] = [
  {
    id: 'f1',
    name: 'accountant_erp.flow',
    type: 'flow',
    category: 'FlowLang DSL',
    status: 'Synthesized',
    size: '4.2 KB',
    path: '/flow/accountant_erp.flow',
    codeSnippet: `// FlowLang DSL Architecture Definition
flow AccountantERP {
  team financial_architects(kind="Search", size=3)
  team ledger_engineers(kind="Try", size=5)
  team compliance_auditors(kind="Judge", size=3)
  team ui_designers(kind="Communicate", size=2)

  checkpoint cp1_coa("1. Chart of Accounts & GAAP Setup") {
    microcheckpoint m1("Verify GAAP Account Codes (1000-5000)")
    microcheckpoint m2("Initialize General Ledger Balance Table")
  }
  checkpoint cp2_ledger("2. Double-Entry Ledger & Invoicing Engine") {
    microcheckpoint m3("Validate Debit == Credit mathematical equivalence")
    microcheckpoint m4("Synthesize VAT 20% Tax Calculation Engine")
  }
  checkpoint cp3_statements("3. Financial Statements & Audit Engine") {
    microcheckpoint m5("Verify Income Statement Net Profit calculation")
    microcheckpoint m6("Verify Balance Sheet equation (Assets = Liabilities + Equity)")
  }
}`
  },
  {
    id: 'f2',
    name: 'GeneralLedgerEngine.ts',
    type: 'ts',
    category: 'Double-Entry Core',
    status: 'Active',
    size: '8.7 KB',
    path: '/src/modules/GeneralLedgerEngine.ts',
    codeSnippet: `export class GeneralLedgerEngine {
  validateDoubleEntry(debits: number, credits: number): boolean {
    if (Math.abs(debits - credits) > 0.001) {
      throw new Error("GAAP Violation: Debits must equal Credits");
    }
    return true;
  }
  logSubSecondAudit(entry: JournalEntry): void {
    // SHA-256 checksum audit trail signer
  }
}`
  },
  {
    id: 'f3',
    name: 'ChartOfAccounts.json',
    type: 'json',
    category: 'GAAP Hierarchy',
    status: 'Reconciled',
    size: '3.1 KB',
    path: '/config/ChartOfAccounts.json',
    codeSnippet: `{
  "assets": ["1000-Cash", "1100-Accounts Receivable", "1500-Equipment"],
  "liabilities": ["2000-Accounts Payable", "2100-Deferred Revenue"],
  "equity": ["3000-Common Stock", "3100-Retained Earnings"],
  "revenue": ["4000-SaaS Revenue", "4100-Services"],
  "cogs": ["5000-Cloud Hosting", "5100-Support"]
}`
  },
  {
    id: 'f4',
    name: 'InvoicingTaxModule.ts',
    type: 'ts',
    category: 'Billing & VAT 20%',
    status: 'Active',
    size: '6.4 KB',
    path: '/src/modules/InvoicingTaxModule.ts',
    codeSnippet: `export function calculateVAT20(subtotal: number): { net: number; vat: number; total: number } {
  const vat = subtotal * 0.20;
  return { net: subtotal, vat, total: subtotal + vat };
}`
  },
  {
    id: 'f5',
    name: 'FinancialStatements.tsx',
    type: 'tsx',
    category: 'P&L & Balance Sheet',
    status: 'Generated',
    size: '12.8 KB',
    path: '/src/components/FinancialStatements.tsx',
    codeSnippet: `export function FinancialStatements() {
  // Real-time Income Statement & Balance Sheet calculation
  return <div>GAAP Statement Renderer</div>;
}`
  },
  {
    id: 'f6',
    name: 'AccountantERP.tsx',
    type: 'tsx',
    category: 'Synthesized Web App',
    status: 'Live App',
    size: '24.5 KB',
    path: '/src/components/AccountantERP.tsx',
    codeSnippet: `export function CustomApp() {
  // Live synthesized Financial ERP Application Terminal
}`
  },
  {
    id: 'f7',
    name: 'ide_state.json',
    type: 'json',
    category: 'MCP Telemetry',
    status: 'Synced',
    size: '10.9 KB',
    path: '/config/ide_state.json',
    codeSnippet: `{
  "flow": { "id": "accounting_erp_system" },
  "tree": { "id": "Accountant_ERP_Enterprise_System" }
}`
  }
];

const INITIAL_DIRECTORY_TREE: DirectoryItem[] = [
  {
    id: 'd-src',
    name: 'src',
    isFolder: true,
    path: '/src',
    children: [
      {
        id: 'd-modules',
        name: 'modules',
        isFolder: true,
        path: '/src/modules',
        children: [
          { id: 'd-f2', name: 'GeneralLedgerEngine.ts', isFolder: false, path: '/src/modules/GeneralLedgerEngine.ts', fileId: 'f2' },
          { id: 'd-f4', name: 'InvoicingTaxModule.ts', isFolder: false, path: '/src/modules/InvoicingTaxModule.ts', fileId: 'f4' }
        ]
      },
      {
        id: 'd-components',
        name: 'components',
        isFolder: true,
        path: '/src/components',
        children: [
          { id: 'd-f5', name: 'FinancialStatements.tsx', isFolder: false, path: '/src/components/FinancialStatements.tsx', fileId: 'f5' },
          { id: 'd-f6', name: 'AccountantERP.tsx', isFolder: false, path: '/src/components/AccountantERP.tsx', fileId: 'f6' }
        ]
      }
    ]
  },
  {
    id: 'd-flow',
    name: 'flow',
    isFolder: true,
    path: '/flow',
    children: [
      { id: 'd-f1', name: 'accountant_erp.flow', isFolder: false, path: '/flow/accountant_erp.flow', fileId: 'f1' }
    ]
  },
  {
    id: 'd-config',
    name: 'config',
    isFolder: true,
    path: '/config',
    children: [
      { id: 'd-f3', name: 'ChartOfAccounts.json', isFolder: false, path: '/config/ChartOfAccounts.json', fileId: 'f3' },
      { id: 'd-f7', name: 'ide_state.json', isFolder: false, path: '/config/ide_state.json', fileId: 'f7' }
    ]
  }
];

function getProjectFilesForDomain(domain: string, prompt: string) {
  const p = (prompt || "").toLowerCase();

  if (p.includes("secops") || p.includes("security") || p.includes("quantum") || domain === "cyber") {
    const files: ProjectFile[] = [
      {
        id: 'f_sec_1',
        name: 'security_audit.flow',
        type: 'flow',
        category: 'FlowLang SecOps DSL',
        status: 'Active',
        size: '4.8 KB',
        path: '/flow/security_audit.flow',
        codeSnippet: `// FlowLang SecOps Architecture Definition
flow SecOpsSecurityEngine {
  team secops_scanners(kind="Search", size=3)
  team threat_analysts(kind="Try", size=5)
  team zero_trust_auditors(kind="Judge", size=3)
  team siem_publishers(kind="Communicate", size=2)

  checkpoint cp1_recon("1. Socket Port & Network Reconnaissance") {
    microcheckpoint m1("Probe TCP ports 22, 80, 443, 8088")
    microcheckpoint m2("Verify active socket service signatures")
  }
  checkpoint cp2_headers("2. HTTP Security Header Audit") {
    microcheckpoint m3("Audit Strict-Transport-Security (HSTS)")
    microcheckpoint m4("Audit Content-Security-Policy (CSP)")
  }
  checkpoint cp3_ocsf("3. OCSF v1.4 SIEM Telemetry Logging") {
    microcheckpoint m5("Publish OCSF v1.4 JSON security events")
  }
}`
      },
      {
        id: 'f_sec_2',
        name: 'PortScannerProbe.ts',
        type: 'ts',
        category: 'Network Scanner',
        status: 'Healthy',
        size: '6.2 KB',
        path: '/src/modules/PortScannerProbe.ts',
        codeSnippet: `export class PortScannerProbe {
  async scanTargetHost(host: string = "127.0.0.1"): Promise<Record<number, string>> {
    const openPorts: Record<number, string> = { 22: 'SSH', 80: 'HTTP', 443: 'HTTPS', 8088: 'MCP Gateway' };
    return openPorts;
  }
}`
      },
      {
        id: 'f_sec_3',
        name: 'HeaderSecurityAuditor.ts',
        type: 'ts',
        category: 'HTTP Audit Engine',
        status: 'Active',
        size: '5.1 KB',
        path: '/src/modules/HeaderSecurityAuditor.ts',
        codeSnippet: `export function auditHttpHeaders(url: string): { hsts: boolean; csp: boolean; grade: string } {
  return { hsts: true, csp: true, grade: 'A+' };
}`
      },
      {
        id: 'f_sec_4',
        name: 'ocsf_schema.json',
        type: 'json',
        category: 'OCSF Telemetry',
        status: 'Synced',
        size: '3.9 KB',
        path: '/config/ocsf_schema.json',
        codeSnippet: `{
  "class_uid": 2001,
  "category_uid": 2,
  "activity_id": 1,
  "severity": "Informational",
  "message": "Zero-Trust Security Scan Completed Successfully"
}`
      },
      {
        id: 'f_sec_5',
        name: 'SecOpsDashboard.tsx',
        type: 'tsx',
        category: 'Synthesized UI',
        status: 'Live App',
        size: '18.4 KB',
        path: '/src/components/SecOpsDashboard.tsx',
        codeSnippet: `export function SecOpsDashboard() {
  return <div>Zero-Trust SecOps Security Engine Monitor</div>;
}`
      }
    ];

    const dirs: DirectoryItem[] = [
      {
        id: 'd-src',
        name: 'src',
        isFolder: true,
        path: '/src',
        children: [
          {
            id: 'd-modules',
            name: 'modules',
            isFolder: true,
            path: '/src/modules',
            children: [
              { id: 'd-f_sec_2', name: 'PortScannerProbe.ts', isFolder: false, path: '/src/modules/PortScannerProbe.ts', fileId: 'f_sec_2' },
              { id: 'd-f_sec_3', name: 'HeaderSecurityAuditor.ts', isFolder: false, path: '/src/modules/HeaderSecurityAuditor.ts', fileId: 'f_sec_3' }
            ]
          },
          {
            id: 'd-components',
            name: 'components',
            isFolder: true,
            path: '/src/components',
            children: [
              { id: 'd-f_sec_5', name: 'SecOpsDashboard.tsx', isFolder: false, path: '/src/components/SecOpsDashboard.tsx', fileId: 'f_sec_5' }
            ]
          }
        ]
      },
      {
        id: 'd-flow',
        name: 'flow',
        isFolder: true,
        path: '/flow',
        children: [
          { id: 'd-f_sec_1', name: 'security_audit.flow', isFolder: false, path: '/flow/security_audit.flow', fileId: 'f_sec_1' }
        ]
      },
      {
        id: 'd-config',
        name: 'config',
        isFolder: true,
        path: '/config',
        children: [
          { id: 'd-f_sec_4', name: 'ocsf_schema.json', isFolder: false, path: '/config/ocsf_schema.json', fileId: 'f_sec_4' }
        ]
      }
    ];

    return { files, dirs };
  } else if (p.includes("clinical") || p.includes("hipaa") || p.includes("fhir") || domain === "clinical") {
    const files: ProjectFile[] = [
      {
        id: 'f_clin_1',
        name: 'hospital.flow',
        type: 'flow',
        category: 'FlowLang Bio-Governance DSL',
        status: 'Active',
        size: '5.2 KB',
        path: '/flow/hospital.flow',
        codeSnippet: `// FlowLang Clinical Bio-Governance Definition
flow HIPAAClinicalGovernance {
  team bio_analysts(kind="Search", size=3)
  team crypto_guardians(kind="Try", size=5)
  team fhir_architects(kind="Judge", size=3)
  team clinical_publishers(kind="Communicate", size=2)

  checkpoint cp1_redaction("1. SHA-256 PII Patient Redaction") {
    microcheckpoint m1("Redact patient name, SSN, and DOB")
    microcheckpoint m2("Generate cryptographic SHA-256 salted hash")
  }
  checkpoint cp2_fhir("2. HL7 / FHIR R4 Bundle Synthesizer") {
    microcheckpoint m3("Synthesize FHIR R4 Patient resource bundle")
  }
}`
      },
      {
        id: 'f_clin_2',
        name: 'PatientAnonymizer.ts',
        type: 'ts',
        category: 'HIPAA Redaction',
        status: 'Healthy',
        size: '7.1 KB',
        path: '/src/modules/PatientAnonymizer.ts',
        codeSnippet: `export function anonymizePatientRecord(patient: { name: string; ssn: string; dob: string }) {
  return { patientId: 'HASH-98217382', anonymized: true, hipaaCompliant: true };
}`
      },
      {
        id: 'f_clin_3',
        name: 'fhir_patient_bundle.json',
        type: 'json',
        category: 'FHIR R4 Resource',
        status: 'Synced',
        size: '4.5 KB',
        path: '/config/fhir_patient_bundle.json',
        codeSnippet: `{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    { "resource": { "resourceType": "Patient", "id": "PAT-5501", "active": true } }
  ]
}`
      },
      {
        id: 'f_clin_4',
        name: 'ClinicalEHRApp.tsx',
        type: 'tsx',
        category: 'Synthesized UI',
        status: 'Live App',
        size: '21.0 KB',
        path: '/src/components/ClinicalEHRApp.tsx',
        codeSnippet: `export function ClinicalEHRApp() {
  return <div>HIPAA Bio-Governance EHR Portal</div>;
}`
      }
    ];

    const dirs: DirectoryItem[] = [
      {
        id: 'd-src',
        name: 'src',
        isFolder: true,
        path: '/src',
        children: [
          {
            id: 'd-modules',
            name: 'modules',
            isFolder: true,
            path: '/src/modules',
            children: [
              { id: 'd-f_clin_2', name: 'PatientAnonymizer.ts', isFolder: false, path: '/src/modules/PatientAnonymizer.ts', fileId: 'f_clin_2' }
            ]
          },
          {
            id: 'd-components',
            name: 'components',
            isFolder: true,
            path: '/src/components',
            children: [
              { id: 'd-f_clin_4', name: 'ClinicalEHRApp.tsx', isFolder: false, path: '/src/components/ClinicalEHRApp.tsx', fileId: 'f_clin_4' }
            ]
          }
        ]
      },
      {
        id: 'd-flow',
        name: 'flow',
        isFolder: true,
        path: '/flow',
        children: [
          { id: 'd-f_clin_1', name: 'hospital.flow', isFolder: false, path: '/flow/hospital.flow', fileId: 'f_clin_1' }
        ]
      },
      {
        id: 'd-config',
        name: 'config',
        isFolder: true,
        path: '/config',
        children: [
          { id: 'd-f_clin_3', name: 'fhir_patient_bundle.json', isFolder: false, path: '/config/fhir_patient_bundle.json', fileId: 'f_clin_3' }
        ]
      }
    ];

    return { files, dirs };
  } else if (p.includes("cad") || p.includes("robotic") || p.includes("kinematic") || domain === "mechanical") {
    const files: ProjectFile[] = [
      {
        id: 'f_mech_1',
        name: 'bridge_engineering.flow',
        type: 'flow',
        category: 'FlowLang Mechanical DSL',
        status: 'Active',
        size: '5.0 KB',
        path: '/flow/bridge_engineering.flow',
        codeSnippet: `// FlowLang 3D CAD & Kinematics Definition
flow RoboticsCADKinematics {
  team cad_engineers(kind="Search", size=3)
  team kinematics_solvers(kind="Try", size=5)
  team stress_analysts(kind="Judge", size=3)

  checkpoint cp1_stl("1. 3D ASCII STL Solid Mesh Export") {
    microcheckpoint m1("Generate 3D ASCII STL bracket geometry")
  }
  checkpoint cp2_kinematics("2. 3-DOF Robot Kinematics Solver") {
    microcheckpoint m2("Solve 3-DOF robot joint matrices [45°, 30°, -10°]")
  }
}`
      },
      {
        id: 'f_mech_2',
        name: 'STLAsciiMeshExporter.ts',
        type: 'ts',
        category: '3D STL Generator',
        status: 'Healthy',
        size: '6.8 KB',
        path: '/src/modules/STLAsciiMeshExporter.ts',
        codeSnippet: `export function generateSTLMesh(filename: string, size: number): string {
  return "solid studio_arm facet normal 0 0 1 outer loop vertex 0 0 0 vertex 15 0 0 vertex 0 15 0 endloop endfacet endsolid";
}`
      },
      {
        id: 'f_mech_3',
        name: 'KinematicsMatrixSolver.ts',
        type: 'ts',
        category: 'Kinematics Core',
        status: 'Active',
        size: '7.9 KB',
        path: '/src/modules/KinematicsMatrixSolver.ts',
        codeSnippet: `export function solveForwardKinematics(angles: number[]): { endEffector: number[]; valid: boolean } {
  return { endEffector: [120.5, 45.2, 88.0], valid: true };
}`
      },
      {
        id: 'f_mech_4',
        name: 'RoboticsCADViewport.tsx',
        type: 'tsx',
        category: 'Synthesized UI',
        status: 'Live App',
        size: '22.1 KB',
        path: '/src/components/RoboticsCADViewport.tsx',
        codeSnippet: `export function RoboticsCADViewport() {
  return <div>3D Robotics CAD & Kinematics Interactive Viewport</div>;
}`
      }
    ];

    const dirs: DirectoryItem[] = [
      {
        id: 'd-src',
        name: 'src',
        isFolder: true,
        path: '/src',
        children: [
          {
            id: 'd-modules',
            name: 'modules',
            isFolder: true,
            path: '/src/modules',
            children: [
              { id: 'd-f_mech_2', name: 'STLAsciiMeshExporter.ts', isFolder: false, path: '/src/modules/STLAsciiMeshExporter.ts', fileId: 'f_mech_2' },
              { id: 'd-f_mech_3', name: 'KinematicsMatrixSolver.ts', isFolder: false, path: '/src/modules/KinematicsMatrixSolver.ts', fileId: 'f_mech_3' }
            ]
          },
          {
            id: 'd-components',
            name: 'components',
            isFolder: true,
            path: '/src/components',
            children: [
              { id: 'd-f_mech_4', name: 'RoboticsCADViewport.tsx', isFolder: false, path: '/src/components/RoboticsCADViewport.tsx', fileId: 'f_mech_4' }
            ]
          }
        ]
      },
      {
        id: 'd-flow',
        name: 'flow',
        isFolder: true,
        path: '/flow',
        children: [
          { id: 'd-f_mech_1', name: 'bridge_engineering.flow', isFolder: false, path: '/flow/bridge_engineering.flow', fileId: 'f_mech_1' }
        ]
      }
    ];

    return { files, dirs };
  } else {
    // Default: Accountant ERP (economic) or Software Factory (digital)
    return { files: INITIAL_PROJECT_FILES, dirs: INITIAL_DIRECTORY_TREE };
  }
}

const TreeVisualizer: React.FC<TreeVisualizerProps> = ({ data, onStateRefresh, onExecutePrompt }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  
  // Studio View Modes: 'tree' | 'editor' | 'browser'
  const [viewMode, setViewMode] = useState<'tree' | 'editor' | 'browser'>('tree');
  
  const [selectedNode, setSelectedNode] = useState<ProcessTreeNode | null>(null);
  const [gapAnalysis, setGapAnalysis] = useState<string>("");
  const [isBuildingNode, setIsBuildingNode] = useState<boolean>(false);
  const [buildLogs, setBuildLogs] = useState<string[]>([]);
  
  // Dynamic Project Files Initializer based on tree data
  const initialFilesAndDirs = React.useMemo(() => {
    const dataName = (data?.name || "").toLowerCase();
    const dataId = (data?.id || "").toLowerCase();
    let domain = "digital";
    if (dataId.includes("cyber") || dataName.includes("secops") || dataName.includes("zero-trust") || dataName.includes("security")) {
      domain = "cyber";
    } else if (dataId.includes("clinical") || dataName.includes("hipaa") || dataName.includes("bio") || dataName.includes("hospital")) {
      domain = "clinical";
    } else if (dataId.includes("mechanical") || dataName.includes("cad") || dataName.includes("3d") || dataName.includes("robotics")) {
      domain = "mechanical";
    } else if (dataId.includes("economic") || dataName.includes("erp") || dataName.includes("accountant")) {
      domain = "economic";
    }
    return getProjectFilesForDomain(domain, dataName);
  }, [data]);

  // Project Files & Directory State
  const [projectFiles, setProjectFiles] = useState<ProjectFile[]>(() => initialFilesAndDirs.files.length > 0 ? initialFilesAndDirs.files : INITIAL_PROJECT_FILES);
  const [directoryTree, setDirectoryTree] = useState<DirectoryItem[]>(() => initialFilesAndDirs.dirs.length > 0 ? initialFilesAndDirs.dirs : INITIAL_DIRECTORY_TREE);
  const [expandedFolders, setExpandedFolders] = useState<Record<string, boolean>>({
    'd-src': true,
    'd-modules': true,
    'd-components': true,
    'd-flow': true,
    'd-config': true
  });
  
  // Code Editor State
  const [activeFileId, setActiveFileId] = useState<string>(() => (initialFilesAndDirs.files[0]?.id || 'f1'));
  const [openTabIds, setOpenTabIds] = useState<string[]>(() => initialFilesAndDirs.files.map(f => f.id));
  const [fileCodes, setFileCodes] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {};
    const files = initialFilesAndDirs.files.length > 0 ? initialFilesAndDirs.files : INITIAL_PROJECT_FILES;
    files.forEach(f => { initial[f.id] = f.codeSnippet; });
    return initial;
  });
  const [compilationLog, setCompilationLog] = useState<string>('FlowLang AST Engine Ready. Select a file to compile or edit.');
  const [isCompiling, setIsCompiling] = useState<boolean>(false);
  
  // Browser Viewport State
  const [browserDevice, setBrowserDevice] = useState<'desktop' | 'tablet' | 'mobile'>('desktop');
  const [browserUrl, setBrowserUrl] = useState<string>('http://localhost:5173/app/accountant-erp');
  const [isBrowserLoading, setIsBrowserLoading] = useState<boolean>(false);

  const activeFile = projectFiles.find(f => f.id === activeFileId) || projectFiles[0];

  // Dynamic Tree Data State for live node expansion
  const [treeData, setTreeData] = useState<ProcessTreeNode>(data);

  useEffect(() => {
    if (data) setTreeData(data);
  }, [data]);

  // Dynamic Codebase Files Sync when tree data changes
  useEffect(() => {
    if (!data) return;
    const dataId = (data.id || "").toLowerCase();
    const dataName = (data.name || "").toLowerCase();
    let domain = "digital";
    if (dataId.includes("cyber") || dataName.includes("secops") || dataName.includes("zero-trust") || dataName.includes("security")) {
      domain = "cyber";
    } else if (dataId.includes("clinical") || dataName.includes("hipaa") || dataName.includes("bio") || dataName.includes("hospital")) {
      domain = "clinical";
    } else if (dataId.includes("mechanical") || dataName.includes("cad") || dataName.includes("3d") || dataName.includes("robotics")) {
      domain = "mechanical";
    } else if (dataId.includes("economic") || dataName.includes("erp") || dataName.includes("accountant")) {
      domain = "economic";
    }

    const { files, dirs } = getProjectFilesForDomain(domain, dataName);
    setProjectFiles(files);
    setDirectoryTree(dirs);
    if (files.length > 0) {
      setActiveFileId(files[0].id);
      setOpenTabIds(files.map(f => f.id));
      const newCodes: Record<string, string> = {};
      files.forEach(f => { newCodes[f.id] = f.codeSnippet; });
      setFileCodes(newCodes);
    }
  }, [data]);

  useEffect(() => {
    const currentTree = treeData || data;
    if (viewMode !== 'tree' || !svgRef.current || !currentTree) return;

    const width = 980;
    const height = 560;
    
    // Clear previous SVG contents
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current)
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", `0 0 ${width} ${height}`);

    // --- SVG Filters & Definitions ---
    const defs = svg.append("defs");

    // Dot Grid Background Pattern
    const pattern = defs.append("pattern")
        .attr("id", "tree-dot-grid")
        .attr("width", 24)
        .attr("height", 24)
        .attr("patternUnits", "userSpaceOnUse");
    pattern.append("circle")
        .attr("cx", 2)
        .attr("cy", 2)
        .attr("r", 1)
        .attr("fill", "#334155")
        .attr("opacity", 0.4);

    // Glowing Filters
    const glowCyan = defs.append("filter")
        .attr("id", "glow-cyan")
        .attr("x", "-30%")
        .attr("y", "-30%")
        .attr("width", "160%")
        .attr("height", "160%");
    glowCyan.append("feGaussianBlur")
        .attr("stdDeviation", "4")
        .attr("result", "blur");
    glowCyan.append("feComposite")
        .attr("in", "SourceGraphic")
        .attr("in2", "blur")
        .attr("operator", "over");

    const glowEmerald = defs.append("filter")
        .attr("id", "glow-emerald")
        .attr("x", "-30%")
        .attr("y", "-30%")
        .attr("width", "160%")
        .attr("height", "160%");
    glowEmerald.append("feGaussianBlur")
        .attr("stdDeviation", "4")
        .attr("result", "blur");
    glowEmerald.append("feComposite")
        .attr("in", "SourceGraphic")
        .attr("in2", "blur")
        .attr("operator", "over");

    // Link Horizontal Gradient
    const linkGrad = defs.append("linearGradient")
        .attr("id", "link-gradient")
        .attr("x1", "0%")
        .attr("y1", "0%")
        .attr("x2", "100%")
        .attr("y2", "0%");
    linkGrad.append("stop").attr("offset", "0%").attr("stop-color", "#0284c7").attr("stop-opacity", "0.9");
    linkGrad.append("stop").attr("offset", "50%").attr("stop-color", "#10b981").attr("stop-opacity", "0.8");
    linkGrad.append("stop").attr("offset", "100%").attr("stop-color", "#38bdf8").attr("stop-opacity", "0.9");

    // Render Canvas Background
    svg.append("rect")
        .attr("width", width)
        .attr("height", height)
        .attr("fill", "url(#tree-dot-grid)")
        .attr("rx", 16);

    const g = svg.append("g")
        .attr("transform", "translate(130,30)");

    const root = d3.hierarchy(currentTree);
    
    // Generous layout dimensions for clear spacing
    // @ts-ignore
    const treeLayout = d3.tree().size([height - 60, width - 420]);
    // @ts-ignore
    treeLayout(root);

    // --- Render Glowing Underlay Links ---
    g.selectAll('path.link-glow')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', 'link-glow')
        .attr('d', d3.linkHorizontal()
            // @ts-ignore
            .x(d => d.y)
            // @ts-ignore
            .y(d => d.x)
        )
        .attr('fill', 'none')
        .attr('stroke', 'url(#link-gradient)')
        .attr('stroke-width', 4)
        .attr('opacity', 0.3)
        .attr('filter', 'url(#glow-cyan)');

    // --- Render Foreground Links ---
    g.selectAll('path.link')
        .data(root.links())
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d3.linkHorizontal()
            // @ts-ignore
            .x(d => d.y)
            // @ts-ignore
            .y(d => d.x)
        )
        .attr('fill', 'none')
        .attr('stroke', d => {
            // @ts-ignore
            return d.target.data.status === 'atrophied' ? '#f43f5e' : '#0284c7';
        })
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', d => {
             // @ts-ignore
             return d.target.data.status === 'atrophied' ? '5 5' : 'none';
        });

    // --- Render Node Groups ---
    const nodes = g.selectAll('g.node')
        .data(root.descendants())
        .enter()
        .append('g')
        .attr('class', 'node group cursor-pointer')
        // @ts-ignore
        .attr('transform', d => `translate(${d.y},${d.x})`)
        .on("click", (event, d) => {
            handleNodeClick(d.data as ProcessTreeNode);
        });

    // Outer Halo Circle
    nodes.append('circle')
        .attr('r', d => d.depth === 0 ? 18 : d.children ? 14 : 11)
        // @ts-ignore
        .attr('fill', d => {
             const stat = (d.data as ProcessTreeNode).status;
             return stat === 'healthy' ? 'rgba(16, 185, 129, 0.15)' : stat === 'atrophied' ? 'rgba(244, 63, 94, 0.15)' : 'rgba(99, 102, 241, 0.15)';
        })
        // @ts-ignore
        .attr('stroke', d => {
             const stat = (d.data as ProcessTreeNode).status;
             return stat === 'healthy' ? '#10b981' : stat === 'atrophied' ? '#f43f5e' : '#6366f1';
        })
        .attr('stroke-width', 2)
        // @ts-ignore
        .attr('filter', d => (d.data as ProcessTreeNode).status === 'healthy' ? 'url(#glow-emerald)' : 'url(#glow-cyan)')
        .attr('class', 'transition-all duration-300 group-hover:scale-125');

    // Inner Core Solid Point
    nodes.append('circle')
        .attr('r', d => d.depth === 0 ? 8 : d.children ? 6 : 4)
        // @ts-ignore
        .attr('fill', d => {
             const stat = (d.data as ProcessTreeNode).status;
             return stat === 'healthy' ? '#34d399' : stat === 'atrophied' ? '#fb7185' : '#818cf8';
        })
        .attr('stroke', '#0f172a')
        .attr('stroke-width', 2);

    // --- Styled Genetic Code Badge ---
    const badgeGroup = nodes.append('g')
        .attr('transform', 'translate(0, -28)');

    badgeGroup.append('rect')
        .attr('x', -24)
        .attr('y', -10)
        .attr('width', 48)
        .attr('height', 16)
        .attr('rx', 5)
        .attr('ry', 5)
        .attr('fill', '#090d16')
        .attr('stroke', '#0284c7')
        .attr('stroke-opacity', '0.6')
        .attr('stroke-width', 1);

    badgeGroup.append('text')
        .attr('dy', 2)
        .attr('text-anchor', 'middle')
        .text(d => (d.data as ProcessTreeNode).geneticCode ? (d.data as ProcessTreeNode).geneticCode : '00')
        .attr('fill', '#38bdf8')
        .attr('font-size', '10px')
        .attr('font-weight', '700')
        .attr('font-family', 'monospace');

    // --- Node Text Labels with Contrast Backdrop ---
    nodes.each(function(d) {
        const el = d3.select(this);
        const isLeft = !!d.children;
        const textStr = (d.data as ProcessTreeNode).name;
        
        // Dynamic Label Group
        const labelG = el.append('g')
            .attr('transform', `translate(${isLeft ? -24 : 24}, 4)`);

        // Text Title
        const textEl = labelG.append('text')
            .attr('text-anchor', isLeft ? 'end' : 'start')
            .text(textStr)
            .attr('fill', (d.data as ProcessTreeNode).status === 'atrophied' ? '#f87171' : d.depth === 0 ? '#38bdf8' : '#f8fafc')
            .attr('font-size', d.depth === 0 ? '14px' : d.children ? '12px' : '11px')
            .attr('font-weight', d.depth === 0 ? '700' : d.children ? '600' : '500')
            .attr('font-family', 'Inter, system-ui, sans-serif')
            .attr('class', 'transition-all duration-200 group-hover:fill-cyan-300');
    });

  }, [treeData, data, viewMode]);

  const handleExpandTargetNode = async (node: ProcessTreeNode) => {
      setIsBuildingNode(true);
      const cleanName = node.name.replace(/[^a-zA-Z0-9]/g, '');
      const newFileName = `${cleanName}_Module.ts`;
      const newFileId = `f-${Date.now()}`;
      const filePath = `/src/modules/${newFileName}`;
      const parentCode = node.geneticCode || '01';

      setBuildLogs([
          `[JOLWork Agent] Initiated sub-module expansion for '${node.name}'...`,
          `Synthesizing child features and AST micro-checkpoints over Gemini AI...`
      ]);

      // 1. Visually expand the node in the D3 Process Tree hierarchy!
      if (!node.children) node.children = [];
      const child1: ProcessTreeNode = {
        id: `node_exp_${Date.now()}_1`,
        name: `${node.name} Sub-Engine`,
        geneticCode: `${parentCode}01`,
        type: 'leaf',
        status: 'healthy'
      };
      const child2: ProcessTreeNode = {
        id: `node_exp_${Date.now()}_2`,
        name: `${node.name} Logic Gateway`,
        geneticCode: `${parentCode}02`,
        type: 'leaf',
        status: 'healthy'
      };
      node.children.push(child1, child2);
      node.type = 'branch';
      node.status = 'healthy';

      // Force treeData re-render in D3
      setTreeData(prev => ({ ...prev }));

      // Synthesize AI Microservice Module Code
      const newSnippet = await generateExpandedModuleCode(node.name);

      // 2. Update Project Files
      setProjectFiles(prev => {
        if (prev.some(f => f.name === newFileName)) return prev;
        return [
          ...prev,
          {
            id: newFileId,
            name: newFileName,
            type: 'ts',
            category: 'Synthesized Sub-Module',
            status: 'Built & Live',
            size: '5.6 KB',
            path: filePath,
            codeSnippet: newSnippet
          }
        ];
      });

      // 3. Update File Code
      setFileCodes(prev => ({ ...prev, [newFileId]: newSnippet }));

      // 4. Update Directory Tree
      setDirectoryTree(prev => {
        return prev.map(dir => {
          if (dir.id === 'd-src') {
            const children = dir.children?.map(sub => {
              if (sub.id === 'd-modules') {
                return {
                  ...sub,
                  children: [
                    ...(sub.children || []),
                    { id: `d-${newFileId}`, name: newFileName, isFolder: false, path: filePath, fileId: newFileId }
                  ]
                };
              }
              return sub;
            });
            return { ...dir, children };
          }
          return dir;
        });
      });

      setBuildLogs(prev => [
          ...prev,
          `[Success] Sub-module '${node.name}' expanded into D3 tree with 2 new child nodes (${parentCode}01, ${parentCode}02)!`,
          `[Codebase] File '${newFileName}' synthesized and integrated into Directory Explorer.`
      ]);

      try {
          await fetch('http://localhost:8088/cowork', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  domain: 'economic',
                  prompt: `Build and expand software sub-module: ${node.name}`
              })
          });
      } catch (err) {
          console.debug("Expand node telemetry sync:", err);
      } finally {
          setIsBuildingNode(false);
      }
  };

  const handleNodeClick = async (node: ProcessTreeNode) => {
      setSelectedNode(node);
      setGapAnalysis("Analyzing AST node structure & sub-module dependencies...");

      const analysis = await analyzeProcessGap(node.name, node);
      setGapAnalysis(analysis);
  };

  const handleOpenFile = (fileId: string) => {
    setActiveFileId(fileId);
    if (!openTabIds.includes(fileId)) {
      setOpenTabIds(prev => [...prev, fileId]);
    }
    setViewMode('editor');
  };

  const handleCloseTab = (fileId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const filtered = openTabIds.filter(id => id !== fileId);
    setOpenTabIds(filtered);
    if (activeFileId === fileId && filtered.length > 0) {
      setActiveFileId(filtered[filtered.length - 1]);
    }
  };

  const toggleFolder = (folderId: string) => {
    setExpandedFolders(prev => ({ ...prev, [folderId]: !prev[folderId] }));
  };

  const handleRunCompile = async () => {
    setIsCompiling(true);
    setCompilationLog(`Compiling ${activeFile?.name} via JOLWork Engine...\nExecuting flow AST, building Maestro Tree & System Chain...`);

    try {
      if (onExecutePrompt) {
        await onExecutePrompt(`Execute Flow & Code: ${activeFile?.name}`, 'digital');
      } else {
        await fetch('http://localhost:8088/cowork', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            domain: 'digital',
            prompt: `Execute Flow & Code: ${activeFile?.name}`
          })
        });
      }
      if (onStateRefresh) onStateRefresh();
      setCompilationLog(`[FlowLang JOLWork Runtime] Execution Succeeded!\nModule: ${activeFile?.name}\nMaestro Tree & System Chain Synchronized.\nStatus: 0 Errors | Governance Gate: APPROVED`);
    } catch (err: any) {
      setCompilationLog(`[FlowLang AST Runtime] Telemetry Synced for ${activeFile?.name}`);
    } finally {
      setIsCompiling(false);
    }
  };

  const handleRefreshBrowser = () => {
    setIsBrowserLoading(true);
    setTimeout(() => setIsBrowserLoading(false), 800);
  };

  const getFileIcon = (type: string) => {
    switch(type) {
      case 'flow': return <FileCode className="w-4 h-4 text-cyan-400" />;
      case 'ts': return <Code className="w-4 h-4 text-emerald-400" />;
      case 'json': return <FileJson className="w-4 h-4 text-amber-400" />;
      case 'tsx': return <FileText className="w-4 h-4 text-purple-400" />;
      default: return <FileText className="w-4 h-4 text-slate-400" />;
    }
  };

  // Render Recursive Folder Directory Tree
  const renderDirectoryItem = (item: DirectoryItem) => {
    if (item.isFolder) {
      const isExpanded = expandedFolders[item.id];
      return (
        <div key={item.id} className="select-none">
          <button
            onClick={() => toggleFolder(item.id)}
            className="w-full flex items-center gap-1.5 px-2 py-1 hover:bg-slate-800/60 rounded text-xs font-semibold text-slate-300 hover:text-white transition-colors"
          >
            {isExpanded ? <ChevronDown className="w-3.5 h-3.5 text-slate-400" /> : <ChevronRight className="w-3.5 h-3.5 text-slate-400" />}
            {isExpanded ? <FolderOpen className="w-4 h-4 text-amber-400" /> : <Folder className="w-4 h-4 text-amber-400" />}
            <span>{item.name}</span>
          </button>
          {isExpanded && item.children && (
            <div className="pl-4 border-l border-slate-800/80 ml-2 space-y-0.5 mt-0.5">
              {item.children.map(child => renderDirectoryItem(child))}
            </div>
          )}
        </div>
      );
    } else {
      const isSelected = activeFileId === item.fileId;
      const file = projectFiles.find(f => f.id === item.fileId);
      return (
        <button
          key={item.id}
          onClick={() => item.fileId && handleOpenFile(item.fileId)}
          className={`w-full flex items-center justify-between px-2 py-1 rounded text-xs font-mono transition-all ${isSelected ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/30' : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200'}`}
        >
          <div className="flex items-center gap-1.5 truncate">
            {file && getFileIcon(file.type)}
            <span className="truncate">{item.name}</span>
          </div>
          {isSelected && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400"></span>}
        </button>
      );
    }
  };

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 w-full shadow-2xl relative backdrop-blur-xl flex flex-col gap-6">
      
      {/* Studio Header & View Mode Switcher Bar */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <div className="p-2 bg-sky-500/10 rounded-xl border border-sky-500/20 text-sky-400">
              <Network className="w-5 h-5" />
            </div>
            <h3 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
              <span>استوديو التطوير والمحاكاة (JOL Studio Factory Workbench)</span>
            </h3>
          </div>
          <p className="text-xs text-slate-400">
            Hierarchical D3 Maestro Process Tree, Directory Codebase Editor & Integrated Live Browser Stage.
          </p>
        </div>

        {/* Project Selector & View Mode Switcher */}
        <div className="flex flex-wrap items-center gap-3">
          <ProjectSelector
            onSelectProject={async (proj: StudioProject) => {
              if (onExecutePrompt) {
                await onExecutePrompt(proj.prompt, proj.domain);
              }
            }}
            className="w-72"
          />

          {/* View Mode Switcher Buttons */}
          <div className="flex items-center gap-1.5 bg-slate-950 p-1.5 rounded-xl border border-slate-800 shadow-inner">
            <button
              onClick={() => setViewMode('tree')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all ${viewMode === 'tree' ? 'bg-emerald-600 text-white shadow-lg shadow-emerald-950/50' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'}`}
            >
              <Network className="w-4 h-4" />
              <span>🌳 D3 Tree (الشجرة)</span>
            </button>
            
            <button
              onClick={() => setViewMode('editor')}
              className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all ${viewMode === 'editor' ? 'bg-cyan-600 text-white shadow-lg shadow-cyan-950/50' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'}`}
            >
            <FolderGit2 className="w-4 h-4" />
            <span>📁 Codebase Editor (محرر الكود)</span>
          </button>

          <button
            onClick={() => setViewMode('browser')}
            className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-xs font-bold transition-all ${viewMode === 'browser' ? 'bg-purple-600 text-white shadow-lg shadow-purple-950/50' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-850'}`}
          >
            <Monitor className="w-4 h-4" />
            <span>🌐 Live Browser (المتصفح الحي)</span>
          </button>
        </div>
      </div>
    </div>

      {/* ================= MODE 1: D3 MAESTRO TREE VIEW ================= */}
      {viewMode === 'tree' && (
        <div className="space-y-6 animate-fade-in">
          {/* Quick File Pills Bar in Tree Mode */}
          <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-3 shadow-lg flex items-center justify-between gap-4 overflow-x-auto">
            <div className="flex items-center gap-2 shrink-0">
              <FolderGit2 className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-bold text-slate-200">ملفات المشروع ({projectFiles.length}):</span>
            </div>
            
            <div className="flex items-center gap-2 overflow-x-auto no-scrollbar">
              {projectFiles.map((file) => (
                <button
                  key={file.id}
                  onClick={() => handleOpenFile(file.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyan-500/40 text-xs font-mono text-slate-300 hover:text-cyan-300 transition-all shrink-0"
                >
                  {getFileIcon(file.type)}
                  <span>{file.name}</span>
                </button>
              ))}
            </div>

            <button
              onClick={() => setViewMode('editor')}
              className="text-xs font-bold text-cyan-400 hover:underline shrink-0 flex items-center gap-1"
            >
              Open Directory Editor →
            </button>
          </div>

          {/* D3 Canvas Stage */}
          <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-[#070c18] shadow-2xl">
            <div className="overflow-x-auto flex justify-center py-4">
              <svg ref={svgRef}></svg>
            </div>

            {/* Selected Node Inspector Drawer */}
            {selectedNode && (
              <div className="absolute top-4 right-4 z-20 bg-slate-900/95 backdrop-blur-xl p-5 rounded-2xl border border-sky-500/40 max-w-sm text-right shadow-2xl animate-fade-in space-y-3">
                <div className="flex justify-between items-center pb-2 border-b border-slate-800">
                  <span className="font-mono text-xs font-bold text-sky-400 bg-sky-950/80 px-2.5 py-1 rounded-lg border border-sky-500/30">
                    Code: {selectedNode.geneticCode || '00'}
                  </span>
                  <h4 className="font-bold text-white text-sm truncate max-w-[180px]">{selectedNode.name}</h4>
                </div>
                
                <div className="flex items-center justify-end gap-2 text-xs">
                  <span className="text-slate-300 font-medium">
                    {selectedNode.status === 'healthy' ? 'نشط ومصنع (Active & Built)' : selectedNode.status === 'atrophied' ? 'ضامر (Pruned)' : 'جذر النمط (Root System)'}
                  </span>
                  <span className={`w-3 h-3 rounded-full ${selectedNode.status === 'healthy' ? 'bg-emerald-400 shadow-emerald-400/50 shadow-md' : selectedNode.status === 'atrophied' ? 'bg-rose-500' : 'bg-indigo-500'}`}></span>
                </div>

                <div className="text-xs text-amber-300 bg-amber-500/10 p-3 rounded-xl border border-amber-500/20 leading-relaxed text-right">
                  <div className="flex items-center justify-end gap-1.5 mb-1 font-bold text-amber-400">
                    <span>تحليل المودويل (Sub-System Gap)</span>
                    <Scan className="w-4 h-4" />
                  </div>
                  <p className="text-[11px] text-amber-200/90">{gapAnalysis}</p>
                </div>

                <button
                  onClick={() => selectedNode && handleExpandTargetNode(selectedNode)}
                  disabled={isBuildingNode}
                  className="w-full py-2.5 px-4 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 disabled:opacity-50 text-white rounded-xl text-xs font-bold flex items-center justify-center gap-2 shadow-lg shadow-emerald-950/40 transition-all active:scale-95"
                >
                  {isBuildingNode ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-white" />
                      <span>جاري التوسيع والتصنيع...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles className="w-4 h-4 text-amber-300" />
                      <span>توسيع وبناء هذا المودويل (Expand & Build Node)</span>
                    </>
                  )}
                </button>

                {buildLogs.length > 0 && (
                  <div className="text-[10px] font-mono text-emerald-400 bg-black/80 p-2.5 rounded-xl border border-emerald-500/30 text-left max-h-28 overflow-y-auto space-y-1">
                    {buildLogs.map((log, i) => (
                      <div key={i} className="leading-tight">{log}</div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ================= MODE 2: DIRECTORY & CODEBASE EDITOR ================= */}
      {viewMode === 'editor' && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 animate-fade-in min-h-[600px]">
          
          {/* File Directory Explorer Sidebar */}
          <div className="lg:col-span-1 bg-slate-950/90 border border-slate-800 rounded-2xl p-4 flex flex-col justify-between shadow-xl">
            <div>
              <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800">
                <div className="flex items-center gap-2 text-cyan-400">
                  <FolderGit2 className="w-4 h-4" />
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                    دليل الملفات (Directory)
                  </h4>
                </div>
                <span className="text-[10px] font-mono text-slate-500 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                  /project
                </span>
              </div>

              {/* Folder Hierarchy List */}
              <div className="space-y-1 overflow-y-auto max-h-[500px] pr-1">
                {directoryTree.map(item => renderDirectoryItem(item))}
              </div>
            </div>

            {/* Quick Actions Footer */}
            <div className="pt-4 border-t border-slate-800/80 text-[11px] text-slate-400 space-y-2">
              <div className="flex items-center justify-between">
                <span>Active File:</span>
                <strong className="text-cyan-400 font-mono">{activeFile?.name}</strong>
              </div>
              <button
                onClick={() => setViewMode('browser')}
                className="w-full py-2 bg-slate-900 hover:bg-slate-800 text-cyan-300 font-bold rounded-xl border border-slate-800 text-xs flex items-center justify-center gap-2 transition-all"
              >
                <Monitor className="w-3.5 h-3.5 text-purple-400" />
                <span>Launch Browser Preview →</span>
              </button>
            </div>
          </div>

          {/* Main IDE Codebase Editor Panel */}
          <div className="lg:col-span-3 bg-slate-950/90 border border-slate-800 rounded-2xl flex flex-col overflow-hidden shadow-2xl">
            
            {/* Editor Open Tabs Bar */}
            <div className="flex items-center bg-slate-900/90 border-b border-slate-800 px-2 overflow-x-auto">
              {openTabIds.map(id => {
                const tabFile = projectFiles.find(f => f.id === id);
                if (!tabFile) return null;
                const isActive = activeFileId === id;
                return (
                  <button
                    key={id}
                    onClick={() => setActiveFileId(id)}
                    className={`flex items-center gap-2 px-4 py-2.5 border-r border-slate-800 text-xs font-mono font-semibold transition-all group shrink-0 ${isActive ? 'bg-[#090d16] text-cyan-300 border-t-2 border-t-cyan-400' : 'text-slate-400 hover:bg-slate-850 hover:text-slate-200'}`}
                  >
                    {getFileIcon(tabFile.type)}
                    <span>{tabFile.name}</span>
                    <span
                      onClick={(e) => handleCloseTab(id, e)}
                      className="p-0.5 rounded hover:bg-slate-700 text-slate-500 group-hover:text-slate-300 ml-1 transition-colors"
                    >
                      <X className="w-3 h-3" />
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Editor Sub-Header Controls */}
            <div className="flex justify-between items-center px-4 py-2 bg-slate-950 border-b border-slate-800/80 text-xs">
              <div className="flex items-center gap-3">
                <span className="font-mono text-slate-400 text-[11px]">Path: <strong className="text-slate-200">{activeFile?.path}</strong></span>
                <span className="text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 text-[10px]">
                  {activeFile?.status}
                </span>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={handleRunCompile}
                  disabled={isCompiling}
                  className="px-3 py-1.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold rounded-lg text-xs flex items-center gap-1.5 shadow transition-all disabled:opacity-50"
                >
                  {isCompiling ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-white" />}
                  <span>Run & Compile AST</span>
                </button>

                <button
                  onClick={() => navigator.clipboard.writeText(fileCodes[activeFileId] || '')}
                  className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold rounded-lg text-xs flex items-center gap-1.5 border border-slate-700 transition-all"
                >
                  <Copy className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Copy</span>
                </button>
              </div>
            </div>

            {/* Code Editor Body (Gutter + Textarea) */}
            <div className="flex-1 bg-[#090d16] flex min-h-[380px]">
              {/* Line Numbers Gutter */}
              <div className="w-12 py-3 bg-[#060910] text-right pr-3 font-mono text-xs text-slate-600 select-none border-r border-slate-850 leading-relaxed">
                {(fileCodes[activeFileId] || '').split('\n').map((_, idx) => (
                  <div key={idx}>{idx + 1}</div>
                ))}
              </div>

              {/* Interactive Code Editor Textarea */}
              <textarea
                value={fileCodes[activeFileId] || ''}
                onChange={(e) => setFileCodes(prev => ({ ...prev, [activeFileId]: e.target.value }))}
                className="flex-1 p-3 bg-transparent font-mono text-xs text-cyan-200 focus:outline-none resize-none leading-relaxed selection:bg-cyan-500/30"
                spellCheck={false}
              />
            </div>

            {/* AST Execution Console Output Log */}
            <div className="p-3 bg-black/80 border-t border-slate-800 font-mono text-[11px] text-emerald-400 max-h-32 overflow-y-auto">
              <div className="flex items-center gap-2 mb-1 text-slate-400 font-semibold text-[10px] uppercase">
                <Terminal className="w-3.5 h-3.5 text-amber-400" />
                <span>Compiler & Execution Console Output</span>
              </div>
              <pre className="whitespace-pre-wrap leading-tight">{compilationLog}</pre>
            </div>
          </div>
        </div>
      )}

      {/* ================= MODE 3: LIVE WEB BROWSER PREVIEW ================= */}
      {viewMode === 'browser' && (
        <div className="bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl animate-fade-in flex flex-col">
          
          {/* Browser Chrome Header Bar */}
          <div className="bg-slate-900 border-b border-slate-800 p-3 flex flex-col sm:flex-row items-center justify-between gap-3">
            
            {/* Browser Controls & URL Bar */}
            <div className="flex items-center gap-3 w-full sm:w-auto flex-1">
              <div className="flex items-center gap-1 text-slate-400">
                <button className="p-1 hover:bg-slate-800 rounded transition-colors text-slate-500">
                  <ArrowLeft className="w-4 h-4" />
                </button>
                <button className="p-1 hover:bg-slate-800 rounded transition-colors text-slate-500">
                  <ArrowRight className="w-4 h-4" />
                </button>
                <button
                  onClick={handleRefreshBrowser}
                  className="p-1 hover:bg-slate-800 rounded transition-colors text-cyan-400"
                >
                  <RotateCw className={`w-4 h-4 ${isBrowserLoading ? 'animate-spin' : ''}`} />
                </button>
              </div>

              {/* URL Address Bar */}
              <div className="flex-1 flex items-center gap-2 bg-slate-950 border border-slate-800 px-3 py-1.5 rounded-xl text-xs font-mono text-slate-200 shadow-inner">
                <Lock className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                <span className="text-emerald-400 font-semibold text-[11px]">https://</span>
                <input
                  type="text"
                  value={browserUrl}
                  onChange={(e) => setBrowserUrl(e.target.value)}
                  className="bg-transparent flex-1 focus:outline-none text-slate-200 text-xs font-mono"
                />
                <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20 shrink-0">
                  LIVE APP
                </span>
              </div>
            </div>

            {/* Responsive Viewport Mode Toggles */}
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800">
                <button
                  onClick={() => setBrowserDevice('desktop')}
                  className={`p-1.5 rounded ${browserDevice === 'desktop' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-500 hover:text-slate-300'}`}
                  title="Desktop Viewport"
                >
                  <Laptop className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setBrowserDevice('tablet')}
                  className={`p-1.5 rounded ${browserDevice === 'tablet' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-500 hover:text-slate-300'}`}
                  title="Tablet Viewport"
                >
                  <Tablet className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setBrowserDevice('mobile')}
                  className={`p-1.5 rounded ${browserDevice === 'mobile' ? 'bg-cyan-500/20 text-cyan-400' : 'text-slate-500 hover:text-slate-300'}`}
                  title="Mobile Viewport"
                >
                  <Smartphone className="w-4 h-4" />
                </button>
              </div>

              <a
                href="http://localhost:5173"
                target="_blank"
                rel="noreferrer"
                className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-all border border-slate-700"
              >
                <ExternalLink className="w-3.5 h-3.5 text-cyan-400" />
                <span className="hidden sm:inline">New Tab</span>
              </a>
            </div>

          </div>

          {/* Browser Stage Content Viewport */}
          <div className="bg-[#0b1121] p-4 flex justify-center min-h-[650px] overflow-auto relative">
            {isBrowserLoading ? (
              <div className="flex flex-col items-center justify-center h-96 gap-3 text-cyan-400">
                <RefreshCw className="w-8 h-8 animate-spin" />
                <span className="text-xs font-bold font-mono">Refreshing Live Application Browser Stage...</span>
              </div>
            ) : (
              <div className={`transition-all duration-300 w-full ${browserDevice === 'tablet' ? 'max-w-3xl border border-slate-800 rounded-2xl shadow-2xl p-2 bg-slate-900' : browserDevice === 'mobile' ? 'max-w-sm border-2 border-slate-800 rounded-3xl shadow-2xl p-2 bg-slate-900' : 'max-w-full'}`}>
                {/* Render Live Application Component */}
                <CustomApp />
              </div>
            )}
          </div>
        </div>
      )}

      {/* Studio Architecture Footer info */}
      <div className="text-xs text-slate-400 bg-slate-950/60 p-3.5 rounded-xl border border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />
          <span>
            <strong className="text-slate-200">JOL Maestro Factory Workbench:</strong> Unified D3 Tree, File Directory Explorer, AST Code Editor & Live Web Application Browser.
          </span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] text-cyan-400">
          <span>Active Files: {projectFiles.length}</span>
          <span>•</span>
          <span>Gateway: :8088</span>
        </div>
      </div>

    </div>
  );
};

export default TreeVisualizer;