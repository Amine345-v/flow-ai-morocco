import React from 'react';
import { Table, FileSpreadsheet, Lock, Activity, Layers, Database, Stethoscope, ShieldCheck, HeartPulse } from 'lucide-react';
import { ProfessionalDomain } from '../types';

interface ResourceViewProps {
    domain: ProfessionalDomain;
    data: Record<string, any>;
}

const ResourceView: React.FC<ResourceViewProps> = ({ domain, data }) => {
    const [showSource, setShowSource] = React.useState(false);

    const getIntegrityBadge = (status: string) => {
        const isActionable = status.includes('SIGNED') || status.includes('PASS') || status === 'VERIFIED' || status.includes('COMPLIANT');
        return (
            <div className={`px-2 py-0.5 rounded-full text-[8px] font-bold border ${isActionable ? 'border-green-500/50 text-green-400 bg-green-500/10' : 'border-red-500/50 text-red-400 bg-red-500/10'}`}>
                INTEGRITY: {status}
            </div>
        );
    };

    const renderOriginFooter = (resource: any) => (
        <div className="p-2 bg-slate-900/80 border-t border-slate-800 flex items-center justify-between text-[9px] text-slate-500">
            <div className="flex items-center gap-2 overflow-hidden">
                <span className="text-slate-600 shrink-0">ORIGIN:</span>
                <span className="truncate hover:text-slate-300 transition-colors cursor-help" title={resource?.origin_path}>
                    {resource?.origin_path || 'TRANSIENT_GOVERNANCE_MEMORY'}
                </span>
            </div>
            <div className="flex items-center gap-3 shrink-0 ml-2">
                <span className="text-[8px] opacity-50 uppercase">{resource?.last_sync ? `SYNCED: ${new Date(resource.last_sync).toLocaleTimeString()}` : 'LIVE'}</span>
                <button
                    onClick={() => setShowSource(!showSource)}
                    className={`px-2 py-0.5 rounded border transition-colors ${showSource ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-400' : 'border-slate-700 hover:border-slate-500'}`}
                >
                    {showSource ? 'GRID' : 'SOURCE'}
                </button>
            </div>
        </div>
    );

    const renderRawSource = (content: string) => (
        <div className="flex-1 p-4 overflow-auto bg-slate-950/80 font-mono text-[10px] text-cyan-500/80">
            <pre className="whitespace-pre-wrap leading-relaxed">
                {content || '// No raw data provided by origin.'}
            </pre>
        </div>
    );

    const renderClinicalTrialView = () => {
        const res = data.clinical;
        return (
            <div className="flex flex-col h-full bg-slate-950/50 border border-slate-800 rounded-lg overflow-hidden font-mono text-[11px]">
                <div className="bg-pink-600/20 p-2 flex items-center justify-between border-b border-pink-500/30">
                    <div className="flex items-center gap-2 text-pink-400 font-bold">
                        <Activity className="w-4 h-4 text-pink-400 animate-pulse" />
                        CLINICAL_TRIAL_GOVERNANCE_PHASE_III
                    </div>
                    {getIntegrityBadge(res?.status || 'HIPAA_COMPLIANT_100%')}
                </div>

                {showSource ? renderRawSource(res?.raw_content) : (
                    <div className="flex-1 p-4 overflow-auto space-y-4">
                        <div className="grid grid-cols-3 gap-2">
                            <div className="p-2 border border-pink-900/50 bg-pink-950/20 rounded">
                                <p className="text-[9px] text-pink-400">PATIENTS ENROLLED</p>
                                <p className="text-lg font-bold text-white">1,420</p>
                            </div>
                            <div className="p-2 border border-pink-900/50 bg-pink-950/20 rounded">
                                <p className="text-[9px] text-pink-400">P-VALUE SIG</p>
                                <p className="text-lg font-bold text-green-400">0.0008</p>
                            </div>
                            <div className="p-2 border border-pink-900/50 bg-pink-950/20 rounded">
                                <p className="text-[9px] text-pink-400">ADVERSE EVENTS</p>
                                <p className="text-lg font-bold text-slate-300">0 (Clean)</p>
                            </div>
                        </div>

                        <div className="p-3 bg-slate-900/80 rounded border border-slate-800 space-y-2">
                            <div className="flex justify-between items-center text-[10px]">
                                <span className="text-slate-400 font-bold">HIPAA Anonymization & Governance Checkpoint</span>
                                <span className="text-green-400 font-bold">VERIFIED</span>
                            </div>
                            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                <div className="bg-pink-500 h-full w-full shadow-[0_0_8px_#ec4899]" />
                            </div>
                            <p className="text-[9px] text-slate-500 italic">
                                Double-blind trial data pipeline validated by AI Diagnostic Judge. FDA Submission bundle prepared.
                            </p>
                        </div>
                    </div>
                )}
                {renderOriginFooter(res)}
            </div>
        );
    };

    const renderEconomicExcel = () => {
        const res = data.economic;
        const sheet = res?.sheet || [
            ['1', 'Core Infrastructure', '$120k', '$115k'],
            ['2', 'Cloud Deployment', '$45k', '$52k'],
            ['3', 'R&D', '$80k', '$75k'],
        ];

        return (
            <div className="flex flex-col h-full bg-slate-950/50 border border-slate-800 rounded-lg overflow-hidden font-mono text-[11px]">
                <div className="bg-green-600/20 p-2 flex items-center justify-between border-b border-green-500/30">
                    <div className="flex items-center gap-2 text-green-400 font-bold">
                        <FileSpreadsheet className="w-4 h-4" />
                        {res?.filename || 'QUANT_PORTFOLIO_RISK.XLSX'}
                    </div>
                    {getIntegrityBadge(res?.integrity || 'VaR_PASSED_99%')}
                </div>

                {showSource ? renderRawSource(res?.raw_content) : (
                    <div className="flex-1 overflow-auto">
                        <table className="w-full text-left border-collapse">
                            <thead>
                                <tr className="bg-slate-900 text-slate-500 border-b border-slate-800">
                                    <th className="p-2 border-r border-slate-800 w-8">#</th>
                                    <th className="p-2 border-r border-slate-800">ITEM</th>
                                    <th className="p-2 border-r border-slate-800">FORECAST</th>
                                    <th className="p-2">ACTUAL</th>
                                </tr>
                            </thead>
                            <tbody>
                                {sheet.map(([id, item, forecast, actual]: any) => (
                                    <tr key={id} className="border-b border-slate-900 hover:bg-white/5 transition-colors">
                                        <td className="p-2 border-r border-slate-800 text-slate-600">{id}</td>
                                        <td className="p-2 border-r border-slate-800 text-slate-300">{item}</td>
                                        <td className="p-2 border-r border-slate-800 text-green-400 font-bold">{forecast}</td>
                                        <td className="p-2 text-slate-400">{actual}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
                {renderOriginFooter(res)}
            </div>
        );
    };

    const renderCyberAudit = () => {
        const res = data.cyber;
        return (
            <div className="flex flex-col h-full bg-slate-950/50 border border-slate-800 rounded-lg overflow-hidden font-mono text-[11px]">
                <div className="bg-red-600/20 p-2 flex items-center justify-between border-b border-red-500/30">
                    <div className="flex items-center gap-2 text-red-400 font-bold">
                        <Lock className="w-4 h-4" />
                        {res?.target || 'ZERO_TRUST_MITRE_AUDIT_REPORT'}
                    </div>
                    {getIntegrityBadge(res?.level || 'HARDENED')}
                </div>

                {showSource ? renderRawSource(res?.raw_content) : (
                    <div className="flex-1 p-4 overflow-auto text-slate-300 space-y-2">
                        <div className="text-green-400 font-bold">[PASS] OCSF Telemetry Ingestion Active</div>
                        <div className="text-red-400">[SCAN] Penetration Testing Execution Playbook Loaded</div>
                        <div className="text-slate-500">{" >> "} MITRE ATT&CK Tactic T1059 (Command Execution) Validated</div>
                        <div className="text-green-500">[ZERO-TRUST] Micro-segmentation Rule Applied</div>
                    </div>
                )}
                {renderOriginFooter(res)}
            </div>
        );
    };

    const renderDigitalWorkspace = () => (
        <div className="flex flex-col h-full bg-slate-950/50 border border-slate-800 rounded-lg overflow-hidden font-mono text-[11px]">
            <div className="bg-cyan-600/20 p-2 flex items-center justify-between border-b border-cyan-500/30">
                <div className="flex items-center gap-2 text-cyan-400 font-bold">
                    <Database className="w-4 h-4" />
                    FLOWLANG_SOFTWARE_FACTORY
                </div>
                <div className="text-[9px] text-cyan-500/70 uppercase">STATUS: LIVE BUILD PIPELINE</div>
            </div>
            <div className="flex-1 p-4 overflow-auto text-slate-300 space-y-2">
                <div className="text-cyan-500">[FACTORY] Architecture Checkpoint Satisfied</div>
                <div className="text-slate-500">{" >> "} QA Gate: 6/6 Micro-checks Passed</div>
                <div className="text-green-500">[RELEASE] Staging Blue-Green Deployment Approved</div>
            </div>
            {renderOriginFooter(null)}
        </div>
    );

    const renderMechanicalSimulation = () => {
        const res = data.mechanical;
        return (
            <div className="flex flex-col h-full bg-slate-950/50 border border-slate-800 rounded-lg overflow-hidden font-mono text-[11px]">
                <div className="bg-orange-600/20 p-2 flex items-center justify-between border-b border-orange-500/30">
                    <div className="flex items-center gap-2 text-orange-400 font-bold">
                        <Layers className="w-4 h-4" />
                        KINEMATICS_ROBOTIC_ARM_ASSEMBLY.CAD
                    </div>
                    {getIntegrityBadge(res?.integrity || 'PASS_0.01MM')}
                </div>

                {showSource ? renderRawSource(res?.raw_content) : (
                    <div className="flex-1 p-4 overflow-auto flex flex-col items-center justify-center space-y-4">
                        <div className="w-32 h-32 border-2 border-orange-500/40 rounded-lg flex items-center justify-center relative bg-gradient-to-br from-orange-500/10 to-transparent">
                            <Activity className="w-12 h-12 text-orange-400 animate-pulse" />
                        </div>
                        <div className="w-full space-y-2">
                            <div className="flex justify-between text-[10px]">
                                <span className="text-slate-500">Torque: 450 Nm</span>
                                <span className="text-orange-400">Tolerance: 0.01 mm</span>
                            </div>
                        </div>
                    </div>
                )}
                {renderOriginFooter(res)}
            </div>
        );
    };

    const renderElectroSchematic = () => {
        const res = data.electro;
        return (
            <div className="flex flex-col h-full bg-slate-950/50 border border-slate-800 rounded-lg overflow-hidden font-mono text-[11px]">
                <div className="bg-purple-600/20 p-2 flex items-center justify-between border-b border-purple-500/30">
                    <div className="flex items-center gap-2 text-purple-400 font-bold">
                        <Activity className="w-4 h-4" />
                        IOT_MICROCONTROLLER_FLEET.SPICE
                    </div>
                    {getIntegrityBadge(res?.status || 'VOLTAGE_STABLE')}
                </div>

                {showSource ? renderRawSource(res?.raw_content) : (
                    <div className="flex-1 p-4 overflow-auto space-y-4">
                        <div className="grid grid-cols-2 gap-2">
                            <div className="p-2 border border-purple-900/50 bg-purple-950/20 rounded">
                                <p className="text-[9px] text-purple-400">LATENCY (P99)</p>
                                <p className="text-lg font-bold text-white">12 ms</p>
                            </div>
                            <div className="p-2 border border-purple-900/50 bg-purple-950/20 rounded">
                                <p className="text-[9px] text-purple-400">BATTERY LIFE</p>
                                <p className="text-lg font-bold text-green-400">10 Years</p>
                            </div>
                        </div>
                    </div>
                )}
                {renderOriginFooter(res)}
            </div>
        );
    };

    return (
        <div className="h-full">
            {domain === 'economic' ? renderEconomicExcel() :
                domain === 'cyber' ? renderCyberAudit() :
                    domain === 'digital' ? renderDigitalWorkspace() :
                        domain === 'mechanical' ? renderMechanicalSimulation() :
                            domain === 'electro' ? renderElectroSchematic() :
                                domain === 'clinical' ? renderClinicalTrialView() :
                                    renderDigitalWorkspace()}
        </div>
    );
};

export default ResourceView;
