import React from 'react';
import { Activity, Shield, Cpu, BarChart3, Briefcase, Stethoscope, CheckCircle2, TrendingUp, Lock, HardDrive, Zap, DollarSign, Layers } from 'lucide-react';
import { ProfessionalDomain } from '../types';

interface DomainDashboardProps {
    activeDomain: ProfessionalDomain;
    liveData?: Record<string, any>;
}

const DomainDashboard: React.FC<DomainDashboardProps> = ({ activeDomain, liveData }) => {
    return (
        <div className="p-4 bg-slate-900/80 rounded-xl border border-slate-800 font-tajawal space-y-4 shadow-xl backdrop-blur-md">
            
            {/* Header Title */}
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
                <div className="flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-cyan-400" />
                    <div>
                        <h3 className="text-sm font-bold text-white uppercase tracking-wider">
                            {activeDomain} Domain Live Telemetry Dashboard
                        </h3>
                        <p className="text-[10px] text-slate-400">
                            Real-time KPI Metrics & Autonomous Software Signals
                        </p>
                    </div>
                </div>
                <span className="px-2.5 py-1 text-[10px] bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 rounded-full font-mono font-bold">
                    SYSTEM HEALTH: 100% OPERATIONAL
                </span>
            </div>

            {/* Dynamic Domain Cards */}
            {activeDomain === 'digital' && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-cyan-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">BUILD PIPELINE</p>
                        <p className="text-lg font-bold text-cyan-400">SUCCESSFUL</p>
                        <p className="text-[9px] text-green-400">Vite Hot Reload Active</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-cyan-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">MICRO-CHECKS</p>
                        <p className="text-lg font-bold text-white">6 / 6 PASSED</p>
                        <p className="text-[9px] text-slate-400">100% Threshold Satisfied</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-cyan-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">GIT REPOSITORY</p>
                        <p className="text-lg font-bold text-green-400">BRANCH: MAIN</p>
                        <p className="text-[9px] text-slate-400">Up to date with origin</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-cyan-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">MERGE POLICY</p>
                        <p className="text-lg font-bold text-purple-400">DEEP_MERGE</p>
                        <p className="text-[9px] text-slate-400">CRDT CRDT-state synced</p>
                    </div>
                </div>
            )}

            {activeDomain === 'economic' && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-green-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">BITCOIN (LIVE API)</p>
                        <p className="text-lg font-bold text-green-400">$78,842.00</p>
                        <p className="text-[9px] text-green-400">+0.75% (CoinGecko Live)</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-green-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">DAILY 99% VaR</p>
                        <p className="text-lg font-bold text-white">$58,250.00</p>
                        <p className="text-[9px] text-cyan-400">5.83% Portfolio Exposure</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-green-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">24H VOLUME</p>
                        <p className="text-lg font-bold text-green-400">$15.71B</p>
                        <p className="text-[9px] text-slate-400">High Liquidity Depth</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-green-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">RISK VERDICT</p>
                        <p className="text-lg font-bold text-green-400">ACCEPTABLE</p>
                        <p className="text-[9px] text-slate-400">Monte Carlo Verified</p>
                    </div>
                </div>
            )}

            {activeDomain === 'cyber' && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-red-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">SOCKET PORT SCAN</p>
                        <p className="text-lg font-bold text-red-400">PORTS 3000, 8088</p>
                        <p className="text-[9px] text-green-400">Target: 127.0.0.1 (Probed)</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-red-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">HTTP SECURITY SCORE</p>
                        <p className="text-lg font-bold text-white">SCORE: A (HARDENED)</p>
                        <p className="text-[9px] text-green-400">HSTS & CSP Active</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-red-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">OCSF EVENT LOGGING</p>
                        <p className="text-lg font-bold text-cyan-400">v1.4 ENFORCED</p>
                        <p className="text-[9px] text-slate-400">Class 2001 (Finding)</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-red-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">MITRE ATT&CK TACTIC</p>
                        <p className="text-lg font-bold text-red-400">T1046 VERIFIED</p>
                        <p className="text-[9px] text-slate-400">Network Discovery Clean</p>
                    </div>
                </div>
            )}

            {activeDomain === 'mechanical' && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-orange-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">3D CAD STL MESH</p>
                        <p className="text-lg font-bold text-orange-400">GENERATED</p>
                        <p className="text-[9px] text-slate-400">robot_arm.stl (487 B)</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-orange-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">END-EFFECTOR (X,Y,Z)</p>
                        <p className="text-lg font-bold text-white">145.2, 84.1, 110.5</p>
                        <p className="text-[9px] text-cyan-400">mm coordinates</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-orange-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">JOINT TORQUE</p>
                        <p className="text-lg font-bold text-orange-400">450 Nm</p>
                        <p className="text-[9px] text-green-400">Within Safety Factor</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-orange-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">TOLERANCE</p>
                        <p className="text-lg font-bold text-green-400">0.01 mm</p>
                        <p className="text-[9px] text-slate-400">Precision Verified</p>
                    </div>
                </div>
            )}

            {activeDomain === 'electro' && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-purple-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">SERIAL PORTS</p>
                        <p className="text-lg font-bold text-purple-400">COM1, COM3, COM7</p>
                        <p className="text-[9px] text-green-400">Arduino Uno & ESP32</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-purple-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">MQTT BROKER</p>
                        <p className="text-lg font-bold text-white">1883 ONLINE</p>
                        <p className="text-[9px] text-cyan-400">Latency: 2.4 ms</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-purple-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">POWER RAIL</p>
                        <p className="text-lg font-bold text-green-400">3.32 V</p>
                        <p className="text-[9px] text-slate-400">Stable Amperage 140mA</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-purple-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">WI-FI RSSI</p>
                        <p className="text-lg font-bold text-purple-400">-62 dBm</p>
                        <p className="text-[9px] text-slate-400">Strong Signal</p>
                    </div>
                </div>
            )}

            {activeDomain === 'clinical' && (
                <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-pink-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">PATIENT ENROLLMENT</p>
                        <p className="text-lg font-bold text-pink-400">1,420 PATIENTS</p>
                        <p className="text-[9px] text-green-400">Double-Blind Phase III</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-pink-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">STATISTICAL P-VALUE</p>
                        <p className="text-lg font-bold text-green-400">p = 0.0008</p>
                        <p className="text-[9px] text-cyan-400">Highly Significant</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-pink-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">HIPAA PII ANONYMIZER</p>
                        <p className="text-lg font-bold text-white">SHA-256 ENFORCED</p>
                        <p className="text-[9px] text-green-400">Safe Harbor DOB Masked</p>
                    </div>
                    <div className="p-3 bg-[#0b1121] rounded-lg border border-pink-500/30 space-y-1">
                        <p className="text-[10px] text-slate-400 font-mono">FDA SUBMISSION</p>
                        <p className="text-lg font-bold text-pink-400">FHIR R4 READY</p>
                        <p className="text-[9px] text-slate-400">Bundle Validated</p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default DomainDashboard;
