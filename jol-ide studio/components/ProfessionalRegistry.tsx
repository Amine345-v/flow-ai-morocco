import React from 'react';
import { Briefcase, Shield, BarChart3, Settings, Cpu, Activity } from 'lucide-react';
import { ProfessionalDomain } from '../types';

interface ProfessionalRegistryProps {
    activeDomain: ProfessionalDomain;
    onDomainChange: (domain: ProfessionalDomain) => void;
}

const ProfessionalRegistry: React.FC<ProfessionalRegistryProps> = ({ activeDomain, onDomainChange }) => {
    const domains = [
        { id: 'digital', name: 'Digital / Software', icon: Settings, color: 'text-cyan-400', bg: 'bg-cyan-400/10', border: 'border-cyan-500/50', dot: 'bg-cyan-400' },
        { id: 'economic', name: 'Economic / Quant', icon: BarChart3, color: 'text-green-400', bg: 'bg-green-400/10', border: 'border-green-500/50', dot: 'bg-green-400' },
        { id: 'cyber', name: 'Cyber / SecOps', icon: Shield, color: 'text-red-400', bg: 'bg-red-400/10', border: 'border-red-500/50', dot: 'bg-red-400' },
        { id: 'mechanical', name: 'Meca / Robotics', icon: Briefcase, color: 'text-orange-400', bg: 'bg-orange-400/10', border: 'border-orange-500/50', dot: 'bg-orange-400' },
        { id: 'electro', name: 'Electro / IoT Fleet', icon: Cpu, color: 'text-purple-400', bg: 'bg-purple-400/10', border: 'border-purple-500/50', dot: 'bg-purple-400' },
        { id: 'clinical', name: 'Clinical / Healthcare', icon: Activity, color: 'text-pink-400', bg: 'bg-pink-400/10', border: 'border-pink-500/50', dot: 'bg-pink-400' },
    ];

    return (
        <div className="p-3 bg-slate-900/50 rounded-lg border border-slate-800 mb-4">
            <div className="flex items-center gap-2 text-[10px] text-slate-500 mb-3 uppercase tracking-widest font-bold">
                <Briefcase className="w-3 h-3 text-cyan-400" />
                PROFESSIONAL DOMAIN
            </div>
            <div className="grid grid-cols-1 gap-2">
                {domains.map((domain) => {
                    const Icon = domain.icon;
                    const isActive = activeDomain === domain.id;
                    return (
                        <button
                            key={domain.id}
                            onClick={() => onDomainChange(domain.id as ProfessionalDomain)}
                            className={`flex items-center justify-between p-2.5 rounded-md transition-all border ${isActive ? `${domain.border} ${domain.bg} shadow-md` : 'border-transparent hover:bg-white/5'}`}
                        >
                            <div className="flex items-center gap-3">
                                <Icon className={`w-4 h-4 ${isActive ? domain.color : 'text-slate-500'}`} />
                                <span className={`text-xs font-semibold ${isActive ? 'text-white' : 'text-slate-400'}`}>
                                    {domain.name}
                                </span>
                            </div>
                            {isActive && (
                                <div className={`w-2 h-2 rounded-full ${domain.dot} shadow-[0_0_10px_currentColor]`} />
                            )}
                        </button>
                    );
                })}
            </div>
        </div>
    );
};

export default ProfessionalRegistry;
