import React from 'react';
import { Order } from '../types';
import { MessageSquare, Terminal } from 'lucide-react';

interface MonolithLogProps {
    logs: Order[];
}

const MonolithLog: React.FC<MonolithLogProps> = ({ logs }) => {
    return (
        <div className="flex flex-col h-full bg-[#0f172a] border border-slate-800 rounded-lg overflow-hidden shadow-xl">
            <div className="bg-slate-800/50 p-2 border-b border-slate-700 flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs font-bold text-cyan-400 uppercase tracking-wider">
                    <MessageSquare className="w-4 h-4" />
                    سجل المهام (Task Log)
                </div>
                <Terminal className="w-3 h-3 text-slate-500" />
            </div>
            <div className="flex-1 overflow-auto p-3 space-y-3 font-mono text-[11px]">
                {logs.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center opacity-20 text-slate-500 italic">
                        <MessageSquare className="w-8 h-8 mb-2" />
                        بانتظار الأوامر المهنية...
                    </div>
                ) : (
                    logs.map((log, idx) => (
                        <div key={log.id || idx} className="group transition-all hover:bg-slate-800/30 p-2 rounded border-l-2 border-cyan-500/50 bg-[#0b1121]/50">
                            <div className="flex justify-between items-start mb-1">
                                <span className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${log.type === 'SEARCH' ? 'bg-blue-500/20 text-blue-400' :
                                        log.type === 'JUDGE' ? 'bg-purple-500/20 text-purple-400' :
                                            'bg-green-500/20 text-green-400'
                                    }`}>
                                    {log.type}
                                </span>
                                <span className="text-[9px] text-slate-600">ID: {log.id?.slice(0, 8)}</span>
                            </div>
                            <p className="text-slate-300 leading-relaxed">{log.content}</p>
                            {log.result && (
                                <div className="mt-2 text-slate-500 bg-black/20 p-2 rounded text-[10px] whitespace-pre-wrap border-t border-slate-800">
                                    {log.result}
                                </div>
                            )}
                        </div>
                    ))
                )}
            </div>
        </div>
    );
};

export default MonolithLog;
