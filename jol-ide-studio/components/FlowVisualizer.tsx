import React, { useState, useEffect } from 'react';
import { Flow, Order, Checkpoint, OrderType } from '../types';
import { Play, CheckCircle, Disc, FileText, Zap } from 'lucide-react';
import { generateCheckpointReport } from '../services/geminiService';

interface FlowVisualizerProps {
  flow: Flow;
  onUpdateFlow: (flow: Flow) => void;
}

const FlowVisualizer: React.FC<FlowVisualizerProps> = ({ flow, onUpdateFlow }) => {
  const [isRunning, setIsRunning] = useState(false);
  const [processingOrderIdx, setProcessingOrderIdx] = useState<number | null>(null);

  const startFlow = async () => {
    if (isRunning) return;
    setIsRunning(true);

    // Simulate processing flow
    for (let i = 0; i < flow.checkpoints.length; i++) {
        const updatedFlow = { ...flow, currentCheckpointIndex: i };
        onUpdateFlow(updatedFlow);
        
        // Simulate "The Zone" - Team Processing
        for(let j = 0; j < flow.team.length; j++) {
            setProcessingOrderIdx(j);
            await new Promise(r => setTimeout(r, 600)); 
        }
        setProcessingOrderIdx(null);

        // Checkpoint Logic - Generate Brief Summary Report
        if (!flow.checkpoints[i].report) {
            const report = await generateCheckpointReport(flow.team, flow.checkpoints[i].name);
            const newCheckpoints = [...flow.checkpoints];
            newCheckpoints[i] = { ...newCheckpoints[i], report };
            onUpdateFlow({ ...updatedFlow, checkpoints: newCheckpoints });
        }
    }
    
    setIsRunning(false);
  };

  const getOrderIcon = (type: string) => {
    switch (type) {
        case 'Search': return '🔍';
        case 'Try': return '🧪';
        case 'Judge': return '⚖️';
        case 'Communicate': return '💬';
        default: return '📦';
    }
  };

  return (
    <div className={`relative bg-slate-900 border border-slate-700 rounded-lg p-6 w-full transition-all duration-700 ${isRunning ? 'ring-2 ring-cyan-500/30 shadow-[0_0_50px_rgba(6,182,212,0.1)]' : ''}`}>
      
      {/* The Zone Indicator */}
      {isRunning && (
        <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-900/50 border border-cyan-500/50 text-cyan-400 text-xs animate-pulse z-20">
            <Zap className="w-3 h-3" />
            <span>في المنطقة (The Zone) - تنفيذ متوازي</span>
        </div>
      )}

      <div className="flex justify-between items-center mb-6">
        <div>
          <h3 className="text-xl font-bold text-cyan-400 flex items-center gap-2">
              <Disc className="w-5 h-5" />
              طبقة المنطق: المسير (Flow & Checkpoints)
          </h3>
          <p className="text-xs text-slate-400 mt-1">سياسات الدمج: <span className="text-cyan-300 font-mono">{flow.mergePolicy || 'deep_merge'}</span></p>
        </div>
        <button
            onClick={startFlow}
            disabled={isRunning}
            className={`px-5 py-2 rounded font-bold flex items-center gap-2 transition-all ${isRunning ? 'bg-slate-800 text-slate-500 cursor-not-allowed' : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-lg shadow-cyan-500/20'}`}
        >
            <Play className="w-4 h-4" />
            {isRunning ? 'المسير جارٍ...' : 'بدء المسير'}
        </button>
      </div>

      {/* The Track */}
      <div className="relative pt-8 pb-12 overflow-x-auto">
        <div className="flex items-start gap-12 min-w-[750px] px-6">
            {flow.checkpoints.map((cp, idx) => {
                const isActive = flow.currentCheckpointIndex === idx;
                const isPast = flow.currentCheckpointIndex > idx;

                return (
                    <div key={cp.id} className="relative flex-1 flex flex-col items-center group">
                        {/* Connecting Line (Flow Thread) */}
                        {idx < flow.checkpoints.length - 1 && (
                            <div className="absolute top-5 left-[50%] w-full h-[2px] bg-slate-800" style={{ width: 'calc(100% + 3rem)' }}>
                                <div 
                                    className={`h-full transition-all duration-1000 ${isPast ? 'bg-cyan-500 w-full' : isActive ? 'bg-gradient-to-r from-cyan-500 to-slate-800 w-1/2' : 'w-0'}`}
                                ></div>
                            </div>
                        )}

                        {/* Checkpoint Node */}
                        <div className={`z-10 w-10 h-10 rounded-full flex items-center justify-center border-4 transition-all duration-500 
                            ${isActive ? 'bg-slate-900 border-cyan-400 scale-125 shadow-[0_0_20px_rgba(6,182,212,0.5)]' 
                            : isPast ? 'bg-cyan-500 border-cyan-600 shadow-md' 
                            : 'bg-slate-800 border-slate-700'}`}>
                            {isPast ? <CheckCircle className="w-5 h-5 text-white" /> : <span className={`text-sm font-bold ${isActive ? 'text-cyan-400' : 'text-slate-500'}`}>{idx + 1}</span>}
                        </div>
                        
                        <div className="mt-4 text-center w-48">
                            <p className={`font-bold text-sm mb-2 transition-colors ${isActive ? 'text-cyan-300' : 'text-slate-500'}`}>{cp.name}</p>
                            
                            {/* Micro-Checkpoints Rendering */}
                            {cp.microCheckpoints && cp.microCheckpoints.length > 0 && (
                              <div className="mb-3 p-2 bg-slate-950/80 rounded border border-cyan-500/30 text-[10px] text-right">
                                <div className="text-cyan-400 font-bold mb-1 flex items-center justify-between">
                                  <span>فحوصات صغرى ({cp.microCheckpoints.length})</span>
                                  <span className="text-[9px] bg-cyan-950 text-cyan-300 px-1 rounded">{cp.microCheckpoints[0].strategy}</span>
                                </div>
                                {cp.microCheckpoints.map(mcp => (
                                  <div key={mcp.id} className="mt-1 bg-slate-900 p-1.5 rounded border border-slate-800">
                                    <div className="flex justify-between text-slate-300 font-mono text-[9px]">
                                      <span>{mcp.name}</span>
                                      <span className="text-green-400">{mcp.passedCount}/{mcp.totalCount} ({Math.round((mcp.passedCount / (mcp.totalCount || 1)) * 100)}%)</span>
                                    </div>
                                    <div className="w-full bg-slate-800 h-1.5 rounded-full mt-1 overflow-hidden">
                                      <div 
                                        className="bg-gradient-to-r from-cyan-500 to-green-400 h-full transition-all duration-500" 
                                        style={{ width: `${(mcp.passedCount / (mcp.totalCount || 1)) * 100}%` }}
                                      ></div>
                                    </div>
                                  </div>
                                ))}
                              </div>
                            )}

                            {/* Context Memory Dump Card */}
                            {cp.report && (
                                <div className="relative p-3 bg-slate-800/80 backdrop-blur-sm border border-slate-600 rounded-md text-[10px] text-slate-300 shadow-lg text-right transform transition-all duration-500 hover:scale-105 hover:border-yellow-500/50 group-hover:z-20">
                                    <div className="flex items-center justify-between text-yellow-500 font-bold border-b border-slate-700 pb-1 mb-1">
                                        <span className="text-[9px] text-slate-400">تفريغ الذاكرة (Pruned)</span>
                                        <div className="flex items-center gap-1">
                                          <span>تقرير موجز</span>
                                          <FileText className="w-3 h-3" />
                                        </div>
                                    </div>
                                    <p className="leading-relaxed opacity-90">{cp.report}</p>
                                </div>
                            )}
                        </div>
                    </div>
                );
            })}
        </div>
      </div>
      
      <div className="mt-4 text-[11px] text-slate-500 bg-black/20 p-3 rounded border border-slate-800/50">
        <strong className="text-cyan-500">هندسة FlowLang:</strong> يتم تفريغ الذاكرة عند كل نقطة تفتيش (Context Pruning) وتمرير "التقرير الموجز" فقط لتجنب الضوضاء في النماذج وتفادي الهلوسة.
      </div>
    </div>
  );
};

export default FlowVisualizer;