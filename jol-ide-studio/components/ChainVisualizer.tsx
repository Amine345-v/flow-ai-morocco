import React, { useState } from 'react';
import { SystemChainNode, OrderType } from '../types';
import { Link2, Zap, Activity } from 'lucide-react';
import { analyzeSystemEcho } from '../services/geminiService';
import ProjectSelector, { StudioProject } from './ProjectSelector';

interface ChainVisualizerProps {
  chain: SystemChainNode[];
  onExecutePrompt?: (prompt: string, domain?: string) => Promise<void>;
}

const ChainVisualizer: React.FC<ChainVisualizerProps> = ({ chain, onExecutePrompt }) => {
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null);
  const [echoes, setEchoes] = useState<Record<string, string>>({});
  const [loadingEcho, setLoadingEcho] = useState(false);

  const handleNodeClick = async (node: SystemChainNode) => {
    if (activeNodeId === node.id) {
        setActiveNodeId(null);
        return;
    }

    setActiveNodeId(node.id);
    
    // Simulate Echo Analysis if not already present for neighbors
    // Find neighbors
    const idx = chain.findIndex(n => n.id === node.id);
    const neighbors = [chain[idx-1], chain[idx+1]].filter(Boolean);
    
    setLoadingEcho(true);
    const newEchoes = { ...echoes };
    
    for (const neighbor of neighbors) {
        if (!newEchoes[neighbor.id]) {
            const analysis = await analyzeSystemEcho(node.order.content, node.order.type);
            newEchoes[neighbor.id] = analysis;
        }
    }
    
    setEchoes(newEchoes);
    setLoadingEcho(false);
  };

  const getRippleClass = (nodeId: string, targetId: string | null, index: number, targetIndex: number) => {
    if (!targetId) return '';
    const distance = Math.abs(index - targetIndex);
    
    // The Active Node (Source of Ripple)
    if (distance === 0) return 'border-purple-500 bg-slate-800 ring-2 ring-purple-500/50 shadow-[0_0_30px_rgba(168,85,247,0.2)] scale-105 z-10';
    
    // Immediate Neighbors (The Echo)
    if (distance === 1) return 'border-purple-400/50 bg-slate-800/50 ring-1 ring-purple-400/30';
    
    return 'opacity-50 blur-[1px]';
  };

  const activeIndex = chain.findIndex(n => n.id === activeNodeId);

  return (
    <div className="bg-slate-900 border border-slate-700 rounded-lg p-6 w-full">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
        <div>
          <h3 className="text-xl font-bold text-purple-400 flex items-center gap-2">
              <Link2 className="w-5 h-5" />
               تسلسل النظام (System Sequence)
          </h3>
          <p className="text-xs text-slate-400 mt-1 flex items-center gap-1">
            <Activity className="w-3 h-3 text-purple-400" />
            <span>اضغط على عنصر لرؤية "صدى التأثير" (Echo Effect)</span>
          </p>
        </div>

        <ProjectSelector
          onSelectProject={async (proj: StudioProject) => {
            if (onExecutePrompt) {
              await onExecutePrompt(proj.prompt, proj.domain);
            }
          }}
          className="w-72"
        />
      </div>

      <div className="flex flex-col md:flex-row items-start justify-center gap-2 min-h-[250px] overflow-x-auto p-8">
        {chain.map((node, idx) => {
            const isNeighbor = activeNodeId && Math.abs(idx - activeIndex) === 1;
            const isSource = activeNodeId === node.id;
            
            return (
            <React.Fragment key={node.id}>
                {/* Node Container */}
                <div className="flex flex-col items-center gap-2">
                    {/* The Node Card */}
                    <div 
                        onClick={() => handleNodeClick(node)}
                        className={`
                            relative w-36 h-36 flex flex-col items-center justify-center p-3 rounded-xl cursor-pointer
                            border transition-all duration-500 ease-out
                            ${!activeNodeId ? 'bg-slate-800 border-slate-600 hover:border-slate-400 hover:bg-slate-750' : getRippleClass(node.id, activeNodeId, idx, activeIndex)}
                        `}
                    >
                        <span className="text-[10px] uppercase font-bold tracking-widest text-slate-500 mb-2">{node.order.type}</span>
                        <p className="text-xs text-center line-clamp-4 text-slate-200 leading-relaxed">{node.order.content}</p>
                        
                        {/* Source Indicator */}
                        {isSource && (
                            <div className="absolute -top-3 -right-3 bg-purple-600 rounded-full p-1.5 shadow-lg animate-ping">
                                <Zap className="w-3 h-3 text-white" />
                            </div>
                        )}
                    </div>

                    {/* Echo Message Bubble */}
                    {isNeighbor && (
                        <div className="w-36 min-h-[40px] mt-2 text-[10px] text-purple-300 bg-purple-900/20 border border-purple-500/30 p-2 rounded text-center animate-fade-in">
                            {loadingEcho ? (
                                <span className="animate-pulse">جاري تحليل الصدى...</span>
                            ) : (
                                <span className="italic">"{echoes[node.id]}"</span>
                            )}
                        </div>
                    )}
                </div>

                {/* Connecting Nerve Thread */}
                {idx < chain.length - 1 && (
                    <div className="h-36 w-8 flex items-center justify-center">
                        <div className={`h-[2px] w-full transition-all duration-500 ${isSource || (activeNodeId && Math.abs((idx + 1) - activeIndex) === 0) ? 'bg-purple-500 shadow-[0_0_10px_rgba(168,85,247,0.8)]' : 'bg-slate-700'}`}></div>
                    </div>
                )}
            </React.Fragment>
        );})}
      </div>
      
       <div className="mt-4 text-[11px] text-slate-500 bg-black/20 p-3 rounded border border-slate-800/50">
        <strong className="text-purple-400">منطق JOL:</strong> السلسلة هي "خيط ناظم" سببي. أي تعديل في أمر لا يهدم النظام، بل يرسل "صدى" (Echo) يؤثر على الجيران (زيادة الأمن قد تبطئ السرعة كصدى)، مما يسمح بالتعديل الجراحي الدقيق.
      </div>
    </div>
  );
};

export default ChainVisualizer;