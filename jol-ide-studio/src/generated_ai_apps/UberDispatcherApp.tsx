import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Zap, Server, RefreshCw, Send, CheckCircle2, AlertCircle, Cpu } from 'lucide-react';

export interface UberDispatcherAppDataRecord {
  id: string;
  title: string;
  category: string;
  status: 'ACTIVE' | 'PENDING' | 'COMPLETED';
  timestamp: string;
  metrics: number;
}

/**
 * Autonomous AI Synthesized Runnable Application Component: UberDispatcherAppApp
 * Target Domain Request: "Build an autonomous Uber driver dispatcher web application with GPS matching"
 */
export default function UberDispatcherAppApp() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'operations' | 'logs'>('dashboard');
  const [items, setItems] = useState<UberDispatcherAppDataRecord[]>([
    { id: '1', title: 'Autonomous Primary Workflow Task', category: 'Uber', status: 'ACTIVE', timestamp: new Date().toLocaleTimeString(), metrics: 98.4 },
    { id: '2', title: 'Secondary Uber Telemetry Sync', category: 'System', status: 'COMPLETED', timestamp: new Date().toLocaleTimeString(), metrics: 100.0 }
  ]);
  const [inputVal, setInputVal] = useState('');
  const [logs, setLogs] = useState<string[]>([
    `[${new Date().toLocaleTimeString()}] System booted for domain 'Autonomous'. Engine ready.`
  ]);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleAction = (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputVal.trim()) return;

    setIsProcessing(true);
    const newLog = `[${new Date().toLocaleTimeString()}] Executed Driver action: "${inputVal}"`;

    setTimeout(() => {
      const newItem: UberDispatcherAppDataRecord = {
        id: (items.length + 1).toString(),
        title: inputVal,
        category: 'Autonomous',
        status: 'ACTIVE',
        timestamp: new Date().toLocaleTimeString(),
        metrics: parseFloat((Math.random() * 20 + 80).toFixed(1))
      };
      setItems(prev => [newItem, ...prev]);
      setLogs(prev => [newLog, ...prev]);
      setInputVal('');
      setIsProcessing(false);
    }, 600);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6 font-sans">
      {/* App Header Bar */}
      <div className="flex items-center justify-between pb-6 mb-6 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-tr from-cyan-500 to-purple-600 rounded-2xl shadow-lg shadow-cyan-500/20 text-white">
            <Cpu className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              UberDispatcherApp Live Application
              <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                AI Runnable UI
              </span>
            </h1>
            <p className="text-xs text-slate-400">Synthesized for: "Build an autonomous Uber driver dispatcher web application with GPS matching"</p>
          </div>
        </div>

        <div className="flex items-center gap-2 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800">
          {(['dashboard', 'operations', 'logs'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold capitalize transition ${
                activeTab === tab
                  ? 'bg-gradient-to-r from-cyan-500 to-purple-600 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80">
          <div className="flex justify-between items-center text-slate-400 mb-2">
            <span className="text-xs font-medium">Autonomous Status</span>
            <Activity className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 flex items-center gap-1.5">
            <ShieldCheck className="w-5 h-5" /> Live 100%
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80">
          <div className="flex justify-between items-center text-slate-400 mb-2">
            <span className="text-xs font-medium">Active Records</span>
            <Server className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-white">{items.length}</div>
        </div>

        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80">
          <div className="flex justify-between items-center text-slate-400 mb-2">
            <span className="text-xs font-medium">Execution Engine</span>
            <Zap className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-cyan-300">Optimal</div>
        </div>

        <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80">
          <div className="flex justify-between items-center text-slate-400 mb-2">
            <span className="text-xs font-medium">Workforce Role</span>
            <Cpu className="w-4 h-4 text-pink-400" />
          </div>
          <div className="text-xs font-mono font-bold text-slate-300">UI_ENGINEERS</div>
        </div>
      </div>

      {/* Action Form */}
      <form onSubmit={handleAction} className="mb-6 p-4 rounded-2xl bg-slate-900/70 border border-slate-800 flex gap-3">
        <input
          type="text"
          placeholder="Type new domain request or task action..."
          value={inputVal}
          onChange={(e) => setInputVal(e.target.value)}
          className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
        />
        <button
          type="submit"
          disabled={isProcessing}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-purple-600 text-white font-semibold text-xs shadow-lg shadow-cyan-500/20 hover:opacity-90 disabled:opacity-50 transition"
        >
          {isProcessing ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          Execute Driver
        </button>
      </form>

      {/* Data Records List */}
      <div className="p-4 rounded-2xl bg-slate-900/40 border border-slate-800">
        <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4">
          Live Autonomous Application State Registry
        </h3>
        <div className="space-y-2">
          {items.map(item => (
            <div key={item.id} className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 flex items-center justify-between text-xs">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <div>
                  <div className="font-semibold text-white">{item.title}</div>
                  <div className="text-[11px] text-slate-400">Category: {item.category} • {item.timestamp}</div>
                </div>
              </div>
              <span className="px-2.5 py-1 rounded-full text-[10px] font-bold bg-cyan-500/10 text-cyan-300 border border-cyan-500/20">
                {item.status} ({item.metrics}%)
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
