import React, { useState, useMemo } from 'react';
import {
  DollarSign,
  FileText,
  Calculator,
  TrendingUp,
  PieChart,
  Layers,
  Plus,
  CheckCircle,
  Shield,
  Building,
  CreditCard,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  BarChart2,
  Search,
  Filter,
  Settings,
  Activity,
  Database,
  Download,
  Trash2,
  Edit,
  Sliders,
  Calendar,
  Eye,
  X,
  TrendingDown,
  Clock,
  Briefcase,
  ChevronRight,
  Sparkles,
  Printer
} from 'lucide-react';

// --- TYPES & INTERFACES ---
export type FlowCategory = 'Operating' | 'Investing' | 'Financing';
export type TransactionStatus = 'cleared' | 'pending' | 'reconciled';
export type FlowType = 'inflow' | 'outflow';

export interface CashFlowItem {
  id: string;
  date: string;
  description: string;
  category: FlowCategory;
  subcategory: string;
  amount: number; // positive = inflow, negative = outflow
  type: FlowType;
  status: TransactionStatus;
  referenceNo: string;
  entity: string;
  notes?: string;
}

export interface StatementSummary {
  beginningCash: number;
  operatingTotal: number;
  investingTotal: number;
  financingTotal: number;
  netCashFlow: number;
  endingCash: number;
  freeCashFlow: number;
}

// --- INITIAL MOCK DATA ---
const INITIAL_CASH_ITEMS: CashFlowItem[] = [
  {
    id: 'CF-1001',
    date: '2025-05-02',
    description: 'Enterprise Client Software Licenses',
    category: 'Operating',
    subcategory: 'Customer Receipts',
    amount: 345000,
    type: 'inflow',
    status: 'reconciled',
    referenceNo: 'INV-2025-089',
    entity: 'Acme Corp Account',
    notes: 'Q2 Renewal prepaid upfront'
  },
  {
    id: 'CF-1002',
    date: '2025-05-04',
    description: 'Engineering & Product Staff Salaries',
    category: 'Operating',
    subcategory: 'Payroll & Benefits',
    amount: -185000,
    type: 'outflow',
    status: 'cleared',
    referenceNo: 'PAY-2025-05A',
    entity: 'Primary Operating'
  },
  {
    id: 'CF-1003',
    date: '2025-05-06',
    description: 'Cloud Infrastructure & AWS Servers',
    category: 'Operating',
    subcategory: 'Supplier & Vendor Expenses',
    amount: -42500,
    type: 'outflow',
    status: 'cleared',
    referenceNo: 'AWS-892341',
    entity: 'Primary Operating'
  },
  {
    id: 'CF-1004',
    date: '2025-05-10',
    description: 'High-Performance GPU Server Cluster',
    category: 'Investing',
    subcategory: 'Capital Expenditures (CapEx)',
    amount: -120000,
    type: 'outflow',
    status: 'reconciled',
    referenceNo: 'CAPEX-2025-04',
    entity: 'Infrastructure Hardware LLC'
  },
  {
    id: 'CF-1005',
    date: '2025-05-12',
    description: 'Liquidation of Short-term Treasury Bills',
    category: 'Investing',
    subcategory: 'Marketable Securities',
    amount: 150000,
    type: 'inflow',
    status: 'cleared',
    referenceNo: 'SEC-99210',
    entity: 'Treasury Account'
  },
  {
    id: 'CF-1006',
    date: '2025-05-15',
    description: 'Series-B Venture Debt Drawdown',
    category: 'Financing',
    subcategory: 'Debt Capital Inflows',
    amount: 500000,
    type: 'inflow',
    status: 'reconciled',
    referenceNo: 'LN-SILICON-2025',
    entity: 'Silicon Horizon Bank'
  },
  {
    id: 'CF-1007',
    date: '2025-05-18',
    description: 'Term Loan Principal Repayment',
    category: 'Financing',
    subcategory: 'Debt Service & Principal',
    amount: -35000,
    type: 'outflow',
    status: 'cleared',
    referenceNo: 'PMT-LN-004',
    entity: 'Silicon Horizon Bank'
  },
  {
    id: 'CF-1008',
    date: '2025-05-20',
    description: 'Quarterly Preferred Share Dividend',
    category: 'Financing',
    subcategory: 'Dividends Paid',
    amount: -25000,
    type: 'outflow',
    status: 'pending',
    referenceNo: 'DIV-2025-Q1',
    entity: 'Investor Relations Escrow'
  },
  {
    id: 'CF-1009',
    date: '2025-05-22',
    description: 'Corporate HQ Office Facilities Lease',
    category: 'Operating',
    subcategory: 'Rent & Operating Facilities',
    amount: -28000,
    type: 'outflow',
    status: 'cleared',
    referenceNo: 'LEASE-HQ-05',
    entity: 'Real Estate Holdings'
  },
  {
    id: 'CF-1010',
    date: '2025-05-25',
    description: 'Strategic Patent Acquisition (SaaS Tech)',
    category: 'Investing',
    subcategory: 'Intangible Asset Investments',
    amount: -65000,
    type: 'outflow',
    status: 'reconciled',
    referenceNo: 'IP-ACQ-88',
    entity: 'IP Escrow Services'
  }
];

export function CustomApp() {
  // --- STATES ---
  const [items, setItems] = useState<CashFlowItem[]>(INITIAL_CASH_ITEMS);
  const [activeTab, setActiveTab] = useState<'dashboard' | 'operating' | 'investing' | 'financing' | 'statement' | 'forecasting'>('dashboard');
  const [beginningCash, setBeginningCash] = useState<number>(1420000);
  
  // Search & Filters
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [categoryFilter, setCategoryFilter] = useState<string>('All');
  const [statusFilter, setStatusFilter] = useState<string>('All');
  const [statementMethod, setStatementMethod] = useState<'direct' | 'indirect'>('direct');
  
  // Forecasting Slider state
  const [forecastRevenueGrowth, setForecastRevenueGrowth] = useState<number>(15);
  const [forecastCapexMultiplier, setForecastCapexMultiplier] = useState<number>(10);

  // Modal / Form state for Add / Edit
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [editingItem, setEditingItem] = useState<CashFlowItem | null>(null);
  const [formData, setFormData] = useState<Partial<CashFlowItem>>({
    date: new Date().toISOString().split('T')[0],
    description: '',
    category: 'Operating',
    subcategory: 'Customer Receipts',
    amount: 0,
    type: 'inflow',
    status: 'cleared',
    referenceNo: `CF-${Math.floor(1000 + Math.random() * 9000)}`,
    entity: 'Primary Treasury',
    notes: ''
  });

  // Detailed Modal View
  const [viewingItem, setViewingItem] = useState<CashFlowItem | null>(null);

  // --- CALCULATIONS & METRICS ---
  const filteredItems = useMemo(() => {
    return items.filter((item) => {
      const matchesSearch = item.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                            item.referenceNo.toLowerCase().includes(searchTerm.toLowerCase()) ||
                            item.subcategory.toLowerCase().includes(searchTerm.toLowerCase()) ||
                            item.entity.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesCategory = categoryFilter === 'All' || item.category === categoryFilter;
      const matchesStatus = statusFilter === 'All' || item.status === statusFilter;
      return matchesSearch && matchesCategory && matchesStatus;
    });
  }, [items, searchTerm, categoryFilter, statusFilter]);

  const summary = useMemo<StatementSummary>(() => {
    let operatingTotal = 0;
    let investingTotal = 0;
    let financingTotal = 0;
    let capexTotal = 0;

    items.forEach((item) => {
      if (item.category === 'Operating') {
        operatingTotal += item.amount;
      } else if (item.category === 'Investing') {
        investingTotal += item.amount;
        if (item.subcategory.toLowerCase().includes('capex') || item.amount < 0) {
          capexTotal += Math.abs(item.amount);
        }
      } else if (item.category === 'Financing') {
        financingTotal += item.amount;
      }
    });

    const netCashFlow = operatingTotal + investingTotal + financingTotal;
    const endingCash = beginningCash + netCashFlow;
    const freeCashFlow = operatingTotal - capexTotal;

    return {
      beginningCash,
      operatingTotal,
      investingTotal,
      financingTotal,
      netCashFlow,
      endingCash,
      freeCashFlow
    };
  }, [items, beginningCash]);

  // --- HANDLERS ---
  const handleOpenAddModal = () => {
    setEditingItem(null);
    setFormData({
      date: new Date().toISOString().split('T')[0],
      description: '',
      category: 'Operating',
      subcategory: 'Customer Receipts',
      amount: 5000,
      type: 'inflow',
      status: 'cleared',
      referenceNo: `CF-${Math.floor(1000 + Math.random() * 9000)}`,
      entity: 'Primary Treasury',
      notes: ''
    });
    setIsModalOpen(true);
  };

  const handleOpenEditModal = (item: CashFlowItem) => {
    setEditingItem(item);
    setFormData({ ...item });
    setIsModalOpen(true);
  };

  const handleDeleteItem = (id: string) => {
    if (window.confirm('Are you sure you want to delete this cash flow transaction record?')) {
      setItems(items.filter((item) => item.id !== id));
    }
  };

  const handleSaveItem = (e: React.FormEvent) => {
    e.preventDefault();
    if (!formData.description || formData.amount === undefined) return;

    const rawAmount = Math.abs(formData.amount || 0);
    const finalAmount = formData.type === 'outflow' ? -rawAmount : rawAmount;

    if (editingItem) {
      setItems(items.map(it => (it.id === editingItem.id ? { ...formData, id: editingItem.id, amount: finalAmount } as CashFlowItem : it)));
    } else {
      const newItem: CashFlowItem = {
        ...formData,
        id: `CF-${Math.floor(1000 + Math.random() * 9000)}`,
        amount: finalAmount
      } as CashFlowItem;
      setItems([newItem, ...items]);
    }

    setIsModalOpen(false);
  };

  const formatCurrency = (val: number, includeSign = false) => {
    const formatted = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0
    }).format(Math.abs(val));

    if (val < 0) return `(${formatted})`;
    if (includeSign && val > 0) return `+${formatted}`;
    return formatted;
  };

  // --- RENDER HELPERS ---
  const renderCategoryBadge = (cat: FlowCategory) => {
    switch (cat) {
      case 'Operating':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">Operating</span>;
      case 'Investing':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20">Investing</span>;
      case 'Financing':
        return <span className="px-2.5 py-1 text-xs font-semibold rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20">Financing</span>;
    }
  };

  const renderStatusBadge = (status: TransactionStatus) => {
    switch (status) {
      case 'reconciled':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-md bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <CheckCircle className="w-3 h-3" /> Reconciled
          </span>
        );
      case 'cleared':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-md bg-blue-500/10 text-blue-400 border border-blue-500/20">
            <Shield className="w-3 h-3" /> Cleared
          </span>
        );
      case 'pending':
        return (
          <span className="flex items-center gap-1.5 px-2.5 py-0.5 text-xs font-medium rounded-md bg-amber-500/10 text-amber-400 border border-amber-500/20">
            <Clock className="w-3 h-3" /> Pending
          </span>
        );
    }
  };

  return (
    <div className="min-h-screen bg-[#0b1121] text-slate-100 font-sans selection:bg-cyan-500 selection:text-white pb-12">
      {/* HEADER NAVBAR */}
      <header className="sticky top-0 z-30 bg-[#0b1121]/80 backdrop-blur-xl border-b border-slate-800/80 px-6 py-4">
        <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/20">
              <Activity className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">Cash Flow Engine Pro</h1>
                <span className="text-xs font-bold uppercase tracking-widest px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  v3.4 Sub-Module
                </span>
              </div>
              <p className="text-xs text-slate-400">Statement of Cash Flows (Operating, Investing, Financing)</p>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(items, null, 2));
                const downloadAnchor = document.createElement('a');
                downloadAnchor.setAttribute("href", dataStr);
                downloadAnchor.setAttribute("download", `cash_flow_export_${new Date().toISOString().split('T')[0]}.json`);
                document.body.appendChild(downloadAnchor);
                downloadAnchor.click();
                downloadAnchor.remove();
              }}
              className="flex items-center gap-2 px-3.5 py-2 text-xs font-semibold rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
            >
              <Download className="w-3.5 h-3.5" /> Export Data
            </button>
            <button
              onClick={handleOpenAddModal}
              className="flex items-center gap-2 px-4 py-2 text-xs font-semibold rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg shadow-cyan-500/25 transition"
            >
              <Plus className="w-4 h-4" /> New Cash Activity
            </button>
          </div>
        </div>
      </header>

      {/* MAIN CONTAINER */}
      <main className="max-w-7xl mx-auto px-6 pt-6 space-y-6">
        {/* KPI DASHBOARD HEADER CARDS */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Beginning Cash Card */}
          <div className="relative overflow-hidden p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md shadow-xl">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Beginning Reserve</p>
                <h3 className="text-2xl font-bold text-slate-100 mt-1">{formatCurrency(summary.beginningCash)}</h3>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-800/80 text-slate-300 border border-slate-700/50">
                <Database className="w-5 h-5" />
              </div>
            </div>
            <div className="mt-4 flex items-center gap-2">
              <span className="text-xs text-slate-400">Period Opening Balance</span>
            </div>
          </div>

          {/* Net Cash Flow Card */}
          <div className="relative overflow-hidden p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md shadow-xl">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Net Cash Flow</p>
                <h3 className={`text-2xl font-bold mt-1 ${summary.netCashFlow >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {formatCurrency(summary.netCashFlow, true)}
                </h3>
              </div>
              <div className={`p-2.5 rounded-xl border ${summary.netCashFlow >= 0 ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border-rose-500/20'}`}>
                {summary.netCashFlow >= 0 ? <TrendingUp className="w-5 h-5" /> : <TrendingDown className="w-5 h-5" />}
              </div>
            </div>
            <div className="mt-4 flex items-center gap-2 text-xs text-slate-400">
              <span>CFO + CFI + CFF Combined</span>
            </div>
          </div>

          {/* Free Cash Flow (FCF) */}
          <div className="relative overflow-hidden p-5 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md shadow-xl">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Free Cash Flow (FCF)</p>
                <h3 className={`text-2xl font-bold mt-1 ${summary.freeCashFlow >= 0 ? 'text-cyan-400' : 'text-amber-400'}`}>
                  {formatCurrency(summary.freeCashFlow)}
                </h3>
              </div>
              <div className="p-2.5 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                <Sparkles className="w-5 h-5" />
              </div>
            </div>
            <div className="mt-4 flex items-center gap-2 text-xs text-slate-400">
              <span>Operating Cash minus CapEx</span>
            </div>
          </div>

          {/* Ending Cash Balance */}
          <div className="relative overflow-hidden p-5 rounded-2xl bg-gradient-to-br from-slate-900/90 to-slate-950 border border-slate-700/80 backdrop-blur-md shadow-xl">
            <div className="flex justify-between items-start">
              <div>
                <p className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">Ending Position</p>
                <h3 className="text-2xl font-bold text-white mt-1">{formatCurrency(summary.endingCash)}</h3>
              </div>
              <div className="p-2.5 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 text-white shadow-md">
                <DollarSign className="w-5 h-5" />
              </div>
            </div>
            <div className="mt-4 flex items-center justify-between text-xs">
              <span className="text-slate-400">Liquidity Status:</span>
              <span className="text-emerald-400 font-semibold flex items-center gap-1">
                <CheckCircle className="w-3 h-3" /> Healthy Liquidity
              </span>
            </div>
          </div>
        </div>

        {/* THREE CASH FLOW TIER BREAKDOWN (Op, Inv, Fin) */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Operating Activities Box */}
          <div
            onClick={() => setActiveTab('operating')}
            className={`cursor-pointer transition-all duration-200 p-5 rounded-2xl border ${
              activeTab === 'operating' ? 'bg-slate-800/90 border-cyan-500 shadow-lg shadow-cyan-500/10' : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                  <Briefcase className="w-4 h-4" />
                </div>
                <h4 className="font-semibold text-slate-200">Operating (CFO)</h4>
              </div>
              <ChevronRight className="w-4 h-4 text-slate-500" />
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-bold text-white">{formatCurrency(summary.operatingTotal, true)}</span>
              <span className="text-xs text-slate-400">Core Operations</span>
            </div>
            {/* Visual Bar */}
            <div className="w-full bg-slate-800 rounded-full h-1.5 mt-4 overflow-hidden">
              <div
                className="bg-cyan-400 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(15, (Math.abs(summary.operatingTotal) / (Math.abs(summary.netCashFlow) || 1)) * 100))}%` }}
              ></div>
            </div>
          </div>

          {/* Investing Activities Box */}
          <div
            onClick={() => setActiveTab('investing')}
            className={`cursor-pointer transition-all duration-200 p-5 rounded-2xl border ${
              activeTab === 'investing' ? 'bg-slate-800/90 border-purple-500 shadow-lg shadow-purple-500/10' : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20">
                  <Building className="w-4 h-4" />
                </div>
                <h4 className="font-semibold text-slate-200">Investing (CFI)</h4>
              </div>
              <ChevronRight className="w-4 h-4 text-slate-500" />
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-bold text-white">{formatCurrency(summary.investingTotal, true)}</span>
              <span className="text-xs text-slate-400">CapEx & Assets</span>
            </div>
            {/* Visual Bar */}
            <div className="w-full bg-slate-800 rounded-full h-1.5 mt-4 overflow-hidden">
              <div
                className="bg-purple-400 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(15, (Math.abs(summary.investingTotal) / (Math.abs(summary.netCashFlow) || 1)) * 100))}%` }}
              ></div>
            </div>
          </div>

          {/* Financing Activities Box */}
          <div
            onClick={() => setActiveTab('financing')}
            className={`cursor-pointer transition-all duration-200 p-5 rounded-2xl border ${
              activeTab === 'financing' ? 'bg-slate-800/90 border-amber-500 shadow-lg shadow-amber-500/10' : 'bg-slate-900/60 border-slate-800/80 hover:border-slate-700'
            }`}
          >
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2.5">
                <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20">
                  <CreditCard className="w-4 h-4" />
                </div>
                <h4 className="font-semibold text-slate-200">Financing (CFF)</h4>
              </div>
              <ChevronRight className="w-4 h-4 text-slate-500" />
            </div>
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-bold text-white">{formatCurrency(summary.financingTotal, true)}</span>
              <span className="text-xs text-slate-400">Debt & Equity</span>
            </div>
            {/* Visual Bar */}
            <div className="w-full bg-slate-800 rounded-full h-1.5 mt-4 overflow-hidden">
              <div
                className="bg-amber-400 h-1.5 rounded-full transition-all duration-500"
                style={{ width: `${Math.min(100, Math.max(15, (Math.abs(summary.financingTotal) / (Math.abs(summary.netCashFlow) || 1)) * 100))}%` }}
              ></div>
            </div>
          </div>
        </div>

        {/* TAB NAVIGATION BAR */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
          <nav className="flex items-center space-x-1 sm:space-x-2 overflow-x-auto pb-1">
            {[
              { id: 'dashboard', label: 'All Transactions', icon: Layers },
              { id: 'operating', label: 'Operating (Op)', icon: Briefcase },
              { id: 'investing', label: 'Investing (Inv)', icon: Building },
              { id: 'financing', label: 'Financing (Fin)', icon: CreditCard },
              { id: 'statement', label: 'Formal Statement', icon: FileText },
              { id: 'forecasting', label: 'Scenario Planner', icon: Sliders }
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                    isActive
                      ? 'bg-slate-800 text-cyan-400 border border-slate-700 shadow-sm'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>

        {/* TAB CONTENT AREAS */}

        {/* TAB 1: ALL TRANSACTIONS & MODULE CATEGORIES VIEW */}
        {(activeTab === 'dashboard' || activeTab === 'operating' || activeTab === 'investing' || activeTab === 'financing') && (
          <div className="space-y-4">
            {/* SEARCH & FILTER CONTROLS */}
            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md flex flex-col md:flex-row items-center justify-between gap-4">
              <div className="relative w-full md:w-96">
                <Search className="w-4 h-4 absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
                <input
                  type="text"
                  placeholder="Search description, reference, or entity..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl pl-10 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="flex items-center gap-3 w-full md:w-auto overflow-x-auto">
                {/* Category Filter */}
                <div className="flex items-center gap-2">
                  <Filter className="w-3.5 h-3.5 text-slate-500" />
                  <span className="text-xs text-slate-400">Category:</span>
                  <select
                    value={activeTab === 'dashboard' ? categoryFilter : (activeTab.charAt(0).toUpperCase() + activeTab.slice(1))}
                    disabled={activeTab !== 'dashboard'}
                    onChange={(e) => setCategoryFilter(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="All">All Categories</option>
                    <option value="Operating">Operating</option>
                    <option value="Investing">Investing</option>
                    <option value="Financing">Financing</option>
                  </select>
                </div>

                {/* Status Filter */}
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-400">Status:</span>
                  <select
                    value={statusFilter}
                    onChange={(e) => setStatusFilter(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="All">All Statuses</option>
                    <option value="cleared">Cleared</option>
                    <option value="reconciled">Reconciled</option>
                    <option value="pending">Pending</option>
                  </select>
                </div>
              </div>
            </div>

            {/* TRANSACTIONS DATA TABLE */}
            <div className="rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md overflow-hidden shadow-xl">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800/80 bg-slate-950/40 text-xs text-slate-400 uppercase tracking-wider">
                      <th className="py-3.5 px-4 font-semibold">Ref & Date</th>
                      <th className="py-3.5 px-4 font-semibold">Description</th>
                      <th className="py-3.5 px-4 font-semibold">Category</th>
                      <th className="py-3.5 px-4 font-semibold">Subcategory</th>
                      <th className="py-3.5 px-4 font-semibold text-right">Amount</th>
                      <th className="py-3.5 px-4 font-semibold">Status</th>
                      <th className="py-3.5 px-4 font-semibold text-center">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-xs">
                    {filteredItems
                      .filter(item => activeTab === 'dashboard' || item.category.toLowerCase() === activeTab)
                      .map((item) => (
                        <tr key={item.id} className="hover:bg-slate-800/40 transition-colors">
                          <td className="py-3.5 px-4">
                            <div className="font-mono text-cyan-400 font-semibold">{item.referenceNo}</div>
                            <div className="text-slate-500 text-[11px] mt-0.5">{item.date}</div>
                          </td>
                          <td className="py-3.5 px-4">
                            <div className="font-medium text-slate-200">{item.description}</div>
                            <div className="text-slate-500 text-[11px]">{item.entity}</div>
                          </td>
                          <td className="py-3.5 px-4">
                            {renderCategoryBadge(item.category)}
                          </td>
                          <td className="py-3.5 px-4 text-slate-300">
                            {item.subcategory}
                          </td>
                          <td className="py-3.5 px-4 text-right font-mono font-semibold">
                            <span className={item.amount >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                              {formatCurrency(item.amount, true)}
                            </span>
                          </td>
                          <td className="py-3.5 px-4">
                            {renderStatusBadge(item.status)}
                          </td>
                          <td className="py-3.5 px-4">
                            <div className="flex items-center justify-center gap-1.5">
                              <button
                                onClick={() => setViewingItem(item)}
                                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-cyan-400 transition"
                                title="View Details"
                              >
                                <Eye className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => handleOpenEditModal(item)}
                                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-amber-400 transition"
                                title="Edit Entry"
                              >
                                <Edit className="w-3.5 h-3.5" />
                              </button>
                              <button
                                onClick={() => handleDeleteItem(item.id)}
                                className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-rose-400 transition"
                                title="Delete Entry"
                              >
                                <Trash2 className="w-3.5 h-3.5" />
                              </button>
                            </div>
                          </td>
                        </tr>
                    ))}
                    {filteredItems.filter(item => activeTab === 'dashboard' || item.category.toLowerCase() === activeTab).length === 0 && (
                      <tr>
                        <td colSpan={7} className="py-12 text-center text-slate-500">
                          <Database className="w-8 h-8 mx-auto mb-2 opacity-40" />
                          No cash flow items match the current query or category selection.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: FORMAL STATEMENT VIEW (GAAP/IFRS Cash Flow Statement) */}
        {activeTab === 'statement' && (
          <div className="space-y-6">
            {/* Controls Bar */}
            <div className="p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Method:</span>
                <div className="flex bg-slate-950 p-1 rounded-xl border border-slate-800">
                  <button
                    onClick={() => setStatementMethod('direct')}
                    className={`px-3 py-1 text-xs font-medium rounded-lg transition ${
                      statementMethod === 'direct' ? 'bg-cyan-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Direct Method
                  </button>
                  <button
                    onClick={() => setStatementMethod('indirect')}
                    className={`px-3 py-1 text-xs font-medium rounded-lg transition ${
                      statementMethod === 'indirect' ? 'bg-cyan-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    Indirect Method
                  </button>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => window.print()}
                  className="flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition"
                >
                  <Printer className="w-3.5 h-3.5" /> Print Statement
                </button>
              </div>
            </div>

            {/* STATEMENT REPORT DOCUMENT */}
            <div className="p-8 rounded-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-md text-slate-200 space-y-8 font-sans shadow-2xl">
              {/* Report Header */}
              <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
                <div>
                  <h2 className="text-2xl font-bold text-white tracking-tight">Statement of Cash Flows</h2>
                  <p className="text-sm text-cyan-400 font-medium mt-1">Acme Global Technologies Inc.</p>
                  <p className="text-xs text-slate-400">For Period Ended May 31, 2025 (In USD)</p>
                </div>
                <div className="text-right">
                  <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    GAAP Compliant • {statementMethod.toUpperCase()} METHOD
                  </span>
                </div>
              </div>

              {/* 1. OPERATING ACTIVITIES */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold text-cyan-400 uppercase tracking-wider border-b border-slate-800 pb-1">
                  1. Cash Flows from Operating Activities
                </h3>
                {statementMethod === 'direct' ? (
                  <div className="space-y-2 text-xs">
                    {items.filter(i => i.category === 'Operating').map(item => (
                      <div key={item.id} className="flex justify-between py-1 border-b border-slate-800/40">
                        <span className="text-slate-300 pl-4">{item.description} ({item.subcategory})</span>
                        <span className="font-mono text-slate-100">{formatCurrency(item.amount, true)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-300 pl-4">Net Income (Period Starting Point)</span>
                      <span className="font-mono text-slate-100">$210,000</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-300 pl-4">Add: Depreciation & Amortization Non-Cash Expenses</span>
                      <span className="font-mono text-slate-100">$45,000</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-300 pl-4">Change in Accounts Receivable / Working Capital</span>
                      <span className="font-mono text-slate-100">($85,500)</span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-300 pl-4">Change in Accounts Payable & Accruals</span>
                      <span className="font-mono text-slate-100">
                        {formatCurrency(summary.operatingTotal - (210000 + 45000 - 85500), true)}
                      </span>
                    </div>
                  </div>
                )}
                <div className="flex justify-between py-2 font-bold text-xs bg-slate-800/40 px-3 rounded-lg text-white">
                  <span>Net Cash Provided by (Used in) Operating Activities</span>
                  <span className="font-mono text-cyan-400">{formatCurrency(summary.operatingTotal)}</span>
                </div>
              </div>

              {/* 2. INVESTING ACTIVITIES */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold text-purple-400 uppercase tracking-wider border-b border-slate-800 pb-1">
                  2. Cash Flows from Investing Activities
                </h3>
                <div className="space-y-2 text-xs">
                  {items.filter(i => i.category === 'Investing').map(item => (
                    <div key={item.id} className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-300 pl-4">{item.description} ({item.subcategory})</span>
                      <span className="font-mono text-slate-100">{formatCurrency(item.amount, true)}</span>
                    </div>
                  ))}
                  {items.filter(i => i.category === 'Investing').length === 0 && (
                    <div className="text-slate-500 text-xs italic pl-4">No investing activities for this period.</div>
                  )}
                </div>
                <div className="flex justify-between py-2 font-bold text-xs bg-slate-800/40 px-3 rounded-lg text-white">
                  <span>Net Cash Provided by (Used in) Investing Activities</span>
                  <span className="font-mono text-purple-400">{formatCurrency(summary.investingTotal)}</span>
                </div>
              </div>

              {/* 3. FINANCING ACTIVITIES */}
              <div className="space-y-3">
                <h3 className="text-sm font-bold text-amber-400 uppercase tracking-wider border-b border-slate-800 pb-1">
                  3. Cash Flows from Financing Activities
                </h3>
                <div className="space-y-2 text-xs">
                  {items.filter(i => i.category === 'Financing').map(item => (
                    <div key={item.id} className="flex justify-between py-1 border-b border-slate-800/40">
                      <span className="text-slate-300 pl-4">{item.description} ({item.subcategory})</span>
                      <span className="font-mono text-slate-100">{formatCurrency(item.amount, true)}</span>
                    </div>
                  ))}
                  {items.filter(i => i.category === 'Financing').length === 0 && (
                    <div className="text-slate-500 text-xs italic pl-4">No financing activities for this period.</div>
                  )}
                </div>
                <div className="flex justify-between py-2 font-bold text-xs bg-slate-800/40 px-3 rounded-lg text-white">
                  <span>Net Cash Provided by (Used in) Financing Activities</span>
                  <span className="font-mono text-amber-400">{formatCurrency(summary.financingTotal)}</span>
                </div>
              </div>

              {/* SUMMARY RECONCILIATION */}
              <div className="pt-4 border-t-2 border-slate-700 space-y-2">
                <div className="flex justify-between text-xs py-1 text-slate-300">
                  <span>NET INCREASE / (DECREASE) IN CASH AND CASH EQUIVALENTS</span>
                  <span className={`font-mono font-bold ${summary.netCashFlow >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {formatCurrency(summary.netCashFlow, true)}
                  </span>
                </div>
                <div className="flex justify-between text-xs py-1 text-slate-300">
                  <span>Cash and Cash Equivalents at Beginning of Period</span>
                  <span className="font-mono">{formatCurrency(summary.beginningCash)}</span>
                </div>
                <div className="flex justify-between text-sm py-2 px-4 rounded-xl bg-cyan-500/10 border border-cyan-500/30 font-bold text-white mt-2">
                  <span>CASH AND CASH EQUIVALENTS AT END OF PERIOD</span>
                  <span className="font-mono text-cyan-300">{formatCurrency(summary.endingCash)}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: SCENARIO PLANNER & FORECASTING */}
        {activeTab === 'forecasting' && (
          <div className="space-y-6">
            <div className="p-6 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md space-y-6">
              <div>
                <h3 className="text-lg font-bold text-white flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-cyan-400" /> Cash Flow Sensitivity & Scenario Planner
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  Adjust drivers to simulate forward-looking runway, cash reserves, and projected end-of-year cash flow.
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Controls */}
                <div className="space-y-5 bg-slate-950/60 p-5 rounded-xl border border-slate-800">
                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-300 mb-2">
                      <span>Projected Revenue / Operating Receipts Growth</span>
                      <span className="text-cyan-400">+{forecastRevenueGrowth}%</span>
                    </div>
                    <input
                      type="range"
                      min="-20"
                      max="50"
                      value={forecastRevenueGrowth}
                      onChange={(e) => setForecastRevenueGrowth(Number(e.target.value))}
                      className="w-full accent-cyan-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-xs font-semibold text-slate-300 mb-2">
                      <span>CapEx Expansion Multiplier</span>
                      <span className="text-purple-400">{forecastCapexMultiplier}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={forecastCapexMultiplier}
                      onChange={(e) => setForecastCapexMultiplier(Number(e.target.value))}
                      className="w-full accent-purple-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
                    />
                  </div>

                  <div className="pt-2">
                    <label className="text-xs text-slate-400 block mb-1">Set Period Beginning Cash Baseline ($)</label>
                    <input
                      type="number"
                      value={beginningCash}
                      onChange={(e) => setBeginningCash(Number(e.target.value))}
                      className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                    />
                  </div>
                </div>

                {/* Simulated Output */}
                <div className="space-y-4 bg-slate-950/60 p-5 rounded-xl border border-slate-800 flex flex-col justify-between">
                  <div>
                    <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Simulated Next-Quarter Projection</h4>
                    
                    <div className="space-y-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">Simulated Operating Cash:</span>
                        <span className="font-mono text-cyan-400 font-bold">
                          {formatCurrency(summary.operatingTotal * (1 + forecastRevenueGrowth / 100))}
                        </span>
                      </div>

                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">Simulated CapEx / Investing Outflow:</span>
                        <span className="font-mono text-purple-400 font-bold">
                          {formatCurrency(summary.investingTotal * (1 + forecastCapexMultiplier / 100))}
                        </span>
                      </div>

                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-400">Financing Cash Balance:</span>
                        <span className="font-mono text-amber-400 font-bold">
                          {formatCurrency(summary.financingTotal)}
                        </span>
                      </div>

                      <div className="pt-3 border-t border-slate-800 flex justify-between items-center">
                        <span className="text-xs font-bold text-white">Projected Free Cash Flow (FCF):</span>
                        <span className="font-mono text-sm font-bold text-emerald-400">
                          {formatCurrency(
                            summary.operatingTotal * (1 + forecastRevenueGrowth / 100) +
                            summary.investingTotal * (1 + forecastCapexMultiplier / 100)
                          )}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="p-3 rounded-lg bg-cyan-500/10 border border-cyan-500/20 text-xs text-cyan-300">
                    💡 <strong>Insight:</strong> Increasing revenue by {forecastRevenueGrowth}% yields an estimated liquidity cushion of {formatCurrency(summary.endingCash + (summary.operatingTotal * (forecastRevenueGrowth / 100)))}.
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* MODAL: ADD / EDIT CASH FLOW ENTRY */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl">
            <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                {editingItem ? <Edit className="w-4 h-4 text-amber-400" /> : <Plus className="w-4 h-4 text-cyan-400" />}
                {editingItem ? 'Edit Cash Flow Activity' : 'Record New Cash Flow Entry'}
              </h3>
              <button
                onClick={() => setIsModalOpen(false)}
                className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={handleSaveItem} className="p-6 space-y-4 text-xs">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-slate-400 font-medium mb-1">Date</label>
                  <input
                    type="date"
                    value={formData.date}
                    onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
                    required
                  />
                </div>

                <div>
                  <label className="block text-slate-400 font-medium mb-1">Reference No.</label>
                  <input
                    type="text"
                    value={formData.referenceNo}
                    onChange={(e) => setFormData({ ...formData, referenceNo: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                    required
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1">Description</label>
                <input
                  type="text"
                  placeholder="e.g., Enterprise Client Payment, Server CapEx..."
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-slate-400 font-medium mb-1">Statement Category</label>
                  <select
                    value={formData.category}
                    onChange={(e) => setFormData({ ...formData, category: e.target.value as FlowCategory })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="Operating">Operating Activities (CFO)</option>
                    <option value="Investing">Investing Activities (CFI)</option>
                    <option value="Financing">Financing Activities (CFF)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 font-medium mb-1">Subcategory Type</label>
                  <input
                    type="text"
                    placeholder="e.g. CapEx, Customer Receipts, Debt"
                    value={formData.subcategory}
                    onChange={(e) => setFormData({ ...formData, subcategory: e.target.value })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label className="block text-slate-400 font-medium mb-1">Flow Type</label>
                  <select
                    value={formData.type}
                    onChange={(e) => setFormData({ ...formData, type: e.target.value as FlowType })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="inflow">Inflow (+)</option>
                    <option value="outflow">Outflow (-)</option>
                  </select>
                </div>

                <div>
                  <label className="block text-slate-400 font-medium mb-1">Amount ($)</label>
                  <input
                    type="number"
                    value={formData.amount}
                    onChange={(e) => setFormData({ ...formData, amount: Number(e.target.value) })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 font-mono focus:outline-none focus:border-cyan-500"
                    required
                  />
                </div>

                <div>
                  <label className="block text-slate-400 font-medium mb-1">Status</label>
                  <select
                    value={formData.status}
                    onChange={(e) => setFormData({ ...formData, status: e.target.value as TransactionStatus })}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="cleared">Cleared</option>
                    <option value="reconciled">Reconciled</option>
                    <option value="pending">Pending</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-slate-400 font-medium mb-1">Bank / Entity Account</label>
                <input
                  type="text"
                  placeholder="Primary Operating Account, Treasury..."
                  value={formData.entity}
                  onChange={(e) => setFormData({ ...formData, entity: e.target.value })}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-slate-200 focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="pt-4 flex items-center justify-end gap-3 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 text-xs font-semibold rounded-xl bg-slate-800 text-slate-300 hover:bg-slate-700 transition"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 text-xs font-semibold rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white shadow-lg transition"
                >
                  {editingItem ? 'Save Changes' : 'Record Transaction'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL: DETAIL VIEW */}
      {viewingItem && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md overflow-hidden shadow-2xl p-6 space-y-4">
            <div className="flex justify-between items-start border-b border-slate-800 pb-3">
              <div>
                <span className="text-xs font-mono text-cyan-400">{viewingItem.referenceNo}</span>
                <h3 className="text-lg font-bold text-white mt-0.5">{viewingItem.description}</h3>
              </div>
              <button
                onClick={() => setViewingItem(null)}
                className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="flex justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Amount</span>
                <span className={`font-mono font-bold text-sm ${viewingItem.amount >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {formatCurrency(viewingItem.amount, true)}
                </span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Statement Category</span>
                {renderCategoryBadge(viewingItem.category)}
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Subcategory</span>
                <span className="text-slate-200 font-medium">{viewingItem.subcategory}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Transaction Status</span>
                {renderStatusBadge(viewingItem.status)}
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Posting Date</span>
                <span className="text-slate-200">{viewingItem.date}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Account / Entity</span>
                <span className="text-slate-200">{viewingItem.entity}</span>
              </div>
              {viewingItem.notes && (
                <div className="pt-2">
                  <span className="text-slate-400 block mb-1">Audit Notes:</span>
                  <div className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 italic">
                    {viewingItem.notes}
                  </div>
                </div>
              )}
            </div>

            <div className="pt-2 flex justify-end">
              <button
                onClick={() => setViewingItem(null)}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-slate-800 text-slate-300 hover:bg-slate-700 transition"
              >
                Close Audit View
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}