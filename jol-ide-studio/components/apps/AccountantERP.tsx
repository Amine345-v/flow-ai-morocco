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
  AlertTriangle,
  Scale,
  X,
  Download,
  Lock,
  CheckCircle2,
  ChevronRight,
  BookOpen,
  Trash2,
  UserCheck,
  Calendar,
  FileSpreadsheet
} from 'lucide-react';

// --- TYPES ---
type TabType = 'dashboard' | 'coa' | 'journal' | 'invoices' | 'statements';
type StatementSubTab = 'income' | 'balance' | 'vat';
type AccountType = 'Asset' | 'Liability' | 'Equity' | 'Revenue' | 'Expense';

interface Account {
  code: string;
  name: string;
  type: AccountType;
  balance: number;
  status: 'Active' | 'Archived';
}

interface JournalLine {
  id: string;
  accountCode: string;
  accountName: string;
  debit: number;
  credit: number;
}

interface JournalEntry {
  id: string;
  ref: string;
  date: string;
  description: string;
  lines: JournalLine[];
  status: 'Posted' | 'Pending Audit';
}

interface Invoice {
  id: string;
  invoiceNumber: string;
  entityName: string;
  type: 'AR' | 'AP'; // Accounts Receivable vs Accounts Payable
  date: string;
  dueDate: string;
  subtotal: number;
  vatAmount: number;
  total: number;
  status: 'Paid' | 'Overdue' | 'Draft' | 'Sent';
}

export default function AccountantERP() {
  // --- STATE MANAGEMENT ---
  const [activeTab, setActiveTab] = useState<TabType>('dashboard');
  const [statementTab, setStatementTab] = useState<StatementSubTab>('income');

  // Modal States
  const [isNewAccountModalOpen, setIsNewAccountModalOpen] = useState(false);
  const [isNewJournalModalOpen, setIsNewJournalModalOpen] = useState(false);
  const [isNewInvoiceModalOpen, setIsNewInvoiceModalOpen] = useState(false);

  // Filter States
  const [coaFilter, setCoaFilter] = useState<string>('All');
  const [coaSearch, setCoaSearch] = useState<string>('');
  const [invoiceTypeFilter, setInvoiceTypeFilter] = useState<'ALL' | 'AR' | 'AP'>('ALL');

  // --- INITIAL DATA STORES ---
  const [accounts, setAccounts] = useState<Account[]>([
    { code: '1010', name: 'Operating Cash Account', type: 'Asset', balance: 520000, status: 'Active' },
    { code: '1020', name: 'Accounts Receivable (AR)', type: 'Asset', balance: 380000, status: 'Active' },
    { code: '1050', name: 'Short-term Treasury Investments', type: 'Asset', balance: 550000, status: 'Active' },
    { code: '2010', name: 'Accounts Payable (AP)', type: 'Liability', balance: 185500, status: 'Active' },
    { code: '2030', name: 'VAT / Sales Tax Payable', type: 'Liability', balance: 34500, status: 'Active' },
    { code: '2050', name: 'Accrued Operating Expenses', type: 'Liability', balance: 200000, status: 'Active' },
    { code: '3010', name: 'Common Share Capital', type: 'Equity', balance: 600000, status: 'Active' },
    { code: '3020', name: 'Retained Earnings', type: 'Equity', balance: 430000, status: 'Active' },
    { code: '4010', name: 'SaaS Software Subscriptions', type: 'Revenue', balance: 480000, status: 'Active' },
    { code: '4020', name: 'Professional Services Revenue', type: 'Revenue', balance: 200000, status: 'Active' },
    { code: '5010', name: 'Payroll & Compensation', type: 'Expense', balance: 290000, status: 'Active' },
    { code: '5020', name: 'Cloud Infrastructure & Hosting', type: 'Expense', balance: 110000, status: 'Active' },
    { code: '5030', name: 'Office Operations & Lease', type: 'Expense', balance: 86800, status: 'Active' }
  ]);

  const [journalEntries, setJournalEntries] = useState<JournalEntry[]>([
    {
      id: 'JE-1001',
      ref: 'JV-2024-089',
      date: '2024-10-24',
      description: 'Client retainer invoice - Enterprise Contract',
      lines: [
        { id: '1', accountCode: '1020', accountName: 'Accounts Receivable (AR)', debit: 45000, credit: 0 },
        { id: '2', accountCode: '4010', accountName: 'SaaS Software Subscriptions', debit: 0, credit: 37500 },
        { id: '3', accountCode: '2030', accountName: 'VAT / Sales Tax Payable', debit: 0, credit: 7500 }
      ],
      status: 'Posted'
    },
    {
      id: 'JE-1002',
      ref: 'JV-2024-090',
      date: '2024-10-25',
      description: 'AWS Cloud Hosting Monthly Settlement',
      lines: [
        { id: '1', accountCode: '5020', accountName: 'Cloud Infrastructure & Hosting', debit: 12500, credit: 0 },
        { id: '2', accountCode: '1010', accountName: 'Operating Cash Account', debit: 0, credit: 12500 }
      ],
      status: 'Posted'
    },
    {
      id: 'JE-1003',
      ref: 'JV-2024-091',
      date: '2024-10-26',
      description: 'Payroll Executive Compensation Clearing',
      lines: [
        { id: '1', accountCode: '5010', accountName: 'Payroll & Compensation', debit: 65000, credit: 0 },
        { id: '2', accountCode: '1010', accountName: 'Operating Cash Account', debit: 0, credit: 65000 }
      ],
      status: 'Posted'
    }
  ]);

  const [invoices, setInvoices] = useState<Invoice[]>([
    { id: 'INV-001', invoiceNumber: 'INV-2024-001', entityName: 'Acme Global Corp', type: 'AR', date: '2024-10-01', dueDate: '2024-10-31', subtotal: 50000, vatAmount: 10000, total: 60000, status: 'Paid' },
    { id: 'INV-002', invoiceNumber: 'INV-2024-002', entityName: 'Nexus Tech Systems', type: 'AR', date: '2024-10-15', dueDate: '2024-11-15', subtotal: 28000, vatAmount: 5600, total: 33600, status: 'Sent' },
    { id: 'INV-003', invoiceNumber: 'INV-2024-003', entityName: 'Vanguard Dynamics', type: 'AR', date: '2024-09-10', dueDate: '2024-10-10', subtotal: 42000, vatAmount: 8400, total: 50400, status: 'Overdue' },
    { id: 'BILL-101', invoiceNumber: 'BILL-2024-088', entityName: 'Cloudflare Inc', type: 'AP', date: '2024-10-18', dueDate: '2024-11-01', subtotal: 8000, vatAmount: 1600, total: 9600, status: 'Sent' },
    { id: 'BILL-102', invoiceNumber: 'BILL-2024-092', entityName: 'Deloitte Advisory', type: 'AP', date: '2024-10-20', dueDate: '2024-11-20', subtotal: 15000, vatAmount: 3000, total: 18000, status: 'Draft' }
  ]);

  // --- FORM STATES FOR MODALS ---
  // Account Form
  const [newAccCode, setNewAccCode] = useState('');
  const [newAccName, setNewAccName] = useState('');
  const [newAccType, setNewAccType] = useState<AccountType>('Asset');
  const [newAccBalance, setNewAccBalance] = useState('');

  // Journal Entry Form
  const [newJeRef, setNewJeRef] = useState(`JV-2024-0${journalEntries.length + 92}`);
  const [newJeDesc, setNewJeDesc] = useState('');
  const [newJeDate, setNewJeDate] = useState(new Date().toISOString().split('T')[0]);
  const [newJeLines, setNewJeLines] = useState<JournalLine[]>([
    { id: '1', accountCode: '1010', accountName: 'Operating Cash Account', debit: 0, credit: 0 },
    { id: '2', accountCode: '4010', accountName: 'SaaS Software Subscriptions', debit: 0, credit: 0 }
  ]);

  // Invoice Form
  const [newInvNum, setNewInvNum] = useState(`INV-2024-00${invoices.length + 1}`);
  const [newInvEntity, setNewInvEntity] = useState('');
  const [newInvType, setNewInvType] = useState<'AR' | 'AP'>('AR');
  const [newInvSubtotal, setNewInvSubtotal] = useState('');

  // --- STATS COMPUTATIONS ---
  const kpiData = useMemo(() => {
    return {
      totalAssets: 1450000,
      totalLiabilities: 420000,
      equity: 1030000,
      revenueYTD: 680000,
      netProfitMargin: 28.4,
      vatPayable: 34500
    };
  }, []);

  const totalJournalDebits = useMemo(() => {
    return newJeLines.reduce((acc, line) => acc + (Number(line.debit) || 0), 0);
  }, [newJeLines]);

  const totalJournalCredits = useMemo(() => {
    return newJeLines.reduce((acc, line) => acc + (Number(line.credit) || 0), 0);
  }, [newJeLines]);

  const isJournalBalanced = useMemo(() => {
    return totalJournalDebits > 0 && Math.abs(totalJournalDebits - totalJournalCredits) < 0.001;
  }, [totalJournalDebits, totalJournalCredits]);

  // --- HANDLERS ---
  const handleAddAccount = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newAccCode || !newAccName) return;
    const newAcc: Account = {
      code: newAccCode,
      name: newAccName,
      type: newAccType,
      balance: parseFloat(newAccBalance) || 0,
      status: 'Active'
    };
    setAccounts([...accounts, newAcc]);
    setIsNewAccountModalOpen(false);
    setNewAccCode('');
    setNewAccName('');
    setNewAccBalance('');
  };

  const handleAddJournalLine = () => {
    setNewJeLines([
      ...newJeLines,
      {
        id: Date.now().toString(),
        accountCode: accounts[0]?.code || '1010',
        accountName: accounts[0]?.name || 'Operating Cash Account',
        debit: 0,
        credit: 0
      }
    ]);
  };

  const handleRemoveJournalLine = (id: string) => {
    if (newJeLines.length <= 2) return; // Keep min 2 lines for double entry
    setNewJeLines(newJeLines.filter((l) => l.id !== id));
  };

  const handleJournalLineChange = (id: string, field: keyof JournalLine, value: any) => {
    setNewJeLines(
      newJeLines.map((line) => {
        if (line.id === id) {
          if (field === 'accountCode') {
            const acc = accounts.find((a) => a.code === value);
            return { ...line, accountCode: value, accountName: acc ? acc.name : line.accountName };
          }
          return { ...line, [field]: value };
        }
        return line;
      })
    );
  };

  const handlePostJournalEntry = (e: React.FormEvent) => {
    e.preventDefault();
    if (!isJournalBalanced) return;

    const newEntry: JournalEntry = {
      id: `JE-${1000 + journalEntries.length + 1}`,
      ref: newJeRef,
      date: newJeDate,
      description: newJeDesc || 'Standard Ledger Adjustment',
      lines: newJeLines.map((l) => ({
        ...l,
        debit: Number(l.debit) || 0,
        credit: Number(l.credit) || 0
      })),
      status: 'Posted'
    };

    setJournalEntries([newEntry, ...journalEntries]);
    setIsNewJournalModalOpen(false);
    setNewJeDesc('');
    setNewJeLines([
      { id: '1', accountCode: '1010', accountName: 'Operating Cash Account', debit: 0, credit: 0 },
      { id: '2', accountCode: '4010', accountName: 'SaaS Software Subscriptions', debit: 0, credit: 0 }
    ]);
  };

  const handleCreateInvoice = (e: React.FormEvent) => {
    e.preventDefault();
    const sub = parseFloat(newInvSubtotal) || 0;
    const vat = sub * 0.2; // 20% VAT standard
    const tot = sub + vat;

    const newInv: Invoice = {
      id: `INV-${Date.now().toString().slice(-3)}`,
      invoiceNumber: newInvNum,
      entityName: newInvEntity || 'Standard Client/Vendor',
      type: newInvType,
      date: new Date().toISOString().split('T')[0],
      dueDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      subtotal: sub,
      vatAmount: vat,
      total: tot,
      status: 'Sent'
    };

    setInvoices([newInv, ...invoices]);
    setIsNewInvoiceModalOpen(false);
    setNewInvEntity('');
    setNewInvSubtotal('');
  };

  // Helper filter for COA
  const filteredAccounts = accounts.filter((acc) => {
    const matchesFilter = coaFilter === 'All' || acc.type === coaFilter;
    const matchesSearch =
      acc.name.toLowerCase().includes(coaSearch.toLowerCase()) ||
      acc.code.includes(coaSearch);
    return matchesFilter && matchesSearch;
  });

  // Helper filter for Invoices
  const filteredInvoices = invoices.filter((inv) => {
    if (invoiceTypeFilter === 'ALL') return true;
    return inv.type === invoiceTypeFilter;
  });

  // Dynamic formatting helper
  const formatCurrency = (val: number) => {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val);
  };

  return (
    <div className="min-h-screen bg-[#0b1121] text-slate-100 font-sans antialiased selection:bg-cyan-500 selection:text-slate-900">
      {/* TOP HEADER / BAR */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-[#0b1121]/90 backdrop-blur-xl px-6 py-3.5">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-600 to-purple-600 shadow-lg shadow-cyan-500/20">
              <Calculator className="w-6 h-6 text-white" />
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
                  AURA ERP
                </h1>
                <span className="px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 rounded-full">
                  Financial Controller Suite
                </span>
              </div>
              <p className="text-xs text-slate-400">JOL Studio Architecture • Multi-Ledger Accounting</p>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="hidden sm:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300">
              <Calendar className="w-3.5 h-3.5 text-cyan-400" />
              <span>Fiscal Period: <strong>FY 2024-Q4</strong></span>
            </div>

            <div className="flex items-center space-x-2 border-l border-slate-800 pl-4">
              <button
                onClick={() => setIsNewJournalModalOpen(true)}
                className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition shadow-sm"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>New Journal Line</span>
              </button>
              <button
                onClick={() => setIsNewInvoiceModalOpen(true)}
                className="flex items-center space-x-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white transition shadow-sm"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>Create Invoice</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* NAVIGATION TABS */}
      <nav className="border-b border-slate-800/60 bg-[#111827]/50 backdrop-blur-md px-6">
        <div className="max-w-7xl mx-auto flex space-x-1 overflow-x-auto py-2">
          {[
            { id: 'dashboard', label: 'Executive Dashboard', icon: BarChart2 },
            { id: 'coa', label: 'Chart of Accounts', icon: Layers },
            { id: 'journal', label: 'Double-Entry Ledger', icon: BookOpen },
            { id: 'invoices', label: 'Invoices & AR/AP', icon: FileText },
            { id: 'statements', label: 'Financial Statements', icon: FileSpreadsheet }
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as TabType)}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-xs font-medium transition-all whitespace-nowrap ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/15 to-indigo-500/15 text-cyan-300 border border-cyan-500/30 shadow-inner'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-400'}`} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </nav>

      {/* MAIN CONTAINER */}
      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">

        {/* ==================== TAB 1: DASHBOARD & KPIS ==================== */}
        {activeTab === 'dashboard' && (
          <div className="space-y-6 animate-fadeIn">
            {/* KPI GRID */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
              {/* Asset Card */}
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-cyan-500/40 transition">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span>Total Assets</span>
                  <Building className="w-4 h-4 text-cyan-400" />
                </div>
                <div className="text-xl font-bold text-white tracking-tight">
                  {formatCurrency(kpiData.totalAssets)}
                </div>
                <div className="flex items-center mt-2 text-[11px] text-emerald-400">
                  <ArrowUpRight className="w-3.5 h-3.5 mr-1" />
                  <span>+12.4% vs last QTR</span>
                </div>
              </div>

              {/* Liabilities Card */}
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-amber-500/40 transition">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span>Total Liabilities</span>
                  <CreditCard className="w-4 h-4 text-amber-400" />
                </div>
                <div className="text-xl font-bold text-white tracking-tight">
                  {formatCurrency(kpiData.totalLiabilities)}
                </div>
                <div className="flex items-center mt-2 text-[11px] text-emerald-400">
                  <ArrowDownRight className="w-3.5 h-3.5 mr-1" />
                  <span>-3.1% reduction</span>
                </div>
              </div>

              {/* Net Equity */}
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-purple-500/40 transition">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span>Net Equity</span>
                  <Shield className="w-4 h-4 text-purple-400" />
                </div>
                <div className="text-xl font-bold text-white tracking-tight">
                  {formatCurrency(kpiData.equity)}
                </div>
                <div className="flex items-center mt-2 text-[11px] text-emerald-400">
                  <ArrowUpRight className="w-3.5 h-3.5 mr-1" />
                  <span>+18.2% YOY Growth</span>
                </div>
              </div>

              {/* Revenue YTD */}
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-emerald-500/40 transition">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span>Revenue YTD</span>
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="text-xl font-bold text-white tracking-tight">
                  {formatCurrency(kpiData.revenueYTD)}
                </div>
                <div className="flex items-center mt-2 text-[11px] text-emerald-400">
                  <ArrowUpRight className="w-3.5 h-3.5 mr-1" />
                  <span>+24.5% target hit</span>
                </div>
              </div>

              {/* Net Profit Margin */}
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-blue-500/40 transition">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span>Net Profit Margin</span>
                  <PieChart className="w-4 h-4 text-blue-400" />
                </div>
                <div className="text-xl font-bold text-white tracking-tight">
                  {kpiData.netProfitMargin}%
                </div>
                <div className="flex items-center mt-2 text-[11px] text-cyan-400">
                  <CheckCircle className="w-3.5 h-3.5 mr-1" />
                  <span>+2.1% efficiency</span>
                </div>
              </div>

              {/* VAT Payable */}
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-rose-500/40 transition">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                  <span>VAT Payable</span>
                  <DollarSign className="w-4 h-4 text-rose-400" />
                </div>
                <div className="text-xl font-bold text-white tracking-tight">
                  {formatCurrency(kpiData.vatPayable)}
                </div>
                <div className="flex items-center mt-2 text-[11px] text-slate-400">
                  <Calendar className="w-3.5 h-3.5 mr-1 text-slate-400" />
                  <span>Due Oct 31, 2024</span>
                </div>
              </div>
            </div>

            {/* DASHBOARD MIDDLE ROW */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Cashflow & Financial Performance Chart (Styled Mock) */}
              <div className="lg:col-span-2 p-5 rounded-2xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md flex flex-col justify-between">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-200">Financial Trajectory & Liquidity</h3>
                    <p className="text-xs text-slate-400">Quarterly comparison of Revenue vs Operating Expenses</p>
                  </div>
                  <div className="flex items-center space-x-3 text-xs">
                    <div className="flex items-center space-x-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block"></span>
                      <span className="text-slate-300">Revenue</span>
                    </div>
                    <div className="flex items-center space-x-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 inline-block"></span>
                      <span className="text-slate-300">Expenses</span>
                    </div>
                  </div>
                </div>

                {/* SVG Visual Bar Chart */}
                <div className="h-56 w-full flex items-end justify-between gap-3 pt-6 pb-2 px-2 border-b border-slate-800">
                  {[
                    { month: 'May', rev: 45, exp: 30 },
                    { month: 'Jun', rev: 60, exp: 38 },
                    { month: 'Jul', rev: 55, exp: 32 },
                    { month: 'Aug', rev: 78, exp: 42 },
                    { month: 'Sep', rev: 85, exp: 48 },
                    { month: 'Oct', rev: 95, exp: 50 }
                  ].map((item, idx) => (
                    <div key={idx} className="flex-1 flex flex-col items-center h-full justify-end group">
                      <div className="w-full flex justify-center items-end gap-1.5 h-full">
                        {/* Rev Bar */}
                        <div
                          style={{ height: `${item.rev}%` }}
                          className="w-1/2 max-w-[28px] bg-gradient-to-t from-cyan-600 to-cyan-400 rounded-t-sm group-hover:brightness-125 transition-all relative"
                        >
                          <span className="opacity-0 group-hover:opacity-100 absolute -top-7 left-1/2 -translate-x-1/2 bg-slate-900 border border-slate-700 text-[10px] text-cyan-300 px-1.5 py-0.5 rounded shadow">
                            ${item.rev}k
                          </span>
                        </div>
                        {/* Exp Bar */}
                        <div
                          style={{ height: `${item.exp}%` }}
                          className="w-1/2 max-w-[28px] bg-gradient-to-t from-indigo-700 to-indigo-500 rounded-t-sm group-hover:brightness-125 transition-all"
                        ></div>
                      </div>
                      <span className="text-[11px] text-slate-400 mt-2 font-medium">{item.month}</span>
                    </div>
                  ))}
                </div>

                <div className="mt-4 flex items-center justify-between text-xs text-slate-400">
                  <div className="flex items-center space-x-2">
                    <Shield className="w-4 h-4 text-emerald-400" />
                    <span>Double-Entry Balance Check: <strong>PASSED (100% Reconciled)</strong></span>
                  </div>
                  <button onClick={() => setActiveTab('statements')} className="text-cyan-400 hover:text-cyan-300 flex items-center space-x-1">
                    <span>View Financial Statements</span>
                    <ChevronRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>

              {/* System Governance & Compliance Widget */}
              <div className="p-5 rounded-2xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-semibold text-slate-200">Compliance & Audit Sentinel</h3>
                    <span className="px-2 py-0.5 text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded">
                      SOC-2 Ready
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-4">Real-time system validation checks</p>

                  <div className="space-y-3">
                    <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-center justify-between">
                      <div className="flex items-center space-x-2.5 text-xs">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        <span className="text-slate-200">Journal Balance Integrity</span>
                      </div>
                      <span className="text-[11px] text-slate-400">Debit = Credit</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-center justify-between">
                      <div className="flex items-center space-x-2.5 text-xs">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        <span className="text-slate-200">VAT / Tax Provision Sync</span>
                      </div>
                      <span className="text-[11px] text-slate-400">20% Standard</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-center justify-between">
                      <div className="flex items-center space-x-2.5 text-xs">
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        <span className="text-slate-200">Unmatched Bank Feeds</span>
                      </div>
                      <span className="text-[11px] text-amber-400 font-semibold">2 Pending</span>
                    </div>

                    <div className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800/80 flex items-center justify-between">
                      <div className="flex items-center space-x-2.5 text-xs">
                        <Lock className="w-4 h-4 text-cyan-400" />
                        <span className="text-slate-200">Period Closing Status</span>
                      </div>
                      <span className="text-[11px] text-cyan-400 font-medium">Q3 Closed</span>
                    </div>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between">
                  <span className="text-xs text-slate-400">Controller Audit Hash:</span>
                  <code className="text-[10px] text-slate-300 font-mono bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                    0x8f2a...c91d
                  </code>
                </div>
              </div>
            </div>

            {/* RECENT JOURNAL ENTRIES MINI TABLE */}
            <div className="p-5 rounded-2xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-200">Recent Ledger Activity</h3>
                  <p className="text-xs text-slate-400">Latest posted general journal entries</p>
                </div>
                <button
                  onClick={() => setActiveTab('journal')}
                  className="text-xs text-cyan-400 hover:text-cyan-300 font-medium flex items-center space-x-1"
                >
                  <span>Open Journal Ledger</span>
                  <ChevronRight className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                      <th className="py-2.5 px-3">Date</th>
                      <th className="py-2.5 px-3">Reference</th>
                      <th className="py-2.5 px-3">Description</th>
                      <th className="py-2.5 px-3 text-right">Total Impact</th>
                      <th className="py-2.5 px-3 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-xs text-slate-300">
                    {journalEntries.slice(0, 3).map((entry) => {
                      const totalVal = entry.lines.reduce((sum, l) => sum + l.debit, 0);
                      return (
                        <tr key={entry.id} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-3 px-3 font-mono text-slate-400">{entry.date}</td>
                          <td className="py-3 px-3 font-semibold text-cyan-400">{entry.ref}</td>
                          <td className="py-3 px-3">{entry.description}</td>
                          <td className="py-3 px-3 text-right font-mono font-medium">{formatCurrency(totalVal)}</td>
                          <td className="py-3 px-3 text-center">
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                              {entry.status}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 2: CHART OF ACCOUNTS ==================== */}
        {activeTab === 'coa' && (
          <div className="space-y-6 animate-fadeIn">
            {/* SEARCH & CONTROLS */}
            <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center space-x-3 flex-1 min-w-[280px]">
                <div className="relative flex-1">
                  <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search accounts by name or code..."
                    value={coaSearch}
                    onChange={(e) => setCoaSearch(e.target.value)}
                    className="w-full bg-slate-900/80 border border-slate-800 rounded-lg pl-9 pr-4 py-2 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-cyan-500/60 transition"
                  />
                </div>

                <div className="flex items-center space-x-1 bg-slate-900/80 border border-slate-800 p-1 rounded-lg text-xs">
                  {['All', 'Asset', 'Liability', 'Equity', 'Revenue', 'Expense'].map((type) => (
                    <button
                      key={type}
                      onClick={() => setCoaFilter(type)}
                      className={`px-3 py-1 rounded-md transition ${
                        coaFilter === type
                          ? 'bg-indigo-600 text-white font-medium shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {type}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={() => setIsNewAccountModalOpen(true)}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs transition shadow"
              >
                <Plus className="w-4 h-4" />
                <span>Add Account</span>
              </button>
            </div>

            {/* ACCOUNTS TABLE */}
            <div className="rounded-2xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-900/40 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                      <th className="py-3.5 px-4">Account Code</th>
                      <th className="py-3.5 px-4">Account Title</th>
                      <th className="py-3.5 px-4">Classification</th>
                      <th className="py-3.5 px-4 text-right">Current Balance</th>
                      <th className="py-3.5 px-4 text-center">Status</th>
                      <th className="py-3.5 px-4 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-xs">
                    {filteredAccounts.map((acc) => {
                      const getTypeBadge = (type: AccountType) => {
                        switch (type) {
                          case 'Asset': return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20';
                          case 'Liability': return 'bg-amber-500/10 text-amber-400 border-amber-500/20';
                          case 'Equity': return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
                          case 'Revenue': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
                          case 'Expense': return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
                        }
                      };

                      return (
                        <tr key={acc.code} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-3.5 px-4 font-mono font-semibold text-cyan-400">{acc.code}</td>
                          <td className="py-3.5 px-4 font-medium text-slate-200">{acc.name}</td>
                          <td className="py-3.5 px-4">
                            <span className={`px-2.5 py-1 rounded-full text-[10px] font-semibold border ${getTypeBadge(acc.type)}`}>
                              {acc.type}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-right font-mono font-medium text-slate-100">
                            {formatCurrency(acc.balance)}
                          </td>
                          <td className="py-3.5 px-4 text-center">
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                              {acc.status}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 text-right">
                            <button className="text-xs text-slate-400 hover:text-cyan-400 transition font-medium">
                              Ledger View
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 3: DOUBLE-ENTRY JOURNAL LEDGER ==================== */}
        {activeTab === 'journal' && (
          <div className="space-y-6 animate-fadeIn">
            {/* LEDGER BANNER */}
            <div className="p-4 rounded-xl bg-gradient-to-r from-indigo-900/40 via-slate-900/80 to-slate-900/80 border border-indigo-500/30 backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 rounded-lg bg-indigo-600/30 text-indigo-400 border border-indigo-500/30">
                  <Scale className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-slate-200">General Journal Integrity Monitor</h3>
                  <p className="text-xs text-slate-400">Strict Debit = Credit verification active on all posted entries</p>
                </div>
              </div>

              <button
                onClick={() => setIsNewJournalModalOpen(true)}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition shadow"
              >
                <Plus className="w-4 h-4" />
                <span>New Balanced Entry</span>
              </button>
            </div>

            {/* JOURNAL ENTRIES LIST */}
            <div className="space-y-4">
              {journalEntries.map((entry) => {
                const entryDebitSum = entry.lines.reduce((s, l) => s + l.debit, 0);
                const entryCreditSum = entry.lines.reduce((s, l) => s + l.credit, 0);

                return (
                  <div key={entry.id} className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md space-y-3">
                    <div className="flex flex-wrap items-center justify-between border-b border-slate-800 pb-2.5 gap-2">
                      <div className="flex items-center space-x-3">
                        <span className="font-mono text-xs text-cyan-400 font-bold">{entry.ref}</span>
                        <span className="text-xs font-medium text-slate-300">{entry.description}</span>
                      </div>
                      <div className="flex items-center space-x-3 text-xs">
                        <span className="text-slate-400 font-mono">{entry.date}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          {entry.status}
                        </span>
                      </div>
                    </div>

                    {/* Entry Lines */}
                    <div className="overflow-x-auto">
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                            <th className="py-1 px-2">Code</th>
                            <th className="py-1 px-2">Account Name</th>
                            <th className="py-1 px-2 text-right">Debit ($)</th>
                            <th className="py-1 px-2 text-right">Credit ($)</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/40 text-xs">
                          {entry.lines.map((line) => (
                            <tr key={line.id}>
                              <td className="py-1.5 px-2 font-mono text-slate-400 text-[11px]">{line.accountCode}</td>
                              <td className="py-1.5 px-2 text-slate-200">{line.accountName}</td>
                              <td className="py-1.5 px-2 text-right font-mono text-emerald-400">
                                {line.debit > 0 ? formatCurrency(line.debit) : '—'}
                              </td>
                              <td className="py-1.5 px-2 text-right font-mono text-cyan-400">
                                {line.credit > 0 ? formatCurrency(line.credit) : '—'}
                              </td>
                            </tr>
                          ))}
                          <tr className="font-semibold text-xs border-t border-slate-700/60 bg-slate-900/40">
                            <td colSpan={2} className="py-1.5 px-2 text-slate-400">Entry Verification Totals:</td>
                            <td className="py-1.5 px-2 text-right font-mono text-emerald-400">{formatCurrency(entryDebitSum)}</td>
                            <td className="py-1.5 px-2 text-right font-mono text-cyan-400">{formatCurrency(entryCreditSum)}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ==================== TAB 4: INVOICES & AR/AP ==================== */}
        {activeTab === 'invoices' && (
          <div className="space-y-6 animate-fadeIn">
            {/* STAT SUMMARY BANNER */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md">
                <span className="text-xs text-slate-400">Accounts Receivable (AR)</span>
                <div className="text-lg font-bold text-emerald-400 mt-1">$144,000.00</div>
                <span className="text-[11px] text-slate-400">Client Outstanding Invoices</span>
              </div>
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md">
                <span className="text-xs text-slate-400">Accounts Payable (AP)</span>
                <div className="text-lg font-bold text-amber-400 mt-1">$27,600.00</div>
                <span className="text-[11px] text-slate-400">Vendor Bills Due</span>
              </div>
              <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md">
                <span className="text-xs text-slate-400">Standard VAT Rate</span>
                <div className="text-lg font-bold text-cyan-400 mt-1">20.0%</div>
                <span className="text-[11px] text-slate-400">Auto Tax Calculation</span>
              </div>
            </div>

            {/* CONTROLS */}
            <div className="p-4 rounded-xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center space-x-2 bg-slate-900/80 border border-slate-800 p-1 rounded-lg text-xs">
                <button
                  onClick={() => setInvoiceTypeFilter('ALL')}
                  className={`px-3 py-1 rounded-md transition ${invoiceTypeFilter === 'ALL' ? 'bg-indigo-600 text-white' : 'text-slate-400'}`}
                >
                  All Invoices
                </button>
                <button
                  onClick={() => setInvoiceTypeFilter('AR')}
                  className={`px-3 py-1 rounded-md transition ${invoiceTypeFilter === 'AR' ? 'bg-indigo-600 text-white' : 'text-slate-400'}`}
                >
                  Receivables (AR)
                </button>
                <button
                  onClick={() => setInvoiceTypeFilter('AP')}
                  className={`px-3 py-1 rounded-md transition ${invoiceTypeFilter === 'AP' ? 'bg-indigo-600 text-white' : 'text-slate-400'}`}
                >
                  Payables (AP)
                </button>
              </div>

              <button
                onClick={() => setIsNewInvoiceModalOpen(true)}
                className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs transition shadow"
              >
                <Plus className="w-4 h-4" />
                <span>Create Invoice / Bill</span>
              </button>
            </div>

            {/* INVOICE TABLE */}
            <div className="rounded-2xl bg-[#151d30]/80 border border-slate-800/80 backdrop-blur-md overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-900/40 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                      <th className="py-3.5 px-4">Invoice #</th>
                      <th className="py-3.5 px-4">Type</th>
                      <th className="py-3.5 px-4">Client / Vendor</th>
                      <th className="py-3.5 px-4">Issue Date</th>
                      <th className="py-3.5 px-4">Due Date</th>
                      <th className="py-3.5 px-4 text-right">Subtotal</th>
                      <th className="py-3.5 px-4 text-right">VAT (20%)</th>
                      <th className="py-3.5 px-4 text-right">Total Amount</th>
                      <th className="py-3.5 px-4 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 text-xs">
                    {filteredInvoices.map((inv) => {
                      const getStatusBadge = (st: string) => {
                        switch (st) {
                          case 'Paid': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
                          case 'Overdue': return 'bg-rose-500/10 text-rose-400 border-rose-500/20';
                          case 'Sent': return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20';
                          default: return 'bg-slate-500/10 text-slate-400 border-slate-500/20';
                        }
                      };

                      return (
                        <tr key={inv.id} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-3.5 px-4 font-mono font-semibold text-cyan-400">{inv.invoiceNumber}</td>
                          <td className="py-3.5 px-4">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${inv.type === 'AR' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'}`}>
                              {inv.type === 'AR' ? 'RECEIVABLE' : 'PAYABLE'}
                            </span>
                          </td>
                          <td className="py-3.5 px-4 font-medium text-slate-200">{inv.entityName}</td>
                          <td className="py-3.5 px-4 text-slate-400 font-mono">{inv.date}</td>
                          <td className="py-3.5 px-4 text-slate-400 font-mono">{inv.dueDate}</td>
                          <td className="py-3.5 px-4 text-right font-mono">{formatCurrency(inv.subtotal)}</td>
                          <td className="py-3.5 px-4 text-right font-mono text-slate-400">{formatCurrency(inv.vatAmount)}</td>
                          <td className="py-3.5 px-4 text-right font-mono font-semibold text-slate-100">{formatCurrency(inv.total)}</td>
                          <td className="py-3.5 px-4 text-center">
                            <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-semibold border ${getStatusBadge(inv.status)}`}>
                              {inv.status}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ==================== TAB 5: FINANCIAL STATEMENTS ==================== */}
        {activeTab === 'statements' && (
          <div className="space-y-6 animate-fadeIn">
            {/* SUB TAB CONTROLS */}
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div className="flex space-x-2">
                {[
                  { id: 'income', label: 'Income Statement (P&L)' },
                  { id: 'balance', label: 'Balance Sheet' },
                  { id: 'vat', label: 'Tax & VAT Return Summary' }
                ].map((st) => (
                  <button
                    key={st.id}
                    onClick={() => setStatementTab(st.id as StatementSubTab)}
                    className={`px-4 py-2 rounded-lg text-xs font-semibold transition ${
                      statementTab === st.id
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : 'text-slate-400 hover:text-slate-200 bg-slate-900/40'
                    }`}
                  >
                    {st.label}
                  </button>
                ))}
              </div>

              <button className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs transition">
                <Download className="w-3.5 h-3.5" />
                <span>Export PDF Report</span>
              </button>
            </div>

            {/* STATEMENT CONTENT AREA */}
            <div className="p-8 rounded-2xl bg-[#151d30]/90 border border-slate-800/80 backdrop-blur-md max-w-4xl mx-auto shadow-2xl">
              {/* SUB TAB 1: PROFIT & LOSS */}
              {statementTab === 'income' && (
                <div className="space-y-6">
                  <div className="text-center border-b border-slate-800 pb-4">
                    <h2 className="text-xl font-bold text-white tracking-wide uppercase">Consolidated Statement of Operations (P&L)</h2>
                    <p className="text-xs text-slate-400 mt-1">For Period Ending October 31, 2024 • USD ($)</p>
                  </div>

                  {/* REVENUE SECTION */}
                  <div className="space-y-2">
                    <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider border-b border-slate-800 pb-1">
                      Revenue & Operating Income
                    </h3>
                    <div className="flex justify-between text-xs py-1 text-slate-300 pl-4">
                      <span>SaaS Software Subscriptions (Acc 4010)</span>
                      <span className="font-mono">$480,000.00</span>
                    </div>
                    <div className="flex justify-between text-xs py-1 text-slate-300 pl-4">
                      <span>Professional Services & Advisory (Acc 4020)</span>
                      <span className="font-mono">$200,000.00</span>
                    </div>
                    <div className="flex justify-between text-xs py-2 font-bold text-slate-100 border-t border-slate-800/60 bg-slate-900/30 px-2 rounded">
                      <span>Total Gross Revenue</span>
                      <span className="font-mono text-emerald-400">$680,000.00</span>
                    </div>
                  </div>

                  {/* OPERATING EXPENSES */}
                  <div className="space-y-2 pt-2">
                    <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider border-b border-slate-800 pb-1">
                      Operating Expenses (OPEX)
                    </h3>
                    <div className="flex justify-between text-xs py-1 text-slate-300 pl-4">
                      <span>Payroll & Executive Compensation (Acc 5010)</span>
                      <span className="font-mono">$290,000.00</span>
                    </div>
                    <div className="flex justify-between text-xs py-1 text-slate-300 pl-4">
                      <span>Cloud Hosting & Server Infrastructure (Acc 5020)</span>
                      <span className="font-mono">$110,000.00</span>
                    </div>
                    <div className="flex justify-between text-xs py-1 text-slate-300 pl-4">
                      <span>Office Facilities & Operational Lease (Acc 5030)</span>
                      <span className="font-mono">$86,800.00</span>
                    </div>
                    <div className="flex justify-between text-xs py-2 font-bold text-slate-100 border-t border-slate-800/60 bg-slate-900/30 px-2 rounded">
                      <span>Total Operating Expenses</span>
                      <span className="font-mono text-rose-400">$486,800.00</span>
                    </div>
                  </div>

                  {/* NET INCOME SUMMARY */}
                  <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-700/80 space-y-2">
                    <div className="flex justify-between text-sm font-bold text-white">
                      <span>Net Operating Income (EBITDA)</span>
                      <span className="font-mono text-emerald-400">$193,200.00</span>
                    </div>
                    <div className="flex justify-between text-xs text-slate-400 pt-1 border-t border-slate-800">
                      <span>Calculated Effective Margin</span>
                      <span className="font-mono text-cyan-400">28.41%</span>
                    </div>
                  </div>
                </div>
              )}

              {/* SUB TAB 2: BALANCE SHEET */}
              {statementTab === 'balance' && (
                <div className="space-y-6">
                  <div className="text-center border-b border-slate-800 pb-4">
                    <h2 className="text-xl font-bold text-white tracking-wide uppercase">Statement of Financial Position (Balance Sheet)</h2>
                    <p className="text-xs text-slate-400 mt-1">As of October 31, 2024 • JOL Accounting Standard</p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* ASSETS COLUMN */}
                    <div className="space-y-3">
                      <h3 className="text-xs font-bold text-cyan-400 uppercase tracking-wider border-b border-slate-800 pb-1">
                        Assets
                      </h3>
                      <div className="flex justify-between text-xs py-1 text-slate-300">
                        <span>Operating Cash Account</span>
                        <span className="font-mono">$520,000.00</span>
                      </div>
                      <div className="flex justify-between text-xs py-1 text-slate-300">
                        <span>Accounts Receivable</span>
                        <span className="font-mono">$380,000.00</span>
                      </div>
                      <div className="flex justify-between text-xs py-1 text-slate-300">
                        <span>Short-term Treasury Investments</span>
                        <span className="font-mono">$550,000.00</span>
                      </div>
                      <div className="flex justify-between text-xs py-2 font-bold text-white border-t border-slate-800 bg-slate-900/40 px-2 rounded">
                        <span>TOTAL ASSETS</span>
                        <span className="font-mono text-cyan-400">$1,450,000.00</span>
                      </div>
                    </div>

                    {/* LIABILITIES & EQUITY COLUMN */}
                    <div className="space-y-3">
                      <h3 className="text-xs font-bold text-amber-400 uppercase tracking-wider border-b border-slate-800 pb-1">
                        Liabilities & Equity
                      </h3>
                      <div className="flex justify-between text-xs py-1 text-slate-300">
                        <span>Accounts Payable</span>
                        <span className="font-mono">$185,500.00</span>
                      </div>
                      <div className="flex justify-between text-xs py-1 text-slate-300">
                        <span>VAT Tax Payable</span>
                        <span className="font-mono">$34,500.00</span>
                      </div>
                      <div className="flex justify-between text-xs py-1 text-slate-300">
                        <span>Accrued Operating Expenses</span>
                        <span className="font-mono">$200,000.00</span>
                      </div>
                      <div className="flex justify-between text-xs py-1.5 font-semibold text-slate-200 border-t border-slate-800/40">
                        <span>Total Liabilities</span>
                        <span className="font-mono text-amber-400">$420,000.00</span>
                      </div>

                      <div className="pt-2">
                        <h4 className="text-[11px] font-bold text-purple-400 uppercase tracking-wider mb-1">Stockholders' Equity</h4>
                        <div className="flex justify-between text-xs py-1 text-slate-300">
                          <span>Common Share Capital</span>
                          <span className="font-mono">$600,000.00</span>
                        </div>
                        <div className="flex justify-between text-xs py-1 text-slate-300">
                          <span>Retained Earnings</span>
                          <span className="font-mono">$430,000.00</span>
                        </div>
                        <div className="flex justify-between text-xs py-1.5 font-semibold text-slate-200 border-t border-slate-800/40">
                          <span>Total Equity</span>
                          <span className="font-mono text-purple-400">$1,030,000.00</span>
                        </div>
                      </div>

                      <div className="flex justify-between text-xs py-2 font-bold text-white border-t border-slate-800 bg-slate-900/40 px-2 rounded">
                        <span>TOTAL LIABILITIES & EQUITY</span>
                        <span className="font-mono text-cyan-400">$1,450,000.00</span>
                      </div>
                    </div>
                  </div>

                  {/* EQUALITY VERIFICATION FOOTER */}
                  <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-center text-xs text-emerald-400 font-medium">
                    ✔ Fundamental Accounting Equation Verified: Assets ($1,450,000) = Liabilities ($420,000) + Equity ($1,030,000)
                  </div>
                </div>
              )}

              {/* SUB TAB 3: TAX & VAT RETURN SUMMARY */}
              {statementTab === 'vat' && (
                <div className="space-y-6">
                  <div className="text-center border-b border-slate-800 pb-4">
                    <h2 className="text-xl font-bold text-white tracking-wide uppercase">VAT / Sales Tax Compliance Return</h2>
                    <p className="text-xs text-slate-400 mt-1">Tax Period Q4-2024 • Filing Deadline Oct 31, 2024</p>
                  </div>

                  <div className="space-y-4">
                    <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-3">
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-300">Output VAT (Collected on Sales @ 20%)</span>
                        <span className="font-mono font-semibold text-emerald-400">$45,000.00</span>
                      </div>
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-slate-300">Input VAT (Deductible on Expenses @ 20%)</span>
                        <span className="font-mono font-semibold text-amber-400">-$10,500.00</span>
                      </div>
                      <div className="border-t border-slate-800 pt-2 flex justify-between items-center text-sm font-bold">
                        <span className="text-white">Net VAT Liability Payable</span>
                        <span className="font-mono text-rose-400">$34,500.00</span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-lg bg-slate-900/40 border border-slate-800 text-xs">
                      <div className="flex items-center space-x-2">
                        <Shield className="w-4 h-4 text-cyan-400" />
                        <span className="text-slate-300">Tax Authority Integration Status: <strong>READY TO FILE</strong></span>
                      </div>
                      <button className="px-3 py-1.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs transition">
                        Submit Digital Return
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* ==================== MODAL 1: ADD ACCOUNT ==================== */}
      {isNewAccountModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-[#151d30] border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl animate-scaleUp">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span>Add Account to COA</span>
              </h3>
              <button onClick={() => setIsNewAccountModalOpen(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleAddAccount} className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Account Code</label>
                <input
                  type="text"
                  placeholder="e.g. 1060"
                  value={newAccCode}
                  onChange={(e) => setNewAccCode(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Account Title Name</label>
                <input
                  type="text"
                  placeholder="e.g. Petty Cash Reserve"
                  value={newAccName}
                  onChange={(e) => setNewAccName(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Account Classification</label>
                <select
                  value={newAccType}
                  onChange={(e) => setNewAccType(e.target.value as AccountType)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500"
                >
                  <option value="Asset">Asset</option>
                  <option value="Liability">Liability</option>
                  <option value="Equity">Equity</option>
                  <option value="Revenue">Revenue</option>
                  <option value="Expense">Expense</option>
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Initial Starting Balance ($)</label>
                <input
                  type="number"
                  placeholder="0.00"
                  value={newAccBalance}
                  onChange={(e) => setNewAccBalance(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500"
                />
              </div>

              <div className="pt-3 flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setIsNewAccountModalOpen(false)}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium shadow"
                >
                  Create Account
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ==================== MODAL 2: NEW BALANCED JOURNAL ENTRY ==================== */}
      {isNewJournalModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 overflow-y-auto">
          <div className="bg-[#151d30] border border-slate-800 rounded-2xl max-w-2xl w-full p-6 space-y-4 shadow-2xl my-8">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                  <BookOpen className="w-4 h-4 text-indigo-400" />
                  <span>Create Double-Entry Journal Entry</span>
                </h3>
                <p className="text-[11px] text-slate-400">Total Debits must equal Total Credits to enable posting</p>
              </div>
              <button onClick={() => setIsNewJournalModalOpen(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handlePostJournalEntry} className="space-y-4 text-xs">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Journal Reference</label>
                  <input
                    type="text"
                    value={newJeRef}
                    onChange={(e) => setNewJeRef(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white font-mono"
                    required
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Posting Date</label>
                  <input
                    type="date"
                    value={newJeDate}
                    onChange={(e) => setNewJeDate(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white"
                    required
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Entry Status</label>
                  <input
                    type="text"
                    value="Posted"
                    disabled
                    className="w-full bg-slate-900/50 border border-slate-800 rounded-lg p-2 text-emerald-400 font-semibold"
                  />
                </div>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Entry Description / Memo</label>
                <input
                  type="text"
                  placeholder="e.g. Monthly Accrual for Software Licenses"
                  value={newJeDesc}
                  onChange={(e) => setNewJeDesc(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white"
                  required
                />
              </div>

              {/* DYNAMIC JOURNAL LINES */}
              <div className="space-y-2">
                <div className="flex justify-between items-center text-[11px] font-semibold text-slate-400 uppercase">
                  <span>Ledger Lines</span>
                  <button
                    type="button"
                    onClick={handleAddJournalLine}
                    className="text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Add Line</span>
                  </button>
                </div>

                {newJeLines.map((line, index) => (
                  <div key={line.id} className="flex items-center space-x-2 bg-slate-900/80 p-2 rounded-lg border border-slate-800">
                    <select
                      value={line.accountCode}
                      onChange={(e) => handleJournalLineChange(line.id, 'accountCode', e.target.value)}
                      className="flex-1 bg-slate-900 border border-slate-700 rounded p-1.5 text-white text-xs"
                    >
                      {accounts.map((acc) => (
                        <option key={acc.code} value={acc.code}>
                          {acc.code} - {acc.name} ({acc.type})
                        </option>
                      ))}
                    </select>

                    <input
                      type="number"
                      placeholder="Debit"
                      value={line.debit || ''}
                      onChange={(e) => handleJournalLineChange(line.id, 'debit', parseFloat(e.target.value) || 0)}
                      className="w-28 bg-slate-900 border border-slate-700 rounded p-1.5 text-emerald-400 font-mono text-right text-xs"
                    />

                    <input
                      type="number"
                      placeholder="Credit"
                      value={line.credit || ''}
                      onChange={(e) => handleJournalLineChange(line.id, 'credit', parseFloat(e.target.value) || 0)}
                      className="w-28 bg-slate-900 border border-slate-700 rounded p-1.5 text-cyan-400 font-mono text-right text-xs"
                    />

                    {newJeLines.length > 2 && (
                      <button
                        type="button"
                        onClick={() => handleRemoveJournalLine(line.id)}
                        className="p-1 text-rose-400 hover:text-rose-300"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {/* BALANCE CALCULATOR BAR */}
              <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-between text-xs font-mono">
                <div>
                  <span className="text-slate-400">Total Debit: </span>
                  <span className="text-emerald-400 font-bold">{formatCurrency(totalJournalDebits)}</span>
                </div>
                <div>
                  <span className="text-slate-400">Total Credit: </span>
                  <span className="text-cyan-400 font-bold">{formatCurrency(totalJournalCredits)}</span>
                </div>
                <div>
                  <span className="text-slate-400">Difference: </span>
                  <span className={isJournalBalanced ? 'text-emerald-400 font-bold' : 'text-rose-400 font-bold'}>
                    {formatCurrency(Math.abs(totalJournalDebits - totalJournalCredits))}
                  </span>
                </div>
              </div>

              {/* POSTING VALIDATION BADGE */}
              <div className="flex items-center justify-between pt-2">
                <div className="flex items-center space-x-1.5 text-xs">
                  {isJournalBalanced ? (
                    <span className="text-emerald-400 flex items-center space-x-1">
                      <CheckCircle2 className="w-4 h-4" />
                      <span>Balanced Entry - Ready for Posting</span>
                    </span>
                  ) : (
                    <span className="text-rose-400 flex items-center space-x-1">
                      <AlertTriangle className="w-4 h-4" />
                      <span>Imbalanced Journal: Debits must equal Credits</span>
                    </span>
                  )}
                </div>

                <div className="flex space-x-2">
                  <button
                    type="button"
                    onClick={() => setIsNewJournalModalOpen(false)}
                    className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium"
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    disabled={!isJournalBalanced}
                    className={`px-4 py-2 rounded-lg font-medium transition shadow ${
                      isJournalBalanced
                        ? 'bg-indigo-600 hover:bg-indigo-500 text-white cursor-pointer'
                        : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                    }`}
                  >
                    Post Journal Entry
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* ==================== MODAL 3: NEW INVOICE / BILL ==================== */}
      {isNewInvoiceModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-[#151d30] border border-slate-800 rounded-2xl max-w-md w-full p-6 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                <FileText className="w-4 h-4 text-cyan-400" />
                <span>Create Invoice or Vendor Bill</span>
              </h3>
              <button onClick={() => setIsNewInvoiceModalOpen(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateInvoice} className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-400 mb-1">Invoice Number</label>
                  <input
                    type="text"
                    value={newInvNum}
                    onChange={(e) => setNewInvNum(e.target.value)}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white font-mono"
                    required
                  />
                </div>
                <div>
                  <label className="block text-slate-400 mb-1">Invoice Type</label>
                  <select
                    value={newInvType}
                    onChange={(e) => setNewInvType(e.target.value as 'AR' | 'AP')}
                    className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white"
                  >
                    <option value="AR">Customer Receivable (AR)</option>
                    <option value="AP">Vendor Payable (AP)</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Client / Vendor Name</label>
                <input
                  type="text"
                  placeholder="e.g. Acme Corporation"
                  value={newInvEntity}
                  onChange={(e) => setNewInvEntity(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Taxable Subtotal ($)</label>
                <input
                  type="number"
                  placeholder="0.00"
                  value={newInvSubtotal}
                  onChange={(e) => setNewInvSubtotal(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2.5 text-white focus:outline-none focus:border-cyan-500 font-mono"
                  required
                />
              </div>

              {/* TAX PREVIEW */}
              <div className="p-3 rounded-lg bg-slate-900/60 border border-slate-800 space-y-1 font-mono">
                <div className="flex justify-between text-slate-400">
                  <span>Auto VAT (20%):</span>
                  <span>{formatCurrency((parseFloat(newInvSubtotal) || 0) * 0.2)}</span>
                </div>
                <div className="flex justify-between text-white font-bold border-t border-slate-800 pt-1">
                  <span>Grand Total:</span>
                  <span className="text-cyan-400">{formatCurrency((parseFloat(newInvSubtotal) || 0) * 1.2)}</span>
                </div>
              </div>

              <div className="pt-3 flex justify-end space-x-2">
                <button
                  type="button"
                  onClick={() => setIsNewInvoiceModalOpen(false)}
                  className="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-medium shadow"
                >
                  Issue Invoice
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}