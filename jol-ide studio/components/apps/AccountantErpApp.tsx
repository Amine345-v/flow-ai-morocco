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
  Calendar,
  AlertTriangle,
  Download,
  Check,
  X,
  Percent,
  ChevronRight,
  Clock,
  ArrowRight
} from 'lucide-react';

// --- TYPES & INTERFACES ---
type TabType = 'overview' | 'ap' | 'vat' | 'debt' | 'amortization';
type BillStatus = 'UNPAID' | 'PARTIAL' | 'PAID' | 'OVERDUE';
type DebtType = 'BOND' | 'BANK_LOAN' | 'REVOLVING_CREDIT' | 'EQUIPMENT_LEASE';
type TaxStatus = 'DRAFT' | 'FILED' | 'SETTLED';

interface APBill {
  id: string;
  vendor: string;
  invoiceNo: string;
  category: string;
  issueDate: string;
  dueDate: string;
  amount: number;
  paidAmount: number;
  vatAmount: number;
  status: BillStatus;
}

interface DebtFacility {
  id: string;
  lender: string;
  type: DebtType;
  principal: number;
  outstandingBalance: number;
  interestRate: number; // percentage e.g. 5.5
  monthlyPayment: number;
  maturityDate: string;
  nextPaymentDate: string;
  status: 'ACTIVE' | 'REFINANCED' | 'PAID_OFF';
}

interface VATPeriod {
  id: string;
  period: string; // e.g. "Q1 2025"
  outputVAT: number; // Sales tax collected
  inputVAT: number;  // Purchase tax paid
  netPayable: number;
  dueDate: string;
  status: TaxStatus;
}

// --- MOCK INITIAL DATA ---
const INITIAL_AP_BILLS: APBill[] = [
  {
    id: 'AP-1001',
    vendor: 'Apex Cloud Solutions',
    invoiceNo: 'INV-2025-089',
    category: 'IT Infrastructure',
    issueDate: '2025-02-01',
    dueDate: '2025-03-03',
    amount: 45200,
    paidAmount: 0,
    vatAmount: 7533,
    status: 'UNPAID',
  },
  {
    id: 'AP-1002',
    vendor: 'Global Logistics Partners',
    invoiceNo: 'GLP-88412',
    category: 'Freight & Shipping',
    issueDate: '2025-01-15',
    dueDate: '2025-02-14',
    amount: 128500,
    paidAmount: 50000,
    vatAmount: 21416,
    status: 'PARTIAL',
  },
  {
    id: 'AP-1003',
    vendor: 'Starlight Energy Corp',
    invoiceNo: 'UTIL-9021',
    category: 'Utilities',
    issueDate: '2025-01-10',
    dueDate: '2025-02-01',
    amount: 14200,
    paidAmount: 0,
    vatAmount: 2366,
    status: 'OVERDUE',
  },
  {
    id: 'AP-1004',
    vendor: 'Vanguard Industrial Supplies',
    invoiceNo: 'VIS-33910',
    category: 'Raw Materials',
    issueDate: '2025-02-10',
    dueDate: '2025-03-25',
    amount: 210000,
    paidAmount: 210000,
    vatAmount: 35000,
    status: 'PAID',
  },
  {
    id: 'AP-1005',
    vendor: 'Precision Tooling Ltd',
    invoiceNo: 'PTL-7712',
    category: 'Equipment Maintenance',
    issueDate: '2025-02-12',
    dueDate: '2025-03-12',
    amount: 38400,
    paidAmount: 0,
    vatAmount: 6400,
    status: 'UNPAID',
  },
];

const INITIAL_DEBT_FACILITIES: DebtFacility[] = [
  {
    id: 'DBT-501',
    lender: 'JPMorgan Chase Commercial Banking',
    type: 'BANK_LOAN',
    principal: 1500000,
    outstandingBalance: 1120000,
    interestRate: 5.75,
    monthlyPayment: 24500,
    maturityDate: '2029-12-31',
    nextPaymentDate: '2025-03-01',
    status: 'ACTIVE',
  },
  {
    id: 'DBT-502',
    lender: 'Silicon Valley Capital',
    type: 'REVOLVING_CREDIT',
    principal: 800000,
    outstandingBalance: 450000,
    interestRate: 6.25,
    monthlyPayment: 11200,
    maturityDate: '2026-06-30',
    nextPaymentDate: '2025-03-15',
    status: 'ACTIVE',
  },
  {
    id: 'DBT-503',
    lender: 'Caterpillar Financial',
    type: 'EQUIPMENT_LEASE',
    principal: 600000,
    outstandingBalance: 290000,
    interestRate: 4.80,
    monthlyPayment: 14100,
    maturityDate: '2027-04-30',
    nextPaymentDate: '2025-03-10',
    status: 'ACTIVE',
  },
  {
    id: 'DBT-504',
    lender: 'Corporate Note Series B',
    type: 'BOND',
    principal: 2000000,
    outstandingBalance: 2000000,
    interestRate: 7.10,
    monthlyPayment: 11833, // Interest only quarterly equivalent
    maturityDate: '2031-10-15',
    nextPaymentDate: '2025-04-15',
    status: 'ACTIVE',
  },
];

const INITIAL_VAT_PERIODS: VATPeriod[] = [
  {
    id: 'VAT-2025-Q1',
    period: 'Q1 2025 (Jan - Mar)',
    outputVAT: 345000,
    inputVAT: 216600,
    netPayable: 128400,
    dueDate: '2025-04-20',
    status: 'DRAFT',
  },
  {
    id: 'VAT-2024-Q4',
    period: 'Q4 2024 (Oct - Dec)',
    outputVAT: 412000,
    inputVAT: 289000,
    netPayable: 123000,
    dueDate: '2025-01-20',
    status: 'SETTLED',
  },
  {
    id: 'VAT-2024-Q3',
    period: 'Q3 2024 (Jul - Sep)',
    outputVAT: 388000,
    inputVAT: 254000,
    netPayable: 134000,
    dueDate: '2024-10-20',
    status: 'SETTLED',
  },
];

export function AccountantErpApp() {
  // --- STATE ---
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [apBills, setApBills] = useState<APBill[]>(INITIAL_AP_BILLS);
  const [debts, setDebts] = useState<DebtFacility[]>(INITIAL_DEBT_FACILITIES);
  const [vatPeriods, setVatPeriods] = useState<VATPeriod[]>(INITIAL_VAT_PERIODS);
  
  // Filters & Search
  const [apSearch, setApSearch] = useState('');
  const [apStatusFilter, setApStatusFilter] = useState<string>('ALL');
  const [debtSearch, setDebtSearch] = useState('');
  
  // Modals & Forms State
  const [isAddApModalOpen, setIsAddApModalOpen] = useState(false);
  const [isAddDebtModalOpen, setIsAddDebtModalOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  // New AP Form
  const [newVendor, setNewVendor] = useState('');
  const [newInvoiceNo, setNewInvoiceNo] = useState('');
  const [newCategory, setNewCategory] = useState('IT Infrastructure');
  const [newAmount, setNewAmount] = useState('');
  const [newDueDate, setNewDueDate] = useState('');

  // New Debt Form
  const [newLender, setNewLender] = useState('');
  const [newDebtType, setNewDebtType] = useState<DebtType>('BANK_LOAN');
  const [newPrincipal, setNewPrincipal] = useState('');
  const [newRate, setNewRate] = useState('');
  const [newMaturity, setNewMaturity] = useState('');

  // Interactive Amortization Calculator State
  const [calcPrincipal, setCalcPrincipal] = useState<number>(500000);
  const [calcRate, setCalcRate] = useState<number>(6.5);
  const [calcYears, setCalcYears] = useState<number>(5);

  // --- HELPER NOTIFICATION TOAST ---
  const triggerToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => setToastMessage(null), 3500);
  };

  // --- COMPUTED VALUES / KPIS ---
  const totalAPOutstanding = useMemo(() => {
    return apBills.reduce((acc, b) => acc + (b.amount - b.paidAmount), 0);
  }, [apBills]);

  const totalOverdueAP = useMemo(() => {
    return apBills
      .filter(b => b.status === 'OVERDUE')
      .reduce((acc, b) => acc + (b.amount - b.paidAmount), 0);
  }, [apBills]);

  const totalDebtBalance = useMemo(() => {
    return debts.reduce((acc, d) => acc + d.outstandingBalance, 0);
  }, [debts]);

  const totalMonthlyDebtService = useMemo(() => {
    return debts.reduce((acc, d) => acc + d.monthlyPayment, 0);
  }, [debts]);

  const currentVATNetPayable = useMemo(() => {
    const current = vatPeriods.find(v => v.status === 'DRAFT');
    return current ? current.netPayable : 0;
  }, [vatPeriods]);

  const totalLiabilities = useMemo(() => {
    return totalAPOutstanding + totalDebtBalance + currentVATNetPayable;
  }, [totalAPOutstanding, totalDebtBalance, currentVATNetPayable]);

  // --- AP ACTIONS ---
  const handleAddAPBill = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newVendor || !newAmount || !newDueDate) return;

    const amt = parseFloat(newAmount);
    const vat = amt * 0.16667; // Simulated ~20% VAT
    const newBill: APBill = {
      id: `AP-${Math.floor(1000 + Math.random() * 9000)}`,
      vendor: newVendor,
      invoiceNo: newInvoiceNo || `INV-${Math.floor(10000 + Math.random() * 90000)}`,
      category: newCategory,
      issueDate: new Date().toISOString().split('T')[0],
      dueDate: newDueDate,
      amount: amt,
      paidAmount: 0,
      vatAmount: Math.round(vat),
      status: 'UNPAID',
    };

    setApBills([newBill, ...apBills]);
    setIsAddApModalOpen(false);
    setNewVendor('');
    setNewInvoiceNo('');
    setNewAmount('');
    setNewDueDate('');
    triggerToast(`AP Liability for "${newVendor}" created successfully!`);
  };

  const handlePayBill = (id: string) => {
    setApBills(prev =>
      prev.map(bill => {
        if (bill.id === id) {
          return {
            ...bill,
            paidAmount: bill.amount,
            status: 'PAID',
          };
        }
        return bill;
      })
    );
    triggerToast(`Payment recorded for Bill #${id}`);
  };

  // --- DEBT ACTIONS ---
  const handleAddDebt = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newLender || !newPrincipal || !newRate || !newMaturity) return;

    const principal = parseFloat(newPrincipal);
    const rate = parseFloat(newRate);
    // Rough monthly PMT estimation formula
    const r = rate / 100 / 12;
    const n = 60; // default 5 yrs if unspecified
    const pmt = (principal * r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);

    const newFacility: DebtFacility = {
      id: `DBT-${Math.floor(500 + Math.random() * 500)}`,
      lender: newLender,
      type: newDebtType,
      principal: principal,
      outstandingBalance: principal,
      interestRate: rate,
      monthlyPayment: Math.round(pmt || principal * 0.01),
      maturityDate: newMaturity,
      nextPaymentDate: new Date(Date.now() + 30 * 86400000).toISOString().split('T')[0],
      status: 'ACTIVE',
    };

    setDebts([...debts, newFacility]);
    setIsAddDebtModalOpen(false);
    setNewLender('');
    setNewPrincipal('');
    setNewRate('');
    setNewMaturity('');
    triggerToast(`Debt obligation facility with "${newLender}" registered!`);
  };

  // --- VAT ACTIONS ---
  const handleFileVAT = (id: string) => {
    setVatPeriods(prev =>
      prev.map(item => (item.id === id ? { ...item, status: 'SETTLED' } : item))
    );
    triggerToast(`VAT Return ${id} filed & tax balance settled!`);
  };

  // --- AMORTIZATION COMPUTATION ---
  const amortizationSchedule = useMemo(() => {
    const schedule = [];
    let balance = calcPrincipal;
    const monthlyRate = calcRate / 100 / 12;
    const totalPayments = calcYears * 12;

    const pmt =
      monthlyRate === 0
        ? calcPrincipal / totalPayments
        : (calcPrincipal * monthlyRate * Math.pow(1 + monthlyRate, totalPayments)) /
          (Math.pow(1 + monthlyRate, totalPayments) - 1);

    for (let month = 1; month <= Math.min(totalPayments, 24); month++) {
      const interest = balance * monthlyRate;
      const principalPaid = pmt - interest;
      balance = Math.max(0, balance - principalPaid);

      schedule.push({
        month,
        payment: pmt,
        principalPaid,
        interestPaid: interest,
        remainingBalance: balance,
      });
    }
    return { pmt, schedule };
  }, [calcPrincipal, calcRate, calcYears]);

  // --- FILTERED DATA ---
  const filteredAP = useMemo(() => {
    return apBills.filter(b => {
      const matchesSearch =
        b.vendor.toLowerCase().includes(apSearch.toLowerCase()) ||
        b.invoiceNo.toLowerCase().includes(apSearch.toLowerCase()) ||
        b.category.toLowerCase().includes(apSearch.toLowerCase());
      const matchesStatus = apStatusFilter === 'ALL' || b.status === apStatusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [apBills, apSearch, apStatusFilter]);

  const filteredDebt = useMemo(() => {
    return debts.filter(
      d =>
        d.lender.toLowerCase().includes(debtSearch.toLowerCase()) ||
        d.type.toLowerCase().includes(debtSearch.toLowerCase())
    );
  }, [debts, debtSearch]);

  // --- FORMATTERS ---
  const formatCurrency = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  return (
    <div className="min-h-screen bg-[#0b1121] text-slate-100 font-sans p-4 sm:p-6 lg:p-8 flex flex-col gap-6 selection:bg-indigo-500 selection:text-white">
      {/* TOAST NOTIFICATION */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-50 bg-emerald-500/90 text-slate-950 font-semibold px-4 py-3 rounded-xl shadow-2xl backdrop-blur-md flex items-center gap-3 animate-bounce border border-emerald-300">
          <CheckCircle className="w-5 h-5 text-slate-950" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* HEADER BAR */}
      <header className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/60 p-5 rounded-2xl border border-slate-800/80 backdrop-blur-md shadow-xl">
        <div className="flex items-center gap-4">
          <div className="p-3 bg-gradient-to-br from-indigo-600 to-violet-700 rounded-xl shadow-lg shadow-indigo-500/20">
            <Building className="w-7 h-7 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-2xl font-bold tracking-tight text-white">
                Liability Accounts & Obligations
              </h1>
              <span className="text-xs bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 px-2.5 py-0.5 rounded-full font-mono">
                Sub-Module v4.8
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Financial ERP Control Hub &bull; Accounts Payable, Tax Provisions & Funded Debt Management
            </p>
          </div>
        </div>

        {/* TOP LEVEL ACTIONS */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => triggerToast('Ledgers updated & synced with main ledger.')}
            className="px-3.5 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-300 text-xs font-medium transition-all flex items-center gap-2 shadow-sm"
          >
            <RefreshCw className="w-4 h-4 text-slate-400" />
            Sync Ledger
          </button>
          <button
            onClick={() => triggerToast('Liability report exported to CSV & PDF format.')}
            className="px-3.5 py-2 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 border border-slate-700 text-slate-300 text-xs font-medium transition-all flex items-center gap-2 shadow-sm"
          >
            <Download className="w-4 h-4 text-slate-400" />
            Export Audit
          </button>
          <button
            onClick={() => setIsAddApModalOpen(true)}
            className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-medium transition-all flex items-center gap-2 shadow-lg shadow-indigo-600/30 active:scale-95"
          >
            <Plus className="w-4 h-4" />
            New AP Bill
          </button>
        </div>
      </header>

      {/* DASHBOARD SUMMARY CARDS (KPIs) */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Card 1 */}
        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-indigo-500/40 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-indigo-500/5 rounded-full blur-2xl group-hover:bg-indigo-500/10 transition-all" />
          <div className="flex justify-between items-start mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Total Liabilities
            </span>
            <div className="p-2 bg-indigo-500/10 text-indigo-400 rounded-lg border border-indigo-500/20">
              <Layers className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-white tracking-tight">
            {formatCurrency(totalLiabilities)}
          </div>
          <div className="flex items-center gap-2 mt-2 text-xs">
            <span className="text-emerald-400 font-medium flex items-center gap-0.5 bg-emerald-500/10 px-1.5 py-0.5 rounded">
              <ArrowUpRight className="w-3.5 h-3.5" /> +2.4%
            </span>
            <span className="text-slate-500">vs last quarter</span>
          </div>
        </div>

        {/* Card 2 */}
        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-amber-500/40 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-amber-500/5 rounded-full blur-2xl group-hover:bg-amber-500/10 transition-all" />
          <div className="flex justify-between items-start mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Accounts Payable (AP)
            </span>
            <div className="p-2 bg-amber-500/10 text-amber-400 rounded-lg border border-amber-500/20">
              <FileText className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-white tracking-tight">
            {formatCurrency(totalAPOutstanding)}
          </div>
          <div className="flex items-center justify-between mt-2 text-xs">
            <span className="text-amber-400 font-medium bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20">
              Overdue: {formatCurrency(totalOverdueAP)}
            </span>
            <span className="text-slate-400 font-mono">{apBills.filter(b => b.status !== 'PAID').length} Pending</span>
          </div>
        </div>

        {/* Card 3 */}
        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-cyan-500/40 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-cyan-500/5 rounded-full blur-2xl group-hover:bg-cyan-500/10 transition-all" />
          <div className="flex justify-between items-start mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              VAT & Sales Tax Payable
            </span>
            <div className="p-2 bg-cyan-500/10 text-cyan-400 rounded-lg border border-cyan-500/20">
              <Shield className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-white tracking-tight">
            {formatCurrency(currentVATNetPayable)}
          </div>
          <div className="flex items-center gap-2 mt-2 text-xs text-slate-400">
            <Clock className="w-3.5 h-3.5 text-cyan-400" />
            <span>Next Filing: <strong className="text-slate-200">Apr 20, 2025</strong></span>
          </div>
        </div>

        {/* Card 4 */}
        <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800/80 backdrop-blur-md relative overflow-hidden group hover:border-violet-500/40 transition-all">
          <div className="absolute top-0 right-0 w-24 h-24 bg-violet-500/5 rounded-full blur-2xl group-hover:bg-violet-500/10 transition-all" />
          <div className="flex justify-between items-start mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Funded Debt Balance
            </span>
            <div className="p-2 bg-violet-500/10 text-violet-400 rounded-lg border border-violet-500/20">
              <CreditCard className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-extrabold text-white tracking-tight">
            {formatCurrency(totalDebtBalance)}
          </div>
          <div className="flex items-center justify-between mt-2 text-xs">
            <span className="text-slate-400">Monthly Debt Service:</span>
            <span className="text-violet-400 font-semibold font-mono">{formatCurrency(totalMonthlyDebtService)}/mo</span>
          </div>
        </div>
      </section>

      {/* NAVIGATION TABS */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2 overflow-x-auto no-scrollbar">
        {[
          { id: 'overview', label: 'Dashboard Analytics', icon: BarChart2 },
          { id: 'ap', label: 'Accounts Payable', icon: FileText, badge: apBills.filter(b => b.status === 'UNPAID' || b.status === 'OVERDUE').length },
          { id: 'vat', label: 'VAT & Tax Provisions', icon: Shield },
          { id: 'debt', label: 'Debt Facilities & Loans', icon: CreditCard },
          { id: 'amortization', label: 'Amortization Calculator', icon: Calculator },
        ].map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as TabType)}
              className={`flex items-center gap-2.5 px-4 py-2.5 rounded-xl text-xs font-semibold transition-all whitespace-nowrap cursor-pointer ${
                isActive
                  ? 'bg-indigo-600/15 text-indigo-400 border border-indigo-500/30 shadow-lg shadow-indigo-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/40 border border-transparent'
              }`}
            >
              <Icon className={`w-4 h-4 ${isActive ? 'text-indigo-400' : 'text-slate-400'}`} />
              <span>{tab.label}</span>
              {tab.badge !== undefined && tab.badge > 0 && (
                <span className="bg-rose-500/20 text-rose-400 border border-rose-500/30 text-[10px] px-1.5 py-0.2 rounded-full font-mono font-bold">
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* TAB 1: OVERVIEW ANALYTICS */}
      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Chart Card Simulation */}
          <div className="lg:col-span-2 bg-slate-900/60 p-6 rounded-2xl border border-slate-800/80 backdrop-blur-md flex flex-col justify-between">
            <div className="flex items-center justify-between mb-6">
              <div>
                <h3 className="text-base font-semibold text-white">Liabilities Cash Outflow Projection</h3>
                <p className="text-xs text-slate-400">30-60-90 Day commitments breakdown</p>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="flex items-center gap-1 text-slate-400"><span className="w-2.5 h-2.5 rounded-full bg-indigo-500"></span> AP Bills</span>
                <span className="flex items-center gap-1 text-slate-400"><span className="w-2.5 h-2.5 rounded-full bg-violet-500"></span> Debt Service</span>
                <span className="flex items-center gap-1 text-slate-400"><span className="w-2.5 h-2.5 rounded-full bg-cyan-500"></span> Tax Payable</span>
              </div>
            </div>

            {/* Custom CSS Bar Graph Visualization */}
            <div className="space-y-5 my-4">
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Next 30 Days Outflow</span>
                  <span className="text-indigo-400 font-mono font-semibold">$214,800</span>
                </div>
                <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden flex">
                  <div className="bg-indigo-500 h-full" style={{ width: '55%' }} />
                  <div className="bg-violet-500 h-full" style={{ width: '30%' }} />
                  <div className="bg-cyan-500 h-full" style={{ width: '15%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">31 - 60 Days Outflow</span>
                  <span className="text-indigo-400 font-mono font-semibold">$348,200</span>
                </div>
                <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden flex">
                  <div className="bg-indigo-500 h-full" style={{ width: '70%' }} />
                  <div className="bg-violet-500 h-full" style={{ width: '20%' }} />
                  <div className="bg-cyan-500 h-full" style={{ width: '10%' }} />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">61 - 90 Days Outflow</span>
                  <span className="text-indigo-400 font-mono font-semibold">$192,000</span>
                </div>
                <div className="h-3 w-full bg-slate-800/80 rounded-full overflow-hidden flex">
                  <div className="bg-indigo-500 h-full" style={{ width: '40%' }} />
                  <div className="bg-violet-500 h-full" style={{ width: '45%' }} />
                  <div className="bg-cyan-500 h-full" style={{ width: '15%' }} />
                </div>
              </div>
            </div>

            <div className="p-4 bg-slate-950/40 rounded-xl border border-slate-800/60 flex items-center justify-between text-xs mt-4">
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
                <span className="text-slate-300">
                  <strong className="text-white">Cash Liquidity Alert:</strong> Total 30-day obligations equal <span className="text-amber-400 font-semibold font-mono">$214,800</span>. Ensure working capital reserves.
                </span>
              </div>
              <button 
                onClick={() => setActiveTab('ap')} 
                className="text-indigo-400 hover:text-indigo-300 font-medium whitespace-nowrap flex items-center gap-1"
              >
                Review AP <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Breakdown & Ratios */}
          <div className="bg-slate-900/60 p-6 rounded-2xl border border-slate-800/80 backdrop-blur-md flex flex-col justify-between">
            <div>
              <h3 className="text-base font-semibold text-white mb-1">Liability Capital Structure</h3>
              <p className="text-xs text-slate-400 mb-6">Distribution across financial debt, trade payables & tax</p>

              <div className="space-y-4">
                <div className="p-3.5 bg-slate-800/40 rounded-xl border border-slate-800 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full bg-violet-500" />
                    <div>
                      <div className="text-xs font-semibold text-slate-200">Bank & Funded Debt</div>
                      <div className="text-[10px] text-slate-400">Long-term & Revolving</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-bold text-white font-mono">{formatCurrency(totalDebtBalance)}</div>
                    <div className="text-[10px] text-slate-400 font-mono">
                      {((totalDebtBalance / (totalLiabilities || 1)) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="p-3.5 bg-slate-800/40 rounded-xl border border-slate-800 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full bg-amber-500" />
                    <div>
                      <div className="text-xs font-semibold text-slate-200">Trade Accounts Payable</div>
                      <div className="text-[10px] text-slate-400">Vendor Invoices</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-bold text-white font-mono">{formatCurrency(totalAPOutstanding)}</div>
                    <div className="text-[10px] text-slate-400 font-mono">
                      {((totalAPOutstanding / (totalLiabilities || 1)) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="p-3.5 bg-slate-800/40 rounded-xl border border-slate-800 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-3 h-3 rounded-full bg-cyan-500" />
                    <div>
                      <div className="text-xs font-semibold text-slate-200">VAT & Sales Tax</div>
                      <div className="text-[10px] text-slate-400">Current Tax Liability</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs font-bold text-white font-mono">{formatCurrency(currentVATNetPayable)}</div>
                    <div className="text-[10px] text-slate-400 font-mono">
                      {((currentVATNetPayable / (totalLiabilities || 1)) * 100).toFixed(1)}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t border-slate-800">
              <div className="flex justify-between items-center text-xs">
                <span className="text-slate-400">Est. Debt-to-Equity Ratio:</span>
                <span className="text-emerald-400 font-mono font-bold">1.42 (Healthy)</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: ACCOUNTS PAYABLE (AP) */}
      {activeTab === 'ap' && (
        <div className="bg-slate-900/60 rounded-2xl border border-slate-800/80 backdrop-blur-md overflow-hidden flex flex-col">
          {/* Table Header Controls */}
          <div className="p-5 border-b border-slate-800/80 flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="relative flex-1 sm:w-72">
                <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search vendor, invoice..."
                  value={apSearch}
                  onChange={e => setApSearch(e.target.value)}
                  className="w-full bg-slate-950/60 border border-slate-800 text-slate-200 text-xs rounded-xl pl-9 pr-4 py-2 focus:outline-none focus:border-indigo-500 transition-colors"
                />
              </div>

              <div className="flex items-center gap-2">
                <Filter className="w-4 h-4 text-slate-400" />
                <select
                  value={apStatusFilter}
                  onChange={e => setApStatusFilter(e.target.value)}
                  className="bg-slate-950/60 border border-slate-800 text-slate-300 text-xs rounded-xl px-3 py-2 focus:outline-none focus:border-indigo-500"
                >
                  <option value="ALL">All Statuses</option>
                  <option value="UNPAID">Unpaid</option>
                  <option value="PARTIAL">Partial</option>
                  <option value="OVERDUE">Overdue</option>
                  <option value="PAID">Paid</option>
                </select>
              </div>
            </div>

            <button
              onClick={() => setIsAddApModalOpen(true)}
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20"
            >
              <Plus className="w-4 h-4" /> Add Vendor Invoice
            </button>
          </div>

          {/* Table Content */}
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-800 bg-slate-950/40 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                  <th className="p-4">Vendor & Details</th>
                  <th className="p-4">Category</th>
                  <th className="p-4">Due Date</th>
                  <th className="p-4 text-right">Invoice Amount</th>
                  <th className="p-4 text-right">VAT Component</th>
                  <th className="p-4 text-right">Balance Due</th>
                  <th className="p-4 text-center">Status</th>
                  <th className="p-4 text-center">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-xs">
                {filteredAP.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="text-center p-8 text-slate-500">
                      No accounts payable records found matching criteria.
                    </td>
                  </tr>
                ) : (
                  filteredAP.map(bill => {
                    const balance = bill.amount - bill.paidAmount;
                    return (
                      <tr key={bill.id} className="hover:bg-slate-800/30 transition-colors">
                        <td className="p-4">
                          <div className="font-semibold text-slate-100">{bill.vendor}</div>
                          <div className="text-[11px] text-slate-500 font-mono">
                            {bill.invoiceNo} &bull; ID: {bill.id}
                          </div>
                        </td>
                        <td className="p-4">
                          <span className="bg-slate-800 border border-slate-700 text-slate-300 px-2.5 py-1 rounded-lg text-[10px]">
                            {bill.category}
                          </span>
                        </td>
                        <td className="p-4 text-slate-300 font-mono">
                          {bill.dueDate}
                        </td>
                        <td className="p-4 text-right font-mono font-semibold text-slate-200">
                          {formatCurrency(bill.amount)}
                        </td>
                        <td className="p-4 text-right font-mono text-slate-400">
                          {formatCurrency(bill.vatAmount)}
                        </td>
                        <td className="p-4 text-right font-mono font-bold text-white">
                          {formatCurrency(balance)}
                        </td>
                        <td className="p-4 text-center">
                          {bill.status === 'PAID' && (
                            <span className="inline-flex items-center gap-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-2.5 py-0.5 rounded-full text-[10px] font-bold">
                              <Check className="w-3 h-3" /> PAID
                            </span>
                          )}
                          {bill.status === 'UNPAID' && (
                            <span className="inline-flex items-center gap-1 bg-blue-500/10 border border-blue-500/30 text-blue-400 px-2.5 py-0.5 rounded-full text-[10px] font-bold">
                              UNPAID
                            </span>
                          )}
                          {bill.status === 'PARTIAL' && (
                            <span className="inline-flex items-center gap-1 bg-amber-500/10 border border-amber-500/30 text-amber-400 px-2.5 py-0.5 rounded-full text-[10px] font-bold">
                              PARTIAL
                            </span>
                          )}
                          {bill.status === 'OVERDUE' && (
                            <span className="inline-flex items-center gap-1 bg-rose-500/10 border border-rose-500/30 text-rose-400 px-2.5 py-0.5 rounded-full text-[10px] font-bold animate-pulse">
                              <AlertTriangle className="w-3 h-3" /> OVERDUE
                            </span>
                          )}
                        </td>
                        <td className="p-4 text-center">
                          {bill.status !== 'PAID' ? (
                            <button
                              onClick={() => handlePayBill(bill.id)}
                              className="bg-indigo-600/20 hover:bg-indigo-600 border border-indigo-500/40 hover:border-indigo-500 text-indigo-300 hover:text-white px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
                            >
                              Settle Bill
                            </button>
                          ) : (
                            <span className="text-slate-500 text-[11px]">Settled</span>
                          )}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 3: VAT & SALES TAX PAYABLE */}
      {activeTab === 'vat' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800/80 backdrop-blur-md">
              <div className="text-xs text-slate-400 uppercase font-semibold">Output VAT Collected (Sales)</div>
              <div className="text-2xl font-bold text-emerald-400 font-mono mt-2">
                {formatCurrency(vatPeriods[0].outputVAT)}
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Tax collected on taxable sales</p>
            </div>
            <div className="bg-slate-900/60 p-5 rounded-2xl border border-slate-800/80 backdrop-blur-md">
              <div className="text-xs text-slate-400 uppercase font-semibold">Input VAT Deductible (Purchases)</div>
              <div className="text-2xl font-bold text-amber-400 font-mono mt-2">
                {formatCurrency(vatPeriods[0].inputVAT)}
              </div>
              <p className="text-[11px] text-slate-500 mt-1">Tax credit paid on operational expenses</p>
            </div>
            <div className="bg-slate-900/60 p-5 rounded-2xl border border-indigo-500/40 bg-indigo-500/5 backdrop-blur-md">
              <div className="text-xs text-indigo-300 uppercase font-semibold">Net VAT Liability Payable</div>
              <div className="text-2xl font-bold text-white font-mono mt-2">
                {formatCurrency(vatPeriods[0].netPayable)}
              </div>
              <p className="text-[11px] text-indigo-400 mt-1">Due for settlement by {vatPeriods[0].dueDate}</p>
            </div>
          </div>

          {/* Tax Periods Ledger */}
          <div className="bg-slate-900/60 rounded-2xl border border-slate-800/80 backdrop-blur-md overflow-hidden">
            <div className="p-5 border-b border-slate-800/80 flex justify-between items-center">
              <div>
                <h3 className="text-base font-semibold text-white">VAT Return Audit History</h3>
                <p className="text-xs text-slate-400">Quarterly tax filing and settlement ledger</p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-800 bg-slate-950/40 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                    <th className="p-4">Tax Period</th>
                    <th className="p-4 text-right">Output VAT (Sales)</th>
                    <th className="p-4 text-right">Input VAT (Purchases)</th>
                    <th className="p-4 text-right">Net Tax Due</th>
                    <th className="p-4">Due Date</th>
                    <th className="p-4 text-center">Filing Status</th>
                    <th className="p-4 text-center">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-xs">
                  {vatPeriods.map(period => (
                    <tr key={period.id} className="hover:bg-slate-800/30 transition-colors">
                      <td className="p-4 font-semibold text-white">
                        {period.period}
                        <div className="text-[10px] text-slate-500 font-mono">{period.id}</div>
                      </td>
                      <td className="p-4 text-right font-mono text-emerald-400">
                        {formatCurrency(period.outputVAT)}
                      </td>
                      <td className="p-4 text-right font-mono text-amber-400">
                        {formatCurrency(period.inputVAT)}
                      </td>
                      <td className="p-4 text-right font-mono font-bold text-white">
                        {formatCurrency(period.netPayable)}
                      </td>
                      <td className="p-4 text-slate-300 font-mono">{period.dueDate}</td>
                      <td className="p-4 text-center">
                        {period.status === 'SETTLED' ? (
                          <span className="bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 px-2.5 py-0.5 rounded-full text-[10px] font-bold">
                            SETTLED
                          </span>
                        ) : (
                          <span className="bg-amber-500/10 border border-amber-500/30 text-amber-400 px-2.5 py-0.5 rounded-full text-[10px] font-bold">
                            DRAFT PENDING
                          </span>
                        )}
                      </td>
                      <td className="p-4 text-center">
                        {period.status !== 'SETTLED' ? (
                          <button
                            onClick={() => handleFileVAT(period.id)}
                            className="bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-lg text-xs font-medium transition-all shadow-md shadow-indigo-600/20"
                          >
                            File & Remit Tax
                          </button>
                        ) : (
                          <span className="text-slate-500 text-[11px] flex items-center justify-center gap-1">
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-500" /> Filed
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: DEBT & LOANS */}
      {activeTab === 'debt' && (
        <div className="space-y-6">
          <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-4">
            <div className="relative flex-1 sm:w-72">
              <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search lender or facility..."
                value={debtSearch}
                onChange={e => setDebtSearch(e.target.value)}
                className="w-full bg-slate-900/60 border border-slate-800 text-slate-200 text-xs rounded-xl pl-9 pr-4 py-2 focus:outline-none focus:border-indigo-500"
              />
            </div>
            <button
              onClick={() => setIsAddDebtModalOpen(true)}
              className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-all flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/20"
            >
              <Plus className="w-4 h-4" /> Add Debt Facility
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {filteredDebt.map(facility => {
              const paidPercent = Math.round(
                ((facility.principal - facility.outstandingBalance) / facility.principal) * 100
              );
              return (
                <div
                  key={facility.id}
                  className="bg-slate-900/60 p-6 rounded-2xl border border-slate-800/80 backdrop-blur-md flex flex-col justify-between hover:border-slate-700 transition-all"
                >
                  <div>
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <span className="text-[10px] font-mono bg-violet-500/10 text-violet-400 border border-violet-500/20 px-2 py-0.5 rounded">
                          {facility.type.replace('_', ' ')}
                        </span>
                        <h4 className="text-base font-bold text-white mt-2">{facility.lender}</h4>
                      </div>
                      <span className="text-xs font-mono font-semibold text-emerald-400 bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">
                        {facility.interestRate}% APR
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-4 my-4 p-4 bg-slate-950/40 rounded-xl border border-slate-800/60">
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase font-semibold">Original Principal</div>
                        <div className="text-sm font-bold text-slate-200 font-mono mt-0.5">
                          {formatCurrency(facility.principal)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase font-semibold">Outstanding Balance</div>
                        <div className="text-sm font-bold text-violet-400 font-mono mt-0.5">
                          {formatCurrency(facility.outstandingBalance)}
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase font-semibold">Monthly Service</div>
                        <div className="text-xs font-semibold text-slate-300 font-mono mt-0.5">
                          {formatCurrency(facility.monthlyPayment)}/mo
                        </div>
                      </div>
                      <div>
                        <div className="text-[10px] text-slate-400 uppercase font-semibold">Maturity Date</div>
                        <div className="text-xs font-semibold text-slate-300 font-mono mt-0.5">
                          {facility.maturityDate}
                        </div>
                      </div>
                    </div>

                    {/* Principal Reduction Progress */}
                    <div className="space-y-1.5 mb-2">
                      <div className="flex justify-between text-[11px] text-slate-400">
                        <span>Principal Repaid</span>
                        <span className="font-mono text-slate-200">{paidPercent}%</span>
                      </div>
                      <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div
                          className="bg-gradient-to-r from-indigo-500 to-violet-500 h-full rounded-full transition-all duration-500"
                          style={{ width: `${paidPercent}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center justify-between pt-4 border-t border-slate-800/80 mt-4 text-xs">
                    <span className="text-slate-400">
                      Next Due: <strong className="text-slate-200">{facility.nextPaymentDate}</strong>
                    </span>
                    <button
                      onClick={() => triggerToast(`Payment of ${formatCurrency(facility.monthlyPayment)} scheduled for ${facility.lender}`)}
                      className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 font-medium transition-all"
                    >
                      Process Payment
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* TAB 5: AMORTIZATION CALCULATOR */}
      {activeTab === 'amortization' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls Form */}
          <div className="bg-slate-900/60 p-6 rounded-2xl border border-slate-800/80 backdrop-blur-md flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-2 mb-4">
                <Calculator className="w-5 h-5 text-indigo-400" />
                <h3 className="text-base font-semibold text-white">Loan Amortization Simulator</h3>
              </div>

              <div className="space-y-4 text-xs">
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Loan Principal ($)</label>
                  <input
                    type="number"
                    value={calcPrincipal}
                    onChange={e => setCalcPrincipal(Number(e.target.value))}
                    className="w-full bg-slate-950/60 border border-slate-800 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Annual Interest Rate (%)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={calcRate}
                    onChange={e => setCalcRate(Number(e.target.value))}
                    className="w-full bg-slate-950/60 border border-slate-800 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:border-indigo-500"
                  />
                </div>

                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Term Duration (Years)</label>
                  <select
                    value={calcYears}
                    onChange={e => setCalcYears(Number(e.target.value))}
                    className="w-full bg-slate-950/60 border border-slate-800 rounded-xl px-3.5 py-2.5 text-white font-mono focus:outline-none focus:border-indigo-500"
                  >
                    <option value={1}>1 Year (12 months)</option>
                    <option value={3}>3 Years (36 months)</option>
                    <option value={5}>5 Years (60 months)</option>
                    <option value={7}>7 Years (84 months)</option>
                    <option value={10}>10 Years (120 months)</option>
                  </select>
                </div>
              </div>
            </div>

            <div className="p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-xl mt-6">
              <div className="text-[11px] text-indigo-300 font-semibold uppercase">Estimated Monthly Payment</div>
              <div className="text-2xl font-extrabold text-white font-mono mt-1">
                {formatCurrency(amortizationSchedule.pmt)}
              </div>
              <div className="text-[10px] text-slate-400 mt-1">
                Total P+I: {formatCurrency(amortizationSchedule.pmt * calcYears * 12)}
              </div>
            </div>
          </div>

          {/* Schedule Table Preview */}
          <div className="lg:col-span-2 bg-slate-900/60 rounded-2xl border border-slate-800/80 backdrop-blur-md overflow-hidden flex flex-col">
            <div className="p-5 border-b border-slate-800/80">
              <h4 className="text-sm font-semibold text-white">Amortization Payment Schedule (First 24 Months)</h4>
              <p className="text-xs text-slate-400">Monthly breakdown of principal vs interest allocations</p>
            </div>

            <div className="overflow-x-auto max-h-[420px]">
              <table className="w-full text-left border-collapse">
                <thead className="sticky top-0 bg-slate-950 border-b border-slate-800 text-[11px] uppercase text-slate-400 font-mono">
                  <tr>
                    <th className="p-3">Mth</th>
                    <th className="p-3 text-right">Payment</th>
                    <th className="p-3 text-right">Principal</th>
                    <th className="p-3 text-right">Interest</th>
                    <th className="p-3 text-right">Remaining Balance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50 text-xs font-mono">
                  {amortizationSchedule.schedule.map(row => (
                    <tr key={row.month} className="hover:bg-slate-800/30">
                      <td className="p-3 text-slate-400 font-semibold">{row.month}</td>
                      <td className="p-3 text-right text-slate-200">{formatCurrency(row.payment)}</td>
                      <td className="p-3 text-right text-emerald-400">{formatCurrency(row.principalPaid)}</td>
                      <td className="p-3 text-right text-rose-400">{formatCurrency(row.interestPaid)}</td>
                      <td className="p-3 text-right text-slate-100">{formatCurrency(row.remainingBalance)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* MODAL: ADD NEW AP BILL */}
      {isAddApModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 w-full max-w-md shadow-2xl relative animate-in fade-in zoom-in duration-200">
            <button
              onClick={() => setIsAddApModalOpen(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
            <h3 className="text-lg font-bold text-white mb-1">Create AP Vendor Invoice</h3>
            <p className="text-xs text-slate-400 mb-5">Record a new trade liability in the accounts payable ledger.</p>

            <form onSubmit={handleAddAPBill} className="space-y-4 text-xs">
              <div>
                <label className="block text-slate-300 mb-1 font-medium">Vendor Name</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Acme Industrial Solutions"
                  value={newVendor}
                  onChange={e => setNewVendor(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Invoice Reference #</label>
                  <input
                    type="text"
                    placeholder="INV-9901"
                    value={newInvoiceNo}
                    onChange={e => setNewInvoiceNo(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Category</label>
                  <select
                    value={newCategory}
                    onChange={e => setNewCategory(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="IT Infrastructure">IT Infrastructure</option>
                    <option value="Raw Materials">Raw Materials</option>
                    <option value="Freight & Shipping">Freight & Shipping</option>
                    <option value="Utilities">Utilities</option>
                    <option value="Maintenance">Maintenance</option>
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Total Amount ($)</label>
                  <input
                    type="number"
                    required
                    placeholder="25000"
                    value={newAmount}
                    onChange={e => setNewAmount(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                </div>
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Due Date</label>
                  <input
                    type="date"
                    required
                    value={newDueDate}
                    onChange={e => setNewDueDate(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsAddApModalOpen(false)}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium shadow-lg shadow-indigo-600/30"
                >
                  Post Liability
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* MODAL: ADD NEW DEBT FACILITY */}
      {isAddDebtModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 w-full max-w-md shadow-2xl relative animate-in fade-in zoom-in duration-200">
            <button
              onClick={() => setIsAddDebtModalOpen(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
            <h3 className="text-lg font-bold text-white mb-1">Register Debt Facility</h3>
            <p className="text-xs text-slate-400 mb-5">Add long-term or revolving debt obligation.</p>

            <form onSubmit={handleAddDebt} className="space-y-4 text-xs">
              <div>
                <label className="block text-slate-300 mb-1 font-medium">Lender Institution</label>
                <input
                  type="text"
                  required
                  placeholder="e.g. Goldman Sachs Credit Corp"
                  value={newLender}
                  onChange={e => setNewLender(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Facility Type</label>
                  <select
                    value={newDebtType}
                    onChange={e => setNewDebtType(e.target.value as DebtType)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                  >
                    <option value="BANK_LOAN">Bank Loan</option>
                    <option value="REVOLVING_CREDIT">Revolving Credit</option>
                    <option value="EQUIPMENT_LEASE">Equipment Lease</option>
                    <option value="BOND">Corporate Bond</option>
                  </select>
                </div>
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Principal Amount ($)</label>
                  <input
                    type="number"
                    required
                    placeholder="1000000"
                    value={newPrincipal}
                    onChange={e => setNewPrincipal(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Interest Rate (% APR)</label>
                  <input
                    type="number"
                    step="0.01"
                    required
                    placeholder="5.5"
                    value={newRate}
                    onChange={e => setNewRate(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"
                  />
                </div>
                <div>
                  <label className="block text-slate-300 mb-1 font-medium">Maturity Date</label>
                  <input
                    type="date"
                    required
                    value={newMaturity}
                    onChange={e => setNewMaturity(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-xl px-3.5 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                  />
                </div>
              </div>

              <div className="flex justify-end gap-3 mt-6 pt-4 border-t border-slate-800">
                <button
                  type="button"
                  onClick={() => setIsAddDebtModalOpen(false)}
                  className="px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium shadow-lg shadow-indigo-600/30"
                >
                  Register Debt
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

export default AccountantErpApp;