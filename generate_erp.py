import os
import sys
import google.generativeai as genai
from pathlib import Path

# Load .env key
key = "AIzaSyBh4AFxbzLHpZCGzMYXlkSKS79svs40iHE"
genai.configure(api_key=key)

model = genai.GenerativeModel("gemini-3.6-flash")

prompt = """
Write a complete, state-of-the-art React TypeScript component named 'AccountantERP'.
This component is a full-featured ERP System for Accountants and Financial Controllers.

Requirements:
- Imports: React, useState, Lucide icons from 'lucide-react' (e.g. DollarSign, FileText, Calculator, TrendingUp, PieChart, Layers, Plus, CheckCircle, Shield, Building, CreditCard, ArrowUpRight, ArrowDownRight, RefreshCw, BarChart2).
- Sections / Tabs:
  1. Dashboard & KPIs: Total Assets ($1,450,000), Total Liabilities ($420,000), Equity ($1,030,000), Revenue YTD ($680,000), Net Profit Margin (28.4%), VAT Payable ($34,500).
  2. Chart of Accounts (COA): Interactive list of accounts (Code, Account Name, Type: Asset/Liability/Equity/Revenue/Expense, Balance, Action).
  3. Double-Entry Journal Ledger: Table of Journal Entries (Date, Reference, Account, Debit, Credit, Status) + Form to add new balanced entry (validating Debit == Credit).
  4. Invoices & AR/AP: List of Customer Invoices with Status (Paid, Overdue, Draft), Tax (VAT 20%), Total Amount + 'Create Invoice' modal button.
  5. Financial Statements: Tabbed view for Income Statement (P&L), Balance Sheet, and Tax/VAT Return Summary.
- Visual Polish: Dark mode matching JOL Studio (#0b1121 theme), glassmorphism, vibrant badges (cyan, green, purple, amber, red), smooth tab switches, and live interactive state!

OUTPUT ONLY VALID TSX CODE. DO NOT INCLUDE MARKDOWN CODE FENCES. START DIRECTLY WITH 'import React...'
"""

print("Synthesizing AccountantERP.tsx using Gemini 3.6 Flash...")
response = model.generate_content(prompt)
code_text = response.text.strip()

if code_text.startswith("```"):
    lines = code_text.splitlines()
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines[-1].startswith("```"):
        lines = lines[:-1]
    code_text = "\n".join(lines)

target_path = Path("c:/Users/asusu/CascadeProjects/flowlang/jol-ide/components/apps/AccountantERP.tsx")
target_path.parent.mkdir(parents=True, exist_ok=True)

with open(target_path, "w", encoding="utf-8") as f:
    f.write(code_text)

print(f"DONE! Written {len(code_text)} bytes to {target_path}")
