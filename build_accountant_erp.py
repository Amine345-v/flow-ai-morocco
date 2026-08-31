"""
Autonomous App Generator for JOL Studio IDE.

Invokes Gemini AI via FlowLang AI Engine to synthesize a complete, professional-grade
Accounting ERP Web Application, and registers it into JOL Studio IDE.
"""

import os
import sys
import json
import time
from pathlib import Path

# Ensure workspace root is in sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# Auto load .env
env_file = os.path.join(workspace_root, "flowlang", ".env")
if os.path.exists(env_file):
    with open(env_file, "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ[k] = v

from flowlang.ai_providers import select_provider

TARGET_APP_PATH = Path(workspace_root) / "jol-ide" / "components" / "apps" / "AccountantERP.tsx"


def generate_accountant_erp():
    print("=== JOL Studio Autonomous App Generator ===")
    print("Initializing Gemini AI Provider...")

    ai = select_provider()
    if not ai:
        print("ERROR: AI Provider not available. Check GEMINI_API_KEY.")
        return False

    print(f"AI Provider Selected: {ai.name}")
    print("Asking Gemini AI to synthesize full Accountant ERP Software Component...")

    # We prompt Gemini AI to generate the complete React TSX code for the Accounting ERP
    prompt = """
    Create a complete, state-of-the-art, feature-rich React TypeScript component named AccountantERP.
    The component must be a full-fledged ERP System for Accountants and Financial Controllers.

    Include the following core modules:
    1. Navigation Header & Dashboard Overview:
       - KPIs: Total Revenue, Total Expenses, Net Income, VAT Payable, Accounts Receivable, Cash Balance.
       - Quick action buttons (New Transaction, Issue Invoice, Run Financial Report).
    2. Chart of Accounts (COA):
       - Assets (Cash, Accounts Receivable, Equipment)
       - Liabilities (Accounts Payable, VAT Payable, Bank Loans)
       - Equity (Owner Capital, Retained Earnings)
       - Revenue (Sales Revenue, Service Income)
       - Expenses (Salaries, Rent, Utilities, Software Licenses)
    3. Double-Entry Journal Ledger:
       - Table displaying Journal Entries with Date, Account, Description, Debit ($), Credit ($), and Status.
       - Form to create new balanced journal entries (validating Debit == Credit).
    4. Invoicing & Billing Module:
       - List of Customer Invoices (Invoice #, Client, Date, Amount, VAT 20%, Paid/Pending status).
       - Modal or form to generate new professional invoices.
    5. Financial Reports & Statements:
       - Income Statement (Profit & Loss)
       - Balance Sheet (Assets = Liabilities + Equity)
       - VAT / Tax Return Summary
    6. Aesthetics & Styling:
       - Modern dark mode matching JOL Studio (#0b1121 theme), with Tailwind CSS styles, sleek badges, glowing gradients, and Lucide React icons.

    Do not include markdown triple backticks in code output if possible, or wrap cleanly.
    Return ONLY valid, compilable React TSX code using standard 'lucide-react' icons.
    """

    res = ai.execute(
        team="software_engineers",
        verb="try",
        args=[prompt],
        kwargs={"max_tokens": 4000, "temperature": 0.2}
    )

    output_text = res.meta.get("output", "") or str(res)
    print(f"AI Output generated ({len(output_text)} bytes).")

    # Clean markdown code fences if present
    if "```" in output_text:
        lines = output_text.splitlines()
        cleaned_lines = []
        in_code = False
        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            cleaned_lines.append(line)
        output_text = "\n".join(cleaned_lines)

    # Ensure parent dir exists
    TARGET_APP_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(TARGET_APP_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Successfully generated AccountantERP.tsx at: {TARGET_APP_PATH}")
    return True


if __name__ == "__main__":
    generate_accountant_erp()
