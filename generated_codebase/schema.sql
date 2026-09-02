-- ============================================================================
-- FlowLang DDL Schema Generation: Accountant ERP Enterprise System
-- Target Engine: PostgreSQL 16 | Synthesized by FlowLang AI Team
-- ============================================================================

CREATE TABLE IF NOT EXISTS enterprise_tenants (
    tenant_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_name VARCHAR(255) NOT NULL,
    tax_id VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chart_of_accounts (
    account_id VARCHAR(20) PRIMARY KEY,
    tenant_id UUID REFERENCES enterprise_tenants(tenant_id) ON DELETE CASCADE,
    account_name VARCHAR(255) NOT NULL,
    account_type VARCHAR(50) CHECK (account_type IN ('ASSET', 'LIABILITY', 'EQUITY', 'REVENUE', 'EXPENSE')),
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS journal_entries (
    entry_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES enterprise_tenants(tenant_id),
    posting_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    description TEXT NOT NULL,
    checksum VARCHAR(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS journal_lines (
    line_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id UUID REFERENCES journal_entries(entry_id) ON DELETE CASCADE,
    account_id VARCHAR(20) REFERENCES chart_of_accounts(account_id),
    debit DECIMAL(18, 4) DEFAULT 0.0000,
    credit DECIMAL(18, 4) DEFAULT 0.0000
);

CREATE INDEX idx_journal_lines_entry ON journal_lines(entry_id);
CREATE INDEX idx_journal_lines_account ON journal_lines(account_id);
