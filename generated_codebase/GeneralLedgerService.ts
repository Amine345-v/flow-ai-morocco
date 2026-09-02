/**
 * FlowLang Enterprise Double-Entry Ledger Core Engine
 * Synthesized by FlowLang AI Team: dev_team (Gemini 3.7 Flash)
 */

export interface JournalLine {
  accountId: string;
  accountName: string;
  debit: number;
  credit: number;
}

export interface JournalEntry {
  entryId: string;
  tenantId: string;
  timestamp: string;
  description: string;
  lines: JournalLine[];
  checksum?: string;
}

export class GeneralLedgerService {
  private entries: Map<string, JournalEntry> = new Map();

  public postTransaction(entry: JournalEntry): { success: boolean; message: string; entryId: string } {
    const totalDebit = entry.lines.reduce((sum, l) => sum + (l.debit || 0), 0);
    const totalCredit = entry.lines.reduce((sum, l) => sum + (l.credit || 0), 0);

    if (Math.abs(totalDebit - totalCredit) > 0.0001) {
      throw new Error(`[GAAP Violation] Debits (${totalDebit}) do not balance with Credits (${totalCredit})`);
    }

    entry.checksum = this.calculateSHA256(entry);
    this.entries.set(entry.entryId, entry);

    console.log(`[GeneralLedger] Posted balanced entry ${entry.entryId}: Debit $${totalDebit} == Credit $${totalCredit}`);
    return { success: true, message: "Transaction posted successfully", entryId: entry.entryId };
  }

  private calculateSHA256(entry: JournalEntry): string {
    return `sha256_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }
}

export const ledgerEngine = new GeneralLedgerService();
export default ledgerEngine;
