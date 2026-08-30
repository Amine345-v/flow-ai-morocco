import * as path from 'path';
import * as fs from 'fs';
import * as cp from 'child_process';
import {
    createConnection,
    TextDocuments,
    ProposedFeatures,
    InitializeParams,
    TextDocumentSyncKind,
    InitializeResult,
    TextDocumentChangeEvent,
    TextDocumentSyncOptions,
    DidChangeConfigurationParams,
    PublishDiagnosticsParams,
    Diagnostic,
    DiagnosticSeverity,
    WorkspaceFolder
} from 'vscode-languageserver/node';
import { TextDocument } from 'vscode-languageserver-textdocument';
import { URI } from 'vscode-uri';

// Configuration settings
interface FlowLangSettings {
    enableLinting: boolean;
    pythonPath: string;
    flowPath: string;
}

// Default settings
const defaultSettings: FlowLangSettings = {
    enableLinting: true,
    pythonPath: 'python',
    flowPath: ''
};

// Global settings
let globalSettings: FlowLangSettings = { ...defaultSettings };

// Cache the settings per document
const documentSettings: Map<string, Thenable<FlowLangSettings>> = new Map();

// Create a connection for the server, using Node's IPC as a transport.
const connection = createConnection(ProposedFeatures.all);

// Create a simple text document manager.
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

// Create a log file for debugging
const logFile = path.join(__dirname, '..', '..', 'flowlang-ls.log');

function log(message: string, data?: any) {
    try {
        const timestamp = new Date().toISOString();
        const logMessage = `[${timestamp}] ${message}${data ? ' ' + JSON.stringify(data) : ''}\n`;
        fs.appendFileSync(logFile, logMessage, 'utf8');
    } catch (e) {
        // Cannot use console.log here as it might break the LSP stdio protocol
    }
}

// Log unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    log('Unhandled promise rejection:', { reason, promise });
});

// Log uncaught exceptions
process.on('uncaughtException', (error) => {
    log('Uncaught exception:', error);
    process.exit(1);
});

connection.onInitialize((params: InitializeParams) => {
    log('Language server initializing with params:', params);

    try {
        // Get the workspace folder
        const workspaceFolders = (params.workspaceFolders as WorkspaceFolder[] | null) || [];
        let workspacePath = '';
        if (workspaceFolders.length > 0) {
            workspacePath = URI.parse(workspaceFolders[0].uri).fsPath;
        }

        // Get settings from environment variables or use defaults
        globalSettings = {
            enableLinting: true,
            pythonPath: process.env.PYTHON_PATH || defaultSettings.pythonPath,
            flowPath: process.env.FLOW_PATH || defaultSettings.flowPath
        };

        log('Language server settings:', globalSettings);

        const result: InitializeResult = {
            capabilities: {
                textDocumentSync: {
                    openClose: true,
                    change: TextDocumentSyncKind.Incremental,
                    willSaveWaitUntil: false,
                    save: {
                        includeText: false
                    }
                },
                completionProvider: {
                    resolveProvider: true
                },
                documentFormattingProvider: true,
                documentRangeFormattingProvider: true,
                documentOnTypeFormattingProvider: {
                    firstTriggerCharacter: '\n'
                }
            }
        };

        log('Language server initialized with capabilities:', result.capabilities);
        return result;
    } catch (error) {
        log('Error during initialization:', error);
        throw error;
    }
});

// Listen for document changes
documents.onDidOpen((event: TextDocumentChangeEvent<TextDocument>) => {
    log('Document opened:', { uri: event.document.uri });
    validateTextDocument(event.document);
});

// The content of a text document has changed. This event is emitted
// when the text document first opened or when its content has changed.
documents.onDidChangeContent(change => {
    log('Document changed:', { uri: change.document.uri });
    validateTextDocument(change.document);
});

documents.onDidClose(event => {
    log('Document closed:', { uri: event.document.uri });
    // Clear diagnostics when document is closed
    connection.sendDiagnostics({ uri: event.document.uri, diagnostics: [] });
});

// Validate the given document
async function validateTextDocument(textDocument: TextDocument): Promise<void> {
    if (!globalSettings.enableLinting || !globalSettings.flowPath || !fs.existsSync(globalSettings.flowPath)) {
        return;
    }

    const text = textDocument.getText();
    const diagnostics: Diagnostic[] = [];

    try {
        // Run the FlowLang linter
        const { stdout, stderr } = cp.spawnSync(
            globalSettings.pythonPath,
            [globalSettings.flowPath, '--lint'],
            {
                input: text,
                encoding: 'utf-8',
                timeout: 5000
            }
        );

        if (stderr) {
            log('Linter stderr:', stderr);
        }

        // Parse the linting results and create diagnostics
        if (stdout) {
            const lines = stdout.split('\n');
            for (const line of lines) {
                // Example format: "line 5, col 10: Error message"
                const match = line.match(/line (\d+), col (\d+):\s*(.+)/);
                if (match) {
                    const line = parseInt(match[1]) - 1;
                    const column = parseInt(match[2]) - 1;
                    const message = match[3];

                    const diagnostic: Diagnostic = {
                        severity: DiagnosticSeverity.Error,
                        range: {
                            start: { line, character: column },
                            end: { line, character: column + 1 }
                        },
                        message,
                        source: 'flowlang'
                    };

                    diagnostics.push(diagnostic);
                }
            }
        }
    } catch (error) {
        log('Error running linter:', error);
    }

    // Send the computed diagnostics to VS Code
    connection.sendDiagnostics({
        uri: textDocument.uri,
        diagnostics
    });
}

// Listen for configuration changes
connection.onDidChangeConfiguration((_change: DidChangeConfigurationParams) => {
    // Revalidate all open documents when configuration changes
    documents.all().forEach(validateTextDocument);
});

// Listen for connection close event
connection.onDidChangeTextDocument((change) => {
    log('Document content changed:', { uri: change.textDocument.uri });
});

connection.onDidCloseTextDocument((params) => {
    log('Document closed:', { uri: params.textDocument.uri });
});

connection.onExit(() => {
    log('Server is shutting down');
    process.exit(0);
});

// Make the text document manager listen on the connection
// (open, change and close text document events)
documents.listen(connection);

// This handler provides the initial list of completion items.
connection.onCompletion(() => {
    return [
        { label: 'flow', kind: 14, data: 1 },
        { label: 'checkpoint', kind: 14, data: 2 },
        { label: 'micro_checkpoint', kind: 14, data: 3 },
        { label: 'micro_check', kind: 14, data: 4 },
        { label: 'team', kind: 14, data: 5 },
        { label: 'chain', kind: 14, data: 6 },
        { label: 'process', kind: 14, data: 7 },
        { label: 'role', kind: 14, data: 8 },
        { label: 'policy', kind: 14, data: 9 },
        { label: 'resource', kind: 14, data: 10 },
        { label: 'result', kind: 14, data: 11 },
        { label: 'Command<Search>', kind: 15, data: 12 },
        { label: 'Command<Try>', kind: 15, data: 13 },
        { label: 'Command<Judge>', kind: 15, data: 14 },
        { label: 'Command<Communicate>', kind: 15, data: 15 },
        { label: 'context.update', kind: 3, data: 16 },
        { label: 'flow.back_to', kind: 3, data: 17 },
        { label: 'flow.end', kind: 3, data: 18 },
        { label: 'par', kind: 14, data: 19 },
        { label: 'race', kind: 14, data: 20 },
    ];
});

// This handler resolves additional information for the item selected in the completion list.
connection.onCompletionResolve((item) => {
    const docs: Record<number, { detail: string; documentation: string }> = {
        1: { detail: 'FlowLang flow', documentation: 'Define an execution flow (The Conductor).' },
        2: { detail: 'FlowLang checkpoint', documentation: 'Execution stage that dumps working memory to prevent AI hallucination.' },
        3: { detail: 'FlowLang micro_checkpoint', documentation: 'High-density batch task delegation across worker teams with threshold gating.' },
        4: { detail: 'FlowLang micro_check', documentation: 'Alias for micro_checkpoint.' },
        5: { detail: 'FlowLang team', documentation: 'Homogeneous pool of worker agents executing typed commands.' },
        6: { detail: 'FlowLang chain', documentation: 'Causal dependency graph with decay propagation.' },
        7: { detail: 'FlowLang process', documentation: 'Hierarchical roadmap and audit tree.' },
        8: { detail: 'FlowLang role', documentation: 'Define capabilities and security access controls.' },
        9: { detail: 'FlowLang policy', documentation: 'Governance constraints enforced at runtime.' },
        10: { detail: 'FlowLang resource', documentation: 'External codebase, dataset, or tool declaration.' },
        11: { detail: 'FlowLang result', documentation: 'Structured result schema declaration.' },
        12: { detail: 'Command<Search>', documentation: 'Search verb command type for information retrieval.' },
        13: { detail: 'Command<Try>', documentation: 'Try verb command type for experimentation and code building.' },
        14: { detail: 'Command<Judge>', documentation: 'Judge verb command type for evaluation and quality gating.' },
        15: { detail: 'Command<Communicate>', documentation: 'Communicate verb command type for monologue and reflection.' },
        16: { detail: 'context.update(...)', documentation: 'Persist variables to flow memory.' },
        17: { detail: 'flow.back_to("stage")', documentation: 'Jump back to a previous checkpoint stage.' },
        18: { detail: 'flow.end', documentation: 'Terminate flow execution.' },
        19: { detail: 'par { ... }', documentation: 'Execute statements concurrently.' },
        20: { detail: 'race { ... }', documentation: 'Execute statements in competition; first result wins.' },
    };

    const doc = docs[item.data as number];
    if (doc) {
        item.detail = doc.detail;
        item.documentation = doc.documentation;
    }
    return item;
});

// Start listening
log('Starting FlowLang Language Server...');
connection.listen();
log('FlowLang Language Server is now running');
