import {
    createConnection,
    TextDocuments,
    ProposedFeatures,
    InitializeParams,
    TextDocumentSyncKind,
    InitializeResult
} from 'vscode-languageserver/node';
import { TextDocument } from 'vscode-languageserver-textdocument';

// Create a connection for the server, using Node's IPC as a transport.
const connection = createConnection(ProposedFeatures.all);

// Create a simple text document manager.
const documents: TextDocuments<TextDocument> = new TextDocuments(TextDocument);

connection.onInitialize((params: InitializeParams) => {
    const result: InitializeResult = {
        capabilities: {
            textDocumentSync: TextDocumentSyncKind.Incremental,
            // Tell the client that this server supports code completion.
            completionProvider: {
                resolveProvider: true
            },
            // Add more capabilities as needed
            // hoverProvider: true,
            // definitionProvider: true,
            // referencesProvider: true,
            // documentSymbolProvider: true,
        }
    };

    return result;
});

// Listen on the connection
connection.listen();

// Make the text document manager listen on the connection
// (open, change and close text document events)
documents.listen(connection);

// This handler provides the initial list of completion items.
connection.onCompletion(() => {
    return [
        {
            label: 'flow',
            kind: 1, // Text
            data: 1
        },
        {
            label: 'chain',
            kind: 1,
            data: 2
        },
        {
            label: 'process',
            kind: 1,
            data: 3
        }
    ];
});

// This handler resolves additional information for the item selected in the completion list.
connection.onCompletionResolve((item) => {
    if (item.data === 1) {
        item.detail = 'FlowLang flow';
        item.documentation = 'Define a new flow in FlowLang';
    } else if (item.data === 2) {
        item.detail = 'FlowLang chain';
        item.documentation = 'Define a chain of processes in FlowLang';
    } else if (item.data === 3) {
        item.detail = 'FlowLang process';
        item.documentation = 'Define a process in FlowLang';
    }
    return item;
});

// Listen on the connection
connection.listen();
