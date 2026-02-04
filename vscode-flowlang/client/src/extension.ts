import * as path from 'path';
import * as vscode from 'vscode';
import { workspace, ExtensionContext, window } from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions, TransportKind } from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('FlowLang extension is now active!');

    // Get the configuration
    const config = workspace.getConfiguration('flowlang');
    const pythonPath = config.get<string>('pythonPath', 'python');
    const flowPath = config.get<string>('pathToFlow', '').replace('${workspaceFolder}', workspace.workspaceFolders?.[0]?.uri.fsPath || '');

    // The server is implemented in node
    const serverModule = context.asAbsolutePath(
        path.join('server', 'out', 'server.js')
    );

    // Server options for both run and debug
    const serverOptions: ServerOptions = {
        run: {
            command: 'node',
            args: [serverModule, '--stdio'],
            options: {
                env: {
                    ...process.env,
                    NODE_OPTIONS: '--no-warnings',
                    PYTHON_PATH: pythonPath,
                    FLOW_PATH: flowPath
                }
            }
        },
        debug: {
            command: 'node',
            args: ['--inspect=6009', serverModule, '--stdio'],
            options: {
                env: {
                    ...process.env,
                    NODE_OPTIONS: '--no-warnings',
                    PYTHON_PATH: pythonPath,
                    FLOW_PATH: flowPath
                }
            }
        }
    };

    // Options to control the language client
    const clientOptions: LanguageClientOptions = {
        // Register the server for FlowLang documents
        documentSelector: [{ scheme: 'file', language: 'flowlang' }],
        synchronize: {
            configurationSection: 'flowlang',
            // Notify the server about file changes to '.flow' files in the workspace
            fileEvents: workspace.createFileSystemWatcher('**/*.flow')
        },
        outputChannel: window.createOutputChannel('FlowLang Language Server'),
        diagnosticCollectionName: 'flowlang',
        revealOutputChannelOn: 1 // 1 = error
    };

    // Create the language client and start the client.
    client = new LanguageClient(
        'flowlang',
        'FlowLang Language Server',
        serverOptions,
        clientOptions
    );

    // Start the client. This will also launch the server
    client.start().catch(error => {
        console.error('Failed to start language server:', error);
        vscode.window.showErrorMessage(
            `Failed to start FlowLang language server: ${error.message}`
        );
    });
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}
