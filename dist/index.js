#!/usr/bin/env node
/**
 * msg-vector-search MCP Server
 *
 * NO CODE setup:
 * 1. Install: npm install -g msg-vector-search
 * 2. Configure in Claude Desktop config:
 *    {
 *      "mcpServers": {
 *        "msg-vector-search": {
 *          "command": "msg-vector-search",
 *          "env": {
 *            "SEATALK_FOLDER": "/path/to/seatalk/folder",
 *            "SEATALK_DB_KEY": "your-db-key"
 *          }
 *        }
 *      }
 *    }
 */
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import * as os from 'os';
// Get the directory of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
// Resolve the path to python_bridge.py
const pythonBridgePath = join(dirname(__dirname), 'python_bridge.py');
// Default SeaTalk folder locations by platform
const defaultSeaTalkFolders = {
    darwin: `${os.homedir()}/Library/Application Support/SeaTalk`,
    win32: `${process.env.APPDATA}\\SeaTalk`,
    linux: `${os.homedir()}/.config/SeaTalk`
};
// Get SeaTalk folder from environment variable or use default
const seatalkFolder = process.env.SEATALK_FOLDER || defaultSeaTalkFolders[process.platform] || '';
// Get SeaTalk DB key from environment variable
const seatalkDbKey = process.env.SEATALK_DB_KEY;
// Validate configuration
if (!seatalkFolder) {
    console.error('Error: SEATALK_FOLDER environment variable not set and default location not available for this platform');
    process.exit(1);
}
if (!seatalkDbKey) {
    console.error('Error: SEATALK_DB_KEY environment variable is required');
    console.error('Please add it to your Claude Desktop config file:');
    console.error('{');
    console.error('  "mcpServers": {');
    console.error('    "msg-vector-search": {');
    console.error('      "command": "msg-vector-search",');
    console.error('      "env": {');
    console.error('        "SEATALK_FOLDER": "/path/to/your/seatalk",');
    console.error('        "SEATALK_DB_KEY": "your-db-key"');
    console.error('      }');
    console.error('    }');
    console.error('  }');
    console.error('}');
    process.exit(1);
}
console.log(`Starting msg-vector-search MCP server...`);
console.log(`SeaTalk folder: ${seatalkFolder}`);
console.log(`DB key: ${seatalkDbKey.substring(0, 5)}...`);
// Create MCP server
const server = new Server({
    name: 'msg-vector-search',
    version: '1.0.0',
});
// Define the search tool
const searchTool = {
    name: 'msg_vector_search',
    description: 'Search SeaTalk messages using semantic search',
    inputSchema: {
        type: 'object',
        properties: {
            query: {
                type: 'string',
                description: 'Search query',
            },
            limit: {
                type: 'integer',
                description: 'Maximum number of results to return',
                default: 10,
            },
        },
        required: ['query'],
    }
};
// Define the stats tool
const statsTool = {
    name: 'msg_vector_search_stats',
    description: 'Get statistics about the SeaTalk message database',
    inputSchema: {
        type: 'object',
        properties: {},
    }
};
// Set up tool handlers
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [searchTool, statsTool]
    };
});
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    try {
        switch (name) {
            case 'msg_vector_search': {
                const { query, limit = 10 } = args;
                const results = await runPythonBridge('search', {
                    query,
                    limit: limit.toString(),
                    'seatalk-folder': seatalkFolder,
                    'db-key': seatalkDbKey,
                });
                return {
                    content: {
                        type: 'text',
                        text: JSON.stringify(results, null, 2)
                    }
                };
            }
            case 'msg_vector_search_stats': {
                const stats = await runPythonBridge('stats', {
                    'seatalk-folder': seatalkFolder,
                    'db-key': seatalkDbKey,
                });
                return {
                    content: {
                        type: 'text',
                        text: JSON.stringify(stats, null, 2)
                    }
                };
            }
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
    catch (error) {
        return {
            content: {
                type: 'text',
                text: `Error: ${error.message}`
            }
        };
    }
});
// Helper function to run Python bridge
async function runPythonBridge(command, args) {
    return new Promise((resolve, reject) => {
        const pythonArgs = [pythonBridgePath, command];
        // Add arguments
        for (const [key, value] of Object.entries(args)) {
            pythonArgs.push(`--${key}`);
            pythonArgs.push(value);
        }
        const pythonProcess = spawn('python3', pythonArgs);
        let stdout = '';
        let stderr = '';
        pythonProcess.stdout.on('data', (data) => {
            stdout += data.toString();
        });
        pythonProcess.stderr.on('data', (data) => {
            stderr += data.toString();
        });
        pythonProcess.on('close', (code) => {
            if (code === 0) {
                try {
                    const results = JSON.parse(stdout);
                    resolve(results);
                }
                catch (error) {
                    reject(new Error(`Failed to parse results: ${error.message}`));
                }
            }
            else {
                reject(new Error(`Command failed with code ${code}: ${stderr}`));
            }
        });
    });
}
// Start the server
const transport = new StdioServerTransport();
server.connect(transport).catch(error => {
    console.error('Failed to start server:', error);
    process.exit(1);
});
