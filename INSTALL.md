# 🔍 SeaTalk Message Search - Installation Guide

This guide helps you set up semantic search for your SeaTalk conversations using MCP.

## 📋 Prerequisites

- **Node.js 18+**: Required for the MCP server

## 🔧 NO CODE Setup

### 1. Install the tool

```bash
# Install directly from GitHub
npm install -g github:agentcluck77/msg-vector-search
```

### 2. Configure in Claude Desktop

Add this to your Claude Desktop config file (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "msg-vector-search": {
      "command": "msg-vector-search",
      "env": {
        "SEATALK_FOLDER": "/Users/YOUR_USERNAME/Library/Application Support/SeaTalk",
        "SEATALK_DB_KEY": "40a3884b8b032e6f732174"
      }
    }
  }
}
```

Replace `YOUR_USERNAME` with your actual username.

### 3. Restart Claude Desktop

That's it! No coding or additional configuration needed.

## 📂 Common SeaTalk Locations

- macOS: `~/Library/Application Support/SeaTalk`
- Windows: `%APPDATA%\\SeaTalk`
- Linux: `~/.config/SeaTalk`

## 🔍 Usage Examples

Once installed, you can use the tool directly from Claude:

- "Find conversations where we discussed the project timeline"
- "When did we last talk about the budget?"
- "Show me messages about the design review"

## 🛠️ Troubleshooting

If you encounter any issues:

1. Make sure Node.js 18+ is installed
2. Check that the SeaTalk folder path is correct
3. Verify that Claude Desktop is configured correctly
4. Restart Claude Desktop after making changes

## 🔒 Security Note

Your messages never leave your computer. All processing happens locally. 