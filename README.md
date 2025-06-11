# 🔍 msg-vector-search

MCP tool for semantic search of SeaTalk conversations using vector embeddings.

## Overview

This tool provides semantic search capabilities for SeaTalk conversations by:

1. Creating vector embeddings of messages using the all-MiniLM-L6-v2 model
2. Storing these embeddings in a vector database for fast similarity search
3. Automatically updating the search index when new messages are available
4. Providing search results ranked by semantic similarity

## 📋 Prerequisites

- **Node.js 18+**: Required for the MCP server

## 🔧 Installation & Setup

### 1. Install the tool

```bash
# Install directly from GitHub (Node.js required)
npm install -g github:agentcluck77/msg-vector-search
```

If you encounter any issues with the global installation, you can also clone and install manually:

```bash
# Clone the repository
git clone https://github.com/agentcluck77/msg-vector-search.git
cd msg-vector-search

# Install dependencies and build
npm install

# Create a global symlink
npm link
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
        "SEATALK_DB_KEY": "your-seatalk-db-key"
      }
    }
  }
}
```

Replace `YOUR_USERNAME` with your actual username and `your-seatalk-db-key` with your actual SeaTalk database key.

### 3. Restart Claude Desktop

That's it! No coding or additional configuration needed.

## 📂 Common SeaTalk Locations

- macOS: `~/Library/Application Support/SeaTalk`
- Windows: `%APPDATA%\\SeaTalk`
- Linux: `~/.config/SeaTalk`

## 🔍 Usage

Once installed, you can use the tool directly from Claude:

1. Ask Claude questions about your SeaTalk conversations
2. Claude will automatically search through your messages using semantic search

### Examples:
- "Find conversations where we discussed the project timeline"
- "When did we last talk about the budget?"
- "Show me messages about the design review"

## 🛠️ Troubleshooting

If you encounter any issues:

1. Make sure Node.js 18+ is installed
2. Check that the SeaTalk folder path is correct
3. Verify that your SeaTalk DB key is correct
4. Restart Claude Desktop after making changes

## 🔒 Security

Your messages never leave your computer. All processing happens locally.

## Development

For developers who want to modify the tool:

```bash
# Clone the repository
git clone https://github.com/agentcluck77/msg-vector-search.git
cd msg-vector-search

# Install dependencies
npm install

# Build the project
npm run build
```

## License

MIT 