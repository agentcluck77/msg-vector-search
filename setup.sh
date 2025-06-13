#!/bin/bash
# Setup script for SeaTalk Search MCP Server
# This pre-installs the package with uvx to avoid timeouts

echo "🔄 Setting up SeaTalk Search MCP Server..."

# Check if uvx is installed
if ! command -v uvx &> /dev/null; then
    echo "❌ Error: uvx is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📦 Pre-installing dependencies with uvx..."
echo "   This will download PyTorch and other ML dependencies (~200MB)"
echo "   This may take a few minutes..."

# Install the package with uvx to cache all dependencies
# We'll run a simple Python import test instead of --help to avoid database initialization  
uvx --from "$SCRIPT_DIR" python -c "import sys; sys.path.insert(0, 'src'); import core.search; print('Dependencies installed successfully!')"

if [ $? -eq 0 ]; then
    echo "✅ Setup complete! Dependencies are now cached."
    echo ""
    echo "📋 Next steps:"
    echo "1. The MCP server is ready to use with Claude Desktop"
    echo "2. Make sure your Claude Desktop config points to:"
    echo "   Command: /Users/$(whoami)/.local/bin/uvx"
    echo "   Args: [\"--from\", \"$SCRIPT_DIR\", \"seatalk-search-server\"]"
    echo ""
    echo "🔍 You can now search your SeaTalk messages in Claude!"
else
    echo "❌ Setup failed. Please check the error messages above."
    exit 1
fi 