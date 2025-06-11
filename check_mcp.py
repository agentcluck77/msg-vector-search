#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "modelcontextprotocol>=0.1.0,<0.2.0",
# ]
# ///

"""
Check if the MCP server can be installed and run with uv
"""

import sys
import os
import subprocess
import json

def check_uv():
    """Check if uv is installed"""
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("✅ uv is installed")
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        print("❌ uv is not installed")
        return False

def check_mcp_dependency():
    """Check if MCP dependency is installed"""
    try:
        import modelcontextprotocol as mcp
        print(f"✅ MCP is installed (version {mcp.__version__})")
        return True
    except ImportError:
        print("❌ MCP is not installed")
        return False
    except Exception as e:
        print(f"⚠️ MCP is partially installed but there was an error: {str(e)}")
        print("   This is likely due to missing system dependencies like cmake")
        return False

def check_system_dependencies():
    """Check if required system dependencies are installed"""
    dependencies = {
        "cmake": "Required for building PyArrow",
    }
    
    for dep, desc in dependencies.items():
        try:
            subprocess.run([dep, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"✅ {dep} is installed ({desc})")
        except (FileNotFoundError, subprocess.SubprocessError):
            print(f"❌ {dep} is not installed ({desc})")

def check_node_dependencies():
    """Check if Node.js dependencies are installed"""
    try:
        result = subprocess.run(["npm", "list", "@modelcontextprotocol/sdk"], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "@modelcontextprotocol/sdk" in result.stdout:
            print("✅ MCP SDK for Node.js is installed")
            return True
        else:
            print("❌ MCP SDK for Node.js is not installed")
            return False
    except (FileNotFoundError, subprocess.SubprocessError):
        print("❌ npm is not available or error occurred")
        return False

def check_seatalk_config():
    """Check if SeaTalk configuration is available"""
    seatalk_folder = os.environ.get("SEATALK_FOLDER")
    seatalk_db_key = os.environ.get("SEATALK_DB_KEY")
    
    if seatalk_folder:
        print(f"✅ SEATALK_FOLDER is set to {seatalk_folder}")
        if os.path.exists(seatalk_folder):
            print(f"✅ SeaTalk folder exists")
        else:
            print(f"❌ SeaTalk folder does not exist")
    else:
        print("❌ SEATALK_FOLDER is not set")
    
    if seatalk_db_key:
        print(f"✅ SEATALK_DB_KEY is set")
    else:
        print("❌ SEATALK_DB_KEY is not set")

def check_build():
    """Check if the MCP server is built"""
    if os.path.exists("dist/index.js"):
        print("✅ MCP server is built")
        return True
    else:
        print("❌ MCP server is not built")
        return False

def main():
    """Main function"""
    print("🔍 Checking MCP server setup...")
    print("\n1. Checking uv installation:")
    check_uv()
    
    print("\n2. Checking system dependencies:")
    check_system_dependencies()
    
    print("\n3. Checking MCP dependency:")
    check_mcp_dependency()
    
    print("\n4. Checking Node.js dependencies:")
    check_node_dependencies()
    
    print("\n5. Checking SeaTalk configuration:")
    check_seatalk_config()
    
    print("\n6. Checking MCP server build:")
    check_build()
    
    print("\n✨ Suggestions:")
    print("- To install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
    print("- To install cmake: brew install cmake")
    print("- To build the MCP server: npm run build")
    print("- To set SeaTalk configuration: export SEATALK_FOLDER=\"/path/to/seatalk\" SEATALK_DB_KEY=\"your-key\"")
    print("- To run the MCP server: node dist/index.js")

if __name__ == "__main__":
    main() 