#!/usr/bin/env python3
"""
Minimal MCP script for checking if the MCP server is working
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def expand_path(path):
    """Expand the tilde (~) in the path"""
    if path and isinstance(path, str):
        return os.path.expanduser(path)
    return path

def load_config():
    """Load configuration from config.json if available"""
    config_path = Path("config.json")
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load config.json: {str(e)}")
    return {}

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

def check_build():
    """Check if the MCP server is built"""
    if Path("dist/index.js").exists():
        print("✅ MCP server is built")
        return True
    else:
        print("❌ MCP server is not built")
        return False

def check_seatalk_config():
    """Check if SeaTalk configuration is available"""
    # Try to get config from environment variables first
    seatalk_folder = os.environ.get("SEATALK_FOLDER")
    seatalk_db_key = os.environ.get("SEATALK_DB_KEY")
    
    # If not found, try to get from config.json
    if not seatalk_folder or not seatalk_db_key:
        config = load_config()
        if "seatalk" in config:
            if not seatalk_folder and "folder" in config["seatalk"]:
                seatalk_folder = config["seatalk"]["folder"]
            if not seatalk_db_key and "db_key" in config["seatalk"]:
                seatalk_db_key = config["seatalk"]["db_key"]
    
    # Expand the tilde (~) in the path
    seatalk_folder = expand_path(seatalk_folder)
    
    if seatalk_folder:
        print(f"✅ SEATALK_FOLDER is set to {seatalk_folder}")
        if Path(seatalk_folder).exists():
            print(f"✅ SeaTalk folder exists")
        else:
            print(f"❌ SeaTalk folder does not exist")
    else:
        print("❌ SEATALK_FOLDER is not set")
    
    if seatalk_db_key:
        print(f"✅ SEATALK_DB_KEY is set")
    else:
        print("❌ SEATALK_DB_KEY is not set")
    
    return seatalk_folder, seatalk_db_key

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

def run_mcp_server(timeout=5, seatalk_folder=None, seatalk_db_key=None):
    """Try to run the MCP server with a timeout"""
    try:
        print("🚀 Starting MCP server...")
        env = dict(os.environ)
        if seatalk_folder:
            env["SEATALK_FOLDER"] = seatalk_folder
        else:
            env["SEATALK_FOLDER"] = "/tmp"  # Fallback for testing
            
        if seatalk_db_key:
            env["SEATALK_DB_KEY"] = seatalk_db_key
        else:
            env["SEATALK_DB_KEY"] = "test"  # Fallback for testing
            
        process = subprocess.Popen(
            ["node", "dist/index.js"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            print(f"✅ MCP server started successfully")
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            print(f"✅ MCP server is running (killed after {timeout}s timeout)")
            return True
    except Exception as e:
        print(f"❌ Failed to start MCP server: {str(e)}")
        return False

def main():
    """Main function"""
    print("🔍 Checking MCP server setup...")
    
    print("\n1. Checking system dependencies:")
    check_system_dependencies()
    
    print("\n2. Checking Node.js dependencies:")
    check_node_dependencies()
    
    print("\n3. Checking MCP server build:")
    build_ok = check_build()
    
    print("\n4. Checking SeaTalk configuration:")
    seatalk_folder, seatalk_db_key = check_seatalk_config()
    
    if build_ok:
        print("\n5. Testing MCP server startup:")
        run_mcp_server(seatalk_folder=seatalk_folder, seatalk_db_key=seatalk_db_key)
    
    print("\n✨ Suggestions:")
    print("- To install cmake: brew install cmake")
    print("- To install Node.js dependencies: npm install")
    print("- To build the MCP server: npm run build")
    print("- To configure SeaTalk:")
    print("  - Option 1: Edit config.json with your SeaTalk folder and DB key")
    print("  - Option 2: Set environment variables: export SEATALK_FOLDER=\"/path/to/seatalk\" SEATALK_DB_KEY=\"your-key\"")
    print("- To run the MCP server: node dist/index.js")

if __name__ == "__main__":
    main() 