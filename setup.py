#!/usr/bin/env python3
"""
Setup script for msg-vector-search
Uses uv for fast dependency installation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_uv():
    """Check if uv is installed, install if not"""
    try:
        subprocess.run(["uv", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("✓ uv is already installed")
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        print("Installing uv...")
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["curl", "-sSf", "https://astral.sh/uv/install.ps1", "|", "powershell", "-ex", "bypass", "-"], 
                    shell=True, check=True
                )
            else:
                subprocess.run(
                    ["curl", "-sSf", "https://astral.sh/uv/install.sh", "|", "sh"],
                    shell=True, check=True
                )
            print("✓ uv installed successfully")
            return True
        except subprocess.SubprocessError:
            print("Failed to install uv. Please install manually from https://github.com/astral-sh/uv")
            return False

def install_dependencies():
    """Install Python dependencies using uv"""
    print("Installing Python dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        subprocess.run(
            ["uv", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print("✓ Python dependencies installed successfully")
        return True
    except subprocess.SubprocessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def install_node_deps():
    """Install Node.js dependencies and build"""
    print("Installing Node.js dependencies...")
    try:
        subprocess.run(["npm", "install"], check=True, cwd=Path(__file__).parent)
        subprocess.run(["npm", "run", "build"], check=True, cwd=Path(__file__).parent)
        print("✓ Node.js dependencies installed and built successfully")
        return True
    except subprocess.SubprocessError as e:
        print(f"Failed to install Node.js dependencies: {e}")
        return False

def setup_config():
    """Set up configuration for msg-vector-search"""
    config_dir = Path.home() / ".config" / "msg-vector-search"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    
    if not config_file.exists():
        print("\nSetting up configuration...")
        seatalk_folder = input("Enter the path to your SeaTalk folder: ")
        db_key = input("Enter your SeaTalk database key: ")
        
        import json
        with open(config_file, "w") as f:
            json.dump({
                "seatalk_folder": seatalk_folder,
                "db_key": db_key
            }, f, indent=2)
        
        print(f"✓ Configuration saved to {config_file}")
    else:
        print(f"✓ Configuration already exists at {config_file}")
    
    return True

def create_launcher():
    """Create launcher script"""
    print("Creating launcher script...")
    bin_dir = Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    
    launcher_path = bin_dir / "msg-vector-search"
    
    with open(launcher_path, "w") as f:
        f.write(f"""#!/bin/bash
# Launcher for msg-vector-search
cd {Path(__file__).parent.absolute()}
export SEATALK_FOLDER=$(cat ~/.config/msg-vector-search/config.json | python3 -c "import sys, json; print(json.load(sys.stdin)['seatalk_folder'])")
export SEATALK_DB_KEY=$(cat ~/.config/msg-vector-search/config.json | python3 -c "import sys, json; print(json.load(sys.stdin)['db_key'])")
node dist/index.js "$@"
""")
    
    os.chmod(launcher_path, 0o755)
    print(f"✓ Launcher script created at {launcher_path}")
    
    # Check if bin_dir is in PATH
    if str(bin_dir) not in os.environ.get("PATH", ""):
        print(f"\nNOTE: Add {bin_dir} to your PATH to run msg-vector-search from anywhere.")
        print(f"Run: export PATH=\"$HOME/.local/bin:$PATH\"")
        print(f"Add this line to your .bashrc or .zshrc to make it permanent.")
    
    return True

def main():
    """Main setup function"""
    print("=== Setting up msg-vector-search ===\n")
    
    # Check dependencies
    if not check_uv():
        return 1
    
    # Install Python dependencies
    if not install_dependencies():
        return 1
    
    # Install Node.js dependencies
    if not install_node_deps():
        return 1
    
    # Set up configuration
    if not setup_config():
        return 1
    
    # Create launcher
    if not create_launcher():
        return 1
    
    print("\n=== Setup completed successfully! ===")
    print("You can now run msg-vector-search from anywhere.")
    print("To search: msg-vector-search \"your search query\"")
    print("To get statistics: msg-vector-search --stats")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 