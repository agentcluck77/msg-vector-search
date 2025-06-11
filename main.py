#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "sentence-transformers",
#     "sqlite-vec",
#     "apsw",
#     "modelcontextprotocol",
# ]
# ///

"""
SeaTalk Message Search - Main Entry Point
Provides a CLI interface to the search engine
"""

import sys
import os
from python_bridge import main

if __name__ == "__main__":
    # Pass command line arguments to the python_bridge main function
    main() 