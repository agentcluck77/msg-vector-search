#!/usr/bin/env python3
"""
SeaTalk Message Vector Search Configuration
Centralized configuration for MCP server
"""

import os
from pathlib import Path

# Base paths - dynamically determined
PROJECT_ROOT = Path(__file__).parent
SRC_DIR = PROJECT_ROOT / "src"

# Data directories - created on demand
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_DATA_DIR = DATA_DIR / "vectors"
LOGS_DIR = PROJECT_ROOT / "logs"

# Database configuration
DB_KEY = os.getenv('SEATALK_DB_KEY')
DB_TYPE = 'WxSQLite3'
DB_PATH = os.getenv('SEATALK_FOLDER')  # Path to SeaTalk database folder
CIPHER = 'sqleet: ChaCha20-Poly1305'

# Embedding configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Default model
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32

# Available embedding models
AVAILABLE_EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2", 
        "dimension": 384,
        "description": "Original English-focused model",
        "languages": "Primarily English"
    },
    "intfloat/multilingual-e5-small": {
        "name": "intfloat/multilingual-e5-small",
        "dimension": 384,
        "description": "Fast multilingual model with excellent performance",
        "languages": "100+ languages including English, Chinese, Southeast Asian, Portuguese"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384, 
        "description": "High-quality multilingual model (slower)",
        "languages": "50+ languages including English, Chinese, Southeast Asian, Portuguese"
    }
}

# API configuration
GPT_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create directories
for dir_path in [DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Search Configuration
SEARCH_CONFIG = {
    # AI-enhanced search parameters (used when AI is enabled)
    'ai_search_limit': 30,
    'ai_similarity_threshold': 0.3,
    'ai_max_tokens': 400,
    'ai_top_results_display': 30,  # Number of source messages to show after AI response
    
    # Basic search parameters (used when AI is disabled)
    'basic_search_limit': 5,
    'basic_similarity_threshold': 0.3,
    'include_context': False,
    
    # Display settings
    'show_user_names': True,  # Show actual names instead of user IDs
    'show_full_content': True,  # Show full message content (not truncated)
} 