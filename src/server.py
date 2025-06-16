#!/usr/bin/env python3
"""
MCP Server for SeaTalk Vector Search
Provides semantic search capabilities for SeaTalk conversations
"""

import os
import logging
import time
import sys
import atexit
from typing import Dict, Any, List, Optional

# Add the src directory to Python path for imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from mcp.server.fastmcp import FastMCP, Context

from core.search import SemanticSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)

logger = logging.getLogger(__name__)

# Set MCP server mode environment variable
os.environ['MCP_SERVER_MODE'] = '1'

# Create MCP server
mcp = FastMCP(
    name="SeaTalk-Search",
    instructions="""
    This MCP server provides semantic search capabilities for your SeaTalk conversations.
    
    To use this server, you need to:
    1. Install it with `fastmcp install path/to/server.py`
    2. Configure your SeaTalk database path and key in the Claude MCP configuration
    
    Required environment variables:
    - SEATALK_DB_PATH: Path to the SeaTalk database directory (default: ~/Library/Application Support/SeaTalk)
    - SEATALK_DB_KEY: Database decryption key (required)
    
    Optional environment variables:
    - SEATALK_EMBEDDING_THRESHOLD: Minimum new messages to trigger embedding update (default: 50)
    """
)

# Initialize search engine
search_engine = None

def initialize_search_engine():
    """Initialize the search engine with database connection"""
    global search_engine
    
    # Check if DB key is provided
    db_key = os.environ.get('SEATALK_DB_KEY')
    if not db_key:
        logger.error("Database key not provided. Please set SEATALK_DB_KEY in your MCP configuration.")
        return None
    
    logger.info("Starting SeaTalk Search MCP server...")

    try:
        # Get database path from environment variable or use default
        db_path = os.environ.get('SEATALK_DB_PATH')
        if not db_path:
            db_path = os.path.expanduser("~/Library/Application Support/SeaTalk")
            os.environ['SEATALK_DB_PATH'] = db_path
        
        # Get embedding update threshold from environment variable or use default
        embedding_threshold = int(os.environ.get('SEATALK_EMBEDDING_THRESHOLD', '50'))
        logger.info(f"Using embedding update threshold: {embedding_threshold} messages")
        
        # Create data directories if they don't exist
        os.makedirs(os.path.join(os.path.dirname(__file__), "../../data/snapshots"), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), "../../data/vectors"), exist_ok=True)
        
        logger.info(f"Using database path: {db_path}")
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(
            db_path=db_path, 
            db_key=db_key, 
            embedding_update_threshold=embedding_threshold
        )
        
        # Pre-load the embedding model for faster searches
        search_engine.preload_model()
        
        # Only update embeddings if not already initialized
        # (setup.sh should have already done this)
        # Force vector DB connection before checking stats
        search_engine.embedding_processor.setup_vector_database()
        
        stats = search_engine.get_database_stats()
        embedded_count = stats.get('embedded_messages', 0)
        
        logger.info(f"Current embedding stats: {embedded_count} embedded messages")
        
        if embedded_count == 0:
            logger.info("No embeddings found, running initial embedding update...")
            update_result = search_engine.update_embeddings()
            logger.info(f"Initial embeddings update: {update_result['new_messages']} new messages")
        else:
            logger.info(f"Found existing embeddings: {embedded_count} messages - skipping initialization")
            # Set last update time to now to prevent immediate re-processing
            search_engine.last_embedding_update_time = time.time()
        
        logger.info("SeaTalk Search MCP server started")
        return search_engine
    except Exception as e:
        logger.error(f"Failed to initialize search engine: {e}")
        return None

# Initialize search engine immediately on server startup
search_engine = initialize_search_engine()

# Register cleanup handler
def cleanup():
    """Clean up resources on shutdown"""
    global search_engine
    
    logger.info("Shutting down SeaTalk Search MCP server...")
    
    if search_engine:
        search_engine.close()
        
    logger.info("SeaTalk Search MCP server stopped")

atexit.register(cleanup)

@mcp.resource("resource://database_stats")
def get_database_stats() -> Dict[str, Any]:
    """
    Get statistics about the SeaTalk database and embeddings
    
    Returns:
        Dictionary with database statistics
    """
    global search_engine
    
    # Try to initialize if not already initialized
    if not search_engine:
        search_engine = initialize_search_engine()
        
    if not search_engine:
        return {
            "status": "error",
            "error": "Search engine not initialized. Please set SEATALK_DB_KEY in your MCP configuration."
        }
    
    return search_engine.get_database_stats()

@mcp.tool()
def update_embeddings(ctx: Context = None) -> Dict[str, Any]:
    """
    Manually update embeddings for new messages.
    
    Returns:
        A dictionary containing update results and statistics
    """
    global search_engine
    
    # Try to initialize if not already initialized
    if not search_engine:
        search_engine = initialize_search_engine()
    
    if not search_engine:
        return {
            "status": "error",
            "error": "Search engine not initialized. Please set SEATALK_DB_KEY in your MCP configuration."
        }
    
    # Log the update request
    if ctx:
        ctx.info("Manually updating embeddings...")
    
    try:
        # Force update embeddings
        results = search_engine.update_embeddings()
        
        # Log update results
        if ctx:
            ctx.info(f"Updated embeddings: {results['new_messages']} new messages processed")
        
        return results
    except Exception as e:
        logger.error(f"Embedding update error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@mcp.tool()
def search_messages(
    query: str,
    limit: int = 30,
    threshold: float = 0.3,
    force_update: bool = False,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Search SeaTalk messages using semantic similarity.
    
    Args:
        query: The search query text
        limit: Maximum number of results to return (default: 30)
        threshold: Minimum similarity threshold (default: 0.3)
        force_update: Force embedding update before search (default: False)
    
    Returns:
        A dictionary containing search results and metadata
    """
    global search_engine
    
    # Try to initialize if not already initialized
    if not search_engine:
        search_engine = initialize_search_engine()
    
    if not search_engine:
        return {
            "status": "error",
            "error": "Search engine not initialized. Please set SEATALK_DB_KEY in your MCP configuration."
        }
    
    # Log the search request
    if ctx:
        ctx.info(f"Searching for: '{query}' (limit={limit}, threshold={threshold}, force_update={force_update})")
    
    try:
        # Perform search with optional forced update
        results = search_engine.search(
            query=query,
            limit=limit,
            similarity_threshold=threshold,
            include_context=True,
            force_update=force_update
        )
        
        # Log search results
        if ctx:
            ctx.info(f"Found {len(results['results'])} results in {results['metadata']['search_time_ms']}ms")
        
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

def main():
    """Entry point for uvx"""
    mcp.run()

if __name__ == "__main__":
    # Run the MCP server
    main() 