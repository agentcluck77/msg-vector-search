#!/usr/bin/env python3
"""
Python Bridge for msg-vector-search MCP Server
Interfaces between Node.js MCP server and Spark Search Python engine
"""

import sys
import json
import argparse
import logging
import os
import shutil
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add src to path to import the search engine
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from src.core.search.engine import SemanticSearchEngine
    from src.core.database.processor import MessageProcessor
    from src.core.embeddings.processor import EmbeddingProcessor
    from src.core.database.user_mapper import UserMapper
    from src.utils.file_manager import DataFileManager
    from config import SEARCH_CONFIG
except ImportError as e:
    print(json.dumps({"success": False, "error": f"Failed to import search engine: {e}"}))
    sys.exit(1)

# Configure logging to be quiet for JSON output
logging.basicConfig(level=logging.WARNING)

def json_response(success: bool, data=None, error=None):
    """Create a standardized JSON response"""
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat()
    }
    if data is not None:
        response["data"] = data
    if error is not None:
        response["error"] = str(error)
    return json.dumps(response, ensure_ascii=False, indent=2)

def find_latest_seatalk_db(seatalk_folder: str) -> Optional[str]:
    """Find the most recently modified main_*.sqlite file in SeaTalk folder"""
    try:
        pattern = os.path.join(seatalk_folder, "main_*.sqlite")
        db_files = glob.glob(pattern)
        
        if not db_files:
            return None
        
        # Get most recently modified file
        latest_db = max(db_files, key=os.path.getmtime)
        return latest_db
    except Exception:
        return None

def get_vector_db_path(seatalk_db_path: str) -> str:
    """Get vector database path next to SeaTalk DB"""
    db_dir = os.path.dirname(seatalk_db_path)
    return os.path.join(db_dir, "msg-vector-search.db")

def get_config_path(seatalk_db_path: str) -> str:
    """Get config file path next to SeaTalk DB"""
    db_dir = os.path.dirname(seatalk_db_path)
    return os.path.join(db_dir, ".msg-vector-search-config")

def save_config(config_path: str, config: Dict[str, Any]):
    """Save configuration to file"""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except:
        return None

def get_last_processed_timestamp(vector_db_path: str) -> int:
    """Get last processed timestamp from vector database metadata"""
    try:
        import sqlite3
        conn = sqlite3.connect(vector_db_path)
        cursor = conn.execute("SELECT last_processed_timestamp FROM search_metadata ORDER BY id DESC LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] else 0
    except:
        return 0

def update_last_processed_timestamp(vector_db_path: str, timestamp: int):
    """Update last processed timestamp in vector database metadata"""
    try:
        import sqlite3
        conn = sqlite3.connect(vector_db_path)
        
        # Check if metadata table exists and has the column
        cursor = conn.execute("PRAGMA table_info(search_metadata)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'last_processed_timestamp' not in columns:
            # Add the column if it doesn't exist
            conn.execute("ALTER TABLE search_metadata ADD COLUMN last_processed_timestamp INTEGER DEFAULT 0")
        
        # Update or insert the timestamp
        conn.execute("UPDATE search_metadata SET last_processed_timestamp = ? WHERE id = (SELECT MAX(id) FROM search_metadata)", (timestamp,))
        if conn.total_changes == 0:
            conn.execute("INSERT INTO search_metadata (last_processed_timestamp) VALUES (?)", (timestamp,))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logging.warning(f"Failed to update last processed timestamp: {e}")

def create_database_snapshot(source_db_path: str, snapshot_path: str, db_key: str) -> bool:
    """Create a robust snapshot of the database using file copy (since SeaTalk may be running)"""
    try:
        # Use file copy with retry logic since the database may be in use by SeaTalk
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.copy2(source_db_path, snapshot_path)
                
                # Verify the snapshot was created successfully
                if os.path.exists(snapshot_path) and os.path.getsize(snapshot_path) > 0:
                    # Verify we can open it
                    try:
                        import apsw
                        verify_conn = apsw.Connection(snapshot_path)
                        verify_conn.pragma("key", db_key)
                        verify_conn.close()
                    except Exception as verify_error:
                        logging.warning(f"Snapshot verification failed: {verify_error}")
                        raise verify_error
                    
                    return True
                else:
                    raise Exception("Snapshot file is empty or not created")
                    
            except Exception as copy_error:
                if attempt < max_retries - 1:
                    logging.warning(f"Copy attempt {attempt + 1} failed: {copy_error}, retrying...")
                    time.sleep(1.0)  # Wait longer before retry
                else:
                    raise copy_error
            
        return False
        
    except Exception as e:
        logging.error(f"Failed to create database snapshot: {e}")
        return False

def setup_search_system(seatalk_folder: str, db_key: str) -> str:
    """Set up the search system with embeddings"""
    try:
        # Find latest SeaTalk database
        seatalk_db_path = find_latest_seatalk_db(seatalk_folder)
        if not seatalk_db_path:
            return json_response(False, error="No main_*.sqlite files found in SeaTalk folder")
        
        # Set up paths
        vector_db_path = get_vector_db_path(seatalk_db_path)
        config_path = get_config_path(seatalk_db_path)
        
        # Create snapshot of database for processing
        snapshot_path = seatalk_db_path + ".snapshot"
        create_database_snapshot(seatalk_db_path, snapshot_path, db_key)
        
        try:
            # Process messages from snapshot
            processor = MessageProcessor(snapshot_path, db_key)
            messages = processor.process_all_messages()
            
            # Build conversation context
            conversations = processor.build_conversation_context()
            
            # Prepare embedding data
            embedding_data = processor.prepare_embedding_data()
            
            # Generate embeddings
            embedding_processor = EmbeddingProcessor(vector_db_path=vector_db_path)
            embedding_processor.process_and_store(embedding_data, batch_size=32)
            
            # Get latest message timestamp for tracking
            latest_timestamp = max([msg.get('create_timestamp', 0) for msg in messages]) if messages else 0
            update_last_processed_timestamp(vector_db_path, latest_timestamp)
            
            # Save configuration
            config = {
                "seatalk_folder": seatalk_folder,
                "seatalk_db_path": seatalk_db_path,
                "vector_db_path": vector_db_path,
                "setup_timestamp": datetime.now().isoformat(),
                "total_messages_processed": len(embedding_data),
                "last_processed_timestamp": latest_timestamp
            }
            save_config(config_path, config)
            
        finally:
            # Clean up snapshot
            if os.path.exists(snapshot_path):
                os.remove(snapshot_path)
        
        return json_response(True, data={
            "message": "Setup completed successfully",
            "vector_db_path": vector_db_path,
            "messages_processed": len(embedding_data),
            "seatalk_db_used": seatalk_db_path
        })
        
    except Exception as e:
        return json_response(False, error=str(e))

def check_and_update_embeddings(seatalk_folder: str, db_key: str) -> Tuple[bool, str]:
    """Check for new messages and update embeddings if needed"""
    try:
        # Find current SeaTalk database
        current_db_path = find_latest_seatalk_db(seatalk_folder)
        if not current_db_path:
            return False, "No SeaTalk database found"
        
        vector_db_path = get_vector_db_path(current_db_path)
        
        # Check if vector database exists
        if not os.path.exists(vector_db_path):
            return False, "Vector database not found. Run setup first."
        
        # Get last processed timestamp
        last_processed = get_last_processed_timestamp(vector_db_path)
        
        # Create snapshot for checking new messages
        snapshot_path = current_db_path + ".snapshot"
        create_database_snapshot(current_db_path, snapshot_path, db_key)
        
        try:
            # Check for new messages using MessageProcessor
            processor = MessageProcessor(snapshot_path, db_key)
            
            # Connect to database and check for new messages
            with processor.database:
                cursor = processor.database.get_cursor()
                cursor.execute("""
                    SELECT COUNT(*) FROM chat_message 
                    WHERE _createAt > ? AND dts IS NULL
                """, (last_processed,))
                new_message_count = cursor.fetchone()[0]
            
            if new_message_count == 0:
                return True, "No new messages to process"
            
            # Process all messages to get the new ones
            all_messages = processor.process_all_messages()
            
            # Filter to only new messages
            new_messages = [
                msg for msg in all_messages 
                if msg.get('create_timestamp', 0) > last_processed
            ]
            
            if not new_messages:
                return True, "No new messages with content to process"
            
            # Build conversation context (needed for embedding preparation)
            processor.build_conversation_context()
            
            # Prepare embedding data for new messages only
            all_embedding_data = processor.prepare_embedding_data()
            new_embedding_data = [
                data for data in all_embedding_data
                if data.get('timestamp', 0) > last_processed
            ]
            
            if new_embedding_data:
                # Generate embeddings for new messages
                embedding_processor = EmbeddingProcessor(vector_db_path=vector_db_path)
                embedding_processor.process_and_store(new_embedding_data, batch_size=32)
                
                # Update last processed timestamp
                latest_timestamp = max([msg.get('create_timestamp', 0) for msg in new_messages])
                update_last_processed_timestamp(vector_db_path, latest_timestamp)
                
                return True, f"Processed {len(new_embedding_data)} new messages"
            else:
                return True, "No new messages with content to embed"
                
        finally:
            # Clean up snapshot
            if os.path.exists(snapshot_path):
                os.remove(snapshot_path)
        
    except Exception as e:
        return False, f"Update failed: {str(e)}"

def search_messages(query: str, seatalk_folder: str, db_key: str, 
                   limit: int = 10, similarity_threshold: float = 0.3, 
                   conversation_type: str = None, session_id: str = None,
                   include_context: bool = False) -> str:
    """Perform semantic search on messages with automatic updates"""
    try:
        # Check and update embeddings first
        update_success, update_message = check_and_update_embeddings(seatalk_folder, db_key)
        
        # Find current SeaTalk database
        seatalk_db_path = find_latest_seatalk_db(seatalk_folder)
        if not seatalk_db_path:
            return json_response(False, error="No SeaTalk database found")
        
        vector_db_path = get_vector_db_path(seatalk_db_path)
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(vector_db_path=vector_db_path, db_key=db_key)
        
        # Perform search
        results = search_engine.search(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            conversation_type=conversation_type,
            session_id=session_id,
            include_context=include_context
        )
        
        # Add update info to response
        results["update_info"] = {
            "update_success": update_success,
            "update_message": update_message
        }
        
        return json_response(True, data=results)
        
    except Exception as e:
        return json_response(False, error=str(e))

def search_conversation(session_id: str, seatalk_folder: str, db_key: str, 
                       query: str = None, limit: int = 50) -> str:
    """Search within a specific conversation"""
    try:
        # Find current SeaTalk database
        seatalk_db_path = find_latest_seatalk_db(seatalk_folder)
        if not seatalk_db_path:
            return json_response(False, error="No SeaTalk database found")
        
        vector_db_path = get_vector_db_path(seatalk_db_path)
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(vector_db_path=vector_db_path, db_key=db_key)
        
        # Get conversation messages
        results = search_engine.get_conversation_messages(
            session_id=session_id,
            query=query,
            limit=limit
        )
        
        return json_response(True, data=results)
        
    except Exception as e:
        return json_response(False, error=str(e))

def get_search_stats(seatalk_folder: str, db_key: str) -> str:
    """Get statistics about the search system"""
    try:
        # Find current SeaTalk database
        seatalk_db_path = find_latest_seatalk_db(seatalk_folder)
        if not seatalk_db_path:
            return json_response(False, error="No SeaTalk database found")
        
        vector_db_path = get_vector_db_path(seatalk_db_path)
        config_path = get_config_path(seatalk_db_path)
        
        # Load setup info
        setup_info = load_config(config_path) or {}
        
        # Initialize search engine
        search_engine = SemanticSearchEngine(vector_db_path=vector_db_path, db_key=db_key)
        
        # Get statistics
        stats = search_engine.get_stats()
        
        # Add setup info
        stats["setup_info"] = setup_info
        
        return json_response(True, data=stats)
        
    except Exception as e:
        return json_response(True, data={"error": str(e)})

def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description="SeaTalk Message Search Bridge")
    
    # Command argument
    parser.add_argument("command", choices=["setup", "search", "search-conversation", "stats", "check-update"],
                        help="Command to execute")
    
    # Common arguments
    parser.add_argument("--seatalk-folder", required=True, help="Path to SeaTalk folder")
    parser.add_argument("--db-key", required=True, help="Database encryption key")
    
    # Search arguments
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--limit", type=int, default=10, help="Maximum results to return")
    parser.add_argument("--similarity-threshold", type=float, default=0.3, help="Similarity threshold (0-1)")
    parser.add_argument("--conversation-type", choices=["group", "private"], help="Filter by conversation type")
    parser.add_argument("--session-id", help="Filter by session ID")
    parser.add_argument("--include-context", action="store_true", help="Include conversation context")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        print(setup_search_system(args.seatalk_folder, args.db_key))
    
    elif args.command == "search":
        if not args.query:
            print(json_response(False, error="Search query is required"))
            sys.exit(1)
            
        print(search_messages(
            query=args.query,
            seatalk_folder=args.seatalk_folder,
            db_key=args.db_key,
            limit=args.limit,
            similarity_threshold=args.similarity_threshold,
            conversation_type=args.conversation_type,
            session_id=args.session_id,
            include_context=args.include_context
        ))
    
    elif args.command == "search-conversation":
        if not args.session_id:
            print(json_response(False, error="Session ID is required"))
            sys.exit(1)
            
        print(search_conversation(
            session_id=args.session_id,
            seatalk_folder=args.seatalk_folder,
            db_key=args.db_key,
            query=args.query,
            limit=args.limit
        ))
    
    elif args.command == "stats":
        print(get_search_stats(args.seatalk_folder, args.db_key))
    
    elif args.command == "check-update":
        success, message = check_and_update_embeddings(args.seatalk_folder, args.db_key)
        print(json_response(success, data={"message": message}))

if __name__ == "__main__":
    main() 