#!/usr/bin/env python3
"""
Initialize Database Script
Processes all messages and generates embeddings
"""

import os
import sys
import logging
import argparse
from pathlib import Path

from core.search import SemanticSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Initialize SeaTalk vector search database")
    
    parser.add_argument("--db-path", help="Path to SeaTalk database directory")
    parser.add_argument("--db-key", help="Database decryption key")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all messages")
    parser.add_argument("--reset-timestamp", action="store_true", help="Reset last processed timestamp to current time")
    parser.add_argument("--check-timestamps", action="store_true", help="Check timestamp ranges and detect issues")
    
    args = parser.parse_args()
    
    # Get database path from arguments or environment variable
    db_path = args.db_path or os.environ.get('SEATALK_DB_PATH')
    if not db_path:
        db_path = os.path.expanduser("~/Library/Application Support/SeaTalk")
        logger.info(f"Using default database path: {db_path}")
    
    # Get database key from arguments or environment variable
    db_key = args.db_key or os.environ.get('SEATALK_DB_KEY')
    
    # Initialize search engine
    search_engine = SemanticSearchEngine(db_path, db_key)
    
    # Handle timestamp debugging options
    if args.check_timestamps:
        logger.info("Checking timestamp ranges...")
        
        # Get database stats
        stats = search_engine.get_database_stats()
        logger.info(f"Total messages: {stats['total_messages']:,}")
        logger.info(f"Embedded messages: {stats['embedded_messages']:,}")
        logger.info(f"Last processed timestamp: {stats['last_processed_timestamp']}")
        
        # Get timestamp ranges from main database
        cursor = search_engine.database.get_cursor()
        cursor.execute("SELECT MIN(_createAt), MAX(_createAt) FROM chat_message")
        min_ts, max_ts = cursor.fetchone()
        
        from datetime import datetime
        logger.info(f"Database timestamp range:")
        logger.info(f"  Oldest: {min_ts} ({datetime.fromtimestamp(min_ts)})")
        logger.info(f"  Newest: {max_ts} ({datetime.fromtimestamp(max_ts)})")
        
        # Check if there's a timestamp issue
        last_processed = stats['last_processed_timestamp']
        if last_processed > max_ts:
            logger.warning("⚠️ Last processed timestamp is newer than newest message!")
            logger.warning("⚠️ This could cause issues on other machines")
        elif last_processed < min_ts:
            logger.warning("⚠️ Last processed timestamp is older than oldest message!")
            logger.warning("⚠️ This will cause all messages to be reprocessed")
        
        search_engine.close()
        return 0
    
    if args.reset_timestamp:
        logger.info("Resetting last processed timestamp to current time...")
        
        import time
        current_time = time.time()
        
        # Reset timestamp in vector database
        vector_db = search_engine.embedding_processor.vector_db
        if not vector_db:
            search_engine.embedding_processor.setup_vector_database()
            vector_db = search_engine.embedding_processor.vector_db
        
        search_engine.message_processor.update_last_processed_timestamp(current_time, vector_db)
        logger.info(f"✓ Reset timestamp to {current_time}")
        
        search_engine.close()
        return 0
    
    # Check if we need to force reprocessing
    if args.force:
        logger.info("Forcing reprocessing of all messages")
        
        # Reset last processed timestamp
        vector_db = search_engine.embedding_processor.vector_db
        if not vector_db:
            search_engine.embedding_processor.setup_vector_database()
            vector_db = search_engine.embedding_processor.vector_db
        
        cursor = vector_db.cursor()
        cursor.execute("""
            DELETE FROM vector_metadata
            WHERE key = 'last_processed_timestamp'
        """)
        
        logger.info("Reset last processed timestamp")
    
    # Update embeddings
    update_result = search_engine.update_embeddings()
    
    # Get database stats
    stats = search_engine.get_database_stats()
    
    # Print results
    print("\n" + "="*50)
    print("SeaTalk Vector Search Initialization")
    print("="*50)
    print(f"Database: {stats['database_file']}")
    print(f"Total messages: {stats['total_messages']:,}")
    print(f"Embedded messages: {stats['embedded_messages']:,}")
    print(f"Embedding coverage: {stats['embedding_coverage']}%")
    print(f"Last processed: {stats['last_processed_time']}")
    print(f"New messages processed: {update_result['new_messages']:,}")
    print(f"Update time: {update_result['update_time_ms'] / 1000:.2f}s")
    print("="*50)
    
    # Close search engine
    search_engine.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 