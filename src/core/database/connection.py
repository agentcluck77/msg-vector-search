#!/usr/bin/env python3
"""
SeaTalk Database Connection Module
Handles encrypted database connections and basic operations
"""

import os
import logging
from typing import Optional

try:
    import apsw
    # Only print version info if not called from MCP server
    if not os.environ.get('MCP_SERVER_MODE'):
        print(f"✓ Using APSW-SQLite3MC version: {apsw.apswversion()}")
except ImportError:
    print("❌ Error: apsw-sqlite3mc not installed. Install with: pip install apsw-sqlite3mc")
    raise

logger = logging.getLogger(__name__)

class SeaTalkDatabase:
    """SeaTalk encrypted database connection manager"""
    
    def __init__(self, db_path: str, db_key: str = None):
        self.db_path = db_path
        self.db_key = db_key or os.getenv('DB_KEY')
        if not self.db_key:
            raise ValueError("Database key is required but not provided")
        self.conn: Optional[apsw.Connection] = None
        
    def connect(self) -> bool:
        """Connect to the SeaTalk database with decryption"""
        try:
            self.conn = apsw.Connection(self.db_path)
            self.conn.pragma("key", self.db_key)
            
            # Test connection
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_message")
            count = cursor.fetchone()[0]
            logger.info(f"✓ Connected to database. Found {count:,} total messages")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def get_cursor(self):
        """Get database cursor"""
        if not self.conn:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.conn.cursor()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close() 