#!/usr/bin/env python3
"""
SeaTalk Database Connection Module
Handles encrypted database connections and snapshot creation
"""

import os
import logging
import shutil
import glob
import time
from typing import Optional
from pathlib import Path
import sys

try:
    import apsw
    # Only print version info if not called from MCP server
    if not os.environ.get('MCP_SERVER_MODE'):
        try:
            # Try different version attribute names
            version = getattr(apsw, 'apswversion', lambda: getattr(apsw, '__version__', 'unknown'))()
            sys.stderr.write(f"Using APSW-SQLite3MC version: {version}\n")
        except:
            sys.stderr.write("Using APSW-SQLite3MC (version unknown)\n")
except ImportError:
    print("Error: apsw-sqlite3mc not installed. Install with: uv sync")
    raise

logger = logging.getLogger(__name__)

class SeaTalkDatabase:
    """SeaTalk encrypted database connection manager"""
    
    def __init__(self, db_path: str = None, db_key: str = None):
        """
        Initialize the database connection manager
        
        Args:
            db_path: Path to the SeaTalk database directory (default: from env var)
            db_key: Database decryption key (default: from env var)
        """
        # Get database path from environment variable if not provided
        self.db_dir = db_path or os.environ.get('SEATALK_DB_PATH')
        if not self.db_dir:
            self.db_dir = os.path.expanduser("~/Library/Application Support/SeaTalk")
            
        # Get database key from environment variable if not provided
        self.db_key = db_key or os.environ.get('SEATALK_DB_KEY')
        
        # Initialize connection
        self.conn: Optional[apsw.Connection] = None
        self.db_path: Optional[str] = None
        self.snapshot_path: Optional[str] = None
        
    def find_latest_database(self) -> str:
        """
        Find the most recent and largest main database file
        
        Returns:
            Path to the latest database file
        """
        # Find all main_*.sqlite files
        pattern = os.path.join(self.db_dir, "main_*.sqlite")
        db_files = glob.glob(pattern)
        
        if not db_files:
            raise FileNotFoundError(f"No SeaTalk database files found in {self.db_dir}")
        
        # Sort by modification time (newest first) and size (largest first)
        db_files.sort(key=lambda f: (os.path.getmtime(f), os.path.getsize(f)), reverse=True)
        
        db_path = db_files[0]
        logger.info(f"Found latest database: {os.path.basename(db_path)}")
        
        return db_path
    
    def create_snapshot(self, db_path: str) -> str:
        """
        Create a snapshot of the database to avoid conflicts with running SeaTalk
        Reuses existing snapshots if they're recent and match the source database
        
        Args:
            db_path: Path to the original database file
            
        Returns:
            Path to the snapshot database file
        """
        # Create snapshots directory if it doesn't exist
        snapshots_dir = os.path.join(os.path.dirname(__file__), "../../../data/snapshots")
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Get source database modification time and size
        source_mtime = os.path.getmtime(db_path)
        source_size = os.path.getsize(db_path)
        base_name = os.path.basename(db_path)
        
        # Look for existing snapshots
        pattern = os.path.join(snapshots_dir, f"snapshot_*_{base_name}")
        existing_snapshots = glob.glob(pattern)
        
        # Check if we can reuse an existing snapshot
        for snapshot in existing_snapshots:
            try:
                snapshot_mtime = os.path.getmtime(snapshot)
                snapshot_size = os.path.getsize(snapshot)
                
                # Reuse if snapshot is newer than source and same size
                # (source hasn't been modified since snapshot was created)
                if snapshot_mtime >= source_mtime and snapshot_size == source_size:
                    logger.info(f"Reusing existing snapshot: {os.path.basename(snapshot)}")
                    return snapshot
            except OSError:
                # Snapshot file might be corrupted, skip it
                continue
        
        # Create new snapshot if none can be reused
        timestamp = int(time.time())
        snapshot_filename = f"snapshot_{timestamp}_{base_name}"
        snapshot_path = os.path.join(snapshots_dir, snapshot_filename)
        
        # Copy the database file
        shutil.copy2(db_path, snapshot_path)
        logger.info(f"Created new database snapshot: {snapshot_filename}")
        
        return snapshot_path
    
    def connect(self) -> bool:
        """
        Connect to the SeaTalk database with decryption
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Find the latest database file
            self.db_path = self.find_latest_database()
            
            # Create a snapshot
            self.snapshot_path = self.create_snapshot(self.db_path)
            
            # Connect to the snapshot
            self.conn = apsw.Connection(self.snapshot_path)
            
            # Set key for decryption if provided
            if self.db_key:
                self.conn.pragma("key", self.db_key)
            
            # Test connection
            cursor = self.conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chat_message")
            count = cursor.fetchone()[0]
            logger.info(f"Connected to database. Found {count:,} total messages")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            self.conn = None  # Reset connection on failure
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
            logger.info("Database connection closed")
            
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close() 