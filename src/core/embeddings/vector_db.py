#!/usr/bin/env python3
"""
Vector Database Module
Handles vector storage and similarity search using SQLite with vector extensions
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages vector storage and similarity search operations"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self.setup_database()
    
    def setup_database(self):
        """Initialize vector database and create tables"""
        self.conn = sqlite3.Connection(self.db_path)
        
        logger.info(f"📂 Setting up vector database at {self.db_path}")
        
        # Create message embeddings table
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            conversation_type TEXT NOT NULL,
            text_content TEXT NOT NULL,
            context_info TEXT,
            full_content TEXT NOT NULL,
            timestamp INTEGER,
            datetime TEXT,
            message_type TEXT,
            has_quote BOOLEAN DEFAULT 0,
            quote_text TEXT DEFAULT '',
            conversation_total_messages INTEGER DEFAULT 0,
            conversation_participants INTEGER DEFAULT 0,
            embedding_vector TEXT NOT NULL,  -- JSON array of floats
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes for better performance
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_id ON message_embeddings(session_id)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversation_type ON message_embeddings(conversation_type)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON message_embeddings(timestamp)
        """)
        
        self.conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_id ON message_embeddings(user_id)
        """)
        
        # Create metadata table for search statistics
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS search_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            total_embeddings INTEGER DEFAULT 0,
            embedding_model TEXT DEFAULT '',
            embedding_dimension INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.conn.commit()
        logger.info("✓ Vector database tables created successfully")
    
    def store_embeddings(self, embedding_data: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Store message embeddings in the database"""
        logger.info(f"💾 Storing {len(embedding_data)} embeddings...")
        
        if len(embedding_data) != len(embeddings):
            raise ValueError("Mismatch between embedding data and embedding vectors")
        
        # Prepare data for insertion
        insert_data = []
        for data, embedding in zip(embedding_data, embeddings):
            # Ensure datetime is string
            datetime_value = data.get('datetime', '')
            if hasattr(datetime_value, 'isoformat'):
                datetime_value = datetime_value.isoformat()
            
            # Handle other potentially problematic fields
            has_quote = 1 if data.get('has_quote') else 0
            quote_text = data.get('quote_text', '')
            conv_total = data.get('conversation_total_messages', 0)
            conv_participants = data.get('conversation_participants', 0)
            
            insert_data.append((
                data.get('message_id', ''),
                data.get('session_id', ''),
                data.get('user_id', ''),
                data.get('conversation_type', ''),
                data.get('text_content', ''),
                data.get('context_info', ''),
                data.get('full_content', ''),
                data.get('timestamp', 0),
                datetime_value,
                data.get('message_type', ''),
                has_quote,
                quote_text,
                conv_total,
                conv_participants,
                json.dumps(embedding)  # Store as JSON string
            ))
        
        # Batch insert
        self.conn.executemany("""
        INSERT OR REPLACE INTO message_embeddings (
            message_id, session_id, user_id, conversation_type, text_content,
            context_info, full_content, timestamp, datetime, message_type,
            has_quote, quote_text, conversation_total_messages, 
            conversation_participants, embedding_vector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, insert_data)
        
        self.conn.commit()
        logger.info(f"✓ Stored {len(insert_data)} embeddings successfully")
        
        # Update metadata
        self.update_metadata(len(insert_data))
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search_similar(self, query_embedding: List[float], 
                      limit: int = 10, 
                      similarity_threshold: float = 0.7,
                      conversation_type: str = None,
                      session_id: str = None) -> List[Dict[str, Any]]:
        """Search for similar messages using cosine similarity"""
        logger.info(f"🔍 Searching for similar messages (limit={limit}, threshold={similarity_threshold})")
        
        # Build query conditions
        where_conditions = []
        params = []
        
        if conversation_type:
            where_conditions.append("conversation_type = ?")
            params.append(conversation_type)
        
        if session_id:
            where_conditions.append("session_id = ?")
            params.append(session_id)
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # Get all embeddings (we'll need to implement vector similarity in Python for now)
        query = f"""
        SELECT 
            message_id, session_id, user_id, conversation_type,
            text_content, context_info, timestamp, datetime,
            message_type, has_quote, quote_text, embedding_vector
        FROM message_embeddings 
        {where_clause}
        """
        
        cursor = self.conn.execute(query, params)
        results = []
        
        for row in cursor:
            # Parse embedding vector
            embedding_vector = json.loads(row[11])
            
            # Calculate similarity
            similarity = self.cosine_similarity(query_embedding, embedding_vector)
            
            if similarity >= similarity_threshold:
                results.append({
                    'message_id': row[0],
                    'session_id': row[1],
                    'user_id': row[2],
                    'conversation_type': row[3],
                    'text_content': row[4],
                    'context_info': row[5],
                    'timestamp': row[6],
                    'datetime': row[7],
                    'message_type': row[8],
                    'has_quote': bool(row[9]),
                    'quote_text': row[10],
                    'similarity': similarity
                })
        
        # Sort by similarity descending and limit results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = results[:limit]
        
        logger.info(f"✓ Found {len(results)} similar messages")
        return results
    
    def get_conversation_context(self, session_id: str, 
                               around_timestamp: int = None,
                               context_window: int = 5) -> List[Dict[str, Any]]:
        """Get conversation context around a specific timestamp"""
        if around_timestamp is None:
            # Get all messages from conversation
            query = """
            SELECT message_id, text_content, user_id, timestamp, datetime
            FROM message_embeddings
            WHERE session_id = ?
            ORDER BY timestamp ASC
            """
            params = [session_id]
        else:
            # Get messages around the timestamp
            window_ms = context_window * 60 * 1000  # Convert minutes to milliseconds
            query = """
            SELECT message_id, text_content, user_id, timestamp, datetime  
            FROM message_embeddings
            WHERE session_id = ?
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
            """
            params = [session_id, around_timestamp - window_ms, around_timestamp + window_ms]
        
        cursor = self.conn.execute(query, params)
        return [
            {
                'message_id': row[0],
                'text_content': row[1], 
                'user_id': row[2],
                'timestamp': row[3],
                'datetime': row[4]
            }
            for row in cursor
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM message_embeddings")
        total_embeddings = cursor.fetchone()[0]
        
        cursor = self.conn.execute("""
        SELECT conversation_type, COUNT(*) 
        FROM message_embeddings 
        GROUP BY conversation_type
        """)
        by_type = dict(cursor.fetchall())
        
        cursor = self.conn.execute("""
        SELECT COUNT(DISTINCT session_id) FROM message_embeddings
        """)
        unique_conversations = cursor.fetchone()[0]
        
        # Calculate group and private messages
        group_messages = by_type.get('group', 0)
        private_messages = by_type.get('private', 0)
        total_messages = total_embeddings  # Each embedding represents one message
        
        return {
            'total_embeddings': total_embeddings,
            'total_messages': total_messages,
            'total_conversations': unique_conversations,
            'group_messages': group_messages,
            'private_messages': private_messages,
            'by_conversation_type': by_type,
            'unique_conversations': unique_conversations
        }
    
    def update_metadata(self, embedding_count: int, model: str = "all-MiniLM-L6-v2", dimension: int = 384):
        """Update search metadata"""
        self.conn.execute("""
        INSERT OR REPLACE INTO search_metadata (id, total_embeddings, embedding_model, embedding_dimension)
        VALUES (1, ?, ?, ?)
        """, (embedding_count, model, dimension))
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 