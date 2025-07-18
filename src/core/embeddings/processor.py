#!/usr/bin/env python3
"""
Embedding Processor Module
Handles embeddings generation and vector database operations
"""

import logging
import json
import os
import time
from typing import List, Dict, Any, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import apsw

from ..database.connection import SeaTalkDatabase
from ..database.user_mapper import UserMapper

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """Processes and generates embeddings for messages using sentence transformers"""
    
    def __init__(self, database, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding processor
        
        Args:
            database: SeaTalkDatabase instance
            model_name: Name of the sentence transformer model to use
        """
        self.database = database
        self.model_name = model_name
        self.model = None
        self.vector_db = None
        self.vector_db_path = None
        
        # Import user mapper
        from ..database.user_mapper import UserMapper
        self.user_mapper = UserMapper(database)
        
        # Create vector database tables if they don't exist
        self.setup_vector_database()
    
    def load_model(self) -> None:
        """Load the sentence transformer model"""
        logger.info(f"Loading embedding model: {self.model_name}")
        start_time = time.time()
        
        try:
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def setup_vector_database(self) -> None:
        """Set up persistent vector database tables"""
        logger.info("Setting up persistent vector database...")
        
        try:
            # Create persistent vector database path
            vectors_dir = os.path.join(os.path.dirname(__file__), "../../../data/vectors")
            os.makedirs(vectors_dir, exist_ok=True)
            
            self.vector_db_path = os.path.join(vectors_dir, "embeddings.db")
            
            # Connect to persistent vector database
            self.vector_db = apsw.Connection(self.vector_db_path)
            
            cursor = self.vector_db.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='message_embeddings'
            """)
            
            if not cursor.fetchone():
                # Create tables
                cursor.execute("""
                    CREATE TABLE message_embeddings (
                        message_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        user_id TEXT,
                        conversation_type TEXT,
                        text_content TEXT,
                        context_info TEXT,
                        timestamp INTEGER,
                        datetime TEXT,
                        message_type INTEGER,
                        has_quote INTEGER,
                        quote_text TEXT,
                        embedding_vector TEXT
                    )
                """)
                
                # Create index
                cursor.execute("""
                    CREATE INDEX idx_message_embeddings_session 
                    ON message_embeddings(session_id)
                """)
                
                # Create metadata table to track last processed timestamp
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS vector_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
                
                logger.info("Persistent vector database tables created")
            else:
                logger.info("Persistent vector database tables already exist")
                
        except Exception as e:
            logger.error(f"Failed to set up persistent vector database: {e}")
            raise
    
    def generate_embeddings(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of messages
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            List of messages with embeddings added
        """
        if not messages:
            logger.info("ℹ️ No messages to embed")
            return []
            
        # Load model if not already loaded
        if not self.model:
            self.load_model()
            
        logger.info(f"Generating embeddings for {len(messages)} messages...")
        
        # Extract text content for embedding
        texts = [msg["text_content"] for msg in messages]
        
        # Generate embeddings in batches
        start_time = time.time()
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        embed_time = time.time() - start_time
        
        logger.info(f"Embeddings generated in {embed_time:.2f}s ({len(messages) / embed_time:.1f} msgs/s)")
        
        # Add embeddings to messages
        for i, msg in enumerate(messages):
            msg["embedding"] = embeddings[i].tolist()
            
        return messages
    
    def store_embeddings(self, messages_with_embeddings: List[Dict[str, Any]]) -> int:
        """
        Store embeddings in the vector database
        
        Args:
            messages_with_embeddings: List of messages with embeddings
            
        Returns:
            Number of embeddings stored
        """
        if not messages_with_embeddings:
            return 0
            
        logger.info(f"Storing {len(messages_with_embeddings)} embeddings...")
        
        try:
            # Ensure vector database is set up
            if not self.vector_db:
                self.setup_vector_database()
                
            cursor = self.vector_db.cursor()
            count = 0
            
            # Begin transaction for better performance
            cursor.execute("BEGIN TRANSACTION")
            
            for msg in tqdm(messages_with_embeddings):
                # Convert context to JSON
                context_json = json.dumps(msg["context"])
                
                # Convert embedding to JSON
                embedding_json = json.dumps(msg["embedding"])
                
                # Insert or replace embedding
                cursor.execute("""
                    INSERT OR REPLACE INTO message_embeddings (
                        message_id, session_id, user_id, conversation_type,
                        text_content, context_info, timestamp, datetime,
                        message_type, has_quote, quote_text, embedding_vector
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    msg["message_id"],
                    msg["session_id"],
                    msg["user_id"],
                    msg["conversation"]["type"],
                    msg["text_content"],
                    context_json,
                    msg["timestamp"],
                    msg["human_time"],
                    1,  # message_type (default to text)
                    0,  # has_quote (default to no)
                    "",  # quote_text (default to empty)
                    embedding_json
                ))
                
                count += 1
            
            # Commit transaction
            cursor.execute("COMMIT")
                
            logger.info(f"✓ Stored {count} embeddings")
            return count
            
        except Exception as e:
            # Rollback transaction on error
            if self.vector_db:
                try:
                    cursor = self.vector_db.cursor()
                    cursor.execute("ROLLBACK")
                except:
                    pass
                
            logger.error(f"❌ Failed to store embeddings: {e}")
            return 0
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
    
    def search_similar_messages(
        self, 
        query: str, 
        limit: int = 30,
        similarity_threshold: float = 0.3,
        conversation_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar messages using vector similarity
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            conversation_type: Filter by conversation type
            session_id: Filter by session ID
            
        Returns:
            List of similar messages
        """
        # Ensure vector database is set up
        if not self.vector_db:
            self.setup_vector_database()
        
        # Load model if not already loaded
        if not self.model:
            self.load_model()
            
        # Generate query embedding
        start_time = time.time()
        query_embedding = self.model.encode([query], convert_to_tensor=False)[0].tolist()
        
        # Search in persistent vector database
        cursor = self.vector_db.cursor()
        
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
        
        # Get all embeddings
        query = f"""
        SELECT 
            message_id, session_id, user_id, conversation_type,
            text_content, context_info, timestamp, datetime,
            message_type, has_quote, quote_text, embedding_vector
        FROM message_embeddings 
        {where_clause}
        """
        
        cursor.execute(query, params)
        results = []
        
        for row in cursor:
            # Parse embedding vector
            embedding_vector = json.loads(row[11])
            
            # Calculate similarity
            similarity = self.cosine_similarity(query_embedding, embedding_vector)
            
            if similarity >= similarity_threshold:
                # Parse context info
                context_info = json.loads(row[5])
                
                results.append({
                    'message_id': row[0],
                    'session_id': row[1],
                    'user_id': row[2],
                    'conversation_type': row[3],
                    'text_content': row[4],
                    'context': context_info,
                    'timestamp': row[6],
                    'human_time': row[7],
                    'message_type': row[8],
                    'has_quote': bool(row[9]),
                    'quote_text': row[10],
                    'similarity_score': similarity
                })
        
        # Sort by similarity descending and limit results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        results = results[:limit]
        
        # Get user names and conversation names using UserMapper
        for result in results:
            # Get user name using UserMapper
            user_id = result['user_id']
            user_name = self.user_mapper.get_user_name(user_id)
            
            # Set user fields in the desired format
            result['user_id'] = user_id
            # For fallback names (User {id}), set user_name to the fallback for better display
            # rather than setting it to None which causes confusion
            if user_name.startswith(f'User {user_id}'):
                result['user_name'] = user_name  # Keep the fallback name for display
            else:
                result['user_name'] = user_name  # Use the real name
            result['user_code'] = f"User {user_id}"
            
            # Get conversation name using similar logic as processor
            conversation_name = None
            if result['conversation_type'] == 'group':
                # Try to get group name from main database
                try:
                    session_id = result['session_id']
                    
                    # Ensure main database is connected
                    if not self.database.conn:
                        if not self.database.connect():
                            raise RuntimeError("Failed to connect to main database")
                    
                    # Method 1: Try to find group name in chat messages with group info
                    main_cursor = self.database.get_cursor()
                    main_cursor.execute("""
                        SELECT c FROM chat_message 
                        WHERE sid = ? AND c LIKE '%"n":%' 
                        AND t IN ('c.g.c.i', 'c.g.a.m', 'system')
                        LIMIT 1
                    """, (session_id,))
                    
                    row = main_cursor.fetchone()
                    if row:
                        try:
                            content = json.loads(row[0])
                            if 'n' in content and isinstance(content['n'], str):
                                group_name = content['n'].strip()
                                if group_name and len(group_name) < 100:  # Reasonable group name length
                                    conversation_name = group_name
                        except Exception:
                            pass
                    
                    # Method 2: If no name found, try to extract from group creation messages
                    if not conversation_name:
                        cursor.execute("""
                            SELECT c FROM chat_message 
                            WHERE sid = ? AND t = 'c.g.c.i'
                            ORDER BY ts ASC LIMIT 1
                        """, (session_id,))
                        
                        row = cursor.fetchone()
                        if row:
                            try:
                                content = json.loads(row[0])
                                # Group creation messages might have group info
                                if 'gn' in content:
                                    conversation_name = content['gn'].strip()
                                elif 'group_name' in content:
                                    conversation_name = content['group_name'].strip()
                            except Exception:
                                pass
                                
                except Exception as e:
                    logger.debug(f"Failed to get group name for {result['session_id']}: {e}")
                    pass
                
                result['conversation'] = {
                    'type': result['conversation_type'],
                    'id': result['session_id'],
                    'conversation_name': conversation_name,
                    'conversation_code': f"Group {result['session_id']}"
                }
            else:
                # For private conversations, use the user name (even if it's a fallback)
                # This ensures consistency with how user_name is displayed
                result['conversation'] = {
                    'type': result['conversation_type'],
                    'id': result['user_id'],
                    'conversation_name': user_name,  # For direct messages, conversation name is the user name
                    'conversation_code': f"User {result['user_id']}"
                }
        
        search_time = time.time() - start_time
        logger.info(f"✓ Found {len(results)} similar messages in {search_time:.2f}s")
        
        return results 