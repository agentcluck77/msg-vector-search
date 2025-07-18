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
        """Load the sentence transformer model with hardware optimizations"""
        logger.info(f"Loading embedding model: {self.model_name}")
        start_time = time.time()
        
        try:
            # Use hardware optimizer for better performance
            from ..utils.hardware_optimizer import get_hardware_optimizer
            hw_optimizer = get_hardware_optimizer()
            
            # Configure PyTorch for optimal performance
            hw_optimizer.configure_pytorch()
            
            # Get optimal device
            device = hw_optimizer.get_pytorch_device()
            logger.info(f"🔧 Using device: {device}")
            
            # Load model with hardware-specific device
            self.model = SentenceTransformer(self.model_name, device=device)
            
            # Set optimal batch size from hardware optimizer
            self.optimal_batch_size = hw_optimizer.get_embedding_batch_size()
            logger.info(f"🔧 Optimal batch size set to: {self.optimal_batch_size}")
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s with {device} acceleration")
            
            # Print performance summary
            hw_optimizer.print_performance_summary()
                
        except Exception as e:
            logger.warning(f"⚠️ Hardware optimization failed: {e}")
            logger.info("⚠️ Falling back to basic model loading")
            
            # Fallback to basic model loading
            try:
                # Try with MPS for Apple Silicon
                import platform
                if platform.system() == 'Darwin' and platform.machine() == 'arm64':
                    try:
                        import torch
                        if torch.backends.mps.is_available():
                            self.model = SentenceTransformer(self.model_name, device='mps')
                            self.optimal_batch_size = 64
                            logger.info("✅ Using Apple Silicon MPS acceleration (basic)")
                        else:
                            raise Exception("MPS not available")
                    except:
                        self.model = SentenceTransformer(self.model_name, device='cpu')
                        self.optimal_batch_size = 32
                        logger.info("✅ Using CPU (basic)")
                else:
                    # Intel Mac or other platform
                    self.model = SentenceTransformer(self.model_name)
                    self.optimal_batch_size = 32
                    logger.info("✅ Using CPU (basic)")
                    
                load_time = time.time() - start_time
                logger.info(f"✅ Model loaded in {load_time:.2f}s (basic mode)")
                
            except Exception as fallback_error:
                logger.error(f"❌ Even basic model loading failed: {fallback_error}")
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
    
    def generate_embeddings(self, messages: List[Dict[str, Any]], batch_size: int = None) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of messages with hardware optimizations
        
        Args:
            messages: List of message dictionaries
            batch_size: Number of messages to process in each embedding batch (auto-optimized if None)
            
        Returns:
            List of messages with embeddings added
        """
        if not messages:
            logger.info("ℹ️ No messages to embed")
            return []
            
        # Load model if not already loaded
        if not self.model:
            self.load_model()
            
        # Use hardware-optimized batch size if not specified
        if batch_size is None:
            batch_size = getattr(self, 'optimal_batch_size', 32)
            
        logger.info(f"Generating embeddings for {len(messages)} messages (batch size: {batch_size})")
        
        start_time = time.time()
        
        # Pre-extract all text content for better memory efficiency
        all_texts = [msg["text_content"] for msg in messages]
        
        try:
            # Use optimized batch processing with parallel encoding
            import concurrent.futures
            import threading
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Try to use hardware optimizer, but fall back if it fails
            try:
                from ..utils.hardware_optimizer import get_hardware_optimizer
                hw_optimizer = get_hardware_optimizer()
                use_parallel = hw_optimizer.should_use_parallel_processing()
            except Exception as hw_error:
                logger.warning(f"⚠️ Hardware optimizer failed: {hw_error}")
                use_parallel = False
            
            if use_parallel and len(messages) > batch_size * 2:
                # Use parallel processing for large datasets on capable hardware
                logger.info("🚀 Using parallel processing for optimized hardware")
                all_embeddings = self._parallel_encode_optimized(all_texts, batch_size, hw_optimizer)
            else:
                # Use standard batched processing
                all_embeddings = self._batch_encode_standard(all_texts, batch_size)
            
            # Add embeddings to messages
            for i, msg in enumerate(messages):
                msg["embedding"] = all_embeddings[i].tolist()
            
            embed_time = time.time() - start_time
            messages_per_second = len(messages) / embed_time
            logger.info(f"✅ Embeddings generated in {embed_time:.2f}s ({messages_per_second:.1f} msgs/s)")
            
            return messages
            
        except Exception as e:
            logger.warning(f"⚠️ Error in optimized embedding generation: {e}")
            logger.info("⚠️ Falling back to simple embedding generation")
            # Fallback to simple batching
            return self._fallback_generate_embeddings(messages, batch_size)
    
    def _parallel_encode_optimized(self, texts: List[str], batch_size: int, hw_optimizer) -> np.ndarray:
        """Parallel encoding optimized for detected hardware"""
        import concurrent.futures
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Get optimal number of workers from hardware optimizer
        num_workers = hw_optimizer.get_parallel_workers()
        chunk_size = max(batch_size, len(texts) // num_workers)
        
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        logger.info(f"🔄 Processing {len(chunks)} chunks with {num_workers} workers (hardware optimized)")
        
        all_embeddings = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._encode_chunk, chunk, idx): (chunk, idx) 
                for idx, chunk in enumerate(chunks)
            }
            
            # Collect results in order
            results = [None] * len(chunks)
            for future in as_completed(future_to_chunk):
                chunk, idx = future_to_chunk[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"❌ Error in chunk {idx}: {e}")
                    # Fallback to sequential processing for this chunk
                    results[idx] = self.model.encode(chunk, convert_to_tensor=False, show_progress_bar=False)
        
        # Concatenate all results
        return np.concatenate(results, axis=0)
    
    def _encode_chunk(self, texts: List[str], chunk_idx: int) -> np.ndarray:
        """Encode a chunk of texts"""
        logger.debug(f"🔄 Processing chunk {chunk_idx + 1} ({len(texts)} texts)")
        return self.model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    
    def _batch_encode_standard(self, texts: List[str], batch_size: int) -> np.ndarray:
        """Standard batched encoding with progress tracking"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_end = min(i + batch_size, len(texts))
            batch_texts = texts[i:batch_end]
            
            batch_num = i // batch_size + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
            
            # Generate embeddings for this batch
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=False, show_progress_bar=False)
            all_embeddings.append(batch_embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def _fallback_generate_embeddings(self, messages: List[Dict[str, Any]], batch_size: int) -> List[Dict[str, Any]]:
        """Fallback to simple embedding generation"""
        logger.info("⚠️ Using fallback embedding generation")
        
        start_time = time.time()
        
        # Extract text content for embedding
        texts = [msg["text_content"] for msg in messages]
        
        # Generate embeddings in one go (simple approach)
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        
        # Add embeddings to messages
        for i, msg in enumerate(messages):
            msg["embedding"] = embeddings[i].tolist()
        
        embed_time = time.time() - start_time
        logger.info(f"✅ Fallback embeddings generated in {embed_time:.2f}s ({len(messages) / embed_time:.1f} msgs/s)")
        
        return messages
    
    def store_embeddings(self, messages_with_embeddings: List[Dict[str, Any]]) -> int:
        """
        Store embeddings in the vector database with optimized bulk operations
        
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
            
            # Optimize SQLite for bulk operations
            cursor.execute("PRAGMA journal_mode = WAL")  # Write-Ahead Logging for better performance
            cursor.execute("PRAGMA synchronous = NORMAL")  # Balanced durability/performance
            cursor.execute("PRAGMA cache_size = 10000")  # Larger cache for better performance
            cursor.execute("PRAGMA temp_store = MEMORY")  # Store temp data in memory
            
            # Begin transaction for better performance
            cursor.execute("BEGIN TRANSACTION")
            
            # Use bulk insert with executemany for better performance
            insert_data = []
            for msg in messages_with_embeddings:
                # Convert context to JSON
                context_json = json.dumps(msg["context"])
                
                # Convert embedding to JSON
                embedding_json = json.dumps(msg["embedding"])
                
                insert_data.append((
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
            
            # Use executemany for bulk insert (much faster than individual inserts)
            cursor.executemany("""
                INSERT OR REPLACE INTO message_embeddings (
                    message_id, session_id, user_id, conversation_type,
                    text_content, context_info, timestamp, datetime,
                    message_type, has_quote, quote_text, embedding_vector
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, insert_data)
            
            # Commit transaction
            cursor.execute("COMMIT")
            
            # Reset SQLite settings to defaults for normal operations
            cursor.execute("PRAGMA synchronous = FULL")
            
            count = len(insert_data)
            logger.info(f"✓ Stored {count} embeddings using bulk operations")
            return count
            
        except Exception as e:
            # Rollback transaction on error
            if self.vector_db:
                try:
                    cursor = self.vector_db.cursor()
                    cursor.execute("ROLLBACK")
                    cursor.execute("PRAGMA synchronous = FULL")  # Reset to default
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