#!/usr/bin/env python3
"""
Semantic Search Engine Module
Integrates database and embeddings for semantic search
"""

import logging
import time
from typing import Dict, Any, List, Optional

from ..database.connection import SeaTalkDatabase
from ..database.processor import MessageProcessor
from ..embeddings.processor import EmbeddingProcessor

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Semantic search engine for SeaTalk messages"""
    
    def __init__(self, db_path: str = None, db_key: str = None, embedding_update_threshold: int = 50):
        """
        Initialize the semantic search engine
        
        Args:
            db_path: Path to the SeaTalk database directory
            db_key: Database decryption key
            embedding_update_threshold: Minimum new messages to trigger embedding update (default: 50)
        """
        # Initialize components
        self.database = SeaTalkDatabase(db_path, db_key)
        self.embedding_update_threshold = embedding_update_threshold
        self.last_embedding_update_time = 0  # Track when embeddings were last updated
        self.embedding_update_cooldown = 60  # 1 minute cooldown for first few searches
        self.long_cooldown = 300  # 5 minutes cooldown after initial period
        self.server_start_time = time.time()  # Track when server started
        self.search_count = 0  # Track number of searches performed
        
        # Connect to database
        if not self.database.connect():
            raise RuntimeError("Failed to connect to database")
            
        self.message_processor = MessageProcessor(self.database)
        self.embedding_processor = EmbeddingProcessor(self.database)
        
    def preload_model(self):
        """
        Pre-load the embedding model to avoid delays during search operations
        """
        logger.info("Pre-loading embedding model...")
        try:
            # This will load the model into memory
            self.embedding_processor.load_model()
            logger.info("✅ Embedding model pre-loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to pre-load embedding model: {e}")
        
    def update_embeddings(self, batch_size: int = 1000, max_messages: int = 10000) -> Dict[str, Any]:
        """
        Update embeddings for new messages
        
        Args:
            batch_size: Number of messages to process in each batch
            max_messages: Maximum total messages to process
            
        Returns:
            Dictionary with update statistics
        """
        start_time = time.time()
        logger.info("Updating embeddings...")
        
        # Get last processed timestamp from vector database
        vector_db = getattr(self.embedding_processor, 'vector_db', None)
        last_timestamp = self.message_processor.get_last_processed_timestamp(vector_db)
        
        # Process new messages with batch processing
        messages = self.message_processor.process_messages(last_timestamp, vector_db, batch_size, max_messages)
        
        if not messages:
            logger.info("No new messages to process")
            return {
                "status": "success",
                "new_messages": 0,
                "update_time_ms": int((time.time() - start_time) * 1000)
            }
        
        # Generate embeddings with hardware-optimized batch size
        messages_with_embeddings = self.embedding_processor.generate_embeddings(messages)
        
        # Store embeddings
        stored_count = self.embedding_processor.store_embeddings(messages_with_embeddings)
        
        # Update last embedding update time
        self.last_embedding_update_time = time.time()
        
        update_time = time.time() - start_time
        logger.info(f"Embeddings updated in {update_time:.2f}s")
        
        return {
            "status": "success",
            "new_messages": stored_count,
            "update_time_ms": int(update_time * 1000),
            "processing_time_seconds": update_time
        }
        
    def search(
        self, 
        query: str, 
        limit: int = 30, 
        similarity_threshold: float = 0.3,
        include_context: bool = True,
        conversation_type: Optional[str] = None,
        session_id: Optional[str] = None,
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Search for messages using semantic similarity
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity threshold
            include_context: Include context information in results
            conversation_type: Filter by conversation type
            session_id: Filter by session ID
            force_update: Force embedding update even if not needed
            
        Returns:
            Dictionary with search results and metadata
        """
        start_time = time.time()
        logger.info(f"Searching for: '{query}'")
        
        # Increment search counter
        self.search_count += 1
        
        # Only update embeddings if forced or if we detect new messages and enough time has passed
        if force_update:
            self.update_embeddings()
        else:
            # Check if enough time has passed since last update (cooldown period)
            current_time = time.time()
            time_since_last_update = current_time - self.last_embedding_update_time
            
            # Use shorter cooldown for first 10 minutes after server startup
            time_since_startup = current_time - self.server_start_time
            active_cooldown = self.embedding_update_cooldown if time_since_startup < 600 else self.long_cooldown
            
            # For the first 5 searches, be extra conservative and use a higher threshold
            effective_threshold = self.embedding_update_threshold * 3 if self.search_count <= 5 else self.embedding_update_threshold
            
            if time_since_last_update < active_cooldown:
                logger.debug(f"Skipping embedding update - cooldown period active ({time_since_last_update:.1f}s < {active_cooldown}s)")
            else:
                # Ensure vector database is set up before checking stats
                if not hasattr(self.embedding_processor, 'vector_db') or not self.embedding_processor.vector_db:
                    self.embedding_processor.setup_vector_database()
                
                # Quick check: only update if we detect significant new messages
                # Get current message count from main database
                if self.database.conn:
                    cursor = self.database.get_cursor()
                    cursor.execute("SELECT COUNT(*) FROM chat_message")
                    total_messages = cursor.fetchone()[0]
                    
                    # Get embedded message count
                    stats = self.get_database_stats()
                    embedded_messages = stats.get('embedded_messages', 0)
                    
                    # Only update if there's a significant difference (more than threshold new messages)
                    if total_messages - embedded_messages > effective_threshold:
                        logger.info(f"Detected {total_messages - embedded_messages} new messages, updating embeddings...")
                        self.update_embeddings()
                    else:
                        logger.debug(f"Skipping embedding update - only {total_messages - embedded_messages} new messages (threshold: {effective_threshold})")
                        logger.info(f"Using existing embeddings ({embedded_messages} messages, {total_messages - embedded_messages} new messages below threshold)")
        
        # Ensure database is connected
        if not self.database.conn:
            if not self.database.connect():
                raise RuntimeError("Failed to connect to database")
        
        # Search for similar messages
        results = self.embedding_processor.search_similar_messages(
            query=query,
            limit=limit,
            similarity_threshold=similarity_threshold,
            conversation_type=conversation_type,
            session_id=session_id
        )
        
        # Format results
        formatted_results = []
        for result in results:
            # Format result
            formatted_result = {
                "message_id": result["message_id"],
                "text_content": result["text_content"],
                "user_id": result["user_id"],
                "user_name": result["user_name"],
                "user_code": result["user_code"],
                "session_id": result["session_id"],
                "human_time": result["human_time"],
                "similarity_score": round(result["similarity_score"], 3),
                "conversation": result["conversation"]
            }
            
            # Include context if requested
            if include_context:
                formatted_result["context"] = result["context"]
                
            formatted_results.append(formatted_result)
            
        search_time = time.time() - start_time
        logger.info(f"Search completed in {search_time:.2f}s, found {len(formatted_results)} results")
        
        # Return results with metadata
        return {
            "status": "success",
            "results": formatted_results,
            "metadata": {
                "total_results": len(formatted_results),
                "query": query,
                "threshold": similarity_threshold,
                "search_time_ms": int(search_time * 1000)
            }
        }
        
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Ensure database is connected
            if not self.database.conn:
                if not self.database.connect():
                    raise RuntimeError("Failed to connect to database")
                    
            cursor = self.database.get_cursor()
            
            # Get total messages
            cursor.execute("SELECT COUNT(*) FROM chat_message")
            total_messages = cursor.fetchone()[0]
            
            # Ensure vector database is set up before querying
            if not hasattr(self.embedding_processor, 'vector_db') or not self.embedding_processor.vector_db:
                self.embedding_processor.setup_vector_database()
            
            # Get embedded messages from persistent vector database
            if hasattr(self.embedding_processor, 'vector_db') and self.embedding_processor.vector_db:
                vector_cursor = self.embedding_processor.vector_db.cursor()
                vector_cursor.execute("SELECT COUNT(*) FROM message_embeddings")
                embedded_messages = vector_cursor.fetchone()[0]
            else:
                embedded_messages = 0
            
            # Get last processed timestamp from vector database
            vector_db = getattr(self.embedding_processor, 'vector_db', None)
            last_timestamp = self.message_processor.get_last_processed_timestamp(vector_db)
            
            # Format last processed time
            last_processed_time = "Never"
            if last_timestamp > 0:
                from datetime import datetime
                dt = datetime.fromtimestamp(last_timestamp)
                last_processed_time = dt.strftime("%b %d, %Y at %I:%M %p")
            
            return {
                "status": "success",
                "total_messages": total_messages,
                "embedded_messages": embedded_messages,
                "embedding_coverage": round(embedded_messages / total_messages * 100, 1) if total_messages > 0 else 0,
                "last_processed_timestamp": last_timestamp,
                "last_processed_time": last_processed_time,
                "database_file": self.database.db_path
            }
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    def close(self):
        """Close database connection"""
        self.database.close()
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close() 