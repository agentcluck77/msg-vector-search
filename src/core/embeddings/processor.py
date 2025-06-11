#!/usr/bin/env python3
"""
Embedding Processor Module
Handles batch embedding generation using sentence transformers
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .vector_db import VectorDatabase

logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    """Processes messages and generates vector embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", vector_db_path: str = "data/vectors/seatalk_vectors.db"):
        self.model_name = model_name
        self.vector_db_path = vector_db_path
        self.model = None
        self.vector_db = None
        self.load_model()
    
    def load_model(self):
        """Load the sentence transformer model"""
        logger.info(f"🤖 Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✓ Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def setup_vector_database(self):
        """Initialize vector database"""
        logger.info(f"🗄️ Setting up vector database at {self.vector_db_path}")
        self.vector_db = VectorDatabase(self.vector_db_path)
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 64, progress_callback=None) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        logger.info(f"🔄 Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches to manage memory
        for i, batch_start in enumerate(tqdm(range(0, len(texts), batch_size), desc="Generating embeddings")):
            batch_texts = texts[batch_start:batch_start+batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_tensor=False)
            
            # Convert to list format
            if hasattr(batch_embeddings, 'tolist'):
                embeddings.extend(batch_embeddings.tolist())
            else:
                embeddings.extend([emb.tolist() if hasattr(emb, 'tolist') else emb for emb in batch_embeddings])
            
            # Update progress if callback provided
            if progress_callback:
                batch_progress = (i + 1) / total_batches
                # Map embedding progress from 85% to 95% of total progress
                overall_progress = 0.85 + (batch_progress * 0.10)
                messages_processed = min(batch_start + batch_size, len(texts))
                progress_callback(overall_progress, f"Generating embeddings... {messages_processed:,}/{len(texts):,} messages ({batch_progress*100:.0f}%)")
        
        logger.info(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def process_embedding_data(self, embedding_data: List[Dict[str, Any]], 
                             batch_size: int = 64, 
                             content_field: str = 'full_content',
                             progress_callback=None) -> List[List[float]]:
        """Process embedding data and generate embeddings"""
        logger.info(f"📊 Processing {len(embedding_data)} messages for embedding generation")
        
        # Extract text content for embedding
        texts = [item[content_field] for item in embedding_data]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size, progress_callback)
        
        return embeddings
    
    def store_embeddings(self, embedding_data: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Store embeddings in vector database"""
        if not self.vector_db:
            self.setup_vector_database()
        
        logger.info(f"💾 Storing {len(embeddings)} embeddings in vector database")
        self.vector_db.store_embeddings(embedding_data, embeddings)
        
        # Get and log statistics
        stats = self.vector_db.get_stats()
        logger.info(f"📊 Database stats: {stats}")
    
    def process_and_store(self, embedding_data: List[Dict[str, Any]], 
                         batch_size: int = 64,
                         content_field: str = 'full_content',
                         progress_callback=None):
        """Complete pipeline: generate and store embeddings"""
        logger.info(f"🚀 Starting complete embedding pipeline for {len(embedding_data)} messages")
        
        # Generate embeddings
        embeddings = self.process_embedding_data(embedding_data, batch_size, content_field, progress_callback)
        
        # Store in vector database
        self.store_embeddings(embedding_data, embeddings)
        
        logger.info("✅ Embedding pipeline completed successfully")
    
    def load_embedding_data_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load embedding data from JSON file"""
        import json
        
        logger.info(f"📂 Loading embedding data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✓ Loaded {len(data)} messages from file")
        return data
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query"""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embedding = self.model.encode([query], convert_to_tensor=False)
        if hasattr(embedding, 'tolist'):
            return embedding[0].tolist() if hasattr(embedding[0], 'tolist') else embedding[0]
        else:
            return embedding[0]
    
    def search_similar_messages(self, query: str, 
                              limit: int = 10,
                              similarity_threshold: float = 0.7,
                              conversation_type: str = None,
                              start_date: str = None,
                              end_date: str = None,
                              session_id: str = None) -> List[Dict[str, Any]]:
        """Search for similar messages using vector similarity"""
        if not self.vector_db:
            self.setup_vector_database()
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        
        # Search in vector database
        results = self.vector_db.search_similar(
            query_embedding, 
            limit=limit,
            similarity_threshold=similarity_threshold,
            conversation_type=conversation_type,
            session_id=session_id
        )
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
        if not self.vector_db:
            self.setup_vector_database()
        
        return self.vector_db.get_stats()
    
    def close(self):
        """Close database connections"""
        if self.vector_db:
            self.vector_db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 