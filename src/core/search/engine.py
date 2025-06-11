#!/usr/bin/env python3
"""
Semantic Search Engine with Temporal Awareness
Handles search queries, vector similarity, temporal ranking, and result processing
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import dateparser
from dateutil.relativedelta import relativedelta

from ..embeddings.processor import EmbeddingProcessor
from ..database.user_mapper import UserMapper

logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """Advanced semantic search engine with temporal awareness for SeaTalk messages"""
    
    def __init__(self, vector_db_path: str = "data/vectors/seatalk_vectors.db", db_key: str = None):
        self.vector_db_path = vector_db_path
        self.embedding_processor = EmbeddingProcessor(vector_db_path=vector_db_path)
        
        # Initialize user mapper to resolve user IDs to names
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent.parent.parent
        sys.path.insert(0, str(project_root))
        from config import DB_PATH, DB_KEY
        
        # Use provided db_key if available, otherwise fall back to config
        db_key_to_use = db_key or DB_KEY
        
        # Initialize user mapper if DB_KEY is available, otherwise defer
        try:
            if db_key_to_use:
                self.user_mapper = UserMapper(str(DB_PATH), db_key_to_use)
            else:
                self.user_mapper = None
                logger.warning("User mapper not initialized - no database key available")
        except Exception as e:
            self.user_mapper = None
            logger.warning(f"Failed to initialize user mapper: {e}")
        
        # Temporal query patterns
        self.temporal_patterns = [
            r'\b(latest|recent|newest|last|current)\b',
            r'\b(today|yesterday|this week|last week|this month|last month)\b',
            r'\b(ago|since|before|after|during)\b',
            r'\b(first|earliest|oldest)\b',
            r'\b(when|what time|what date)\b',
            r'\b(\d+\s*(day|week|month|year)s?\s*(ago|back))\b'
        ]
        
        # Recency keywords that indicate temporal intent
        self.recency_keywords = {
            'latest', 'recent', 'newest', 'last', 'current',
            'today', 'yesterday', 'now', 'just', 'recently'
        }
        
        # Historical keywords that indicate older content
        self.historical_keywords = {
            'first', 'earliest', 'oldest', 'initial', 'beginning'
        }
    
    def _classify_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify query to determine temporal intent"""
        query_lower = query.lower()
        
        # Check for temporal patterns
        is_temporal = any(re.search(pattern, query_lower, re.IGNORECASE) 
                         for pattern in self.temporal_patterns)
        
        # Determine temporal direction
        has_recency_intent = any(keyword in query_lower for keyword in self.recency_keywords)
        has_historical_intent = any(keyword in query_lower for keyword in self.historical_keywords)
        
        # Parse specific time references
        time_reference = self._parse_time_reference(query)
        
        return {
            'is_temporal': is_temporal,
            'has_recency_intent': has_recency_intent,
            'has_historical_intent': has_historical_intent,
            'time_reference': time_reference,
            'temporal_strength': self._calculate_temporal_strength(query_lower)
        }
    
    def _parse_time_reference(self, query: str) -> Optional[Dict[str, Any]]:
        """Parse specific time references from query"""
        try:
            # Use dateparser to extract time references
            parsed_time = dateparser.parse(query, settings={'PREFER_DATES_FROM': 'past'})
            if parsed_time:
                return {
                    'parsed_datetime': parsed_time,
                    'is_relative': any(word in query.lower() for word in ['ago', 'since', 'last', 'recent'])
                }
        except:
            pass
        
        # Manual parsing for common patterns
        query_lower = query.lower()
        now = datetime.now()
        
        if 'today' in query_lower:
            return {'parsed_datetime': now.replace(hour=0, minute=0, second=0, microsecond=0), 'period': 'day'}
        elif 'yesterday' in query_lower:
            return {'parsed_datetime': now - timedelta(days=1), 'period': 'day'}
        elif 'this week' in query_lower:
            return {'parsed_datetime': now - timedelta(days=now.weekday()), 'period': 'week'}
        elif 'last week' in query_lower:
            return {'parsed_datetime': now - timedelta(days=now.weekday() + 7), 'period': 'week'}
        elif 'this month' in query_lower:
            return {'parsed_datetime': now.replace(day=1), 'period': 'month'}
        elif 'last month' in query_lower:
            return {'parsed_datetime': now.replace(day=1) - relativedelta(months=1), 'period': 'month'}
        
        return None
    
    def _calculate_temporal_strength(self, query: str) -> float:
        """Calculate how strongly temporal the query is (0.0 to 1.0)"""
        temporal_indicators = 0
        total_words = len(query.split())
        
        # Count temporal keywords
        for keyword in self.recency_keywords | self.historical_keywords:
            if keyword in query:
                temporal_indicators += 1
        
        # Count temporal patterns
        for pattern in self.temporal_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                temporal_indicators += 1
        
        # Normalize by query length
        if total_words == 0:
            return 0.0
        
        strength = min(temporal_indicators / total_words * 2, 1.0)  # Scale and cap at 1.0
        return strength
    
    def _get_temporal_search_params(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Convert temporal intent to search parameters"""
        params = {}
        
        if intent['time_reference']:
            time_ref = intent['time_reference']
            if 'parsed_datetime' in time_ref:
                params['reference_time'] = time_ref['parsed_datetime']
                params['time_period'] = time_ref.get('period', 'day')
        
        if intent['has_recency_intent']:
            params['sort_order'] = 'newest_first'
            params['temporal_boost'] = 0.3  # Boost recent messages
        elif intent['has_historical_intent']:
            params['sort_order'] = 'oldest_first'
            params['temporal_boost'] = 0.3  # Boost older messages
        
        return params
    
    def _apply_temporal_ranking(self, results: List[Dict[str, Any]], 
                              temporal_params: Dict[str, Any],
                              temporal_strength: float) -> List[Dict[str, Any]]:
        """Apply temporal ranking to search results"""
        if not results or temporal_strength < 0.1:
            return results
        
        now = datetime.now()
        
        for result in results:
            # Parse message timestamp
            try:
                if result.get('datetime'):
                    msg_time = datetime.fromisoformat(result['datetime'].replace('Z', '+00:00'))
                elif result.get('timestamp'):
                    msg_time = datetime.fromtimestamp(result['timestamp'] / 1000)  # Convert ms to seconds
                else:
                    continue
                
                # Calculate time-based score
                time_diff = abs((now - msg_time).total_seconds())
                
                # Recency score (higher for newer messages)
                max_age_seconds = 365 * 24 * 3600  # 1 year
                recency_score = max(0, 1 - (time_diff / max_age_seconds))
                
                # Apply temporal boost based on intent
                temporal_boost = temporal_params.get('temporal_boost', 0.0)
                sort_order = temporal_params.get('sort_order', 'newest_first')
                
                if sort_order == 'oldest_first':
                    # Boost older messages
                    temporal_score = (1 - recency_score) * temporal_boost
                else:
                    # Boost newer messages (default)
                    temporal_score = recency_score * temporal_boost
                
                # Combine semantic similarity with temporal score
                original_score = result.get('similarity_score', result.get('similarity', 0))
                hybrid_score = (original_score * (1 - temporal_strength)) + (temporal_score * temporal_strength)
                
                result['hybrid_score'] = hybrid_score
                result['temporal_score'] = temporal_score
                result['recency_score'] = recency_score
                
            except Exception as e:
                logger.warning(f"Failed to apply temporal ranking to result: {e}")
                result['hybrid_score'] = result.get('similarity_score', result.get('similarity', 0))
        
        # Sort by hybrid score
        results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        return results
    
    def search(self, query: str, 
               limit: int = 10,
               similarity_threshold: float = 0.7,
               conversation_type: str = None,
               session_id: str = None,
               include_context: bool = False) -> Dict[str, Any]:
        """
        Perform hybrid semantic and temporal search on messages
        
        Args:
            query: Search query string
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0.0-1.0)
            conversation_type: Filter by 'group' or 'private'
            session_id: Filter by specific conversation
            include_context: Whether to include conversation context
        
        Returns:
            Search results with metadata including temporal ranking
        """
        logger.info(f"🔍 Searching for: '{query}' (limit={limit}, threshold={similarity_threshold})")
        
        try:
            # Analyze query for temporal intent
            query_intent = self._classify_query_intent(query)
            temporal_params = self._get_temporal_search_params(query_intent)
            
            # Log temporal analysis for debugging
            if query_intent['is_temporal']:
                logger.info(f"🕒 Detected temporal query (strength: {query_intent['temporal_strength']:.2f})")
                if query_intent['has_recency_intent']:
                    logger.info("⏰ Recency intent detected - boosting recent messages")
                elif query_intent['has_historical_intent']:
                    logger.info("📅 Historical intent detected - boosting older messages")
            
            # Perform vector similarity search
            results = self.embedding_processor.search_similar_messages(
                query=query,
                limit=limit * 2,  # Get more results for temporal filtering
                similarity_threshold=similarity_threshold,
                conversation_type=conversation_type,
                session_id=session_id
            )
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = self._process_search_result(result)
                
                # Add conversation context if requested
                if include_context:
                    context = self._get_conversation_context(
                        result['session_id'], 
                        result['timestamp']
                    )
                    processed_result['context'] = context
                
                processed_results.append(processed_result)
            
            # Apply temporal ranking if query has temporal intent
            if query_intent['is_temporal'] and query_intent['temporal_strength'] > 0.1:
                processed_results = self._apply_temporal_ranking(
                    processed_results, 
                    temporal_params, 
                    query_intent['temporal_strength']
                )
                logger.info(f"🔄 Applied temporal ranking (strength: {query_intent['temporal_strength']:.2f})")
            
            # Limit results after temporal ranking
            processed_results = processed_results[:limit]
            
            search_metadata = {
                'query': query,
                'total_results': len(processed_results),
                'temporal_analysis': {
                    'is_temporal': query_intent['is_temporal'],
                    'temporal_strength': query_intent['temporal_strength'],
                    'has_recency_intent': query_intent['has_recency_intent'],
                    'has_historical_intent': query_intent['has_historical_intent'],
                    'temporal_boost_applied': query_intent['temporal_strength'] > 0.1
                },
                'search_parameters': {
                    'limit': limit,
                    'similarity_threshold': similarity_threshold,
                    'conversation_type': conversation_type,
                    'session_id': session_id,
                    'include_context': include_context
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✓ Found {len(processed_results)} results")
            
            return {
                'results': processed_results,
                'metadata': search_metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return {
                'results': [],
                'metadata': {'error': str(e), 'query': query},
                'error': str(e)
            }
    
    def _process_search_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enrich a single search result"""
        user_id = result['user_id']
        
        # Get user name if user mapper is available
        if self.user_mapper:
            user_name = self.user_mapper.get_user_name(user_id)
        else:
            user_name = f"User {user_id}"
        
        processed = {
            'message_id': result['message_id'],
            'text_content': result['text_content'],
            'similarity_score': round(result['similarity'], 4),
            'conversation': {
                'session_id': result['session_id'],
                'type': result['conversation_type']
            },
            'user_id': user_id,
            'user_name': user_name,
            'timestamp': result['timestamp'],
            'datetime': result['datetime'],
            'message_type': result['message_type']
        }
        
        # Add hybrid scoring information if available
        if 'hybrid_score' in result:
            processed['hybrid_score'] = round(result['hybrid_score'], 4)
            processed['temporal_score'] = round(result.get('temporal_score', 0), 4)
            processed['recency_score'] = round(result.get('recency_score', 0), 4)
        
        # Add quote information if available
        if result.get('has_quote') and result.get('quote_text'):
            processed['quote'] = {
                'has_quote': True,
                'quote_text': result['quote_text']
            }
        
        # Add formatted datetime
        if result['datetime']:
            try:
                dt = datetime.fromisoformat(result['datetime'].replace('Z', '+00:00'))
                processed['formatted_date'] = dt.strftime("%Y-%m-%d %H:%M:%S")
                processed['human_time'] = self._format_human_time(dt)
            except:
                processed['formatted_date'] = result['datetime']
                processed['human_time'] = "Unknown time"
        
        return processed
    
    def _get_conversation_context(self, session_id: str, timestamp: int, context_window: int = 3) -> List[Dict[str, Any]]:
        """Get conversation context around a message"""
        try:
            context = self.embedding_processor.vector_db.get_conversation_context(
                session_id=session_id,
                around_timestamp=timestamp,
                context_window=context_window
            )
            
            # Process context messages
            processed_context = []
            for msg in context:
                processed_context.append({
                    'message_id': msg['message_id'],
                    'text_content': msg['text_content'],
                    'user_id': msg['user_id'],
                    'timestamp': msg['timestamp'],
                    'datetime': msg['datetime']
                })
            
            return processed_context
            
        except Exception as e:
            logger.warning(f"Failed to get context for {session_id}: {e}")
            return []
    
    def _format_human_time(self, dt: datetime) -> str:
        """Format datetime in a human-readable way"""
        now = datetime.now()
        diff = now - dt
        
        if diff.days > 365:
            years = diff.days // 365
            return f"{years} year{'s' if years > 1 else ''} ago"
        elif diff.days > 30:
            months = diff.days // 30
            return f"{months} month{'s' if months > 1 else ''} ago"
        elif diff.days > 0:
            return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
        else:
            return "Just now"
    
    def search_by_conversation(self, session_id: str, 
                             query: str = None,
                             limit: int = 50) -> Dict[str, Any]:
        """Search within a specific conversation"""
        logger.info(f"🔍 Searching conversation {session_id}")
        
        if query:
            # Semantic search within conversation
            return self.search(
                query=query,
                limit=limit,
                session_id=session_id,
                similarity_threshold=0.5,  # Lower threshold for within-conversation search
                include_context=True
            )
        else:
            # Get all messages from conversation
            try:
                context = self.embedding_processor.vector_db.get_conversation_context(session_id)
                processed_results = []
                
                for msg in context:
                    processed_results.append({
                        'message_id': msg['message_id'],
                        'text_content': msg['text_content'],
                        'user_id': msg['user_id'],
                        'timestamp': msg['timestamp'],
                        'datetime': msg['datetime']
                    })
                
                return {
                    'results': processed_results[:limit],
                    'metadata': {
                        'session_id': session_id,
                        'total_results': len(processed_results),
                        'query_type': 'conversation_browse'
                    }
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to get conversation {session_id}: {e}")
                return {
                    'results': [],
                    'metadata': {'error': str(e), 'session_id': session_id},
                    'error': str(e)
                }
    
    def get_popular_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversations with the most messages"""
        try:
            stats = self.embedding_processor.get_database_stats()
            
            # This would need to be implemented in the vector database
            # For now, return basic stats
            return [
                {
                    'conversation_type': conv_type,
                    'message_count': count
                }
                for conv_type, count in stats.get('by_conversation_type', {}).items()
            ]
            
        except Exception as e:
            logger.error(f"❌ Failed to get popular conversations: {e}")
            return []
    
    def suggest_queries(self, partial_query: str) -> List[str]:
        """Suggest related queries based on existing messages"""
        # This would be enhanced with actual query suggestion logic
        # For now, return some common patterns
        suggestions = []
        
        if partial_query.lower() in ['what', 'when', 'who', 'how', 'why']:
            suggestions = [
                f"{partial_query} did we discuss about the project?",
                f"{partial_query} was the meeting scheduled?",
                f"{partial_query} mentioned the deadline?"
            ]
        elif 'project' in partial_query.lower():
            suggestions = [
                "project timeline and milestones",
                "project budget discussions",
                "project team assignments"
            ]
        elif 'meeting' in partial_query.lower():
            suggestions = [
                "meeting notes and action items",
                "meeting schedule changes",
                "meeting participants and decisions"
            ]
        
        return suggestions[:5]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        try:
            db_stats = self.embedding_processor.get_database_stats()
            
            return {
                'database_stats': db_stats,
                'search_capabilities': {
                    'semantic_search': True,
                    'conversation_filtering': True,
                    'context_retrieval': True,
                    'similarity_threshold_range': [0.0, 1.0]
                },
                'vector_model': {
                    'name': self.embedding_processor.model_name,
                    'dimension': 384
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {'error': str(e)} 