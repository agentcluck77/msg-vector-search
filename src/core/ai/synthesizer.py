#!/usr/bin/env python3
"""
GPT Synthesis Module
Handles OpenAI GPT integration and search result synthesis
"""

import openai
import logging
import json
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from config import OPENAI_API_KEY, GPT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Result of GPT synthesis"""
    response: str
    search_results_used: int
    token_usage: Dict[str, int]
    conversation_sources: List[str]
    confidence_score: float


class SparkAISynthesizer:
    """
    GPT-4.1-nano integration for SeaTalk Spark Search
    Synthesizes search results into natural language responses
    """
    
    def __init__(self, api_key: str = None, model: str = None):
        """
        Initialize the synthesizer
        
        Args:
            api_key: OpenAI API key (defaults to config)
            model: GPT model to use (defaults to config)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or GPT_MODEL
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info(f"Initialized SparkAISynthesizer with model: {self.model}")
    
    def synthesize_search_results(
        self,
        query: str,
        search_results: List[Dict],
        max_tokens: int = 500,
        temperature: float = 0.3,
        include_citations: bool = True
    ) -> SynthesisResult:
        """
        Synthesize search results into a natural language response
        
        Args:
            query: Original user query
            search_results: List of search results from semantic search
            max_tokens: Maximum tokens for GPT response
            temperature: GPT temperature (0.0-1.0)
            include_citations: Whether to include conversation citations
            
        Returns:
            SynthesisResult with response and metadata
        """
        if not search_results:
            return SynthesisResult(
                response="I couldn't find any relevant conversations related to your query.",
                search_results_used=0,
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                conversation_sources=[],
                confidence_score=0.0
            )
        
        # Prepare context from search results
        context = self._prepare_context(search_results, include_citations)
        
        # Create system and user prompts
        system_prompt = self._create_system_prompt()
        user_prompt = self._create_user_prompt(query, context)
        
        try:
            # Call GPT API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Extract response and metadata
            synthesis_text = response.choices[0].message.content
            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            # Calculate confidence based on search result similarities
            confidence_score = self._calculate_confidence(search_results)
            
            # Extract unique conversation sources
            conversation_sources = list(set([
                result.get('conversation', {}).get('name', 'Unknown')
                for result in search_results[:10]  # Top 10 results
            ]))
            
            logger.info(f"GPT synthesis completed. Tokens used: {token_usage['total_tokens']}")
            
            return SynthesisResult(
                response=synthesis_text,
                search_results_used=len(search_results),
                token_usage=token_usage,
                conversation_sources=conversation_sources,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"GPT synthesis failed: {e}")
            return SynthesisResult(
                response=f"I encountered an error while processing your query: {str(e)}",
                search_results_used=len(search_results),
                token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                conversation_sources=[],
                confidence_score=0.0
            )
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for GPT"""
        return """You are SeaTalk Spark Search, an AI assistant that helps users find and understand information from their SeaTalk conversations.

Your role:
1. Analyze the provided search results from the user's message history
2. Provide a natural, conversational response that directly answers their query
3. Always include citations to specific conversations and timestamps when relevant
4. Synthesize information across multiple conversations when patterns emerge
5. If no relevant information is found, say so clearly and suggest alternative search terms

Guidelines:
- Be concise but comprehensive in your responses
- Use natural language, not technical jargon
- Focus on the most relevant and recent information
- When citing conversations, use format: "In your conversation with [Person/Group] on [Date]..."
- If information spans multiple conversations, summarize the key themes
- Maintain user privacy by not speculating beyond the provided data

Response style:
- Direct and helpful
- Conversational but professional
- Include specific quotes when they add value
- Highlight important decisions, action items, or conclusions"""
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create the user prompt with query and context"""
        return f"""
Query: "{query}"

Search Results from SeaTalk Conversations:
{context}

Based on this conversation history, please provide a comprehensive answer to the user's query. Include relevant citations and synthesize information across conversations when appropriate.
"""
    
    def _prepare_context(self, search_results: List[Dict], include_citations: bool = True) -> str:
        """
        Prepare context string from search results
        
        Args:
            search_results: List of search results
            include_citations: Whether to include detailed citations
            
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant conversations found."
        
        context_parts = []
        max_results = min(15, len(search_results))  # Limit to top 15 results
        
        for i, result in enumerate(search_results[:max_results]):
            # Extract result data
            message_text = result.get('text_content', '')
            sender = result.get('user_id', 'Unknown User')
            conversation = result.get('conversation', {})
            conv_name = conversation.get('name', 'Unknown Conversation')
            conv_type = conversation.get('type', 'unknown')
            timestamp = result.get('timestamp')
            similarity = result.get('similarity_score', 0.0)
            
            # Format timestamp
            if timestamp:
                try:
                    if isinstance(timestamp, (int, float)):
                        # Convert milliseconds to seconds if needed
                        if timestamp > 1e10:
                            timestamp = timestamp / 1000
                        time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
                    else:
                        time_str = str(timestamp)
                except:
                    time_str = "Unknown time"
            else:
                time_str = "Unknown time"
            
            # Create context entry
            if include_citations:
                context_parts.append(
                    f"[Result {i+1}] Similarity: {similarity:.3f}\n"
                    f"From: {sender} | Chat: {conv_name} ({conv_type}) | Time: {time_str}\n"
                    f"Message: {message_text}\n"
                )
            else:
                context_parts.append(f"- {message_text}")
        
        return "\n".join(context_parts)
    
    def _calculate_confidence(self, search_results: List[Dict]) -> float:
        """
        Calculate confidence score based on search result quality
        
        Args:
            search_results: List of search results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not search_results:
            return 0.0
        
        # Consider top 5 results for confidence calculation
        top_results = search_results[:5]
        similarities = [result.get('similarity_score', 0.0) for result in top_results]
        
        if not similarities:
            return 0.0
        
        # Calculate weighted average (higher weight for top results)
        weights = [0.4, 0.3, 0.2, 0.1, 0.05][:len(similarities)]
        weighted_similarity = sum(sim * weight for sim, weight in zip(similarities, weights))
        
        # Normalize to 0-1 range (assuming similarity scores are 0-1)
        confidence = min(1.0, max(0.0, weighted_similarity))
        
        return confidence
    
    def format_response_with_metadata(self, result: SynthesisResult) -> str:
        """
        Format synthesis result with metadata for display
        
        Args:
            result: SynthesisResult object
            
        Returns:
            Formatted string with response and metadata
        """
        lines = [
            "🤖 SeaTalk Spark AI Response",
            "═" * 50,
            "",
            result.response,
            "",
            "📊 Search Metadata:",
            f"• Results analyzed: {result.search_results_used}",
            f"• Confidence score: {result.confidence_score:.1%}",
            f"• Token usage: {result.token_usage['total_tokens']} tokens",
            ""
        ]
        
        if result.conversation_sources:
            lines.extend([
                "💬 Source conversations:",
                *[f"• {source}" for source in result.conversation_sources[:5]],
                ""
            ])
        
        return "\n".join(lines)


class SparkSearchOrchestrator:
    """
    Main orchestrator that combines semantic search with GPT synthesis
    Implements the complete Spark Search pipeline
    """
    
    def __init__(self, search_engine, synthesizer: SparkAISynthesizer = None):
        """
        Initialize the orchestrator
        
        Args:
            search_engine: Semantic search engine instance
            synthesizer: GPT synthesizer instance (optional)
        """
        self.search_engine = search_engine
        self.synthesizer = synthesizer or SparkAISynthesizer()
        logger.info("Initialized SparkSearchOrchestrator")
    
    def spark_search(
        self,
        query: str,
        search_limit: int = 15,
        similarity_threshold: float = 0.3,
        use_ai_synthesis: bool = True,
        max_tokens: int = 500,
        session_id: str = None
    ) -> Dict:
        """
        Perform complete Spark Search with AI synthesis
        
        Args:
            query: User search query
            search_limit: Maximum number of search results to analyze
            similarity_threshold: Minimum similarity threshold for search
            use_ai_synthesis: Whether to use GPT synthesis
            max_tokens: Maximum tokens for GPT response
            session_id: Optional session ID to filter by conversation
            
        Returns:
            Complete search and synthesis results
        """
        logger.info(f"Starting Spark Search for query: '{query}'")
        
        # Step 1: Semantic search
        try:
            search_results = self.search_engine.search(
                query=query,
                limit=search_limit,
                similarity_threshold=similarity_threshold,
                include_context=False,
                session_id=session_id
            )
            
            raw_results = search_results.get('results', [])
            logger.info(f"Found {len(raw_results)} semantic search results")
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return {
                'status': 'search_error',
                'error': str(e),
                'query': query,
                'results': []
            }
        
        # Step 2: GPT synthesis (if enabled and API key available)
        synthesis_result = None
        if use_ai_synthesis and self.synthesizer.api_key:
            try:
                synthesis_result = self.synthesizer.synthesize_search_results(
                    query=query,
                    search_results=raw_results,
                    max_tokens=max_tokens
                )
                logger.info("GPT synthesis completed successfully")
                
            except Exception as e:
                logger.error(f"GPT synthesis failed: {e}")
                synthesis_result = SynthesisResult(
                    response=f"Search completed but AI synthesis failed: {str(e)}",
                    search_results_used=len(raw_results),
                    token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    conversation_sources=[],
                    confidence_score=0.0
                )
        
        # Return comprehensive results
        result = {
            'status': 'success',
            'query': query,
            'semantic_search': {
                'results_count': len(raw_results),
                'results': raw_results,  # Return all results, let caller decide how many to display
                'search_time': search_results.get('search_time', 0)
            }
        }
        
        if synthesis_result:
            result['ai_synthesis'] = {
                'response': synthesis_result.response,
                'confidence_score': synthesis_result.confidence_score,
                'token_usage': synthesis_result.token_usage,
                'conversation_sources': synthesis_result.conversation_sources,
                'results_analyzed': synthesis_result.search_results_used
            }
        
        return result 