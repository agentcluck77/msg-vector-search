#!/usr/bin/env python3
"""
SeaTalk Message Processor Module
Extracts and processes messages from the SeaTalk database
"""

import json
import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from .connection import SeaTalkDatabase
from .user_mapper import UserMapper

logger = logging.getLogger(__name__)

class MessageProcessor:
    """Processes messages from SeaTalk database"""
    
    def __init__(self, database: SeaTalkDatabase):
        """
        Initialize the message processor
        
        Args:
            database: SeaTalkDatabase instance
        """
        self.database = database
        self.user_mapper = UserMapper(database)
        
    def get_last_processed_timestamp(self, vector_db=None) -> float:
        """
        Get the timestamp of the last processed message from vector database
        
        Args:
            vector_db: Vector database connection (optional)
        
        Returns:
            Timestamp as float, or 0.0 if no messages have been processed
        """
        try:
            # Use provided vector_db or try to use main database
            if vector_db:
                cursor = vector_db.cursor()
            else:
                cursor = self.database.get_cursor()
                
            cursor.execute("""
                SELECT value FROM vector_metadata
                WHERE key = 'last_processed_timestamp'
            """)
            result = cursor.fetchone()
            
            if result and result[0]:
                # Keep full precision as float
                return float(result[0])
            return 0.0
        except Exception as e:
            logger.warning(f"⚠️ Could not get last processed timestamp: {e}")
            return 0.0
            
    def update_last_processed_timestamp(self, timestamp: float, vector_db=None) -> None:
        """
        Update the timestamp of the last processed message in vector database
        
        Args:
            timestamp: Timestamp to store (float for precision)
            vector_db: Vector database connection (optional)
        """
        try:
            # Use provided vector_db or try to use main database
            if vector_db:
                cursor = vector_db.cursor()
            else:
                cursor = self.database.get_cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='vector_metadata'
            """)
            
            if not cursor.fetchone():
                # Create table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE vector_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT
                    )
                """)
            
            # Store timestamp with full precision
            cursor.execute("""
                INSERT OR REPLACE INTO vector_metadata (key, value)
                VALUES ('last_processed_timestamp', ?)
            """, (str(timestamp),))
            
            logger.info(f"✓ Updated last processed timestamp: {timestamp}")
        except Exception as e:
            logger.error(f"❌ Failed to update last processed timestamp: {e}")
    
    def extract_text_from_content(self, content: str, message_type: int) -> str:
        """
        Extract plain text from message content
        
        Args:
            content: Raw message content
            message_type: Message type code
            
        Returns:
            Extracted plain text
        """
        # Skip empty content
        if not content:
            return ""
            
        # Handle different message types
        if message_type == 1:  # Text message
            # Try to parse as JSON
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "text" in data:
                    return data["text"]
                elif isinstance(data, list):
                    # Concatenate text from all elements
                    text_parts = []
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            text_parts.append(item["text"])
                    return " ".join(text_parts)
            except json.JSONDecodeError:
                # Not JSON, return as is
                return content
                
        # Default: return content as is
        return content
        
    def clean_text_content(self, text: str) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not text:
            return ""
            
        # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
        
    def get_user_name(self, user_id: str) -> str:
        """
        Get user name from user ID
        
        Args:
            user_id: User ID
            
        Returns:
            User name or ID if not found
        """
        return self.user_mapper.get_user_name(user_id)
            
    def get_conversation_name(self, session_id: str) -> Tuple[str, str]:
        """
        Get conversation name and type from session ID
        
        Args:
            session_id: Session ID
            
        Returns:
            Tuple of (conversation_name, conversation_type)
        """
        # Determine conversation type
        conv_type = "group" if session_id.startswith("group-") else "private"
        conv_name = None
        
        try:
            cursor = self.database.get_cursor()
            
            if conv_type == "group":
                # Try to get group name from message content
                cursor.execute("""
                    SELECT c FROM chat_message 
                    WHERE sid = ? AND c LIKE '%"n":%' 
                    AND t IN ('c.g.c.i', 'c.g.a.m', 'system')
                    LIMIT 1
                """, (session_id,))
                
                row = cursor.fetchone()
                if row:
                    try:
                        content = json.loads(row[0])
                        if 'n' in content and isinstance(content['n'], str):
                            group_name = content['n'].strip()
                            if group_name and len(group_name) < 100:  # Reasonable group name length
                                conv_name = group_name
                    except Exception:
                        pass
                
                # If no name found, try to extract from group creation messages
                if not conv_name:
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
                                conv_name = content['gn'].strip()
                            elif 'group_name' in content:
                                conv_name = content['group_name'].strip()
                        except Exception:
                            pass
            else:
                # For private conversations, extract user ID and get user name
                if session_id.startswith("buddy-"):
                    user_id = session_id.replace("buddy-", "")
                    conv_name = self.get_user_name(user_id)
                else:
                    # Direct user ID
                    conv_name = self.get_user_name(session_id)
                
        except Exception as e:
            logger.debug(f"Failed to get conversation name for {session_id}: {e}")
            
        # Fallback to session ID if no name found
        if not conv_name:
            conv_name = session_id
            
        return (conv_name, conv_type)
        
    def get_message_context(self, message_id: str, session_id: str) -> Dict[str, str]:
        """
        Get previous and next messages for context
        
        Args:
            message_id: Current message ID
            session_id: Session ID
            
        Returns:
            Dictionary with previous and next message content
        """
        context = {
            "previous_message": "",
            "next_message": ""
        }
        
        try:
            cursor = self.database.get_cursor()
            
            # Get previous message
            cursor.execute("""
                SELECT c, _createAt FROM chat_message 
                WHERE sid = ? AND _createAt < (
                    SELECT _createAt FROM chat_message WHERE _mid = ?
                )
                ORDER BY _createAt DESC LIMIT 1
            """, (session_id, message_id))
            
            prev = cursor.fetchone()
            if prev:
                prev_content = self.extract_text_from_content(prev[0], 1)
                prev_content = self.clean_text_content(prev_content)
                context["previous_message"] = prev_content[:100] + "..." if len(prev_content) > 100 else prev_content
                
            # Get next message
            cursor.execute("""
                SELECT c, _createAt FROM chat_message 
                WHERE sid = ? AND _createAt > (
                    SELECT _createAt FROM chat_message WHERE _mid = ?
                )
                ORDER BY _createAt ASC LIMIT 1
            """, (session_id, message_id))
            
            next_msg = cursor.fetchone()
            if next_msg:
                next_content = self.extract_text_from_content(next_msg[0], 1)
                next_content = self.clean_text_content(next_content)
                context["next_message"] = next_content[:100] + "..." if len(next_content) > 100 else next_content
                
        except Exception as e:
            logger.debug(f"Could not get message context: {e}")
            
        return context
    
    def process_messages(self, since_timestamp: Optional[float] = None, vector_db=None) -> List[Dict[str, Any]]:
        """
        Process messages from the database
        
        Args:
            since_timestamp: Only process messages after this timestamp (float for precision)
            vector_db: Vector database connection for timestamp tracking
            
        Returns:
            List of processed messages
        """
        logger.info("🔄 Processing messages...")
        
        # Use provided timestamp or get last processed timestamp
        if since_timestamp is None:
            since_timestamp = self.get_last_processed_timestamp(vector_db)
            
        # Add safety check to prevent processing too many messages at once
        try:
            # Ensure database is connected
            if not self.database.conn:
                if not self.database.connect():
                    raise RuntimeError("Failed to connect to database")
                    
            cursor = self.database.get_cursor()
            
            # First, check how many messages we would process
            count_query = """
                SELECT COUNT(*) FROM chat_message 
                WHERE _createAt > ?
            """
            cursor.execute(count_query, (since_timestamp,))
            message_count = cursor.fetchone()[0]
            
            # Safety check: if more than 5000 messages, something might be wrong
            if message_count > 5000:
                logger.warning(f"⚠️ About to process {message_count:,} messages (since timestamp {since_timestamp})")
                logger.warning("⚠️ This seems excessive - there might be a timestamp issue")
                
                # Get the oldest message timestamp to compare
                cursor.execute("SELECT MIN(_createAt) FROM chat_message")
                oldest_timestamp = cursor.fetchone()[0]
                
                # Get the newest message timestamp
                cursor.execute("SELECT MAX(_createAt) FROM chat_message")
                newest_timestamp = cursor.fetchone()[0]
                
                logger.info(f"📊 Database timestamp range: {oldest_timestamp} to {newest_timestamp}")
                logger.info(f"📊 Last processed timestamp: {since_timestamp}")
                
                # If the since_timestamp is older than the oldest message, reset it
                if since_timestamp < oldest_timestamp:
                    logger.warning(f"⚠️ Last processed timestamp ({since_timestamp}) is older than oldest message ({oldest_timestamp})")
                    logger.warning("⚠️ Resetting to process only recent messages")
                    # Set to process only messages from the last 7 days
                    import time
                    since_timestamp = time.time() - (7 * 24 * 60 * 60)
                    
                    # Recount with new timestamp
                    cursor.execute(count_query, (since_timestamp,))
                    message_count = cursor.fetchone()[0]
                    logger.info(f"📊 After reset: will process {message_count:,} messages from last 7 days")
            
            # Query for messages after the timestamp
            query = """
                SELECT 
                    sid, _mid, c, t, _createAt, u
                FROM chat_message 
                WHERE _createAt > ?
                ORDER BY _createAt ASC
            """
            
            cursor.execute(query, (since_timestamp,))
            
            messages = []
            latest_timestamp = since_timestamp
            
            for row in cursor:
                session_id, message_id, content, msg_type, create_timestamp, user_id = row
                
                # Extract and clean text content
                text_content = self.extract_text_from_content(content, msg_type)
                clean_text = self.clean_text_content(text_content)
                
                # Skip empty messages
                if not clean_text:
                    continue
                    
                # Get user name
                user_name = self.get_user_name(user_id)
                
                # Get conversation info
                conv_name, conv_type = self.get_conversation_name(session_id)
                
                # Get message context
                context = self.get_message_context(message_id, session_id)
                
                # Format datetime
                dt = datetime.fromtimestamp(create_timestamp)
                human_time = dt.strftime("%b %d, %Y at %I:%M %p")
                
                # Update latest timestamp (keep as float for precision)
                if create_timestamp > latest_timestamp:
                    latest_timestamp = float(create_timestamp)
                
                # Build message object
                message = {
                    "message_id": message_id,
                    "session_id": session_id,
                    "user_id": user_id,
                    "user_name": user_name,
                    "text_content": clean_text,
                    "timestamp": create_timestamp,
                    "human_time": human_time,
                    "conversation": {
                        "type": conv_type,
                        "name": conv_name
                    },
                    "context": context
                }
                
                messages.append(message)
            
            # Update last processed timestamp
            if latest_timestamp > since_timestamp:
                self.update_last_processed_timestamp(latest_timestamp, vector_db)
            
            logger.info(f"✓ Processed {len(messages)} new messages")
            return messages
            
        except Exception as e:
            logger.error(f"❌ Failed to process messages: {e}")
            return [] 