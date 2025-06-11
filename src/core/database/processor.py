#!/usr/bin/env python3
"""
SeaTalk Message Processor
Clean, modular message processing and content extraction
"""

import json
import re
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from .connection import SeaTalkDatabase

logger = logging.getLogger(__name__)

class MessageProcessor:
    """Processes SeaTalk messages and extracts clean content"""
    
    def __init__(self, db_path: str, db_key: str = None):
        self.database = SeaTalkDatabase(db_path, db_key)
        self.processed_messages = []
        self.conversations = {}
    
    def extract_text_from_content(self, content: str, message_type: str) -> str:
        """Extract clean text from JSON content based on message type"""
        if not content:
            return ""
        
        try:
            # Parse JSON content
            if content.startswith('{'):
                content_obj = json.loads(content)
            else:
                return content  # Plain text
            
            # Extract text based on message type
            if message_type == "text":
                return content_obj.get('c', '')
            elif message_type == "image":
                return f"[Image: {content_obj.get('w', 0)}x{content_obj.get('h', 0)}]"
            elif message_type == "video":
                return f"[Video: {content_obj.get('s', 0)} bytes]"
            elif message_type == "file":
                filename = content_obj.get('name', 'unknown')
                return f"[File: {filename}]"
            elif message_type.startswith("sticker"):
                return "[Sticker]"
            elif message_type == "c.b.n":
                return "[Business Notification]"
            elif message_type == "c.g.m":
                return "[Group Management Action]"
            elif message_type == "history":
                return "[Chat History]"
            elif message_type.startswith("neocall"):
                return "[Voice/Video Call]"
            else:
                # Try to extract any text content
                if 'c' in content_obj:
                    return content_obj['c']
                return f"[{message_type}]"
                
        except json.JSONDecodeError:
            return content
        except Exception as e:
            logger.warning(f"Error extracting text from content: {e}")
            return ""
    
    def clean_text_content(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove control characters but keep Chinese characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def extract_quote_content(self, quote_json: str) -> Dict[str, Any]:
        """Extract content from quoted messages"""
        if not quote_json:
            return {}
        
        try:
            quote_obj = json.loads(quote_json)
            
            quote_info = {
                'quoted_message_id': quote_obj.get('mid', ''),
                'quoted_user_id': quote_obj.get('u', ''),
                'quoted_timestamp': quote_obj.get('sts', 0),
                'quoted_text': ''
            }
            
            # Extract quoted text content
            if 'c' in quote_obj and isinstance(quote_obj['c'], dict):
                quote_info['quoted_text'] = self.clean_text_content(
                    quote_obj['c'].get('c', '')
                )
            
            return quote_info
            
        except Exception as e:
            logger.warning(f"Error parsing quote content: {e}")
            return {}
    
    def process_all_messages(self) -> List[Dict[str, Any]]:
        """Process all messages from the database"""
        logger.info("🔄 Processing all messages...")
        
        with self.database:
            cursor = self.database.get_cursor()
            
            # Extract all messages with comprehensive data
            cursor.execute("""
                SELECT 
                    sid, _mid, c, q, t, cid, cts, ts, u, _createAt, 
                    rmid, _quoteMid, rtmid, smid, o, pts, w, ec, 
                    _attribute, dts
                FROM chat_message 
                ORDER BY _createAt ASC
            """)
            
            messages = []
            for i, row in enumerate(cursor):
                # Build comprehensive message object
                message = {
                    # Core identifiers
                    'session_id': row[0],
                    'message_id': row[1],
                    'conversation_id': row[5],
                    'user_id': row[8],
                    
                    # Content processing
                    'raw_content': row[2],
                    'quote_content': row[3],
                    'message_type': row[4],
                    
                    # Timestamps
                    'client_timestamp': row[6],
                    'server_timestamp': row[7],
                    'create_timestamp': row[9],
                    'create_datetime': datetime.fromtimestamp(row[9]) if row[9] else None,
                    
                    # Threading and replies
                    'reply_message_id': row[10],
                    'quote_message_id': row[11],
                    'thread_message_id': row[12],
                    'sequence_message_id': row[13],
                    
                    # Additional metadata
                    'orientation': row[14],
                    'priority_timestamp': row[15],
                    'weight': row[16],
                    'error_code': row[17],
                    'attributes': row[18],
                    'delete_timestamp': row[19],
                    
                    # Processed content
                    'clean_text': '',
                    'quote_info': {},
                    'is_text_message': False,
                    'has_content': False,
                    'conversation_type': 'group' if row[0].startswith('group-') else 'private'
                }
                
                # Extract and clean text content
                message['clean_text'] = self.clean_text_content(
                    self.extract_text_from_content(message['raw_content'], message['message_type'])
                )
                
                # Process quote information
                if message['quote_content']:
                    message['quote_info'] = self.extract_quote_content(message['quote_content'])
                
                # Set content flags
                message['is_text_message'] = message['message_type'] == 'text'
                message['has_content'] = bool(message['clean_text'] and 
                                            not message['clean_text'].startswith('[') and 
                                            len(message['clean_text']) > 1)
                
                messages.append(message)
                
                # Progress logging
                if (i + 1) % 1000 == 0:
                    logger.info(f"  Processed {i + 1:,} messages...")
            
            logger.info(f"✓ Processed {len(messages):,} total messages")
            self.processed_messages = messages
            return messages
    
    def build_conversation_context(self) -> Dict[str, Dict[str, Any]]:
        """Build conversation context and statistics"""
        logger.info("🏗️ Building conversation context...")
        
        conversations = {}
        
        for message in self.processed_messages:
            session_id = message['session_id']
            
            if session_id not in conversations:
                conversations[session_id] = {
                    'session_id': session_id,
                    'conversation_type': message['conversation_type'],
                    'total_messages': 0,
                    'text_messages': 0,
                    'content_messages': 0,
                    'first_message_time': None,
                    'last_message_time': None,
                    'participants': set(),
                    'message_types': {},
                    'messages': []
                }
            
            conv = conversations[session_id]
            
            # Update statistics
            conv['total_messages'] += 1
            conv['participants'].add(message['user_id'])
            
            if message['is_text_message']:
                conv['text_messages'] += 1
            
            if message['has_content']:
                conv['content_messages'] += 1
            
            # Track message types
            msg_type = message['message_type']
            conv['message_types'][msg_type] = conv['message_types'].get(msg_type, 0) + 1
            
            # Update time bounds
            msg_time = message['create_timestamp']
            if msg_time:
                if conv['first_message_time'] is None or msg_time < conv['first_message_time']:
                    conv['first_message_time'] = msg_time
                if conv['last_message_time'] is None or msg_time > conv['last_message_time']:
                    conv['last_message_time'] = msg_time
            
            conv['messages'].append(message)
        
        # Convert sets to lists for JSON serialization
        for conv in conversations.values():
            conv['participants'] = list(conv['participants'])
            conv['participant_count'] = len(conv['participants'])
        
        logger.info(f"✓ Built context for {len(conversations)} conversations")
        self.conversations = conversations
        return conversations
    
    def prepare_embedding_data(self) -> List[Dict[str, Any]]:
        """Prepare clean data for vector embedding generation"""
        logger.info("📝 Preparing data for embedding generation...")
        
        embedding_data = []
        
        for message in self.processed_messages:
            # Only include messages with meaningful text content
            if not message['has_content']:
                continue
            
            # Build context string for embedding
            context_parts = []
            
            # Add conversation type context
            if message['conversation_type'] == 'group':
                context_parts.append(f"Group conversation {message['session_id']}")
            else:
                context_parts.append(f"Private conversation {message['session_id']}")
            
            # Add quote context if available
            if message['quote_info'] and message['quote_info'].get('quoted_text'):
                context_parts.append(f"Replying to: {message['quote_info']['quoted_text'][:100]}")
            
            # Build embedding record
            embedding_record = {
                # Core identifiers
                'message_id': message['message_id'],
                'session_id': message['session_id'],
                'user_id': message['user_id'],
                'conversation_type': message['conversation_type'],
                
                # Content for embedding
                'text_content': message['clean_text'],
                'context_info': ' | '.join(context_parts),
                'full_content': f"{' | '.join(context_parts)} | {message['clean_text']}",
                
                # Metadata
                'timestamp': message['create_timestamp'],
                'datetime': message['create_datetime'].isoformat() if message['create_datetime'] else None,
                'message_type': message['message_type'],
                'has_quote': bool(message['quote_info']),
                'quote_text': message['quote_info'].get('quoted_text', '') if message['quote_info'] else '',
                
                # Conversation stats
                'conversation_total_messages': self.conversations[message['session_id']]['total_messages'],
                'conversation_participants': self.conversations[message['session_id']]['participant_count']
            }
            
            embedding_data.append(embedding_record)
        
        logger.info(f"✓ Prepared {len(embedding_data):,} messages for embedding")
        return embedding_data 