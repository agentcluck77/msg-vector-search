#!/usr/bin/env python3
"""
User Name Mapper
Extracts and caches user display names from SeaTalk database
"""

import json
import logging
import re
import time
from typing import Dict, Optional, Set
from .connection import SeaTalkDatabase
from pathlib import Path

logger = logging.getLogger(__name__)

class UserMapper:
    """Maps user IDs to display names from SeaTalk database"""
    
    def __init__(self, database: SeaTalkDatabase):
        self.database = database
        self._user_cache: Dict[int, str] = {}
        self._cache_loaded = False
        self._re_email_pattern = re.compile(r'Email: (\w+)\.(\w+)@shopee\.com')
        self._re_user_id_pattern = re.compile(r'User ID: (\d+)')
    
    def _load_user_names(self) -> None:
        """Load user names from cache file or database"""
        if self._cache_loaded:
            return
        
        # Try to load from cache file first
        cache_file = Path(__file__).parent.parent.parent.parent / "data" / "user_cache.json"
        if cache_file.exists():
            logger.info(f"🔄 Loading user names from cache file: {cache_file}")
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # Convert string keys back to int
                self._user_cache = {int(uid): name for uid, name in cache_data.items()}
                self._cache_loaded = True
                logger.info(f"✓ Loaded {len(self._user_cache)} user mappings from cache")
                
                # Check if cache is recent (less than 24 hours old)
                cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if cache_age_hours < 24:
                    logger.info(f"✓ Cache is recent ({cache_age_hours:.1f}h old), skipping database rebuild")
                    return
                else:
                    logger.info(f"⚠️ Cache is old ({cache_age_hours:.1f}h), will refresh if needed")
                
            except Exception as e:
                logger.warning(f"Failed to load cache file: {e}, falling back to database")
        
        # Only do expensive database extraction if cache is missing or very old
        if not self._user_cache or len(self._user_cache) < 100:
            logger.info("🔄 Loading user names from database with enhanced extraction...")
            self._extract_user_names_from_database()
        else:
            logger.info("✓ Using existing cache, skipping expensive database rebuild")
        
        self._cache_loaded = True
        logger.info(f"✓ Loaded {len(self._user_cache)} user mappings")
    
    def _extract_user_names_from_database(self) -> None:
        """Optimized comprehensive user name extraction from database"""
        # Ensure database is connected
        if not self.database.conn:
            if not self.database.connect():
                logger.error("Failed to connect to database for user name extraction")
                return
                
        cursor = self.database.get_cursor()
        
        logger.info("🔍 Method 1: Extracting from user info messages...")
        self._extract_from_user_info_messages(cursor)
        
        logger.info("🔍 Method 2: Extracting from debug logs...")
        self._extract_from_debug_content_messages(cursor)
        
        logger.info("🔍 Method 3: Deep search for remaining users...")
        self._deep_search_remaining_users(cursor)
        
        # Add fallback names for users without names
        logger.info("🔄 Adding fallback names for remaining users...")
        self._add_fallback_names(cursor)
        
        # Save the improved cache
        self._save_cache_to_file()
    
    def _extract_from_user_info_messages(self, cursor) -> None:
        """Optimized extraction from all messages containing user info"""
        # Single optimized query that covers history, group info, and message content
        cursor.execute("""
            SELECT c FROM chat_message 
            WHERE c LIKE '%"uid":%' 
            AND c LIKE '%"n":%'
            AND t IN ('history', 'c.g.c.i', 'c.g.a.m', 'c.g.r.m', 'text', 'system')
        """)
        
        count = 0
        for row in cursor:
            try:
                content = json.loads(row[0])
                count += self._extract_users_from_content(content)
            except Exception:
                continue
        
        logger.info(f"   ✓ Found {count} user names from user info messages")
    
    def _extract_users_from_content(self, content: dict) -> int:
        """Optimized extraction of users from message content"""
        count = 0
        
        # Handle user arrays (most common case)
        if 'u' in content and isinstance(content['u'], list):
            for user in content['u']:
                if (isinstance(user, dict) and 
                    'uid' in user and 'n' in user and 
                    user['uid'] not in self._user_cache):
                    
                    name = user['n'].strip()
                    if name and not self._is_file_name(name):
                        self._user_cache[user['uid']] = name
                        count += 1
        
        # Handle direct user name in content (less common)
        elif 'u' in content and 'n' in content and isinstance(content['u'], int):
            user_id = content['u']
            if user_id not in self._user_cache:
                name = content['n'].strip()
                if name and not self._is_file_name(name):
                    self._user_cache[user_id] = name
                    count += 1
        
        return count
    
    def _extract_from_debug_content_messages(self, cursor) -> None:
        """Optimized extraction from debug logs"""
        cursor.execute("""
            SELECT c FROM chat_message 
            WHERE c LIKE '%User ID:%' 
            AND c LIKE '%Email:%shopee.com%'
        """)
        
        count = 0
        for row in cursor:
            try:
                content_str = row[0]
                count += self._extract_from_debug_content(content_str)
            except Exception:
                continue
        
        logger.info(f"   ✓ Found {count} additional user names from debug logs")
    
    def _extract_from_debug_content(self, content_str: str) -> int:
        """Extract user names from debug log content"""
        count = 0
        
        # Find user ID in the content
        user_id_match = self._re_user_id_pattern.search(content_str)
        if user_id_match:
            user_id = int(user_id_match.group(1))
            
            # Look for email in the same content
            email_match = self._re_email_pattern.search(content_str)
            if email_match and user_id not in self._user_cache:
                first_name, last_name = email_match.groups()
                display_name = f"{first_name.capitalize()} {last_name.capitalize()}"
                self._user_cache[user_id] = display_name
                count += 1
                logger.info(f"      ✅ Found name from debug log: {user_id} -> {display_name}")
        
        return count
    
    def _deep_search_remaining_users(self, cursor) -> None:
        """Deep search for users that still don't have names"""
        # Get all user IDs that don't have names yet
        cursor.execute("SELECT DISTINCT u FROM chat_message WHERE u IS NOT NULL")
        all_users = {row[0] for row in cursor}
        unnamed_users = all_users - set(self._user_cache.keys())
        
        if not unnamed_users:
            logger.info("   ✓ All users already have names")
            return
        
        logger.info(f"   🔍 Deep searching for {len(unnamed_users)} unnamed users...")
        
        # Batch search for multiple users at once
        user_ids_str = ','.join(map(str, list(unnamed_users)[:50]))  # Limit to 50 at a time
        
        cursor.execute(f"""
            SELECT c FROM chat_message 
            WHERE (u IN ({user_ids_str}) OR c LIKE '%"uid":' || u || ',%')
            AND c LIKE '%"n":%'
            LIMIT 500
        """)
        
        count = 0
        for row in cursor:
            try:
                content = json.loads(row[0])
                count += self._extract_users_from_content(content)
            except Exception:
                continue
        
        logger.info(f"   ✓ Found {count} additional user names from deep search")
    
    def _add_fallback_names(self, cursor) -> None:
        """Add fallback names for users without names"""
        cursor.execute("SELECT DISTINCT u FROM chat_message WHERE u IS NOT NULL")
        fallback_count = 0
        for row in cursor:
            user_id = row[0]
            if user_id not in self._user_cache:
                self._user_cache[user_id] = f"User {user_id}"
                fallback_count += 1
        
        logger.info(f"   ✓ Added {fallback_count} fallback names")
    
    def _save_cache_to_file(self) -> None:
        """Save the improved cache to file"""
        cache_file = Path(__file__).parent.parent.parent.parent / "data" / "user_cache.json"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert int keys to strings for JSON
        cache_data = {str(uid): name for uid, name in self._user_cache.items()}
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        logger.info(f"💾 Saved improved user cache to {cache_file}")
    
    def _is_file_name(self, name: str) -> bool:
        """Optimized check if a name is likely a file name rather than a user name"""
        if not name or len(name.strip()) < 2:
            return True
        
        name_lower = name.lower()
        
        # Quick checks first (most common cases)
        if (name_lower.startswith('user ') and name_lower[5:].isdigit() or
            len(name) > 100 or
            any(ext in name_lower for ext in ['.pdf', '.mov', '.mp4', '.jpg', '.png', '.doc', '.txt', '.zip']) or
            any(pattern in name_lower for pattern in ['screen recording', 'screenshot', 'debug log']) or
            '/' in name or '\\' in name or name.startswith(('http', 'www'))):
            return True
        
        return False
    
    def get_user_name(self, user_id) -> str:
        """Get display name for a user ID (accepts int or str)"""
        if not self._cache_loaded:
            self._load_user_names()
        
        # Convert to int if it's a string
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            user_id_int = user_id
        
        return self._user_cache.get(user_id_int, f"User {user_id}")
    
    def get_all_users(self) -> Dict[int, str]:
        """Get all user mappings"""
        if not self._cache_loaded:
            self._load_user_names()
        
        return self._user_cache.copy()
    
    def refresh_cache(self) -> None:
        """Force refresh of the user cache"""
        self._cache_loaded = False
        self._user_cache.clear()
        self._load_user_names()
    
    def force_rebuild_cache(self) -> None:
        """Force complete rebuild of user cache from database"""
        logger.info("🔄 Force rebuilding user cache from database...")
        
        # Remove existing cache file
        cache_file = Path(__file__).parent.parent.parent.parent / "data" / "user_cache.json"
        if cache_file.exists():
            cache_file.unlink()
        
        # Clear memory cache
        self._cache_loaded = False
        self._user_cache.clear()
        
        # Rebuild from database
        self._load_user_names()
        
        logger.info("✓ User cache rebuilt successfully") 