#!/usr/bin/env python3
"""
File Management Utilities
Handles saving and loading processed data with consistent naming and formats
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

logger = logging.getLogger(__name__)

class DataFileManager:
    """Manages saving and loading of processed SeaTalk data"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_processing_results(self, 
                              processed_messages: List[Dict[str, Any]],
                              conversations: Dict[str, Dict[str, Any]],
                              embedding_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Save all processing results with timestamped filenames"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        files = {
            'processed_messages': self.output_dir / f"processed_messages_{timestamp}.json",
            'conversations': self.output_dir / f"conversations_{timestamp}.json", 
            'embedding_data': self.output_dir / f"embedding_data_{timestamp}.json",
            'embedding_csv': self.output_dir / f"embedding_data_{timestamp}.csv",
            'summary': self.output_dir / f"processing_summary_{timestamp}.md"
        }
        
        logger.info(f"💾 Saving processed data to {self.output_dir}")
        
        # Save processed messages
        self._save_json(files['processed_messages'], processed_messages)
        
        # Save conversations 
        self._save_json(files['conversations'], conversations)
        
        # Save embedding data (JSON and CSV)
        self._save_json(files['embedding_data'], embedding_data)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(embedding_data)
        df.to_csv(files['embedding_csv'], index=False, encoding='utf-8')
        
        # Generate summary
        self._generate_summary(files['summary'], processed_messages, conversations, embedding_data)
        
        logger.info(f"✓ Saved {len(files)} files successfully")
        return {k: str(v) for k, v in files.items()}
    
    def _save_json(self, filepath: Path, data: Any):
        """Save data as JSON with proper encoding"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def load_latest_embedding_data(self) -> List[Dict[str, Any]]:
        """Load the most recent embedding data file"""
        pattern = "embedding_data_*.json"
        files = sorted(self.output_dir.glob(pattern), reverse=True)
        
        if not files:
            raise FileNotFoundError("No embedding data files found")
        
        latest_file = files[0]
        logger.info(f"Loading embedding data from {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _generate_summary(self, 
                         filepath: Path,
                         processed_messages: List[Dict[str, Any]],
                         conversations: Dict[str, Dict[str, Any]], 
                         embedding_data: List[Dict[str, Any]]):
        """Generate a summary report of the processing results"""
        
        total_messages = len(processed_messages)
        text_messages = sum(1 for m in processed_messages if m['is_text_message'])
        content_messages = sum(1 for m in processed_messages if m['has_content'])
        embedding_ready = len(embedding_data)
        
        # Language analysis
        chinese_messages = sum(1 for m in embedding_data 
                             if any('\u4e00' <= char <= '\u9fff' for char in m['text_content']))
        english_messages = embedding_ready - chinese_messages
        
        # Conversation stats
        group_convs = sum(1 for c in conversations.values() if c['conversation_type'] == 'group')
        private_convs = len(conversations) - group_convs
        
        # Top conversations
        sorted_convs = sorted(conversations.values(), 
                            key=lambda x: x['content_messages'], reverse=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# SeaTalk Processing Summary\n\n")
            f.write(f"**Processing Date:** {datetime.now().isoformat()}\n\n")
            
            f.write("## Processing Results\n\n")
            f.write(f"- **Total Messages:** {total_messages:,}\n")
            f.write(f"- **Text Messages:** {text_messages:,} ({text_messages/total_messages*100:.1f}%)\n")
            f.write(f"- **Content Messages:** {content_messages:,} ({content_messages/total_messages*100:.1f}%)\n")
            f.write(f"- **Embedding Ready:** {embedding_ready:,} ({embedding_ready/total_messages*100:.1f}%)\n\n")
            
            f.write("## Content Analysis\n\n")
            f.write(f"- **Chinese Messages:** {chinese_messages:,} ({chinese_messages/embedding_ready*100:.1f}%)\n")
            f.write(f"- **English Messages:** {english_messages:,} ({english_messages/embedding_ready*100:.1f}%)\n\n")
            
            f.write("## Conversations\n\n")
            f.write(f"- **Total:** {len(conversations)}\n")
            f.write(f"- **Groups:** {group_convs}\n")
            f.write(f"- **Private:** {private_convs}\n\n")
            
            f.write("## Top 10 Active Conversations\n\n")
            for i, conv in enumerate(sorted_convs[:10], 1):
                conv_type = "🏢 Group" if conv['conversation_type'] == 'group' else "👤 Private"
                f.write(f"{i}. {conv_type} `{conv['session_id']}`: {conv['content_messages']:,} messages\n") 