#!/bin/bash
# Setup script for SeaTalk Search MCP Server
# This pre-installs dependencies, pre-warms models, and initializes the database

echo "🔄 Setting up SeaTalk Search MCP Server..."

# Check if uvx is installed
if ! command -v uvx &> /dev/null; then
    echo "❌ Error: uvx is not installed. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📦 Step 1/4: Pre-installing dependencies with uvx..."
echo "   This will download PyTorch and other ML dependencies (~200MB)"
echo "   This may take a few minutes..."

# Install the package with uvx to cache all dependencies
uvx --from "$SCRIPT_DIR" python -c "
import sys
sys.path.insert(0, 'src')

print('✅ Installing core dependencies...')

try:
    import numpy as np
    numpy_version = np.__version__
    print(f'   📦 NumPy version: {numpy_version}')
    
    # Check for NumPy 2.x compatibility issues
    if numpy_version.startswith('2.'):
        print('   ⚠️  NumPy 2.x detected - checking PyTorch compatibility...')
    
    import torch
    print(f'   📦 PyTorch version: {torch.__version__}')
    
    import sentence_transformers
    print(f'   📦 sentence-transformers loaded successfully')
    
    import apsw
    print('   📦 APSW-SQLite3MC loaded successfully')
    
    print('✅ Core dependencies installed successfully!')
    
except ImportError as e:
    print(f'❌ Dependency import error: {e}')
    print('   Try updating your dependencies or check for compatibility issues')
    sys.exit(1)
except Exception as e:
    print(f'❌ Unexpected error loading dependencies: {e}')
    if 'numpy' in str(e).lower():
        print('   This may be a NumPy compatibility issue.')
        print('   Try: pip install \"numpy<2\" torch --upgrade')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🤖 Step 2/4: Pre-warming sentence transformer model..."

uvx --from "$SCRIPT_DIR" python -c "
import sys
sys.path.insert(0, 'src')

print('   📥 Downloading model: all-MiniLM-L6-v2...')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

print('   🧪 Testing model encoding...')
test_embeddings = model.encode(['test message'])
print(f'   ✅ Model loaded! Embedding dimension: {test_embeddings.shape[1]}')
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to pre-warm model"
    exit 1
fi

echo ""
echo "🗄️  Step 3/4: Testing MCP server imports..."

uvx --from "$SCRIPT_DIR" python -c "
import sys
sys.path.insert(0, 'src')
import os

# Suppress APSW version output
import io
import contextlib

# Capture stdout to suppress version messages
captured_output = io.StringIO()
with contextlib.redirect_stdout(captured_output):
    import core.search
    import core.database  
    import core.embeddings

print('✅ All MCP server imports successful!')
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to import MCP server modules"
    exit 1
fi

echo ""
echo "🚀 Step 4/4: Initializing database and embeddings..."
echo "   This will connect to your SeaTalk database and create initial embeddings"
echo "   Progress will be shown below:"

# Check if environment variables are set for initialization
if [ -z "$SEATALK_DB_PATH" ] || [ -z "$SEATALK_DB_KEY" ]; then
    echo ""
    echo "⚠️  Database environment variables not set - skipping initialization"
    echo "   To complete full initialization, set these environment variables and run:"
    echo "   SEATALK_DB_PATH='/path/to/SeaTalk' SEATALK_DB_KEY='your-key' ./setup.sh"
    echo ""
    echo "✅ Setup complete! Dependencies and models are cached."
    echo "   First MCP server run will initialize the database (may take 30-60s)"
else
    # Run full initialization with progress
    uvx --from "$SCRIPT_DIR" python -c "
import sys
import os
import time
sys.path.insert(0, 'src')

# Lightweight progress bar implementation
class ProgressBar:
    def __init__(self, total, width=50):
        self.total = total
        self.width = width
        self.current = 0
        
    def update(self, current, message=''):
        self.current = current
        if self.total > 0:
            percent = min(100, int(100 * current / self.total))
            filled = int(self.width * current / self.total)
        else:
            percent = 0
            filled = 0
            
        bar = '█' * filled + '░' * (self.width - filled)
        print(f'\r   [{bar}] {percent:3d}% {message}', end='', flush=True)
        
    def finish(self, message='Complete'):
        self.update(self.total, message)
        print()

print('   🔗 Connecting to SeaTalk database...')
progress = ProgressBar(100)
progress.update(10, 'Initializing...')

try:
    from core.search import SemanticSearchEngine
    progress.update(20, 'Database connected')
    
    # Initialize search engine
    engine = SemanticSearchEngine()
    progress.update(30, 'Search engine created')
    
    # Get database stats to show what we're working with
    stats = engine.get_database_stats()
    if stats.get('status') == 'success':
        total_messages = stats.get('total_messages', 0)
        embedded_messages = stats.get('embedded_messages', 0)
        
        progress.update(40, f'Found {total_messages:,} messages')
        
        if embedded_messages == 0:
            print()
            print(f'   📊 Database has {total_messages:,} messages, creating embeddings...')
            
            # Update embeddings with progress tracking
            print('   ⚡ Processing messages (this may take several minutes)...')
            
            progress.update(50, 'Starting embedding process...')
            
            # Show that we're actively processing
            import threading
            import time
            
            # Flag to control the progress animation
            processing_complete = False
            
            def animate_progress():
                dots = 0
                while not processing_complete:
                    dots = (dots + 1) % 4
                    dot_str = '.' * dots + ' ' * (3 - dots)
                    progress.update(50 + (dots * 5), f'Processing embeddings{dot_str}')
                    time.sleep(1)
            
            # Start animation thread
            animation_thread = threading.Thread(target=animate_progress, daemon=True)
            animation_thread.start()
            
            # Run the actual embedding process
            update_result = engine.update_embeddings()
            
            # Stop animation
            processing_complete = True
            animation_thread.join(timeout=0.1)
            
            new_messages = update_result.get('new_messages', 0)
            processing_time = update_result.get('processing_time_seconds', 0)
            
            progress.update(90, f'Processed {new_messages:,} messages')
            progress.finish(f'Complete! ({processing_time:.1f}s)')
            
            print(f'   ✅ Successfully embedded {new_messages:,} messages')
            print(f'   ⏱️  Processing time: {processing_time:.1f} seconds')
            
        else:
            coverage = (embedded_messages / total_messages * 100) if total_messages > 0 else 0
            remaining_messages = total_messages - embedded_messages
            
            if remaining_messages > 0:
                print()
                print(f'   📊 Database has {embedded_messages:,}/{total_messages:,} messages embedded ({coverage:.1f}% coverage)')
                print(f'   ⚡ Processing remaining {remaining_messages:,} messages...')
                
                progress.update(60, 'Processing remaining messages...')
                
                # Show that we're actively processing
                import threading
                import time
                
                # Flag to control the progress animation
                processing_complete = False
                
                def animate_progress():
                    dots = 0
                    while not processing_complete:
                        dots = (dots + 1) % 4
                        dot_str = '.' * dots + ' ' * (3 - dots)
                        progress.update(60 + (dots * 5), f'Processing remaining messages{dot_str}')
                        time.sleep(1)
                
                # Start animation thread
                animation_thread = threading.Thread(target=animate_progress, daemon=True)
                animation_thread.start()
                
                # Run the actual embedding process
                update_result = engine.update_embeddings()
                
                # Stop animation
                processing_complete = True
                animation_thread.join(timeout=0.1)
                
                new_messages = update_result.get('new_messages', 0)
                processing_time = update_result.get('processing_time_seconds', 0)
                
                progress.update(90, f'Processed {new_messages:,} additional messages')
                progress.finish(f'Complete! ({processing_time:.1f}s)')
                
                print(f'   ✅ Successfully embedded {new_messages:,} additional messages')
                print(f'   ⏱️  Processing time: {processing_time:.1f} seconds')
            else:
                progress.update(90, f'{embedded_messages:,} already embedded')
                progress.finish(f'Database ready ({coverage:.1f}% coverage)')
                
                print(f'   ✅ Database already fully initialized ({embedded_messages:,}/{total_messages:,} messages)')
    else:
        progress.finish('Error')
        print(f'   ❌ Database error: {stats.get(\"error\", \"Unknown error\")}')
        sys.exit(1)
        
    # Test a quick search to verify everything works
    print('   🔍 Testing search functionality...')
    test_result = engine.search('test', limit=1)
    if test_result.get('status') == 'success':
        print('   ✅ Search test successful!')
    else:
        print('   ❌ Search test failed')
        sys.exit(1)
        
except Exception as e:
    progress.finish('Failed')
    print(f'   ❌ Initialization error: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 Complete setup finished! Database fully initialized."
    else
        echo ""
        echo "❌ Database initialization failed"
        exit 1
    fi
fi

echo ""
echo "📋 Next steps:"
echo "1. The MCP server is ready for instant use with Claude Desktop"
echo "2. Make sure your Claude Desktop config points to:"
echo "   Command: /Users/$(whoami)/.local/bin/uvx"
echo "   Args: [\"--from\", \"$SCRIPT_DIR\", \"seatalk-search-server\"]"
echo ""
echo "⚡ First MCP call should now be instant (no initialization delay)"
echo "🔍 You can now search your SeaTalk messages in Claude!" 