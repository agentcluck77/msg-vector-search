#!/bin/bash
# Setup script for SeaTalk Search MCP Server
# This pre-installs dependencies, pre-warms models, and initializes the database

# Default values
MAX_MESSAGES=10000
BATCH_SIZE=1000
DEBUG=false
DRY_RUN=false
SKIP_INIT=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-messages)
            MAX_MESSAGES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-init)
            SKIP_INIT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --max-messages N    Maximum messages to process (default: 10000)"
            echo "  --batch-size N      Batch size for processing (default: 1000)"
            echo "  --debug            Enable debug logging"
            echo "  --dry-run          Show what would be processed without doing it"
            echo "  --skip-init        Skip database initialization (deps only)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🔄 Setting up SeaTalk Search MCP Server..."
if [ "$DEBUG" = true ]; then
    echo "🐛 Debug mode enabled"
    echo "📊 Max messages: $MAX_MESSAGES"
    echo "📊 Batch size: $BATCH_SIZE"
fi

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
if [ "$SKIP_INIT" = true ]; then
    echo ""
    echo "⚠️  Skipping database initialization (--skip-init flag)"
    echo "   To initialize later, run without --skip-init flag"
    echo ""
    echo "✅ Setup complete! Dependencies and models are cached."
    echo "   First MCP server run will initialize the database"
elif [ -z "$SEATALK_DB_PATH" ] || [ -z "$SEATALK_DB_KEY" ]; then
    echo ""
    echo "⚠️  Database environment variables not set - skipping initialization"
    echo "   To complete full initialization, set these environment variables and run:"
    echo "   SEATALK_DB_PATH='/path/to/SeaTalk' SEATALK_DB_KEY='your-key' ./setup.sh"
    echo ""
    echo "✅ Setup complete! Dependencies and models are cached."
    echo "   First MCP server run will initialize the database (may take 30-60s)"
else
    # Run full initialization with progress
    LOG_FILE=""
    if [ "$DEBUG" = true ]; then
        LOG_FILE="/tmp/seatalk_setup_$(date +%Y%m%d_%H%M%S).log"
        echo "🐛 Debug logging to: $LOG_FILE"
    fi
    
    uvx --from "$SCRIPT_DIR" python -c "
import sys
import os
import time
import logging
sys.path.insert(0, 'src')

# Setup logging if debug mode
if '$DEBUG' == 'true':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('$LOG_FILE'),
            logging.StreamHandler()
        ]
    )
    print('🐛 Debug logging enabled')
else:
    logging.basicConfig(level=logging.INFO)

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
        
        # Show processing estimate
        remaining_messages = total_messages - embedded_messages
        if remaining_messages > 0:
            print()
            print(f'   📊 Database Analysis:')
            print(f'   📊 Total messages: {total_messages:,}')
            print(f'   📊 Already embedded: {embedded_messages:,}')
            print(f'   📊 Remaining to process: {remaining_messages:,}')
            print()
            
            # Automatically adjust parameters based on database size
            auto_max_messages = $MAX_MESSAGES
            auto_batch_size = $BATCH_SIZE
            processing_mode = 'normal'
            
            if remaining_messages > 50000:
                # Very large database - use conservative settings
                processing_mode = 'large'
                auto_max_messages = 5000
                auto_batch_size = 250
                print(f'   🔍 LARGE DATABASE DETECTED ({remaining_messages:,} messages)')
                print(f'   🔍 Automatically switching to SAFE MODE:')
                print(f'   🔍 • Will process {auto_max_messages:,} messages this run')
                print(f'   🔍 • Using smaller batches ({auto_batch_size} messages per batch)')
                print(f'   🔍 • You can run setup.sh multiple times to process more')
                print()
            elif remaining_messages > 20000:
                # Medium database - use moderate settings
                processing_mode = 'medium'
                auto_max_messages = 10000
                auto_batch_size = 500
                print(f'   📊 MEDIUM DATABASE DETECTED ({remaining_messages:,} messages)')
                print(f'   📊 Using optimized settings:')
                print(f'   📊 • Will process {auto_max_messages:,} messages this run')
                print(f'   📊 • Using batch size: {auto_batch_size}')
                print()
            else:
                # Small database - use standard settings
                print(f'   ⚡ SMALL DATABASE - using standard processing')
                print()
            
            # Calculate and show estimates
            messages_to_process = min(remaining_messages, auto_max_messages)
            estimated_batches = (messages_to_process + auto_batch_size - 1) // auto_batch_size
            estimated_time = estimated_batches * (15 if processing_mode == 'large' else 10)
            
            print(f'   ⏱️  This run will process: {messages_to_process:,} messages')
            print(f'   ⏱️  Estimated time: {estimated_time // 60}m {estimated_time % 60}s')
            
            if remaining_messages > auto_max_messages:
                remaining_after = remaining_messages - auto_max_messages
                additional_runs = (remaining_after + auto_max_messages - 1) // auto_max_messages
                print(f'   ⏱️  Additional runs needed: {additional_runs} (for remaining {remaining_after:,} messages)')
            
            print()
            
            # Auto-enable debug logging for large databases
            if processing_mode == 'large':
                import logging
                import tempfile
                import datetime
                
                log_file = f'/tmp/seatalk_setup_{datetime.datetime.now().strftime(\"%Y%m%d_%H%M%S\")}.log'
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ]
                )
                print(f'   🐛 Debug logging enabled automatically: {log_file}')
                print()
            
            # Wait for user confirmation for large databases
            if processing_mode == 'large':
                print('   ⚠️  LARGE DATABASE - Please confirm:')
                print('   ⚠️  This will take several minutes and process 5,000 messages.')
                print('   ⚠️  You can stop anytime with Ctrl+C and resume later.')
                print('   ⚠️  The system will automatically save progress as it goes.')
                print()
                
                # Give user 10 seconds to cancel
                import time
                for i in range(10, 0, -1):
                    print(f'   ⏳ Starting in {i} seconds... (Press Ctrl+C to cancel)', end='\r')
                    time.sleep(1)
                print('   ⏳ Starting processing...                                     ')
                print()
                
                # Add signal handler for graceful shutdown
                import signal
                def signal_handler(signum, frame):
                    print('\\n   ⚠️  Stopping processing... (progress is saved)')
                    print('   ⚠️  You can resume by running ./setup.sh again')
                    sys.exit(0)
                signal.signal(signal.SIGINT, signal_handler)
            
            # Update the parameters that will be used
            max_messages_to_use = auto_max_messages
            batch_size_to_use = auto_batch_size
            
            # Check if this is a dry run
            if '$DRY_RUN' == 'true':
                print(f'   🔍 DRY RUN: Would process {messages_to_process:,} messages')
                print(f'   🔍 Batch size: {batch_size_to_use} messages per batch')
                print(f'   🔍 Estimated processing time: {estimated_time // 60}m {estimated_time % 60}s')
                print(f'   🔍 Use without --dry-run to actually process')
                sys.exit(0)
        else:
            # No new messages to process
            max_messages_to_use = $MAX_MESSAGES
            batch_size_to_use = $BATCH_SIZE
        
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
            
            # Run the actual embedding process with automatically determined parameters
            update_result = engine.update_embeddings(batch_size=batch_size_to_use, max_messages=max_messages_to_use)
            
            # Stop animation
            processing_complete = True
            animation_thread.join(timeout=0.1)
            
            new_messages = update_result.get('new_messages', 0)
            processing_time = update_result.get('processing_time_seconds', 0)
            
            progress.update(90, f'Processed {new_messages:,} messages')
            progress.finish(f'Complete! ({processing_time:.1f}s)')
            
            print(f'   ✅ Successfully embedded {new_messages:,} messages')
            print(f'   ⏱️  Processing time: {processing_time:.1f} seconds')
            
            # Show completion status for large databases
            if processing_mode == 'large' and remaining_messages > max_messages_to_use:
                processed_so_far = embedded_messages + new_messages
                total_completion = (processed_so_far / total_messages) * 100
                print(f'   📊 Database completion: {total_completion:.1f}% ({processed_so_far:,}/{total_messages:,} messages)')
                print(f'   📊 Run ./setup.sh again to process more messages')
                print(f'   📊 Estimated additional runs needed: {(total_messages - processed_so_far) // max_messages_to_use}')
            
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
                
                # Run the actual embedding process with automatically determined parameters
                update_result = engine.update_embeddings(batch_size=batch_size_to_use, max_messages=max_messages_to_use)
                
                # Stop animation
                processing_complete = True
                animation_thread.join(timeout=0.1)
                
                new_messages = update_result.get('new_messages', 0)
                processing_time = update_result.get('processing_time_seconds', 0)
                
                progress.update(90, f'Processed {new_messages:,} additional messages')
                progress.finish(f'Complete! ({processing_time:.1f}s)')
                
                print(f'   ✅ Successfully embedded {new_messages:,} additional messages')
                print(f'   ⏱️  Processing time: {processing_time:.1f} seconds')
                
                # Show completion status for large databases
                if processing_mode == 'large':
                    processed_so_far = embedded_messages + new_messages
                    total_completion = (processed_so_far / total_messages) * 100
                    print(f'   📊 Database completion: {total_completion:.1f}% ({processed_so_far:,}/{total_messages:,} messages)')
                    if processed_so_far < total_messages:
                        print(f'   📊 Run ./setup.sh again to process more messages')
                        print(f'   📊 Estimated additional runs needed: {(total_messages - processed_so_far) // max_messages_to_use}')
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
        echo "🎉 Setup completed successfully!"
        echo ""
        
        # Check if we processed a large database and need more runs
        if [ -n "$SEATALK_DB_PATH" ] && [ -n "$SEATALK_DB_KEY" ]; then
            echo "📊 Database Processing Summary:"
            echo "• This run processed some of your messages"
            echo "• If you have a large database (50K+ messages), you may need to run ./setup.sh again"
            echo "• Each run will process more messages until complete"
            echo "• The MCP server will work even with partial processing"
            echo ""
        fi
        
        echo "📋 Next steps:"
        echo "1. The MCP server is ready for use with Claude Desktop"
        echo "2. Make sure your Claude Desktop config points to:"
        echo "   Command: /Users/$(whoami)/.local/bin/uvx"
        echo "   Args: [\"--from\", \"$SCRIPT_DIR\", \"seatalk-search-server\"]"
        echo ""
        
        if [ -n "$SEATALK_DB_PATH" ] && [ -n "$SEATALK_DB_KEY" ]; then
            echo "⚡ Your database is initialized - searches should be fast!"
            echo "🔍 You can now search your SeaTalk messages in Claude!"
            echo ""
            echo "💡 TIP: For large databases, run ./setup.sh again periodically to"
            echo "   process more messages and improve search coverage."
        else
            echo "⚡ First MCP call will initialize the database (may take 30-60s)"
            echo "🔍 You can now search your SeaTalk messages in Claude!"
        fi
    else
        echo ""
        echo "❌ Database initialization failed"
        echo ""
        echo "🔧 Troubleshooting:"
        echo "• Check that SEATALK_DB_PATH and SEATALK_DB_KEY are correct"
        echo "• Try running with debug logging: ./setup.sh --debug"
        echo "• For large databases, try: ./setup.sh --max-messages 1000"
        echo "• Check /tmp/seatalk_setup_*.log for detailed error information"
        exit 1
    fi
fi 