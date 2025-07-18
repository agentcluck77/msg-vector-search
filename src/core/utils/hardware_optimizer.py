#!/usr/bin/env python3
"""
Hardware Optimization Utilities
Detects and optimizes for Apple Silicon and Intel Mac hardware
"""

import logging
import platform
import os
from typing import Dict, Any, Tuple

# Try to import psutil, but don't fail if it's not available
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)

class HardwareOptimizer:
    """Hardware detection and optimization utilities"""
    
    def __init__(self):
        self.system_info = self._detect_hardware()
        self.optimization_settings = self._get_optimization_settings()
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware specifications"""
        system = platform.system()
        machine = platform.machine()
        processor = platform.processor()
        
        # CPU info
        cpu_count = os.cpu_count()
        
        # Get physical CPU count with fallback if psutil not available
        if HAS_PSUTIL:
            cpu_count_physical = psutil.cpu_count(logical=False)
            # Memory info
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
        else:
            # Fallback estimates without psutil
            cpu_count_physical = cpu_count // 2  # Rough estimate
            memory_gb = 8.0  # Conservative default
            logger.warning("⚠️ psutil not available, using hardware fallback estimates")
        
        # Apple-specific detection
        is_apple_silicon = system == 'Darwin' and machine == 'arm64'
        is_intel_mac = system == 'Darwin' and machine == 'x86_64'
        
        # Detect specific Apple chip
        chip_name = "Unknown"
        if is_apple_silicon:
            # Try to detect specific Apple chip
            try:
                import subprocess
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    chip_name = result.stdout.strip()
            except:
                pass
        
        info = {
            'system': system,
            'machine': machine,
            'processor': processor,
            'cpu_count': cpu_count,
            'cpu_count_physical': cpu_count_physical,
            'memory_gb': memory_gb,
            'is_apple_silicon': is_apple_silicon,
            'is_intel_mac': is_intel_mac,
            'chip_name': chip_name
        }
        
        logger.info(f"🔍 Hardware detected: {chip_name if is_apple_silicon else processor}")
        logger.info(f"🔍 CPU cores: {cpu_count_physical} physical, {cpu_count} logical")
        logger.info(f"🔍 Memory: {memory_gb:.1f} GB{' (estimated)' if not HAS_PSUTIL else ''}")
        
        return info
    
    def _get_optimization_settings(self) -> Dict[str, Any]:
        """Get hardware-specific optimization settings"""
        settings = {
            'embedding_batch_size': 32,
            'processing_batch_size': 1000,
            'parallel_workers': 2,
            'use_gpu': False,
            'memory_limit_mb': 2048,
            'enable_parallel_processing': False
        }
        
        # Apple Silicon optimizations
        if self.system_info['is_apple_silicon']:
            # Apple Silicon has excellent performance
            settings.update({
                'embedding_batch_size': 128,  # Can handle larger batches
                'processing_batch_size': 2000,  # Larger processing batches
                'parallel_workers': min(4, self.system_info['cpu_count_physical']),
                'use_gpu': True,  # Enable MPS
                'memory_limit_mb': min(8192, int(self.system_info['memory_gb'] * 1024 * 0.5)),
                'enable_parallel_processing': True
            })
            
            # M2/M3 specific optimizations
            if 'M2' in self.system_info['chip_name'] or 'M3' in self.system_info['chip_name']:
                settings['embedding_batch_size'] = 256  # Even larger for newer chips
                settings['processing_batch_size'] = 4000
                
        # Intel Mac optimizations  
        elif self.system_info['is_intel_mac']:
            # Intel Macs - more conservative but still optimized
            settings.update({
                'embedding_batch_size': 64,
                'processing_batch_size': 1500,
                'parallel_workers': min(3, self.system_info['cpu_count_physical']),
                'use_gpu': False,  # No MPS support
                'memory_limit_mb': min(4096, int(self.system_info['memory_gb'] * 1024 * 0.4)),
                'enable_parallel_processing': True
            })
            
            # High-end Intel Macs (16+ GB RAM)
            if self.system_info['memory_gb'] >= 16:
                settings['embedding_batch_size'] = 96
                settings['processing_batch_size'] = 2000
        
        # Adjust for available memory
        if self.system_info['memory_gb'] < 8:
            # Low memory systems
            settings['embedding_batch_size'] = max(16, settings['embedding_batch_size'] // 2)
            settings['processing_batch_size'] = max(500, settings['processing_batch_size'] // 2)
            settings['parallel_workers'] = 1
            settings['enable_parallel_processing'] = False
        
        logger.info(f"🔧 Optimization settings: {settings}")
        return settings
    
    def get_pytorch_device(self) -> str:
        """Get optimal PyTorch device"""
        try:
            import torch
            
            if self.system_info['is_apple_silicon']:
                if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                    return 'mps'
                else:
                    logger.warning("⚠️ MPS not available, falling back to CPU")
                    return 'cpu'
            elif torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'
    
    def configure_pytorch(self) -> None:
        """Configure PyTorch for optimal performance"""
        try:
            import torch
            
            # Set number of threads
            if self.system_info['is_apple_silicon'] or self.system_info['is_intel_mac']:
                # Use all available cores on Mac
                torch.set_num_threads(self.system_info['cpu_count'])
            else:
                # More conservative on other platforms
                torch.set_num_threads(min(4, self.system_info['cpu_count']))
            
            # Memory management
            if hasattr(torch, 'set_default_tensor_type'):
                if self.system_info['is_apple_silicon']:
                    # Use float32 for better MPS compatibility
                    torch.set_default_tensor_type(torch.FloatTensor)
            
            logger.info(f"🔧 PyTorch configured for {self.system_info['machine']} with {torch.get_num_threads()} threads")
            
        except ImportError:
            logger.warning("⚠️ PyTorch not available for configuration")
    
    def get_embedding_batch_size(self) -> int:
        """Get optimal embedding batch size"""
        return self.optimization_settings['embedding_batch_size']
    
    def get_processing_batch_size(self) -> int:
        """Get optimal processing batch size"""
        return self.optimization_settings['processing_batch_size']
    
    def get_parallel_workers(self) -> int:
        """Get optimal number of parallel workers"""
        return self.optimization_settings['parallel_workers']
    
    def should_use_parallel_processing(self) -> bool:
        """Check if parallel processing should be enabled"""
        return self.optimization_settings['enable_parallel_processing']
    
    def get_memory_limit_mb(self) -> int:
        """Get memory limit in MB"""
        return self.optimization_settings['memory_limit_mb']
    
    def print_performance_summary(self) -> None:
        """Print performance optimization summary"""
        print("\n🚀 HARDWARE OPTIMIZATION SUMMARY")
        print("=" * 50)
        
        # Hardware info
        if self.system_info['is_apple_silicon']:
            print(f"🔥 Apple Silicon: {self.system_info['chip_name']}")
            print(f"🚀 GPU Acceleration: {'✅ MPS Enabled' if self.optimization_settings['use_gpu'] else '❌ MPS Not Available'}")
        elif self.system_info['is_intel_mac']:
            print(f"🔧 Intel Mac: {self.system_info['processor']}")
            print(f"🚀 GPU Acceleration: ❌ Not Available")
        else:
            print(f"🔧 Other Platform: {self.system_info['processor']}")
        
        print(f"⚡ CPU Cores: {self.system_info['cpu_count_physical']} physical, {self.system_info['cpu_count']} logical")
        print(f"💾 Memory: {self.system_info['memory_gb']:.1f} GB")
        
        # Optimization settings
        print(f"\n🔧 OPTIMIZATION SETTINGS:")
        print(f"📦 Embedding Batch Size: {self.optimization_settings['embedding_batch_size']}")
        print(f"📦 Processing Batch Size: {self.optimization_settings['processing_batch_size']}")
        print(f"⚡ Parallel Workers: {self.optimization_settings['parallel_workers']}")
        print(f"🔄 Parallel Processing: {'✅ Enabled' if self.optimization_settings['enable_parallel_processing'] else '❌ Disabled'}")
        print(f"💾 Memory Limit: {self.optimization_settings['memory_limit_mb']} MB")
        
        # Performance estimates
        print(f"\n📊 ESTIMATED PERFORMANCE:")
        if self.system_info['is_apple_silicon']:
            if 'M3' in self.system_info['chip_name']:
                print(f"🚀 Expected Speed: 150-200 messages/second")
            elif 'M2' in self.system_info['chip_name']:
                print(f"🚀 Expected Speed: 100-150 messages/second")
            else:
                print(f"🚀 Expected Speed: 80-120 messages/second")
        elif self.system_info['is_intel_mac']:
            print(f"🚀 Expected Speed: 50-80 messages/second")
        else:
            print(f"🚀 Expected Speed: 30-50 messages/second")
        
        print("=" * 50)
        
# Global instance
_hardware_optimizer = None

def get_hardware_optimizer() -> HardwareOptimizer:
    """Get global hardware optimizer instance"""
    global _hardware_optimizer
    if _hardware_optimizer is None:
        _hardware_optimizer = HardwareOptimizer()
    return _hardware_optimizer