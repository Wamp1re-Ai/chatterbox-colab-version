#!/usr/bin/env python3
"""
GPU Discovery Script for Local Machine (GTX 1650)

This script detects and tests your NVIDIA GTX 1650 GPU capabilities
before running the notebook on Google Colab.
"""

import sys
import subprocess
import torch
import platform
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🔍 {title}")
    print("=" * 60)

def check_system_info():
    """Display system information."""
    print_header("System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

def check_nvidia_driver():
    """Check NVIDIA driver installation."""
    print_header("NVIDIA Driver Check")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA Driver is installed and working!")
            print("\nNVIDIA-SMI Output:")
            print("-" * 40)
            # Parse and display key information
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line or 'CUDA Version' in line:
                    print(line.strip())
                elif 'GTX' in line or '1650' in line:
                    print(line.strip())
                elif 'MiB' in line and ('/' in line):
                    print(line.strip())
            print("-" * 40)
            return True, result.stdout
        else:
            print("❌ NVIDIA-SMI failed to run")
            print(f"Error: {result.stderr}")
            return False, None
    except subprocess.TimeoutExpired:
        print("❌ NVIDIA-SMI command timed out")
        return False, None
    except FileNotFoundError:
        print("❌ NVIDIA-SMI not found")
        print("💡 Please install NVIDIA drivers from: https://www.nvidia.com/drivers")
        return False, None
    except Exception as e:
        print(f"❌ Error running NVIDIA-SMI: {e}")
        return False, None

def check_cuda_pytorch():
    """Check CUDA availability in PyTorch."""
    print_header("CUDA & PyTorch Check")
    
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("✅ CUDA is available in PyTorch!")
        
        # Device count
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices: {device_count}")
        
        # Device details
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multiprocessors: {props.multi_processor_count}")
            print(f"  Max Threads per Block: {props.max_threads_per_block}")
            
            # Check if it's GTX 1650
            if '1650' in props.name:
                print("🎯 GTX 1650 Detected!")
                print("  Recommended settings for ChatterboxTTS:")
                print("  - chunk_size: 150-200 characters")
                print("  - max_mel_tokens: 800-1000")
                print("  - optimize_memory_between_chunks: True")
                print("  - temperature: 0.7-0.8")
        
        # CUDA version info
        print(f"\nCUDA Version (PyTorch): {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        return True
    else:
        print("❌ CUDA is not available in PyTorch")
        print("\n💡 Possible solutions:")
        print("1. Install PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("2. Or for CUDA 12.x:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

def test_gpu_performance():
    """Test basic GPU performance."""
    print_header("GPU Performance Test")
    
    if not torch.cuda.is_available():
        print("❌ Skipping GPU test - CUDA not available")
        return False
    
    try:
        device = torch.device('cuda')
        print(f"Testing on: {torch.cuda.get_device_name(0)}")
        
        # Memory test
        print("\n📊 Memory Test:")
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated(0) / 1024**2
        print(f"Memory before: {memory_before:.1f} MB")
        
        # Create test tensors
        print("Creating test tensors...")
        start_time = datetime.now()
        
        # Test tensor operations
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        memory_after = torch.cuda.memory_allocated(0) / 1024**2
        print(f"Memory after: {memory_after:.1f} MB")
        print(f"Memory used: {memory_after - memory_before:.1f} MB")
        print(f"Matrix multiplication time: {duration:.3f} seconds")
        
        # Cleanup
        del x, y, z
        torch.cuda.empty_cache()
        
        print("✅ GPU performance test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ GPU performance test failed: {e}")
        return False

def check_chatterbox_compatibility():
    """Check if ChatterboxTTS can be imported and used."""
    print_header("ChatterboxTTS Compatibility Check")
    
    try:
        # Clear any cached imports
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith('chatterbox')]
        for module in modules_to_remove:
            del sys.modules[module]
        
        # Try to import
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterboxTTS imported successfully!")
        
        # Check for generate_long_text method
        if hasattr(ChatterboxTTS, 'generate_long_text'):
            print("✅ generate_long_text method is available!")
            
            # Get method signature
            import inspect
            sig = inspect.signature(ChatterboxTTS.generate_long_text)
            print(f"Method signature: generate_long_text{sig}")
            
            return True
        else:
            print("❌ generate_long_text method not found")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import ChatterboxTTS: {e}")
        print("💡 Make sure ChatterboxTTS is installed:")
        print("   pip install -e .")
        return False
    except Exception as e:
        print(f"❌ Error checking ChatterboxTTS: {e}")
        return False

def generate_colab_recommendations():
    """Generate recommendations for Colab usage."""
    print_header("Google Colab Recommendations")
    
    print("📝 For optimal performance on Google Colab:")
    print("\n🚀 Runtime Settings:")
    print("   - Select 'GPU' runtime (T4, V100, or A100)")
    print("   - Use 'High-RAM' if available for longer texts")
    
    print("\n⚙️  ChatterboxTTS Settings for GTX 1650 equivalent:")
    print("   - chunk_method: 'sentences' (most natural)")
    print("   - max_chunk_size: 150-200 characters")
    print("   - max_mel_tokens: 800-1000")
    print("   - optimize_memory_between_chunks: True")
    print("   - temperature: 0.7-0.8")
    
    print("\n📊 Memory Management:")
    print("   - Use model.estimate_memory_usage() before generation")
    print("   - Clear GPU cache with torch.cuda.empty_cache()")
    print("   - Monitor memory usage during long generations")
    
    print("\n🔧 Installation Commands for Colab:")
    print("   !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   !git clone <your-repo-url>")
    print("   !cd chatterbox-colab-version && pip install -e .")

def main():
    """Main function to run all checks."""
    print("🚀 GPU Discovery for GTX 1650 - ChatterboxTTS")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all checks
    checks = [
        ("System Info", check_system_info),
        ("NVIDIA Driver", check_nvidia_driver),
        ("CUDA & PyTorch", check_cuda_pytorch),
        ("GPU Performance", test_gpu_performance),
        ("ChatterboxTTS", check_chatterbox_compatibility)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            if name == "System Info":
                check_func()
                results[name] = True
            else:
                results[name] = check_func()
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results[name] = False
    
    # Generate Colab recommendations
    generate_colab_recommendations()
    
    # Summary
    print_header("Summary")
    print("Check Results:")
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    # Overall status
    critical_checks = ["NVIDIA Driver", "CUDA & PyTorch", "ChatterboxTTS"]
    critical_passed = all(results.get(check, False) for check in critical_checks)
    
    print("\n" + "=" * 60)
    if critical_passed:
        print("🎉 Your GTX 1650 is ready for ChatterboxTTS!")
        print("✅ You can proceed to test on Google Colab")
    else:
        print("⚠️  Some issues need to be resolved before Colab testing")
        print("💡 Please fix the failed checks above")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()