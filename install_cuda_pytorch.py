#!/usr/bin/env python3
"""
CUDA PyTorch Installation Script for GTX 1650

This script will install the correct CUDA-enabled PyTorch version
for your NVIDIA GTX 1650 with CUDA 12.9 drivers.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n🔄 {description}...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during {description}: {e}")
        return False

def check_current_pytorch():
    """Check current PyTorch installation."""
    print("🔍 Checking current PyTorch installation...")
    
    try:
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("❌ CUDA not available in current PyTorch")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def uninstall_cpu_pytorch():
    """Uninstall CPU-only PyTorch packages."""
    packages_to_remove = [
        'torch',
        'torchvision', 
        'torchaudio',
        'torchtext',
        'torchdata'
    ]
    
    print("\n🗑️  Uninstalling CPU-only PyTorch packages...")
    
    for package in packages_to_remove:
        cmd = [sys.executable, '-m', 'pip', 'uninstall', package, '-y']
        print(f"Removing {package}...")
        try:
            subprocess.run(cmd, check=False, capture_output=True)
        except Exception as e:
            print(f"Note: Could not remove {package}: {e}")
    
    print("✅ CPU PyTorch packages removed")

def install_cuda_pytorch():
    """Install CUDA-enabled PyTorch for GTX 1650."""
    print("\n🚀 Installing CUDA-enabled PyTorch...")
    
    # For CUDA 12.x (your system has CUDA 12.9)
    cuda_commands = [
        # Try CUDA 12.1 first (most stable)
        {
            'name': 'CUDA 12.1',
            'cmd': [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', 
                   '--index-url', 'https://download.pytorch.org/whl/cu121']
        },
        # Fallback to CUDA 11.8 (more compatible)
        {
            'name': 'CUDA 11.8 (fallback)',
            'cmd': [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', 
                   '--index-url', 'https://download.pytorch.org/whl/cu118']
        }
    ]
    
    for cuda_option in cuda_commands:
        print(f"\n🎯 Trying {cuda_option['name']}...")
        success = run_command(cuda_option['cmd'], f"Installing PyTorch with {cuda_option['name']}")
        
        if success:
            # Test the installation
            print("\n🧪 Testing CUDA installation...")
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"✅ Success! CUDA is now available")
                    print(f"PyTorch version: {torch.__version__}")
                    print(f"CUDA version: {torch.version.cuda}")
                    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    return True
                else:
                    print(f"❌ CUDA still not available with {cuda_option['name']}")
                    continue
            except Exception as e:
                print(f"❌ Error testing {cuda_option['name']}: {e}")
                continue
    
    return False

def verify_installation():
    """Verify the final installation."""
    print("\n🔍 Final verification...")
    
    try:
        # Clear any cached imports
        modules_to_remove = [key for key in sys.modules.keys() if key.startswith('torch')]
        for module in modules_to_remove:
            if module in sys.modules:
                del sys.modules[module]
        
        import torch
        import torchvision
        import torchaudio
        
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ TorchVision: {torchvision.__version__}")
        print(f"✅ TorchAudio: {torchaudio.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA: {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # Quick performance test
            print("\n🧪 Quick GPU test...")
            device = torch.device('cuda')
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            print(f"✅ GPU computation successful! Result shape: {z.shape}")
            
            return True
        else:
            print("❌ CUDA still not available")
            return False
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def create_test_script():
    """Create a test script for ChatterboxTTS with GPU."""
    test_script = '''
#!/usr/bin/env python3
"""
Quick GPU Test for ChatterboxTTS
"""

import torch
from chatterbox.tts import ChatterboxTTS

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Test ChatterboxTTS
    print("\\nTesting ChatterboxTTS with GPU...")
    model = ChatterboxTTS.from_pretrained(device='cuda')
    print("✅ Model loaded on GPU successfully!")
    
    # Test generate_long_text
    if hasattr(model, 'generate_long_text'):
        print("✅ generate_long_text method available!")
        
        # Quick test
        test_text = "Hello, this is a test of the ChatterboxTTS system with GPU acceleration."
        audio = model.generate_long_text(
            text=test_text,
            chunk_method='sentences',
            max_chunk_size=150,
            optimize_memory_between_chunks=True
        )
        print(f"✅ Generated audio shape: {audio.shape}")
        print("🎉 GPU test completed successfully!")
    else:
        print("❌ generate_long_text method not found")
else:
    print("❌ No GPU available")
'''
    
    with open('test_gpu_chatterbox.py', 'w') as f:
        f.write(test_script)
    
    print("\n📝 Created test_gpu_chatterbox.py")
    print("Run it with: python test_gpu_chatterbox.py")

def main():
    """Main installation process."""
    print("🚀 CUDA PyTorch Installation for GTX 1650")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check current state
    current_cuda_works = check_current_pytorch()
    
    if current_cuda_works:
        print("\n✅ CUDA PyTorch is already working!")
        print("No installation needed.")
    else:
        print("\n🔧 Installing CUDA-enabled PyTorch...")
        
        # Uninstall CPU version
        uninstall_cpu_pytorch()
        
        # Install CUDA version
        success = install_cuda_pytorch()
        
        if not success:
            print("\n❌ Failed to install CUDA PyTorch")
            print("💡 Try manual installation:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            return False
    
    # Verify installation
    if verify_installation():
        print("\n🎉 CUDA PyTorch installation successful!")
        
        # Create test script
        create_test_script()
        
        print("\n📋 Next steps:")
        print("1. Run: python test_gpu_chatterbox.py")
        print("2. Run: python discover_gpu_local.py (should now pass all tests)")
        print("3. Upload your project to Google Colab for testing")
        
        return True
    else:
        print("\n❌ Installation verification failed")
        return False

if __name__ == "__main__":
    try:
        success = main()
        print("\n" + "=" * 60)
        if success:
            print("✅ Installation completed successfully!")
        else:
            print("❌ Installation failed. Please check the errors above.")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")