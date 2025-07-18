#!/usr/bin/env python3
"""
Verification Script for generate_long_text Method

This script helps diagnose and fix the 'generate_long_text' AttributeError.
Run this script to verify that the method exists and works properly.
"""

import sys
import importlib
import torch

def clear_import_cache():
    """Clear Python import cache for chatterbox modules."""
    print("🔄 Clearing import cache...")
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('chatterbox')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"   Removed cached module: {module}")
    print("✅ Import cache cleared!\n")

def verify_methods():
    """Verify that all required methods exist."""
    print("🔍 Verifying ChatterboxTTS methods...")
    
    try:
        # Import with fresh cache
        from chatterbox.tts import ChatterboxTTS
        
        # Check for required methods
        required_methods = [
            'generate',
            'generate_long_text', 
            'generate_streaming',
            'estimate_memory_usage'
        ]
        
        missing_methods = []
        for method in required_methods:
            if hasattr(ChatterboxTTS, method):
                print(f"   ✅ {method} - Available")
            else:
                print(f"   ❌ {method} - Missing")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"\n❌ Missing methods: {missing_methods}")
            print("\n🔧 SOLUTION: Restart your Python interpreter/kernel and try again.")
            return False
        else:
            print("\n✅ All methods are available!")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_generate_long_text():
    """Test the generate_long_text method with a simple example."""
    print("\n🧪 Testing generate_long_text method...")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # Check device - prioritize GPU
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"   Using device: {device} (GPU detected)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print(f"   Using device: {device} (No GPU available)")
        
        # Initialize model (this might take a moment)
        print("   Loading model...")
        model = ChatterboxTTS.from_pretrained(device=device)
        
        # Test text
        test_text = "This is a test of the generate_long_text method. It should work without any errors."
        
        # Test memory estimation
        print("   Testing memory estimation...")
        memory_info = model.estimate_memory_usage(test_text)
        print(f"   Memory estimate: {memory_info['total_estimated_mb']:.1f}MB")
        
        # Test generate_long_text
        print("   Testing generate_long_text...")
        audio = model.generate_long_text(
            text=test_text,
            chunk_method='sentences',
            max_chunk_size=100,
            optimize_memory_between_chunks=True
        )
        
        print(f"   ✅ Success! Generated audio shape: {audio.shape}")
        return True
        
    except AttributeError as e:
        if 'generate_long_text' in str(e):
            print(f"   ❌ AttributeError: {e}")
            print("\n🔧 SOLUTION:")
            print("   1. Restart your Python interpreter/kernel")
            print("   2. Run this script again")
            print("   3. Make sure you're using the latest version")
            return False
        else:
            print(f"   ❌ Unexpected AttributeError: {e}")
            return False
    except Exception as e:
        print(f"   ❌ Error during testing: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 ChatterboxTTS generate_long_text Verification\n")
    print("=" * 50)
    
    # Step 1: Clear cache
    clear_import_cache()
    
    # Step 2: Verify methods exist
    if not verify_methods():
        print("\n❌ Verification failed. Please restart your Python interpreter.")
        return
    
    # Step 3: Test the method (optional, requires model download)
    print("\n" + "=" * 50)
    print("\n🤔 Would you like to test the method with actual model loading?")
    print("   This will download the model if not already cached (~2GB)")
    
    try:
        response = input("\nTest with model? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            if test_generate_long_text():
                print("\n🎉 All tests passed! generate_long_text is working correctly.")
            else:
                print("\n❌ Test failed. Please restart your Python interpreter.")
        else:
            print("\n✅ Method verification complete. The generate_long_text method exists.")
            print("   If you still get AttributeError, restart your Python interpreter.")
    except KeyboardInterrupt:
        print("\n\n✅ Verification complete. The generate_long_text method exists.")
    
    print("\n" + "=" * 50)
    print("\n📚 For more help, see: TROUBLESHOOTING_GENERATE_LONG_TEXT.md")

if __name__ == "__main__":
    main()