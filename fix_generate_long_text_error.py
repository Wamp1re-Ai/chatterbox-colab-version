#!/usr/bin/env python3
"""
Comprehensive fix for 'ChatterboxTTS' object has no attribute 'generate_long_text' error.
"""

import sys
import os
import importlib

# Ensure we're using the local version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def fix_import_cache():
    """Clear import cache and reload modules."""
    print("🔄 Clearing import cache and reloading modules...")
    
    # Remove any cached chatterbox modules
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('chatterbox')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"   Removed cached module: {module}")
    
    # Force reload
    try:
        import chatterbox.tts
        importlib.reload(chatterbox.tts)
        print("   ✅ Successfully reloaded chatterbox.tts")
    except Exception as e:
        print(f"   ❌ Error reloading: {e}")
        return False
    
    return True

def verify_installation():
    """Verify the ChatterboxTTS class has all required methods."""
    print("\n🔍 Verifying ChatterboxTTS installation...")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        required_methods = ['generate', 'generate_long_text', 'generate_streaming', 'estimate_memory_usage']
        missing_methods = []
        
        for method in required_methods:
            if hasattr(ChatterboxTTS, method):
                print(f"   ✅ {method} - Available")
            else:
                print(f"   ❌ {method} - Missing")
                missing_methods.append(method)
        
        if missing_methods:
            print(f"\n❌ Missing methods: {missing_methods}")
            return False
        else:
            print("\n✅ All required methods are available!")
            return True
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def demonstrate_correct_usage():
    """Show correct usage examples."""
    print("\n📚 Correct usage examples:")
    print("=" * 40)
    
    # Example 1: Basic usage
    print("\n1️⃣ Basic long text generation:")
    print("```python")
    print("from chatterbox.tts import ChatterboxTTS")
    print("import torch")
    print("")
    print("# Initialize model")
    print("device = 'cuda' if torch.cuda.is_available() else 'cpu'")
    print("model = ChatterboxTTS.from_pretrained(device=device)")
    print("")
    print("# Generate long audio")
    print("long_text = 'Your very long text here...'")
    print("audio = model.generate_long_text(long_text)")
    print("```")
    
    # Example 2: Advanced usage with IndexTTS features
    print("\n2️⃣ Advanced usage with IndexTTS-inspired features:")
    print("```python")
    print("audio = model.generate_long_text(")
    print("    text=long_text,")
    print("    chunk_method='semantic',  # Better for natural speech")
    print("    max_mel_tokens=1000,      # IndexTTS-inspired control")
    print("    sentences_bucket_max_size=10,")
    print("    optimize_memory_between_chunks=True")
    print(")")
    print("```")
    
    # Example 3: Memory estimation
    print("\n3️⃣ Memory usage estimation:")
    print("```python")
    print("# Estimate memory before generation")
    print("memory_info = model.estimate_memory_usage(long_text)")
    print("print(f\"Estimated memory: {memory_info['total_estimated_mb']:.1f} MB\")")
    print("print(f\"Recommended chunk size: {memory_info['recommended_chunk_size']}\")")
    print("```")

def create_working_example():
    """Create a working example file."""
    print("\n📝 Creating working example file...")
    
    example_code = '''#!/usr/bin/env python3
"""
Working example of ChatterboxTTS with generate_long_text method.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.tts import ChatterboxTTS
import torch
import torchaudio as ta

def main():
    print("🎵 Testing ChatterboxTTS generate_long_text method")
    print("=" * 50)
    
    # Sample long text
    long_text = """
    This is a demonstration of the enhanced ChatterboxTTS system with IndexTTS-inspired improvements.
    The system now features advanced chunking mechanisms that handle long texts much more effectively.
    You can use different chunking methods like sentences, semantic boundaries, clauses, or character-based chunking.
    The memory management has been significantly improved to prevent out-of-memory errors during long audio generation.
    """.strip()
    
    print(f"📝 Text to synthesize ({len(long_text)} characters):")
    print(long_text[:100] + "..." if len(long_text) > 100 else long_text)
    
    try:
        # Check if method exists
        if not hasattr(ChatterboxTTS, 'generate_long_text'):
            print("❌ ERROR: generate_long_text method not found!")
            print("Please run fix_generate_long_text_error.py first")
            return
        
        print("\n✅ generate_long_text method found!")
        
        # For actual usage, uncomment the following lines:
        # (This requires downloading the model files)
        
        print("\n🔧 To use this method with actual model:")
        print("1. Uncomment the lines below")
        print("2. Make sure you have the model files downloaded")
        print("3. Run this script")
        
        print("\n# Uncomment these lines for actual usage:")
        print("# device = 'cuda' if torch.cuda.is_available() else 'cpu'")
        print("# model = ChatterboxTTS.from_pretrained(device=device)")
        print("# audio = model.generate_long_text(long_text, chunk_method='semantic')")
        print("# ta.save('long_audio_output.wav', audio, model.sr)")
        print("# print('✅ Audio saved to long_audio_output.wav')")
        
        # Show method signature
        import inspect
        sig = inspect.signature(ChatterboxTTS.generate_long_text)
        print(f"\n📖 Method signature:")
        print(f"generate_long_text{sig}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Restart your Python interpreter/kernel")
        print("2. Run fix_generate_long_text_error.py")
        print("3. Check your import statements")

if __name__ == "__main__":
    main()
'''
    
    with open('working_example_generate_long_text.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("   ✅ Created: working_example_generate_long_text.py")

def main():
    print("🔧 ChatterboxTTS generate_long_text Error Fix")
    print("=" * 50)
    
    # Step 1: Clear import cache
    if fix_import_cache():
        # Step 2: Verify installation
        if verify_installation():
            # Step 3: Show usage examples
            demonstrate_correct_usage()
            
            # Step 4: Create working example
            create_working_example()
            
            print("\n🎉 Fix completed successfully!")
            print("\n📋 Next steps:")
            print("1. Restart your Python interpreter/kernel")
            print("2. Use the correct import: from chatterbox.tts import ChatterboxTTS")
            print("3. Initialize with: model = ChatterboxTTS.from_pretrained(device='cpu')")
            print("4. Use: audio = model.generate_long_text('your text here')")
            print("5. Check working_example_generate_long_text.py for complete example")
        else:
            print("\n❌ Verification failed. Please check your installation.")
    else:
        print("\n❌ Failed to clear import cache. Please restart Python interpreter.")

if __name__ == "__main__":
    main()