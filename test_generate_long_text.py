#!/usr/bin/env python3
"""
Test script to reproduce and fix the 'generate_long_text' attribute error.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.tts import ChatterboxTTS
import torch

def test_generate_long_text():
    print("🧪 Testing generate_long_text method availability...")
    print("=" * 55)
    
    # Check if the method exists on the class
    if hasattr(ChatterboxTTS, 'generate_long_text'):
        print("✅ generate_long_text method EXISTS on ChatterboxTTS class")
    else:
        print("❌ generate_long_text method MISSING from ChatterboxTTS class")
        return
    
    print("\n🔧 Common solutions for 'object has no attribute' errors:")
    print("1. Restart Python interpreter/kernel to clear import cache")
    print("2. Check if you're using the correct import path")
    print("3. Verify you're using the latest version of the code")
    print("4. Make sure you're instantiating ChatterboxTTS correctly")
    
    print("\n📋 Correct usage examples:")
    print("\n# Method 1: Using from_pretrained (recommended)")
    print("from chatterbox.tts import ChatterboxTTS")
    print("model = ChatterboxTTS.from_pretrained(device='cpu')")
    print("audio = model.generate_long_text('Your long text here')")
    
    print("\n# Method 2: Using from_local (if you have local model files)")
    print("model = ChatterboxTTS.from_local('path/to/model', device='cpu')")
    print("audio = model.generate_long_text('Your long text here')")
    
    print("\n🚨 Common mistakes to avoid:")
    print("❌ Don't do: model = ChatterboxTTS()  # Missing required parameters")
    print("❌ Don't do: from chatterbox import ChatterboxTTS  # Wrong import path")
    print("✅ Do: from chatterbox.tts import ChatterboxTTS")
    
    print("\n🔍 Debugging steps if error persists:")
    print("1. Check the type of your model object:")
    print("   print(type(model))")
    print("   print(dir(model))")
    
    print("\n2. Verify the method exists:")
    print("   print(hasattr(model, 'generate_long_text'))")
    
    print("\n3. Check for import issues:")
    print("   import importlib")
    print("   importlib.reload(chatterbox.tts)")
    
    # Test with a mock instance to show the method signature
    print("\n📖 Method signature:")
    import inspect
    sig = inspect.signature(ChatterboxTTS.generate_long_text)
    print(f"generate_long_text{sig}")
    
    print("\n🎯 IndexTTS-inspired parameters available:")
    print("- max_mel_tokens: Controls speech generation length")
    print("- max_text_tokens_per_sentence: Limits tokens per sentence")
    print("- sentences_bucket_max_size: Optimizes sentence grouping")
    print("- chunk_method: 'sentences', 'semantic', 'clauses', 'character'")
    print("- optimize_memory_between_chunks: Memory management")

if __name__ == "__main__":
    test_generate_long_text()