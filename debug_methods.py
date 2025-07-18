#!/usr/bin/env python3
"""
Debug script to check ChatterboxTTS methods availability.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.tts import ChatterboxTTS
import inspect

def check_methods():
    print("🔍 Checking ChatterboxTTS class methods...")
    print("=" * 50)
    
    # Get all methods of ChatterboxTTS class
    methods = [method for method in dir(ChatterboxTTS) if not method.startswith('_')]
    
    print(f"📋 Available methods in ChatterboxTTS:")
    for i, method in enumerate(sorted(methods), 1):
        print(f"  {i:2d}. {method}")
    
    # Check specifically for the methods we're interested in
    target_methods = ['generate', 'generate_long_text', 'generate_streaming', 'estimate_memory_usage']
    
    print(f"\n🎯 Checking target methods:")
    for method in target_methods:
        if hasattr(ChatterboxTTS, method):
            print(f"  ✅ {method} - EXISTS")
            # Get method signature
            try:
                sig = inspect.signature(getattr(ChatterboxTTS, method))
                print(f"     Signature: {method}{sig}")
            except Exception as e:
                print(f"     Could not get signature: {e}")
        else:
            print(f"  ❌ {method} - MISSING")
    
    print(f"\n📦 ChatterboxTTS class info:")
    print(f"  Module: {ChatterboxTTS.__module__}")
    print(f"  File: {inspect.getfile(ChatterboxTTS)}")
    
    # Check if we can create an instance (without loading models)
    print(f"\n🏗️  Testing class instantiation (mock):")
    try:
        # We can't actually instantiate without models, but we can check the __init__ signature
        init_sig = inspect.signature(ChatterboxTTS.__init__)
        print(f"  __init__ signature: {init_sig}")
        print(f"  ✅ Class definition looks correct")
    except Exception as e:
        print(f"  ❌ Issue with class definition: {e}")

if __name__ == "__main__":
    check_methods()