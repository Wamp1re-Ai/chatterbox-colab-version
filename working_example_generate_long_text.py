#!/usr/bin/env python3
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
        print("# # Initialize model - prioritize GPU for better performance")
        print("# if torch.cuda.is_available():")
        print("#     device = 'cuda'")
        print("#     print(f'🚀 Using GPU: {torch.cuda.get_device_name(0)}')")
        print("#     print(f'   CUDA Version: {torch.version.cuda}')")
        print("#     print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')")
        print("# else:")
        print("#     device = 'cpu'")
        print("#     print('⚠️  Using CPU (GPU not available)')")
        print("#     print('   For better performance, ensure CUDA is installed and GPU is available')")
        print("# print(f'Device: {device}')")
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
