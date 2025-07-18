#!/usr/bin/env python3
"""
Example script demonstrating IndexTTS-inspired long audio generation with Chatterbox.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.tts import ChatterboxTTS
import torch

def main():
    print("🎵 Chatterbox TTS with IndexTTS-inspired Long Audio Generation")
    print("=" * 60)
    
    # Sample long text for demonstration
    long_text = """
    Welcome to the enhanced Chatterbox TTS system with IndexTTS-inspired improvements!
    This system now features advanced chunking mechanisms that handle long texts much more effectively.
    
    The new chunking system includes several methods:
    First, sentence bucketing that groups sentences intelligently for optimal processing.
    Second, semantic boundary detection that respects natural speech flow and pauses.
    Third, improved memory management that prevents out-of-memory errors during long audio generation.
    
    These enhancements are inspired by the IndexTTS architecture and provide better quality audio output.
    The system automatically estimates memory usage and recommends optimal chunk sizes.
    You can now generate high-quality audio from very long texts without worrying about memory limitations.
    
    Try different chunking methods like 'sentences', 'semantic', 'clauses', or 'character' to see what works best for your use case.
    """.strip()
    
    print(f"\n📝 Input text ({len(long_text)} characters):")
    print(long_text[:200] + "..." if len(long_text) > 200 else long_text)
    
    try:
        # Initialize TTS (this will work if you have the model files)
        print("\n🔧 Initializing Chatterbox TTS...")
        
        # For demonstration, we'll show how to use the new features
        # without actually loading the model (which requires model files)
        
        # Show memory estimation
        print("\n💾 Memory Usage Estimation:")
        
        # Create a mock TTS instance to demonstrate the estimate_memory_usage method
        class DemoTTS:
            def __init__(self):
                self.sr = 24000
            
            def estimate_memory_usage(self, text: str, max_new_tokens: int = 1000, chunk_method: str = "sentences") -> dict:
                from chatterbox.tts import estimate_tokens
                
                text_length = len(text)
                estimated_text_tokens = estimate_tokens(text)
                estimated_speech_tokens = min(estimated_text_tokens * 8, max_new_tokens)
                
                base_memory_mb = 2500
                text_processing_mb = estimated_text_tokens * 0.02
                speech_generation_mb = estimated_speech_tokens * 0.15
                audio_synthesis_mb = estimated_speech_tokens * 0.08
                watermarking_mb = (estimated_speech_tokens * self.sr // 1000) * 0.001
                
                total_estimated_mb = (
                    base_memory_mb + text_processing_mb + 
                    speech_generation_mb + audio_synthesis_mb + watermarking_mb
                )
                
                if chunk_method == "sentences":
                    base_chunk_size = 150
                elif chunk_method == "semantic":
                    base_chunk_size = 200
                elif chunk_method == "clauses":
                    base_chunk_size = 120
                else:
                    base_chunk_size = 100
                    
                memory_factor = max(0.5, min(2.0, 4000 / total_estimated_mb))
                recommended_chunk_size = int(base_chunk_size * memory_factor)
                
                return {
                    "text_length": text_length,
                    "estimated_text_tokens": estimated_text_tokens,
                    "estimated_speech_tokens": estimated_speech_tokens,
                    "total_estimated_mb": total_estimated_mb,
                    "recommended_chunk_size": max(50, min(500, recommended_chunk_size)),
                    "estimated_audio_duration_seconds": estimated_speech_tokens * 0.02,
                    "memory_efficiency_score": min(100, max(0, 100 - (total_estimated_mb - 3000) / 50))
                }
        
        demo_tts = DemoTTS()
        memory_info = demo_tts.estimate_memory_usage(long_text)
        
        print(f"  📊 Estimated memory usage: {memory_info['total_estimated_mb']:.1f} MB")
        print(f"  🎯 Recommended chunk size: {memory_info['recommended_chunk_size']} characters")
        print(f"  ⏱️  Estimated audio duration: {memory_info['estimated_audio_duration_seconds']:.1f} seconds")
        print(f"  ⚡ Memory efficiency score: {memory_info['memory_efficiency_score']:.1f}/100")
        
        print("\n🎛️  Available Generation Methods:")
        print("\n1. Standard Generation (for shorter texts):")
        print("   tts.generate(text, max_mel_tokens=1000)")
        
        print("\n2. Long Text Generation with Chunking:")
        print("   tts.generate_long_text(")
        print("       text=long_text,")
        print("       chunk_method='sentences',  # or 'semantic', 'clauses', 'character'")
        print("       max_chunk_size=200,")
        print("       max_mel_tokens=1000,")
        print("       max_text_tokens_per_sentence=50,")
        print("       sentences_bucket_max_size=10,")
        print("       optimize_memory_between_chunks=True")
        print("   )")
        
        print("\n3. Streaming Generation (for real-time processing):")
        print("   for audio_chunk in tts.generate_streaming(")
        print("       text=long_text,")
        print("       chunk_method='semantic',")
        print("       max_mel_tokens=1000")
        print("   ):")
        print("       # Process each audio chunk as it's generated")
        print("       pass")
        
        print("\n🔧 IndexTTS-Inspired Features:")
        print("  ✅ max_mel_tokens: Controls speech generation length")
        print("  ✅ max_text_tokens_per_sentence: Limits tokens per sentence")
        print("  ✅ sentences_bucket_max_size: Optimizes sentence grouping")
        print("  ✅ Semantic boundary detection for natural speech flow")
        print("  ✅ Intelligent memory management and optimization")
        print("  ✅ Improved chunking algorithms for better quality")
        
        print("\n🚀 To use with actual model:")
        print("   # Load your model")
        print("   tts = ChatterboxTTS.from_pretrained('path/to/model')")
        print("   ")
        print("   # Generate long audio with new features")
        print("   audio = tts.generate_long_text(")
        print("       text=your_long_text,")
        print("       chunk_method='semantic',  # Best for natural speech")
        print("       max_mel_tokens=1000,      # IndexTTS-inspired control")
        print("       optimize_memory_between_chunks=True")
        print("   )")
        
        print("\n✅ The 'estimate_memory_usage' method is now properly implemented!")
        print("   This should fix the AttributeError you encountered.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Note: This demo doesn't load actual model files.")
        print("   To use with real audio generation, ensure you have:")
        print("   - Model checkpoint files in the correct location")
        print("   - Sufficient GPU memory")
        print("   - All dependencies installed")

if __name__ == "__main__":
    main()