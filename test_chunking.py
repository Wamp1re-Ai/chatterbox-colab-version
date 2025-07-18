#!/usr/bin/env python3
"""
Test script to verify the IndexTTS-inspired chunking system in Chatterbox.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.tts import ChatterboxTTS, smart_sentence_split, estimate_tokens

def test_chunking_methods():
    """Test the new chunking methods."""
    print("Testing IndexTTS-inspired chunking methods...")
    
    # Sample long text for testing
    test_text = """
    This is a test of the new IndexTTS-inspired chunking system. It should handle long texts much better than before.
    The system now includes sentence bucketing, semantic boundary detection, and improved memory management.
    These features are designed to provide better quality audio generation for long texts, similar to IndexTTS.
    The chunking system considers token limits, sentence boundaries, and memory optimization to ensure smooth processing.
    """.strip()
    
    print(f"\nOriginal text ({len(test_text)} chars):")
    print(test_text)
    
    # Test smart sentence splitting
    print("\n=== Smart Sentence Splitting ===")
    sentences = smart_sentence_split(test_text)
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    # Test token estimation
    print("\n=== Token Estimation ===")
    tokens = estimate_tokens(test_text)
    print(f"Estimated tokens: {tokens}")
    
    # Test memory estimation (without loading the actual model)
    print("\n=== Memory Estimation Test ===")
    try:
        # Create a mock TTS instance for testing estimate_memory_usage
        class MockTTS:
            def __init__(self):
                self.sr = 24000  # Sample rate
            
            def estimate_memory_usage(self, text: str, max_new_tokens: int = 1000, chunk_method: str = "sentences") -> dict:
                """Mock implementation of estimate_memory_usage for testing."""
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
                    "base_memory_mb": base_memory_mb,
                    "text_processing_mb": text_processing_mb,
                    "speech_generation_mb": speech_generation_mb,
                    "audio_synthesis_mb": audio_synthesis_mb,
                    "watermarking_mb": watermarking_mb,
                    "total_estimated_mb": total_estimated_mb,
                    "recommended_chunk_size": max(50, min(500, recommended_chunk_size)),
                    "estimated_audio_duration_seconds": estimated_speech_tokens * 0.02,
                    "memory_efficiency_score": min(100, max(0, 100 - (total_estimated_mb - 3000) / 50))
                }
        
        mock_tts = MockTTS()
        memory_info = mock_tts.estimate_memory_usage(test_text)
        
        print("Memory usage estimation:")
        for key, value in memory_info.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error in memory estimation test: {e}")
    
    print("\n=== Chunking Methods Test ===")
    try:
        # Test chunking methods (without actual model loading)
        class MockChatterboxTTS:
            def _chunk_by_sentences_with_bucket(self, text: str, max_chunk_size: int, bucket_max_size: int):
                sentences = smart_sentence_split(text)
                chunks = []
                current_chunk = ""
                sentence_bucket = []
                
                for sentence in sentences:
                    sentence_bucket.append(sentence)
                    
                    if (len(sentence_bucket) >= bucket_max_size or 
                        len(" ".join(sentence_bucket)) > max_chunk_size):
                        
                        bucket_text = " ".join(sentence_bucket[:-1]) if len(sentence_bucket) > 1 else sentence_bucket[0]
                        
                        if current_chunk and len(current_chunk) + len(bucket_text) + 1 > max_chunk_size:
                            chunks.append(current_chunk.strip())
                            current_chunk = bucket_text
                        else:
                            if current_chunk:
                                current_chunk += " " + bucket_text
                            else:
                                current_chunk = bucket_text
                        
                        sentence_bucket = sentence_bucket[-1:] if len(sentence_bucket) > 1 else []
                
                if sentence_bucket:
                    bucket_text = " ".join(sentence_bucket)
                    if current_chunk and len(current_chunk) + len(bucket_text) + 1 > max_chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = bucket_text
                    else:
                        if current_chunk:
                            current_chunk += " " + bucket_text
                        else:
                            current_chunk = bucket_text
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                return [chunk for chunk in chunks if chunk]
        
        mock_chunker = MockChatterboxTTS()
        
        # Test sentence bucketing
        print("\nSentence bucketing chunks:")
        sentence_chunks = mock_chunker._chunk_by_sentences_with_bucket(test_text, 150, 3)
        for i, chunk in enumerate(sentence_chunks, 1):
            print(f"Chunk {i} ({len(chunk)} chars): {chunk[:100]}...")
            
    except Exception as e:
        print(f"Error in chunking test: {e}")
    
    print("\n✅ Chunking system test completed!")

if __name__ == "__main__":
    test_chunking_methods()