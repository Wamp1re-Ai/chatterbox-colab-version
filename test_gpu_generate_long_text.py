#!/usr/bin/env python3
"""
GPU-Optimized Test for generate_long_text Method

This script demonstrates the generate_long_text method with GPU acceleration
and shows GPU utilization information.
"""

import sys
import importlib
import torch
import torchaudio as ta
from datetime import datetime

def clear_import_cache():
    """Clear Python import cache for chatterbox modules."""
    print("🔄 Clearing import cache...")
    modules_to_remove = [key for key in sys.modules.keys() if key.startswith('chatterbox')]
    for module in modules_to_remove:
        del sys.modules[module]
        print(f"   Removed cached module: {module}")
    print("✅ Import cache cleared!\n")

def check_gpu_info():
    """Display detailed GPU information."""
    print("🔍 GPU Information:")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"   📊 CUDA Devices Available: {device_count}")
        
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            memory_total = props.total_memory / 1024**3
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_cached = torch.cuda.memory_reserved(i) / 1024**3
            
            print(f"   🚀 GPU {i}: {props.name}")
            print(f"      Memory: {memory_total:.1f} GB total, {memory_allocated:.1f} GB allocated, {memory_cached:.1f} GB cached")
            print(f"      Compute Capability: {props.major}.{props.minor}")
            print(f"      Multiprocessors: {props.multi_processor_count}")
        
        print(f"   🔧 CUDA Version: {torch.version.cuda}")
        print(f"   🔧 PyTorch Version: {torch.__version__}")
        return True
    else:
        print("   ❌ No CUDA-capable GPU detected")
        print("   💡 Install CUDA and PyTorch with GPU support for better performance")
        return False

def test_generate_long_text_gpu():
    """Test generate_long_text with GPU optimization."""
    print("\n🧪 Testing generate_long_text with GPU...")
    
    try:
        # Clear cache and import
        clear_import_cache()
        from chatterbox.tts import ChatterboxTTS
        
        # Setup device
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"   ✅ Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("   ⚠️  Falling back to CPU")
        
        # Initialize model
        print("   📥 Loading ChatterboxTTS model...")
        start_time = datetime.now()
        model = ChatterboxTTS.from_pretrained(device=device)
        load_time = (datetime.now() - start_time).total_seconds()
        print(f"   ⏱️  Model loaded in {load_time:.1f} seconds")
        
        # Show GPU memory usage after model loading
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_cached = torch.cuda.memory_reserved(0) / 1024**3
            print(f"   📊 GPU Memory after loading: {memory_allocated:.1f} GB allocated, {memory_cached:.1f} GB cached")
        
        # Test text - moderately long to demonstrate chunking
        test_text = """
        Welcome to the advanced ChatterboxTTS system with IndexTTS-inspired features. 
        This demonstration showcases the new generate_long_text method that can handle ultra-long texts efficiently. 
        The system uses intelligent chunking algorithms to break down long texts while preserving natural speech patterns. 
        Memory optimization ensures that even very long texts can be processed without running out of GPU memory. 
        The chunking can be done by sentences, semantic boundaries, clauses, or character count. 
        Sentence-based chunking preserves the most natural speech flow and is recommended for most use cases. 
        The system also includes memory usage estimation to help you plan your generation tasks effectively.
        """
        
        # Estimate memory usage
        print("\n   🔍 Estimating memory usage...")
        memory_info = model.estimate_memory_usage(test_text)
        print(f"   📝 Text length: {memory_info['text_length']} characters")
        print(f"   💾 Estimated memory: {memory_info['total_estimated_mb']:.1f} MB")
        print(f"   📏 Recommended chunk size: {memory_info['recommended_chunk_size']} characters")
        print(f"   ⏱️  Estimated audio duration: {memory_info['estimated_audio_duration_seconds']:.1f} seconds")
        
        # Test different chunking methods
        chunk_methods = ['sentences', 'semantic', 'clauses']
        
        for i, chunk_method in enumerate(chunk_methods):
            print(f"\n   🎯 Testing chunking method: {chunk_method}")
            
            # Monitor GPU memory before generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache
                memory_before = torch.cuda.memory_allocated(0) / 1024**3
                print(f"      GPU memory before: {memory_before:.1f} GB")
            
            # Generate audio
            start_time = datetime.now()
            audio = model.generate_long_text(
                text=test_text,
                chunk_method=chunk_method,
                max_chunk_size=200,
                max_mel_tokens=1000,  # IndexTTS-inspired parameter
                sentences_bucket_max_size=8,
                optimize_memory_between_chunks=True,
                temperature=0.8,
                cfg_weight=0.5
            )
            generation_time = (datetime.now() - start_time).total_seconds()
            
            # Monitor GPU memory after generation
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(0) / 1024**3
                print(f"      GPU memory after: {memory_after:.1f} GB")
                print(f"      Memory increase: {memory_after - memory_before:.1f} GB")
            
            print(f"      ✅ Generated audio shape: {audio.shape}")
            print(f"      ⏱️  Generation time: {generation_time:.1f} seconds")
            print(f"      🎵 Audio duration: {audio.shape[-1] / model.sr:.1f} seconds")
            
            # Save audio file
            output_file = f"test_gpu_long_audio_{chunk_method}.wav"
            ta.save(output_file, audio, model.sr)
            print(f"      💾 Saved: {output_file}")
        
        # Final GPU memory status
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated(0) / 1024**3
            print(f"\n   📊 Final GPU memory: {final_memory:.1f} GB")
        
        print("\n🎉 All GPU tests completed successfully!")
        return True
        
    except AttributeError as e:
        if 'generate_long_text' in str(e):
            print(f"   ❌ AttributeError: {e}")
            print("\n🔧 SOLUTION:")
            print("   1. Restart your Python interpreter/kernel")
            print("   2. Run this script again")
            print("   3. Ensure you're using the latest version")
            return False
        else:
            print(f"   ❌ Unexpected AttributeError: {e}")
            return False
    except Exception as e:
        print(f"   ❌ Error during GPU testing: {e}")
        return False

def main():
    """Main function for GPU testing."""
    print("🚀 ChatterboxTTS GPU Performance Test\n")
    print("=" * 60)
    
    # Check GPU availability
    gpu_available = check_gpu_info()
    
    if not gpu_available:
        print("\n⚠️  No GPU detected. The test will run on CPU but will be slower.")
        print("   For optimal performance, ensure CUDA is installed and GPU is available.")
    
    print("\n" + "=" * 60)
    
    # Run the test
    success = test_generate_long_text_gpu()
    
    print("\n" + "=" * 60)
    if success:
        print("\n✅ GPU test completed successfully!")
        print("\n📁 Generated files:")
        print("   - test_gpu_long_audio_sentences.wav")
        print("   - test_gpu_long_audio_semantic.wav")
        print("   - test_gpu_long_audio_clauses.wav")
    else:
        print("\n❌ GPU test failed. Please check the error messages above.")
    
    print("\n💡 Tips for GPU optimization:")
    print("   - Use smaller chunk sizes for limited GPU memory")
    print("   - Enable optimize_memory_between_chunks=True")
    print("   - Monitor GPU memory usage during generation")
    print("   - Use torch.cuda.empty_cache() to free unused memory")

if __name__ == "__main__":
    main()