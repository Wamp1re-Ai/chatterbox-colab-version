from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import re
import gc
from typing import List, Generator, Optional, Iterator
import numpy as np

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


def chunk_text_by_sentences(text: str, max_chunk_size: int = 200) -> List[str]:
    """
    Split text into chunks by sentence boundaries while respecting max_chunk_size.
    Preserves natural speech patterns for better TTS quality.
    """
    # Split by sentence boundaries (periods, question marks, exclamation points)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed max_chunk_size, start a new chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks


def chunk_text_by_clauses(text: str, max_chunk_size: int = 200) -> List[str]:
    """
    Split text by clauses and sentence boundaries for more natural speech.
    Uses grammatical rules to identify natural breakpoints.
    """
    # Pattern to split on:
    # 1. Sentence endings: . ? ! ;
    # 2. Comma + coordinating conjunctions: , and|but|or|nor|for|yet|so
    pattern = r'(?<=[.!?;])\s+|(?<=,)\s+(?=(?:and|but|or|nor|for|yet|so)\s)'
    
    clauses = re.split(pattern, text.strip())
    
    chunks = []
    current_chunk = ""
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
            
        # If adding this clause would exceed max_chunk_size, start a new chunk
        if current_chunk and len(current_chunk) + len(clause) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = clause
        else:
            if current_chunk:
                current_chunk += " " + clause
            else:
                current_chunk = clause
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if chunk]  # Remove empty chunks


def optimize_memory():
    """
    Force garbage collection and clear CUDA cache to free up memory.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def smart_sentence_split(text: str) -> List[str]:
    """
    IndexTTS-inspired smart sentence splitting that handles various punctuation.
    """
    import re
    
    # Enhanced sentence splitting pattern that handles multiple languages
    sentence_pattern = r'(?<=[.!?。！？])\s+|(?<=[.!?。！？])[""\'\'\)\]]*\s+'
    sentences = re.split(sentence_pattern, text.strip())
    
    # Clean up empty sentences and whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text (rough approximation).
    """
    # Simple token estimation: split by whitespace and punctuation
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return len(tokens)


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        max_mel_tokens=1000,
    ):
        """
        Generate audio from text using the loaded model with IndexTTS-inspired optimizations.
        
        Args:
            text: Input text to convert to speech
            repetition_penalty: Penalty for repetition in generation
            min_p: Minimum probability threshold
            top_p: Top-p sampling parameter
            audio_prompt_path: Path to audio prompt file
            exaggeration: Exaggeration factor for speech
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature
            max_new_tokens: Maximum number of new tokens to generate
            max_mel_tokens: Controls the length of generated speech (IndexTTS-inspired)
        
        Returns:
            Generated audio tensor
        """
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            # IndexTTS-inspired: Use max_mel_tokens to control speech generation length
            effective_max_tokens = min(max_new_tokens, max_mel_tokens)
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=effective_max_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def generate_long_text(
        self,
        text: str,
        chunk_method: str = "sentences",
        max_chunk_size: int = 200,
        max_mel_tokens: int = 1000,
        max_text_tokens_per_sentence: int = 50,
        sentences_bucket_max_size: int = 10,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_new_tokens: int = 1000,
        optimize_memory_between_chunks: bool = True,
    ) -> torch.Tensor:
        """
        Generate audio from long text using IndexTTS-inspired chunking and memory optimization.
        
        Args:
            text: Input text to convert to speech
            chunk_method: Method to chunk text ('sentences', 'clauses', 'semantic', or 'character')
            max_chunk_size: Maximum characters per chunk
            max_mel_tokens: Controls the length of generated speech (IndexTTS-inspired)
            max_text_tokens_per_sentence: Maximum tokens per sentence for better control
            sentences_bucket_max_size: Maximum capacity for sentence bucketing
            optimize_memory_between_chunks: Whether to clear memory between chunks
            **kwargs: Other generation parameters
        
        Returns:
            Combined audio tensor
        """
        # IndexTTS-inspired intelligent chunking
        if chunk_method == "sentences":
            chunks = self._chunk_by_sentences_with_bucket(text, max_chunk_size, sentences_bucket_max_size)
        elif chunk_method == "clauses":
            chunks = chunk_text_by_clauses(text, max_chunk_size)
        elif chunk_method == "semantic":
            chunks = self._chunk_by_semantic_boundaries(text, max_chunk_size, max_text_tokens_per_sentence)
        elif chunk_method == "character":
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        else:
            raise ValueError(f"Unknown chunk_method: {chunk_method}")
        
        print(f"Processing {len(chunks)} chunks with {chunk_method} method (max_mel_tokens: {max_mel_tokens})...")
        
        audio_chunks = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            try:
                # Generate audio for this chunk
                chunk_audio = self.generate(
                    text=chunk,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    audio_prompt_path=audio_prompt_path if i == 0 else None,  # Only use prompt for first chunk
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                
                audio_chunks.append(chunk_audio)
                
                # Optimize memory between chunks
                if optimize_memory_between_chunks and i < len(chunks) - 1:
                    optimize_memory()
                    
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                # Continue with next chunk instead of failing completely
                continue
        
        if not audio_chunks:
            raise RuntimeError("Failed to generate audio for any chunks")
        
        # Concatenate all audio chunks
        combined_audio = torch.cat(audio_chunks, dim=-1)
        
        # Final memory cleanup
        if optimize_memory_between_chunks:
            optimize_memory()
        
        print(f"Successfully generated {combined_audio.shape[-1] / self.sr:.2f} seconds of audio")
        return combined_audio

    def _chunk_by_sentences_with_bucket(self, text: str, max_chunk_size: int, bucket_max_size: int) -> List[str]:
        """
        IndexTTS-inspired sentence bucketing for optimal chunking.
        """
        sentences = smart_sentence_split(text)
        chunks = []
        current_chunk = ""
        sentence_bucket = []
        
        for sentence in sentences:
            sentence_bucket.append(sentence)
            
            # Check if bucket is full or if adding would exceed chunk size
            if (len(sentence_bucket) >= bucket_max_size or 
                len(" ".join(sentence_bucket)) > max_chunk_size):
                
                # Process the bucket
                bucket_text = " ".join(sentence_bucket[:-1]) if len(sentence_bucket) > 1 else sentence_bucket[0]
                
                if current_chunk and len(current_chunk) + len(bucket_text) + 1 > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = bucket_text
                else:
                    if current_chunk:
                        current_chunk += " " + bucket_text
                    else:
                        current_chunk = bucket_text
                
                # Keep the last sentence for next bucket if we removed it
                sentence_bucket = sentence_bucket[-1:] if len(sentence_bucket) > 1 else []
        
        # Process remaining sentences in bucket
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
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk]

    def _chunk_by_semantic_boundaries(self, text: str, max_chunk_size: int, max_tokens_per_sentence: int) -> List[str]:
        """
        IndexTTS-inspired semantic boundary detection for natural speech flow.
        """
        import re
        
        sentences = smart_sentence_split(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Estimate tokens in sentence
            sentence_tokens = estimate_tokens(sentence)
            
            # If sentence is too long, split it further
            if sentence_tokens > max_tokens_per_sentence:
                # Split by clauses within the sentence
                clause_pattern = r'(?<=,)\s+|(?<=;)\s+|(?<=:)\s+'
                clauses = re.split(clause_pattern, sentence)
                
                for clause in clauses:
                    if current_chunk and len(current_chunk) + len(clause) + 1 > max_chunk_size:
                        chunks.append(current_chunk.strip())
                        current_chunk = clause
                    else:
                        if current_chunk:
                            current_chunk += " " + clause
                        else:
                            current_chunk = clause
            else:
                # Normal sentence processing
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk]

    def generate_streaming(
        self,
        text: str,
        chunk_method: str = "sentences",
        max_chunk_size: int = 200,
        max_mel_tokens: int = 1000,
        max_text_tokens_per_sentence: int = 50,
        sentences_bucket_max_size: int = 10,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        max_new_tokens: int = 1000,
        optimize_memory_between_chunks: bool = True,
    ) -> Iterator[torch.Tensor]:
        """
        Generate audio from long text in streaming fashion.
        Yields audio chunks as they are generated to reduce memory usage.
        
        Args:
            text: Input text to convert to speech
            chunk_method: Method to chunk text ('sentences', 'clauses', or 'character')
            max_chunk_size: Maximum characters per chunk
            **kwargs: Other generation parameters
        
        Yields:
            Audio tensor for each chunk
        """
        # IndexTTS-inspired intelligent chunking
        if chunk_method == "sentences":
            chunks = self._chunk_by_sentences_with_bucket(text, max_chunk_size, sentences_bucket_max_size)
        elif chunk_method == "clauses":
            chunks = chunk_text_by_clauses(text, max_chunk_size)
        elif chunk_method == "semantic":
            chunks = self._chunk_by_semantic_boundaries(text, max_chunk_size, max_text_tokens_per_sentence)
        elif chunk_method == "character":
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        else:
            raise ValueError(f"Unknown chunk_method: {chunk_method}")
        
        print(f"Streaming {len(chunks)} chunks with {chunk_method} method (max_mel_tokens: {max_mel_tokens})...")
        
        for i, chunk in enumerate(chunks):
            print(f"Streaming chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...")
            
            try:
                # Generate audio for this chunk
                chunk_audio = self.generate(
                    text=chunk,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                    audio_prompt_path=audio_prompt_path if i == 0 else None,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
                
                yield chunk_audio
                
                # Optimize memory between chunks
                if optimize_memory_between_chunks and i < len(chunks) - 1:
                    optimize_memory()
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                continue

    def estimate_memory_usage(self, text: str, max_new_tokens: int = 1000, chunk_method: str = "sentences") -> dict:
        """
        Estimate memory usage for generating audio from given text with improved accuracy.
        
        Args:
            text: Input text
            max_new_tokens: Maximum tokens to generate
            chunk_method: Chunking method to use
        
        Returns:
            Dictionary with memory estimates
        """
        text_length = len(text)
        estimated_text_tokens = estimate_tokens(text)
        estimated_speech_tokens = min(estimated_text_tokens * 8, max_new_tokens)  # More accurate estimate
        
        # More accurate memory estimates based on model architecture
        base_memory_mb = 2500  # Base model memory (T3 + S3Gen + VE)
        text_processing_mb = estimated_text_tokens * 0.02  # Text tokenization and processing
        speech_generation_mb = estimated_speech_tokens * 0.15  # Speech token generation
        audio_synthesis_mb = estimated_speech_tokens * 0.08  # Audio synthesis from tokens
        watermarking_mb = (estimated_speech_tokens * self.sr // 1000) * 0.001  # Watermarking overhead
        
        total_estimated_mb = (
            base_memory_mb + text_processing_mb + 
            speech_generation_mb + audio_synthesis_mb + watermarking_mb
        )
        
        # Calculate optimal chunk size based on available memory and method
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
            "estimated_audio_duration_seconds": estimated_speech_tokens * 0.02,  # Rough duration estimate
            "memory_efficiency_score": min(100, max(0, 100 - (total_estimated_mb - 3000) / 50))
        }