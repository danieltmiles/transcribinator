import os
import subprocess
import sys
import json
import ssl
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio
from pydub import AudioSegment
from starlette.websockets import WebSocket
from torch import Tensor
from transformers import AutoTokenizer


@dataclass
class TranscriptionJob:
    job_id: str
    filename: str
    human_readable_filename: str
    status: str
    progress: int
    websocket: Optional[WebSocket] = None
    transcript: Optional[str] = None
    error: Optional[str] = None

# Iterator to create audio segments from diarization timestamps
def diarized_segment_iter(signal, diarization, sample_rate):
    """
    Yields audio segments based on diarization timestamps.

    Args:
        signal: Audio signal tensor
        diarization: DiarizeOutput object with speaker_diarization attribute
        sample_rate: Sample rate of the audio

    Yields:
        dict with 'audio', 'start', 'end', and 'speaker' keys
    """
    for diar_seg in diarization.speaker_diarization:
        start_time = diar_seg[0].start
        end_time = diar_seg[0].end
        speaker = diar_seg[1]

        # Convert time to sample indices
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        # Extract audio segment
        audio_segment = signal[start_sample:end_sample]

        yield {
            'audio': audio_segment.squeeze().numpy(),
            'start': start_time,
            'end': end_time,
            'speaker': speaker
        }

# Create speaker assignment for all segments based on temporal overlap
def assign_speaker_to_segment(diarization, segment_start, segment_end):
    """Assign speaker to a segment based on temporal overlap with diarization segments"""
    best_overlap = 0
    best_speaker = "Speaker_Unknown"  # default fallback

    for i, (diar_seg, speaker) in enumerate(diarization.speaker_diarization):
        # Calculate overlap between transcription segment and diarization segment
        overlap_start = max(segment_start, diar_seg.start)
        overlap_end = min(segment_end, diar_seg.end)
        overlap = max(0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker

    return best_speaker


def normalize_audio(audio_file_path: str) -> tuple[Tensor, int]:
    import io
    import numpy as np
    
    file_extension = os.path.splitext(audio_file_path)[1].lower()
    print(f"Loading audio file {audio_file_path}")
    
    if file_extension != ".wav":
        # Load audio using pydub and convert to in-memory WAV
        audio = AudioSegment.from_file(audio_file_path)
        
        # Export to in-memory bytes buffer as WAV
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)
        
        # Load from in-memory buffer using torchaudio
        signal, sr = torchaudio.load(wav_buffer)
    else:
        # Directly load WAV files
        signal, sr = torchaudio.load(audio_file_path)
    
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    signal = signal.squeeze()

    # whisper needs a sample rate of 16000
    if sr != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        sr = 16000
    return signal, sr


def load_config(config_file):
    """Load configuration from JSON file.
    
    Expected JSON format:
    {
        "work_queue": "queue_name",
        "host": "localhost",
        "port": 5672,
        "username": "guest",
        "password": "guest"
    }
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate required fields
        required_fields = ['work_queue', 'host', 'port', 'username', 'password']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in config file: {', '.join(missing_fields)}")
        
        # Ensure port is an integer
        config['port'] = int(config['port'])
        
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


def create_ssl_context(cert_file='server_certificate.pem', verify=True):
    """Create SSL context for RabbitMQ connections.
    
    Args:
        cert_file: Path to the certificate file
        verify: Whether to verify certificates (set to False for self-signed)
    
    Returns:
        ssl.SSLContext configured for RabbitMQ
    """
    ssl_context = ssl.create_default_context(cafile=cert_file)
    if not verify:
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
    return ssl_context


def load_quantized_llm_model(device: str, model_path: str = None):
    """
    Load the LLM model for speaker identification.

    Supports different hardware backends:
    - MPS (Apple Metal): Uses MLX library for optimal performance
    - CUDA (NVIDIA): Uses llama-cpp-python for GPU acceleration
    - CPU: Not optimized for quantized models

    Args:
        model_path: Path to the model (optional, uses default if not provided)

    Returns:
        tuple: (model, tokenizer) - tokenizer may be None for llama-cpp
    """
    if device == "mps" and "MLX" in model_path:
        # Import MLX libraries for Apple MPS hardware
        try:
            import mlx.core as mx
            from mlx_lm import load
            print("Loading MLX model for MPS device...")
            model_name = model_path or "./Qwen3-32B-MLX-4bit"
            model, tokenizer = load(model_name)
            model_type = "mlx"
            print(f"Successfully loaded MLX model: {model_name}")
            return model, tokenizer, model_type
        except ImportError:
            print("MLX libraries not available. Install mlx-lm for MPS support.")
            raise
        except Exception as e:
            print(f"Error loading MLX model: {e}")
            raise

    elif device == "cuda" or (device == "mps" and "GGUF" in model_path.upper()):
        # Import llama-cpp-python for NVIDIA CUDA hardware
        try:
            from llama_cpp import Llama
            print(f"Loading GGUF model for {device} device...")
            model_path = model_path or "./Qwen3-32B-Q4_K_M.gguf"

            # Temporarily suppress llama.cpp warnings
            os.environ['LLAMA_LOG_DISABLE'] = '1'
            model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Use all GPU layers
                # n_ctx=8192,  # Context window size
                n_ctx=10240,  # Context window size
                verbose=False
            )
            tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-32B-Think")

            # Re-enable logging after model load
            if 'LLAMA_LOG_DISABLE' in os.environ:
                del os.environ['LLAMA_LOG_DISABLE']

            model_type = "llamacpp"
            print(f"Successfully loaded GGUF model: {model_path}")
            return model, tokenizer, model_type  # llama-cpp handles tokenization internally
        except ImportError:
            print("llama-cpp-python not available. Install llama-cpp-python for CUDA support.")
            raise
        except Exception as e:
            print(f"Error loading GGUF model: {e}")
            raise

    else:
        print("CPU inference not optimized for quantized models. Please use an MPS or CUDA device.")
        raise RuntimeError("Unsupported device: CPU")


def quantized_generate_from_prompt(prompt: str, model, tokenizer, model_type, max_tokens: int = 12288) -> str:
    """
    Generate text from a prompt using the appropriate backend.

    Handles both MLX and llama-cpp model types with their respective APIs.
    For MLX, uses stream_generate and stops naturally on EOS token.

    Args:
        prompt: The input prompt
        model: The loaded model
        tokenizer: The tokenizer (None for llama-cpp)
        max_tokens: Maximum number of tokens to generate (default: 8192, acts as safety limit)

    Returns:
        str: The generated text
    """
    if model_type == "mlx":
        # MLX streaming generation - stops naturally on EOS token
        try:
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler

            # Get EOS token ID from tokenizer
            eos_token_id = tokenizer.eos_token_id

            # Accumulate the generated text
            generated_text = ""

            for token_info in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,  # Acts as safety limit
                sampler=make_sampler(
                    temp=0.7,
                    # top_p=0.9,
                    # top_k=40,
                    # min_p=0.0,
                    # min_tokens_to_keep=1,
                    # xtc_probability=0.0,
                    # xtc_threshold=0.0
                ),
                max_kv_size=32768,
            ):
                # stream_generate yields GenerationResponse objects
                generated_text += token_info.text

                # Check if we hit EOS token
                if token_info.token == eos_token_id:
                    print(f"EOS token detected, stopping generation. Generated {token_info.generation_tokens} tokens, {token_info.generation_tps} per second")
                    break

            return generated_text.strip()
        except Exception as e:
            print(f"MLX generation error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    elif model_type == "llamacpp":
        # llama-cpp-python streaming generation - stops naturally on EOS token
        try:
            generated_text = ""

            # Stream the response with stream=True
            stream = model(
                prompt,
                max_tokens=max_tokens,  # Acts as safety limit
                temperature=0.7,
                # top_p=0.9,
                # top_k=40,
                repeat_penalty=1.1,
                echo=False,
                stream=True  # Enable streaming
            )

            # Accumulate tokens from the stream
            for chunk in stream:
                # Each chunk has the structure: {'choices': [{'text': '...', 'finish_reason': ...}]}
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    choice = chunk['choices'][0]
                    token_text = choice.get('text', '')
                    generated_text += token_text

                    # Check for finish reason (None, 'stop', 'length')
                    finish_reason = choice.get('finish_reason', None)
                    if finish_reason == 'stop':
                        print("EOS token detected, stopping generation.")
                        break
                    elif finish_reason == 'length':
                        print("Max tokens reached.")
                        break

            return generated_text.strip()
        except Exception as e:
            print(f"GGUF generation error: {e}")
            import traceback
            traceback.print_exc()
            return ""

    else:
        print(f"Unknown model type: {model_type}")
        return ""
