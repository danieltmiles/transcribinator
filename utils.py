import os
import sys
import json
import ssl
from dataclasses import dataclass
from typing import Optional

import torch
import torchaudio
from pydub import AudioSegment
from starlette.websockets import WebSocket
from torch import Tensor


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
    file_extension = os.path.splitext(audio_file_path)[1].lower()
    if file_extension != ".wav":
        audio = AudioSegment.from_file(audio_file_path)
        audio.export(f"{audio_file_path}_temp.wav", format="wav")
        audio_file_path = f"{audio_file_path}_temp.wav"
    print(f"Loading audio file {audio_file_path}")
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
