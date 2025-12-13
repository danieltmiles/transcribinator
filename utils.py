import os
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


async def normalize_audio(audio_file_path: str) -> tuple[Tensor, int]:
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
