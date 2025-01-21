import asyncio
from io import StringIO

import torch
import tqdm
import torchaudio
from anyio import WouldBlock
from anyio.streams.memory import MemoryObjectSendStream
from speechbrain.pretrained import SpeakerRecognition
from pydub import AudioSegment

try:
    import whisper
except AttributeError:
    raise ImportError(
        "Please install the correct Whisper package using: pip install openai-whisper\n"
        "If you have the 'whisper' package installed, first uninstall it with: pip uninstall whisper"
    )
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
import warnings

import logging
LOGGER = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

def similar(a, b, threshold=0.85):
    """Check if two strings are similar using sequence matcher"""
    return SequenceMatcher(None, a, b).ratio() > threshold

def clean_overlapping_text(text_list):
    """Remove overlapping phrases from consecutive segments"""
    if not text_list:
        return text_list
    
    cleaned = [text_list[0]]
    for current_text in text_list[1:]:
        last_text = cleaned[-1]
        
        # Check if current_text is completely contained in last_text
        if current_text in last_text:
            continue
            
        # Check if last_text is completely contained in current_text
        if last_text in current_text:
            cleaned[-1] = current_text
            continue
            
        # Check for partial overlap
        words_current = current_text.split()
        words_last = last_text.split()
        
        # Look for overlapping phrases
        overlap_found = False
        for i in range(min(len(words_last), len(words_current))):
            last_phrase = " ".join(words_last[-i-1:])
            current_phrase = " ".join(words_current[:i+1])
            if similar(last_phrase, current_phrase):
                # Remove overlapping part from current text
                cleaned.append(" ".join(words_current[i+1:]))
                overlap_found = True
                break
        
        if not overlap_found:
            cleaned.append(current_text)
    
    return [text for text in cleaned if text.strip()]

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(datetime.fromtimestamp(seconds).strftime('%H:%M:%S'))

async def process_audio(audio_file_path: str, num_speakers: int, min_segment_length: float, progress_send_stream: MemoryObjectSendStream[int], transcript_send_stream: MemoryObjectSendStream[str]):
    """
    Process audio file for speaker diarization and transcription

    Parameters:
    - audio_file_path: Path to the audio file
    - num_speakers: Expected number of speakers
    - min_segment_length: Minimum segment length in seconds
    """
    import os
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
    print("Loading models...")
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.mps.is_available():
        device = "mps"
    speaker_recognition = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    try:
        whisper_model = whisper.load_model("large")
    except AttributeError:
        raise ImportError(
            "Error loading Whisper model. Please ensure you have openai-whisper installed, not whisper"
        )
    await asyncio.sleep(0.1)

    # Parameters for segmentation
    window_size = int(sr * min_segment_length * 3)  # 3x min_segment_length windows
    stride = int(sr * min_segment_length * 2)       # 2x min_segment_length stride
    
    segments = []
    embeddings = []
    
    print(f"Processing audio segments from 0 through {len(signal)} with stride {stride}")
    for start in tqdm.tqdm(range(0, len(signal), stride), total=len(signal)//stride):
        end = min(start + window_size, len(signal))
        segment = signal[start:end]
        
        if len(segment) < sr * min_segment_length:
            continue
            
        embedding = speaker_recognition.encode_batch(segment.unsqueeze(0))
        embeddings.append(embedding.squeeze().cpu().numpy())
        
        segments.append({
            'start': start / sr,
            'end': end / sr,
            'audio': segment
        })
    await asyncio.sleep(0.1)
    
    print(f"Clustering speakers (target: {num_speakers} speakers)...")
    embeddings = np.array(embeddings)
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=num_speakers)
    labels = clustering.fit_predict(embeddings)
    
    # Process segments with speaker labels and transcription
    raw_segments = []
    import os
    import soundfile as sf
    temp_dir = "temp_segments"
    os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Transcribing {len(segments)} segments...")
    last_progress = 0
    await asyncio.sleep(0.1)
    for i, segment in tqdm.tqdm(enumerate(segments), total=len(segments)):
        current_progress = int((i / len(segments)) * 100)
        if current_progress > last_progress:
            await progress_send_stream.send(current_progress)
            await asyncio.sleep(0.1)
            last_progress = current_progress
        speaker = f"Speaker_{labels[i]}"
        
        temp_file = os.path.join(temp_dir, f"segment_{i}.wav")
        sf.write(temp_file, segment['audio'].numpy(), sr)

        await asyncio.sleep(0.1)
        result = whisper_model.transcribe(temp_file)
        text = result["text"].strip()
        
        os.remove(temp_file)
        
        if text:  # Only add segments with text
            raw_segments.append({
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'text': text
            })
    
    # Merge segments from the same speaker and clean overlapping text
    transcript = []
    current_speaker = None
    current_texts = []
    current_start = None
    
    for segment in raw_segments:
        if segment['speaker'] != current_speaker:
            if current_speaker and current_texts:
                cleaned_texts = clean_overlapping_text(current_texts)
                if cleaned_texts:
                    transcript.append({
                        'speaker': current_speaker,
                        'start': format_timestamp(current_start),
                        'end': format_timestamp(segment['start']),
                        'text': ' '.join(cleaned_texts)
                    })
            current_speaker = segment['speaker']
            current_texts = [segment['text']]
            current_start = segment['start']
        else:
            current_texts.append(segment['text'])
    
    # Add final segment
    if current_texts:
        cleaned_texts = clean_overlapping_text(current_texts)
        if cleaned_texts:
            transcript.append({
                'speaker': current_speaker,
                'start': format_timestamp(current_start),
                'end': format_timestamp(raw_segments[-1]['end']),
                'text': ' '.join(cleaned_texts)
            })
    
    # Clean up temp directory
    os.rmdir(temp_dir)

    await progress_send_stream.send(100)
    await transcript_send_stream.send(produce_transcript(transcript))
    LOGGER.info(f"{transcript}")
    return transcript

def save_transcript(transcript, output_file):
    """Save the transcript to a file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in transcript:
            f.write(f"[{entry['start']} - {entry['end']}] {entry['speaker']}:\n")
            f.write(f"{entry['text']}\n\n")


def produce_transcript(transcript) -> str:
    """Save the transcript to a file"""
    f = StringIO()
    for entry in transcript:
        f.write(f"[{entry['start']} - {entry['end']}] {entry['speaker']}:\n")
        f.write(f"{entry['text']}\n\n")
    return f.getvalue()

# Example usage
if __name__ == "__main__":
    audio_file = "example.mp3"
    output_file = "transcript.txt"
    
    print("Processing audio file...")
    transcript = process_audio(
        audio_file,
        num_speakers=3,
        min_segment_length=1.0
    )
    save_transcript(transcript, output_file)
    print(f"Transcript saved to {output_file}")
