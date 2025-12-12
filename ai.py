import asyncio
import json
import time
from io import StringIO

import torch
import tqdm
import torchaudio
from anyio.streams.memory import MemoryObjectSendStream
from speechbrain.pretrained import SpeakerRecognition
from pydub import AudioSegment
from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

from speaker_counting_parallel import find_optimal_speakers_multi_metric_parallel

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
if torch.mps.is_available():
    device = "mps"

try:
    import whisper
except AttributeError:
    raise ImportError(
        "Please install the correct Whisper package using: pip install openai-whisper\n"
        "If you have the 'whisper' package installed, first uninstall it with: pip uninstall whisper"
    )
class ModelHaver(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
            cls._instance.whisper_model = whisper.load_model("large")
            # Put any initialization here.
        return cls._instance

import numpy as np
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
    seconds = int(seconds)
    one_hour = 60 * 60
    one_minute = 60
    hours = int(seconds / one_hour)
    remaining_seconds = seconds % one_hour
    minutes = int(remaining_seconds / one_minute)
    remaining_seconds = remaining_seconds % one_minute
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"

def find_optimal_speakers(embeddings, max_speakers=30):
    """
    Find optimal number of speakers using silhouette analysis and clustering metrics.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    
    print("Finding optimal number of speakers...")
    best_score = -1
    best_n_speakers = 2
    
    # Try different numbers of speakers
    for n_speakers in range(2, min(max_speakers + 1, len(embeddings))):
        clustering = AgglomerativeClustering(n_clusters=n_speakers, linkage='average')
        labels = clustering.fit_predict(embeddings)
        
        # Calculate silhouette score (higher is better)
        score = silhouette_score(embeddings, labels)
        print(f"  {n_speakers} speakers: silhouette score = {score:.3f}")
        
        if score > best_score:
            best_score = score
            best_n_speakers = n_speakers
    
    print(f"Optimal number of speakers: {best_n_speakers} (score: {best_score:.3f})")
    return best_n_speakers

async def process_audio(audio_file_path: str, min_segment_length: float, progress_send_stream: MemoryObjectSendStream[dict], transcript_send_stream: MemoryObjectSendStream[str]):
    """
    Process audio file for speaker identification and transcription

    Parameters:
    - audio_file_path: Path to the audio file
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

    # whisper needs a sample rate of 16000
    if sr != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        sr = 16000

    start = time.time()
    print("Loading models...")
    speaker_recognition = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    
    try:
        whisper_model = ModelHaver.instance().whisper_model
    except AttributeError:
        raise ImportError(
            "Error loading Whisper model. Please ensure you have openai-whisper installed, not whisper"
        )
    end = time.time()
    await asyncio.sleep(0.1)
    print(f"loaded models in {end - start} seconds")

    # Parameters for segmentation
    window_size = int(sr * min_segment_length * 3)  # 3x min_segment_length windows
    stride = int(sr * min_segment_length * 2)       # 2x min_segment_length stride
    
    # Separate segments for diarization (speaker identification) and transcription
    diarization_segments = []
    all_segments = []
    embeddings = []
    
    print(f"Processing audio segments from 0 through {len(signal)} with stride {stride}")
    await progress_send_stream.send({"stage": "diarization", "progress": 0})
    diar_last_progress = 0
    diar_total_iters = max(1, len(range(0, len(signal), stride)))
    for idx, start in enumerate(tqdm.tqdm(range(0, len(signal), stride), total=diar_total_iters)):
        end = min(start + window_size, len(signal))
        segment = signal[start:end]
        
        # Always add to transcription segments (covers all audio)
        all_segments.append({
            'start': start / sr,
            'end': end / sr,
            'audio': segment
        })
        
        # Only add to diarization if segment is long enough for good speaker embeddings
        if len(segment) >= sr * min_segment_length:
            embedding = speaker_recognition.encode_batch(segment.unsqueeze(0))
            embeddings.append(embedding.squeeze().cpu().numpy())
            
            diarization_segments.append({
                'start': start / sr,
                'end': end / sr,
                'audio': segment,
                'segment_index': len(all_segments) - 1  # Link back to all_segments
            })
        
        current_progress = int((idx / diar_total_iters) * 100)
        if current_progress > diar_last_progress:
            await progress_send_stream.send({"stage": "diarization", "progress": current_progress})
            await asyncio.sleep(0.05)
            diar_last_progress = current_progress
    await progress_send_stream.send({"stage": "diarization", "progress": 100})
    await asyncio.sleep(0.1)
    
    # print(f"Clustering speakers")
    # embeddings = np.array(embeddings)
    # from sklearn.cluster import AgglomerativeClustering
    # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.05)
    # labels = clustering.fit_predict(embeddings)
    # num_speakers = len(np.unique(labels))
    # print(f"num speakers: {num_speakers}")
    print(f"Performing speaker diarization on {len(embeddings)} segments")
    embeddings = np.array(embeddings)
    
    # Normalize embeddings
    from sklearn.preprocessing import normalize
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
    
    # Use optimal number of speakers detection
    # num_speakers = find_optimal_speakers(embeddings_normalized, max_speakers=10)
    num_speakers, labels = find_optimal_speakers_multi_metric_parallel(
        embeddings_normalized,
        max_speakers=250,
        n_processes=None  # Auto-detect optimal number of processes
    )
    print(f"Estimated number of speakers: {num_speakers}")
    
    # Perform final clustering with determined number of speakers
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(n_clusters=num_speakers, linkage='average')
    labels = clustering.fit_predict(embeddings_normalized)
    
    # Assign speaker labels to segments and print some statistics
    unique_speakers = np.unique(labels)
    print(f"Final speaker count: {len(unique_speakers)}")
    
    # Print segment distribution per speaker
    for speaker_id in unique_speakers:
        speaker_segments = np.sum(labels == speaker_id)
        percentage = (speaker_segments / len(labels)) * 100
        print(f"Speaker {speaker_id}: {speaker_segments} segments ({percentage:.1f}%)")
    
    # Create speaker assignment for all segments based on temporal overlap
    def assign_speaker_to_segment(segment_start, segment_end):
        """Assign speaker to a segment based on temporal overlap with diarization segments"""
        best_overlap = 0
        best_speaker = "Speaker_0"  # default fallback
        
        for i, diar_seg in enumerate(diarization_segments):
            # Calculate overlap between transcription segment and diarization segment
            overlap_start = max(segment_start, diar_seg['start'])
            overlap_end = min(segment_end, diar_seg['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = f"Speaker_{labels[i]}"
        
        return best_speaker
    
    # Process segments with speaker labels and transcription
    raw_segments = []
    import os
    # temp_dir = "temp_segments"
    # os.makedirs(temp_dir, exist_ok=True)
    
    print(f"Transcribing {len(all_segments)} segments...")
    await progress_send_stream.send({"stage": "transcription", "progress": 0})
    last_progress = 0
    await asyncio.sleep(0.1)
    results = []
    for i, segment in tqdm.tqdm(enumerate(all_segments), total=len(all_segments)):
        current_progress = int((i / len(all_segments)) * 100)
        if current_progress > last_progress:
            await progress_send_stream.send({"stage": "transcription", "progress": current_progress})
            await asyncio.sleep(0.1)
            last_progress = current_progress
        
        # Assign speaker based on temporal overlap with diarization segments
        speaker = assign_speaker_to_segment(segment['start'], segment['end'])
        
        # temp_file = os.path.join(temp_dir, f"segment_{i}.wav")
        # sf.write(temp_file, segment['audio'].numpy(), sr)

        if i % 10 == 0:
            await asyncio.sleep(0.1)
        result = whisper_model.transcribe(
            segment['audio'],
            temperature=0.2,  # Low temperature. Conservative, but small creative freedom
            language='en',      # Explicitly specify English
            # initial_prompt="Um, uh, and other hesitation sounds should be transcribed as such.",
            word_timestamps=True,
        )
        text = ""
        for segment_segment in result.get("segments", []):
            segment_words = segment_segment.get("words", [])
            for j, word in enumerate(segment_words):
                word_word = word.get("word", "")
                probability = word.get("probability", 0.0)
                word_duration = word["end"] - word["start"]
                if word_duration < 0.1:
                    continue
                # if this is the first or last word in the segment and it is unusually
                # short, there is a chance it has been cut off in the middle and we
                # should allow our overlapping to pick it up in its entirity in a different
                # segment.
                if word_duration < 0.25 and (j == 0 or j == len(segment_words) - 1):
                    probability *= 0.5
                if probability > 0.3:
                    text += word_word
        text = text.strip()

        # os.remove(temp_file)
        
        if text:  # Only add segments with text
            raw_segments.append({
                'speaker': speaker,
                'start': segment['start'],
                'end': segment['end'],
                'text': text
            })
            results.append(result)
    with open("results.json", "w") as fl:
        json.dump(results, fl, indent=4)
    
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
                'text': " ".join(cleaned_texts)
            })

    del whisper_model
    del ModelHaver._instance
    ModelHaver._instance = None
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Force FP16
        # device_map="cuda:0",  # Explicitly map to GPU if you have one
        device_map="auto",
        trust_remote_code=True,
        max_memory={0: "22GiB"},  # Reserve a bit less than total VRAM
        # low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    await progress_send_stream.send({"stage": "transcription", "progress": 100})

    await progress_send_stream.send({"stage": "cleanup", "progress": 0})
    for i, segment in enumerate(transcript):
        current_progress = int((i / max(1, len(transcript))) * 100)
        await progress_send_stream.send({"stage": "cleanup", "progress": current_progress})
        await asyncio.sleep(0.1)
        print(f"\ntext:\n{segment['text']}\n")
        cleaned_segment_text = llm_clean(segment["text"], model, tokenizer)
        print(f"\ncleaned text:\n{cleaned_segment_text}\n")
        transcript[i]["text"] = cleaned_segment_text
    
    # Clean up temp directory
    # os.rmdir(temp_dir)

    await progress_send_stream.send({"stage": "cleanup", "progress": 100})
    await transcript_send_stream.send(produce_transcript(transcript))
    LOGGER.info(f"{transcript}")
    
    # Clean up all models from memory
    del model
    del tokenizer  
    del speaker_recognition
    unload_models_from_memory()
    
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

def generate_from_prompt(prompt: str, model: Qwen2ForCausalLM, tokenizer: Qwen2TokenizerFast) -> str:
    test_encoding = tokenizer(prompt, return_tensors="pt")
    prompt_length = test_encoding.input_ids.size(1)
    max_length = min(prompt_length + 200, 32000)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_length,
    )
    attention_mask = encoded['attention_mask'].to(device)
    outputs = model.generate(
        encoded.input_ids.to(device),
        attention_mask=attention_mask,
        max_new_tokens=prompt_length,  # Allow some buffer for expanded text
        temperature=0.2,  # Lower temperature for more consistent outputs
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.2
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=False)

def unload_models_from_memory():
    """
    Comprehensive function to unload all models from memory and free GPU/CPU resources
    """
    import gc
    
    print("Unloading models from memory...")
    
    # Clear ModelHaver instance (Whisper model)
    if hasattr(ModelHaver, '_instance') and ModelHaver._instance is not None:
        if hasattr(ModelHaver._instance, 'whisper_model'):
            del ModelHaver._instance.whisper_model
        del ModelHaver._instance
        ModelHaver._instance = None
        print("- Whisper model unloaded")
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("- CUDA memory cache cleared")
    
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("- MPS memory cache cleared")
    
    print("Memory cleanup completed")

def llm_clean(text: str, model: Qwen2ForCausalLM, tokenizer: Qwen2TokenizerFast) -> str | None:
    prompt_template = """You are a transcript editor. The following text was transcribed from an audio recording by an unskilled person who
made errors grouping the words into sentences and sometimes typed a word or phrase multiple times, when the speaker did not say it.
Please identify and correct these transcriber errors without altering the speakers' original language. Follow these rules then mark
the end of the cleaned up text with "END OUTPUT TEXT"

Rules:
1. Remove only the most obvious speech disfluencies like "uh", "um", "er". Keep redundancies if they seem intentional.
2. Correct homophones (two, too, to) but maintain the original spelling if it appears to be a typo or error.
3. Maintain the exact wording of the speaker, including errors and run-on sentences. Create an incorrect sentence if necessary to preserve the speaker's intended meaning.
4. Preserve repetitions that appear to be intentional, even if they could be considered redundant or awkward.
5. Avoid rephrasing, condensing, or adding nuance to the content even when the speaker's intended message is unclear.
6. Preserve the exact wording used by the speaker, even if it creates ambiguity or awkwardness.
7. Capitalize proper nouns correctly, but leave other capitalization inconsistencies intact.

BEGIN INPUT TEXT:
{text}
END INPUT TEXT

BEGIN OUTPUT TEXT:
"""
    text = text.replace("...", "")
    prompt = prompt_template.format(text=text)
    full_output = ""
    end_delimiter = "END OUTPUT TEXT"
    while end_delimiter not in full_output[len(prompt):]:
        full_output = generate_from_prompt(prompt, model, tokenizer)
        begin_delim = "BEGIN OUTPUT TEXT:"
        begin_indexes = [i for i in range(len(full_output)) if full_output.startswith(begin_delim, i)]
        end_delim = "END OUTPUT TEXT"
        end_indexes = [i for i in range(len(full_output)) if full_output.startswith(end_delim, i)]
        if len(end_indexes) > 1:
            chunk = full_output[begin_indexes[-1] + len(begin_delim):end_indexes[1]]
            return chunk.strip()
