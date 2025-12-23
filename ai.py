import asyncio
import json
import os
import ssl
import time
import pickle
import base64
import uuid
import hashlib
from contextlib import contextmanager
from io import StringIO
from queue import Queue
from typing import Any

import anyio
import aio_pika
import torch
from numpy import ndarray

from shared_disks import WebDavRemoteStorage

if torch.cuda.is_available():
    _original_load = torch.load
    torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
import tqdm
from anyio.streams.memory import MemoryObjectSendStream
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput
from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast, AutoModelForCausalLM, AutoTokenizer

from speaker_counting_parallel import find_optimal_speakers_multi_metric_parallel
from utils import diarized_segment_iter, assign_speaker_to_segment, normalize_audio, create_ssl_context


def load_hf_token(token_file="hf_token.txt"):
    """Load HuggingFace token from a file.
    
    Args:
        token_file: Path to the token file (default: hf_token.txt)
        
    Returns:
        str: The token string
        
    Raises:
        FileNotFoundError: If the token file doesn't exist
        ValueError: If the token file is empty
    """
    token_path = os.path.join(os.path.dirname(__file__), token_file)
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Token file not found: {token_path}")
    
    with open(token_path, 'r') as f:
        token = f.read().strip()
    
    if not token:
        raise ValueError(f"Token file is empty: {token_path}")
    
    return token


class TqdmProgressHook:
    """Custom hook using tqdm for progress display"""

    def __init__(self, important_step_name: str = None):
        self.pbar = None
        self.step_name = None
        self.important_step_name = important_step_name
        self.progress_queue = Queue()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.pbar is not None:
            self.pbar.close()

    def __call__(
            self,
            step_name,
            step_artifact,
            file=None,
            total=None,
            completed=None,
    ):
        if completed is None:
            completed = total = 1

        # Create new progress bar when step changes
        if step_name != self.step_name:
            if self.pbar is not None:
                self.pbar.close()
            self.step_name = step_name
            self.pbar = tqdm.tqdm(total=total, desc=step_name, unit="it")

        if self.important_step_name == None or step_name == self.important_step_name:
            progress_percentage = int(completed / total * 100)
            # Put progress update in queue instead of creating orphaned task
            self.progress_queue.put({"stage": "diarization", "progress": progress_percentage})

        # Update progress
        if self.pbar is not None:
            self.pbar.n = completed
            self.pbar.total = total
            self.pbar.refresh()

    async def flush_progress(self):
        """Send all queued progress updates to the stream"""
        while not self.progress_queue.empty():
            try:
                progress_data = self.progress_queue.get_nowait()
            except:
                break


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


def sliding_window(iterable, n, stride=1):
    """
    Create sliding windows of size n with specified stride.
    
    Args:
        iterable: The sequence to create windows from
        n: Window size (number of items per window)
        stride: Step size between windows (stride=1 gives overlapping windows,
                stride=n gives non-overlapping windows)
    
    Yields:
        Tuples of size n (or smaller for the last window if not enough items)
    
    Examples:
        stride=1 gives overlapping windows (1-7, 2-8, 3-9...)
        stride=3 gives less overlapping windows (1-7, 4-11, 7-14...)
    """
    items = list(iterable)
    for i in range(0, len(items) - n + 1, stride):
        yield tuple(items[i:i + n])


def format_segment_for_speaker_identification(segment: dict[str, Any]) -> str:
    """
    Format a transcript segment for speaker identification.
    
    Args:
        segment: A transcript segment with speaker, start, end, and text fields
    
    Returns:
        Formatted string in the format "[HH:MM:SS - HH:MM:SS] Speaker_XX:\ntext"
    """
    start = segment.get('start', 0)
    end = segment.get('end', 0)
    speaker = segment.get('speaker', 'Unknown')
    text = segment.get('text', '')
    
    start_str = format_timestamp(start)
    end_str = format_timestamp(end)
    
    return f"[{start_str} - {end_str}] {speaker}:\n{text}"


async def send_diarization_job_to_queue(audio_file_path: str, rabbitmq_config: dict[str, Any]) -> None | DiarizeOutput:
    """Send diarization job to RabbitMQ and wait for result using aio_pika."""
    from utils import create_ssl_context
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    # TODO: make this robust against weird filenames
    file_extension = audio_file_path.split(".")[-1]
    remote_file_path = f"{job_id}.{file_extension}"

    # Calculate SHA256 checksum of the audio file
    sha256_hash = hashlib.sha256()
    with open(audio_file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    audio_file_sha256 = sha256_hash.hexdigest()

    webdav_remote_storage = WebDavRemoteStorage("https://webdav.doodledome.org", "dmiles", "secret123")
    print("uploading file to shared storage")
    webdav_remote_storage.send(audio_file_path, remote_file_path)
    print("finished uploading file to shared storage")

    work_queue = "pyannote/speaker-diarization-community-1"
    response_queue = f"{work_queue}-{uuid.uuid4()}"

    # Create job message
    job_message = {
        'job_id': job_id,
        'remote_file_type': "webdav",
        "remote_file_info": {
            "server": "https://webdav.doodledome.org",
            "filename": remote_file_path,
        },
        'reply_to': response_queue,
        'sha256sum': audio_file_sha256,
    }
    
    # Connect to RabbitMQ
    ssl_context = create_ssl_context()
    connection = await aio_pika.connect_robust(
        host=rabbitmq_config['host'],
        port=rabbitmq_config['port'],
        login=rabbitmq_config['username'],
        password=rabbitmq_config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )
    
    result = None
    
    async with connection:
        channel = await connection.channel()
        
        # Declare queues
        work_queue_obj = await channel.declare_queue(work_queue, durable=True)
        response_queue_obj = await channel.declare_queue(response_queue, durable=True)
        
        # Send job to work queue
        await channel.default_exchange.publish(
            aio_pika.Message(body=json.dumps(job_message).encode()),
            routing_key=work_queue,
        )
        
        print(f"Sent diarization job {job_id} to queue {work_queue}")
        print(f"Waiting for diarization result for job {job_id}...")
        
        # Wait for response
        async with response_queue_obj.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    response = json.loads(message.body.decode())
                    
                    if response.get('job_id') == job_id:
                        if response.get('status') == 'success':
                            # Deserialize diarization result
                            diarization_encoded = response.get('diarization')
                            diarization_bytes = base64.b64decode(diarization_encoded)
                            result = pickle.loads(diarization_bytes)
                            print(f"Received diarization result for job {job_id}")
                        else:
                            error = response.get('error', 'Unknown error')
                            raise RuntimeError(f"Diarization job failed: {error}")
                        break
        
        # Delete the reply-to queue now that we're done consuming
        await channel.queue_delete(response_queue)
    
    if result is None:
        raise RuntimeError("Failed to receive diarization result")
    
    return result

async def send_whisper_jobs(
    audio_file_path: str,
    rabbitmq_config: dict[str, Any],
    diarization: DiarizeOutput,
    segment_stream_send: anyio.streams.memory.MemoryObjectSendStream[dict[str, Any]],
) -> None:
    """
    Send whisper jobs and stream results as they arrive via memory object stream.
    
    Args:
        audio_file_path: Path to audio file
        rabbitmq_config: RabbitMQ configuration
        diarization: Diarization output
        segment_stream_send: Stream to send completed segments to
    """
    from utils import create_ssl_context
    
    work_queue = "whisper/large"
    response_queue = f"{work_queue}-{uuid.uuid4()}"
    signal, sr = normalize_audio(audio_file_path)
    segments_list = list(diarized_segment_iter(signal, diarization, sr))
    job_id = str(uuid.uuid4())
    received_segments = []
    total_segments = len(segments_list)
    out_of_order_responses = []

    current_speaker = None
    accumulated_segment = None
    
    # Connect to RabbitMQ
    ssl_context = create_ssl_context()
    connection = await aio_pika.connect_robust(
        host=rabbitmq_config['host'],
        port=rabbitmq_config['port'],
        login=rabbitmq_config['username'],
        password=rabbitmq_config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )
    
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        
        # Declare queues
        work_queue_obj = await channel.declare_queue(work_queue, durable=True)
        response_queue_obj = await channel.declare_queue(response_queue, durable=True)
        
        # Send all jobs first
        for i, segment in enumerate(segments_list):
            job_message = {
                'job_id': job_id,
                'reply_to': response_queue,
                'audio_segment': {
                    'audio': segment['audio'].tolist(),
                    'start': segment["start"],
                    "end": segment["end"],
                    "speaker": segment["speaker"],
                },
                'speaker': assign_speaker_to_segment(diarization, segment['start'], segment['end']),
                'temperature': 0.2,
                'language': 'en',
                'word_timestamps': True,
                'segment_count': i,
                'total_segments': total_segments,
            }
            await channel.default_exchange.publish(
                aio_pika.Message(body=json.dumps(job_message).encode()),
                routing_key=work_queue,
            )
        
        print(f"Sent {total_segments} whisper jobs, waiting for responses...")
        
        # Now consume responses
        async with response_queue_obj.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    response = json.loads(message.body.decode())
                    
                    if response.get("job_id") != job_id:
                        continue
                        # TODO: nack
                    
                    segment_count = response.get("segment_count")
                    if segment_count is None:
                        print("bad message, no segment count, refusing to process")
                        continue
                    print(f"Received segment {segment_count}")
                    
                    # Track this segment
                    if len(received_segments) == 0 or segment_count == received_segments[-1] + 1:
                        received_segments.append(segment_count)
                    else:
                        out_of_order_responses.append(response)
                        continue

                    accumulated_segment, current_speaker = await process_segment(
                        accumulated_segment,
                        current_speaker,
                        response,
                        segment_count,
                        segment_stream_send,
                    )

                    # reconcile any out-of-order segments
                    out_of_order_responses = sorted(out_of_order_responses,
                                                   key=lambda x: x["segment_count"])
                    unreconciled_out_of_order_segments = []
                    for out_of_order_response in out_of_order_responses:
                        out_of_order_segment_number = out_of_order_response["segment_count"]
                        if out_of_order_segment_number == received_segments[-1] + 1:
                            received_segments.append(out_of_order_segment_number)
                            accumulated_segment, current_speaker = await process_segment(
                                accumulated_segment,
                                current_speaker,
                                out_of_order_response,
                                out_of_order_segment_number,
                                segment_stream_send,
                            )
                        else:
                            unreconciled_out_of_order_segments.append(out_of_order_response)
                    out_of_order_responses = unreconciled_out_of_order_segments

                    # Stop when all segments received
                    if len(received_segments) == total_segments:
                        print(f"Received all {total_segments} segments")
                        # send the last accumulated segment
                        if accumulated_segment is not None:
                            await segment_stream_send.send(accumulated_segment)
                        await segment_stream_send.aclose()
                        break
        
        # Delete the reply-to queue now that we're done consuming
        await channel.queue_delete(response_queue)

    print("Finished streaming all whisper segments")


async def process_segment(
    accumulated_segment: dict[str, Any] | None,
    current_speaker: str | None,
    response: dict[str, Any],
    segment_count: int,
    segment_stream_send: MemoryObjectSendStream,
) -> tuple[dict[str, Any], str]:
    speaker = response["speaker"]
    text = get_text_from_segment(response)

    if current_speaker != speaker:
        # Send accumulated segment if exists
        if accumulated_segment is not None:
            await segment_stream_send.send(accumulated_segment)

        # Start new accumulated segment
        accumulated_segment = {
            'speaker': speaker,
            'start': response.get("audio_segment", {}).get("start", -1.0),
            'end': response.get("audio_segment", {}).get("end", -1.0),
            'text': text,
            'segment_count': segment_count,
            'subsegment_numbers': [response.get("segment_count")],
        }
        current_speaker = speaker
    else:
        # Merge with current segment
        if accumulated_segment is not None:
            accumulated_segment["text"] += text
            accumulated_segment["end"] = response.get("audio_segment", {}).get("end", -1.0)
            accumulated_segment["subsegment_numbers"] += [response.get("segment_count")]
    return accumulated_segment, current_speaker


def get_text_from_segment(response) -> str:
    text = ""
    for segment in response.get("transcription", {}).get("segments", []):
        segment_words = segment.get("words", [])
        for word in segment_words:
            word_word = word.get("word", "")
            probability = word.get("probability", 0.0)
            word_duration = word["end"] - word["start"]
            if word_duration < 0.1:
                continue
            if probability > 0.3:
                text += word_word
    return text


async def process_audio(audio_file_path: str, min_segment_length: float, transcript_send_stream: MemoryObjectSendStream[str]):
    """
    Process audio file for speaker identification and transcription.
    
    Pipeline stages (running concurrently where possible):
    1. Diarization - identify speaker segments in audio
    2. Whisper transcription - transcribe each segment (streams to step 3)
    3. Journalistic courtesy (LLM cleanup) - clean up transcription errors (streams to step 4)
    4. Speaker identification - identify speaker names from context clues (runs as windows become ready)

    Parameters:
    - audio_file_path: Path to the audio file
    - min_segment_length: Minimum segment length in seconds
    - transcript_send_stream: Stream for sending final transcript
    """
    with open("rabbitmq_config.json", "r") as fl:
        rabbitmq_config = json.load(fl)
    
    # Step 1: Diarization must complete first
    print("Step 1: Starting diarization...")
    diarization = await send_diarization_job_to_queue(audio_file_path, rabbitmq_config)
    
    # Steps 2, 3, & 4 run concurrently with streaming between them
    # Create memory object streams for the pipeline:
    # whisper -> cleanup -> speaker_identification -> final transcript
    print("Steps 2-4: Starting transcription, cleanup, and speaker identification (concurrent)...")
    
    # Stream from whisper to cleanup
    whisper_to_cleanup_send, whisper_to_cleanup_receive = anyio.create_memory_object_stream[dict[str, Any]](max_buffer_size=100)
    
    # Stream from cleanup to speaker identification
    cleanup_to_speaker_id_send, cleanup_to_speaker_id_receive = anyio.create_memory_object_stream[dict[str, Any]](max_buffer_size=100)
    
    async with anyio.create_task_group() as tg:
        # Start whisper jobs - streams results to cleanup
        tg.start_soon(
            send_whisper_jobs,
            audio_file_path,
            rabbitmq_config,
            diarization,
            whisper_to_cleanup_send
        )
        
        # Start journalistic courtesy - consumes from whisper, streams to speaker ID
        tg.start_soon(
            journalistic_courtesy_streaming,
            whisper_to_cleanup_receive,
            cleanup_to_speaker_id_send,
            rabbitmq_config
        )
        
        # Start speaker identification - consumes cleaned segments, produces final transcript
        tg.start_soon(
            speaker_identification_streaming,
            cleanup_to_speaker_id_receive,
            transcript_send_stream,
            rabbitmq_config,
            5,  # window_size
            2,  # stride
        )
    
    print("Audio processing completed successfully")


async def send_speaker_identification_jobs(
    cleaned_segments: list[dict[str, Any]],
    rabbitmq_config: dict[str, Any],
    window_size: int = 5,
    stride: int = 2,
) -> dict[str, dict[str, Any]]:
    """
    Send speaker identification jobs to RabbitMQ using a sliding window approach.
    
    This function takes cleaned transcript segments and sends them to an LLM worker
    in overlapping windows. The LLM analyzes each window to identify speaker names
    from context clues, and results are merged by keeping the highest confidence
    identification for each speaker.
    
    Args:
        cleaned_segments: List of cleaned transcript segments, sorted by segment_count
        rabbitmq_config: RabbitMQ connection configuration
        window_size: Number of segments per window (default: 5)
        stride: Number of segments to move between windows (default: 2)
        
    Returns:
        dict: Speaker tally mapping speaker IDs to their identified names and confidence
              e.g., {"Speaker_01": {"name": "John Smith", "confidence": 9}, ...}
    """
    if not cleaned_segments:
        return {}
    
    work_queue = "llm/speaker-identification"
    response_queue = f"{work_queue}-{uuid.uuid4()}"
    job_id = str(uuid.uuid4())
    
    # Sort segments by segment_count to ensure proper ordering
    sorted_segments = sorted(cleaned_segments, key=lambda x: x.get("segment_count", 0))
    
    # Calculate total windows
    total_windows = max(0, (len(sorted_segments) - window_size) // stride + 1)
    if total_windows == 0 and len(sorted_segments) > 0:
        total_windows = 1  # At least one window if we have any segments
    
    print(f"Speaker identification: Processing {len(sorted_segments)} segments in {total_windows} windows")
    print(f"  Window size: {window_size}, Stride: {stride}")
    print(f"Job ID: {job_id}")
    
    # Connect to RabbitMQ
    ssl_context = create_ssl_context()
    connection = await aio_pika.connect_robust(
        host=rabbitmq_config['host'],
        port=rabbitmq_config['port'],
        login=rabbitmq_config['username'],
        password=rabbitmq_config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )
    
    results = {}
    received_windows = set()
    speaker_tally = {}
    
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)
        
        # Declare queues
        await channel.declare_queue(work_queue, durable=True)
        response_queue_obj = await channel.declare_queue(response_queue, durable=True)
        
        # Send all window jobs
        window_id = 0
        for window_segments in sliding_window(sorted_segments, window_size, stride):
            # Format each segment in the window
            formatted_segments = [
                format_segment_for_speaker_identification(seg) 
                for seg in window_segments
            ]
            transcript_window = "\n\n".join(formatted_segments)
            
            job_message = {
                'job_id': job_id,
                'reply_to': response_queue,
                'window_id': window_id,
                'transcript_window': transcript_window,
                'total_windows': total_windows,
            }
            
            await channel.default_exchange.publish(
                aio_pika.Message(body=json.dumps(job_message).encode()),
                routing_key=work_queue,
            )
            window_id += 1
        
        # Handle edge case: if we have fewer segments than window_size, send one window
        if window_id == 0 and len(sorted_segments) > 0:
            formatted_segments = [
                format_segment_for_speaker_identification(seg) 
                for seg in sorted_segments
            ]
            transcript_window = "\n\n".join(formatted_segments)
            
            job_message = {
                'job_id': job_id,
                'reply_to': response_queue,
                'window_id': 0,
                'transcript_window': transcript_window,
                'total_windows': 1,
            }
            
            await channel.default_exchange.publish(
                aio_pika.Message(body=json.dumps(job_message).encode()),
                routing_key=work_queue,
            )
            total_windows = 1
        
        print(f"Sent {total_windows} speaker identification windows, waiting for responses...")
        
        # Consume responses
        async with response_queue_obj.iterator() as queue_iter:
            async for message in queue_iter:
                async with message.process():
                    response = json.loads(message.body.decode())
                    
                    if response.get("job_id") != job_id:
                        continue
                    
                    window_id = response.get("window_id")
                    if window_id is None:
                        print("Bad message: no window_id, skipping")
                        continue
                    
                    print(f"Received speaker identification response for window {window_id}")
                    received_windows.add(window_id)
                    
                    if response.get("status") == "success":
                        parsed_result = response.get("result", {})
                        results[window_id] = parsed_result
                        
                        # Merge results into speaker_tally with confidence-based replacement
                        for speaker_id, speaker_info in parsed_result.items():
                            if isinstance(speaker_info, dict) and 'name' in speaker_info and 'confidence' in speaker_info:
                                new_confidence = speaker_info['confidence']
                                new_name = speaker_info['name']
                                
                                # Add or update speaker if confidence is higher
                                if speaker_id not in speaker_tally or new_confidence > speaker_tally[speaker_id]['confidence']:
                                    speaker_tally[speaker_id] = {
                                        'name': new_name,
                                        'confidence': new_confidence
                                    }
                                    print(f"  Window {window_id}: Updated {speaker_id} -> {new_name} (confidence: {new_confidence})")
                    else:
                        error = response.get("error", "Unknown error")
                        print(f"  Window {window_id} error: {error}")
                    
                    # Stop when all windows received
                    if len(received_windows) >= total_windows:
                        print(f"Received all {total_windows} speaker identification responses")
                        break
        
        # Delete the reply-to queue
        await channel.queue_delete(response_queue)
    
    print(f"Speaker identification complete. Identified {len(speaker_tally)} speakers:")
    for speaker_id, info in sorted(speaker_tally.items()):
        print(f"  {speaker_id}: {info['name']} (confidence: {info['confidence']})")
    
    return speaker_tally


def apply_speaker_identifications(
    cleaned_segments: list[dict[str, Any]],
    speaker_tally: dict[str, dict[str, Any]],
    min_confidence: int = 2
) -> list[dict[str, Any]]:
    """
    Apply speaker identifications to transcript segments.
    
    Replaces generic speaker labels (e.g., "Speaker_01") with identified names
    when the confidence meets the minimum threshold.
    
    Args:
        cleaned_segments: List of cleaned transcript segments
        speaker_tally: Mapping of speaker IDs to names and confidence scores
        min_confidence: Minimum confidence required to apply identification (default: 2)
        
    Returns:
        List of segments with updated speaker names
    """
    updated_segments = []
    
    for segment in cleaned_segments:
        updated_segment = segment.copy()
        speaker_id = segment.get('speaker', '')
        
        if speaker_id in speaker_tally:
            speaker_info = speaker_tally[speaker_id]
            confidence = speaker_info.get('confidence', 0)
            name = speaker_info.get('name', 'Unknown')
            
            # Only apply identification if confidence meets threshold and name is not "Unknown"
            if confidence >= min_confidence and name.lower() != 'unknown':
                updated_segment['speaker'] = name
                updated_segment['original_speaker_id'] = speaker_id
                updated_segment['speaker_confidence'] = confidence
        
        updated_segments.append(updated_segment)
    
    return updated_segments


async def journalistic_courtesy_streaming(
    segment_stream_receive: anyio.streams.memory.MemoryObjectReceiveStream[dict[str, Any]],
    cleaned_segment_send_stream: MemoryObjectSendStream[dict[str, Any]],
    rabbitmq_config: dict[str, Any]
) -> None:
    """
    Clean up transcript using LLM via RabbitMQ worker, streaming cleaned segments as they arrive.
    
    Consolidates consecutive segments from the same speaker before sending to cleanup,
    then streams cleaned segments to the next pipeline stage as responses arrive.
    
    Args:
        segment_stream_receive: Stream to receive raw transcript segments from
        cleaned_segment_send_stream: Stream for sending cleaned segments to next stage
        rabbitmq_config: RabbitMQ connection configuration
    """
    work_queue = "llm/cleanup"
    response_queue = f"{work_queue}-{uuid.uuid4()}"
    job_id = str(uuid.uuid4())
    
    # Create stream for consolidated segments
    consolidated_send, consolidated_receive = anyio.create_memory_object_stream[dict[str, Any]](max_buffer_size=100)
    
    async def consolidate_segments_by_speaker():
        """Consolidate consecutive segments from the same speaker and forward to cleanup."""
        current_speaker = None
        accumulated_segment = None
        segment_count = 0
        
        async with segment_stream_receive, consolidated_send:
            async for segment in segment_stream_receive:
                speaker = segment.get('speaker')
                print(f"Received segment for consolidation: {speaker} - {segment.get('text', '')[:50]}...")
                
                if current_speaker != speaker:
                    # Different speaker - send accumulated segment if it exists
                    if accumulated_segment is not None:
                        accumulated_segment['segment_count'] = segment_count
                        await consolidated_send.send(accumulated_segment)
                        print(f"Sent consolidated segment {segment_count}: {accumulated_segment.get('speaker')}")
                        segment_count += 1
                    
                    # Start new accumulated segment
                    accumulated_segment = {
                        'speaker': speaker,
                        'start': segment.get('start'),
                        'end': segment.get('end'),
                        'text': segment.get('text', ''),
                    }
                    current_speaker = speaker
                else:
                    # Same speaker - merge with current segment
                    if accumulated_segment is not None:
                        accumulated_segment['text'] += ' ' + segment.get('text', '')
                        accumulated_segment['end'] = segment.get('end')
            
            # Send the last accumulated segment
            if accumulated_segment is not None:
                accumulated_segment['segment_count'] = segment_count
                await consolidated_send.send(accumulated_segment)
                print(f"Sent final consolidated segment {segment_count}: {accumulated_segment.get('speaker')}")
    
    async def send_to_rabbitmq_and_stream():
        """Send consolidated segments to RabbitMQ and stream responses to next stage."""
        received_segments = set()
        segment_count = 0
        total_segments = None
        out_of_order_responses = {}
        next_segment_to_send = 0
        
        # Connect to RabbitMQ
        ssl_context = create_ssl_context()
        connection = await aio_pika.connect_robust(
            host=rabbitmq_config['host'],
            port=rabbitmq_config['port'],
            login=rabbitmq_config['username'],
            password=rabbitmq_config['password'],
            ssl=True,
            ssl_context=ssl_context,
        )
        
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=10)
            
            # Declare queues
            await channel.declare_queue(work_queue, durable=True)
            response_queue_obj = await channel.declare_queue(response_queue, durable=True)
            
            async def consume_and_stream_responses():
                nonlocal total_segments, next_segment_to_send
                async with response_queue_obj.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            response = json.loads(message.body.decode())
                            
                            if response.get("job_id") != job_id:
                                continue
                            
                            segment_num = response.get("segment_count")
                            print(f"Received cleanup response for segment {segment_num}")
                            
                            if segment_num is None:
                                continue
                            
                            received_segments.add(segment_num)
                            
                            if response.get("status") == "success":
                                segment = response.get("segment")
                            else:
                                print(f"Error cleaning segment {segment_num}: {response.get('error')}")
                                segment = response.get("segment")
                            
                            # Handle ordering - buffer out-of-order responses
                            if segment_num == next_segment_to_send:
                                # Send this segment
                                await cleaned_segment_send_stream.send(segment)
                                next_segment_to_send += 1
                                
                                # Send any buffered segments that are now in order
                                while next_segment_to_send in out_of_order_responses:
                                    buffered_segment = out_of_order_responses.pop(next_segment_to_send)
                                    await cleaned_segment_send_stream.send(buffered_segment)
                                    next_segment_to_send += 1
                            else:
                                # Buffer for later
                                out_of_order_responses[segment_num] = segment
                            
                            # Stop when all segments received
                            if total_segments is not None and len(received_segments) == total_segments:
                                print(f"Received all {total_segments} cleaned segments")
                                break
            
            # Start response consumer task
            async with anyio.create_task_group() as tg:
                tg.start_soon(consume_and_stream_responses)
                
                # Send consolidated segments as they arrive
                async with consolidated_receive:
                    async for segment in consolidated_receive:
                        job_message = {
                            'job_id': job_id,
                            'reply_to': response_queue,
                            'segment': {
                                'speaker': segment.get('speaker'),
                                'start': segment.get('start'),
                                'end': segment.get('end'),
                                'text': segment.get('text'),
                                'segment_count': segment.get('segment_count'),
                            },
                            'segment_count': segment.get('segment_count'),
                            'total_segments': None,
                        }
                        
                        await channel.default_exchange.publish(
                            aio_pika.Message(body=json.dumps(job_message).encode()),
                            routing_key=work_queue,
                        )
                        print(f"Sent consolidated segment {segment.get('segment_count')} to cleanup queue")
                        segment_count += 1
                
                # Now we know the total
                total_segments = segment_count
                print(f"All {total_segments} consolidated segments sent to cleanup queue")
            
            # Delete the reply-to queue
            await channel.queue_delete(response_queue)
        
        # Close the output stream
        await cleaned_segment_send_stream.aclose()
    
    # Run consolidation and RabbitMQ tasks concurrently
    async with anyio.create_task_group() as tg:
        tg.start_soon(consolidate_segments_by_speaker)
        tg.start_soon(send_to_rabbitmq_and_stream)


async def speaker_identification_streaming(
    cleaned_segment_receive_stream: anyio.streams.memory.MemoryObjectReceiveStream[dict[str, Any]],
    transcript_send_stream: MemoryObjectSendStream[str],
    rabbitmq_config: dict[str, Any],
    window_size: int = 5,
    stride: int = 2,
) -> None:
    """
    Identify speakers from cleaned transcript segments using a streaming sliding window approach.
    
    As cleaned segments arrive, they are buffered until a complete window is ready.
    Windows are sent to RabbitMQ for LLM processing as soon as they're complete.
    Final transcript is produced after all segments are received and speaker IDs are applied.
    
    Args:
        cleaned_segment_receive_stream: Stream to receive cleaned segments from
        transcript_send_stream: Stream for sending final transcript
        rabbitmq_config: RabbitMQ connection configuration
        window_size: Number of segments per window (default: 5)
        stride: Number of segments to move between windows (default: 2)
    """
    work_queue = "llm/speaker-identification"
    response_queue = f"{work_queue}-{uuid.uuid4()}"
    job_id = str(uuid.uuid4())
    
    print(f"Speaker identification starting (window_size={window_size}, stride={stride})")
    print(f"Job ID: {job_id}")
    
    # Buffer for segments and results
    segment_buffer = []
    all_segments = []
    windows_sent = 0
    windows_received = set()
    speaker_tally = {}
    stream_closed = False
    
    # Connect to RabbitMQ
    ssl_context = create_ssl_context()
    connection = await aio_pika.connect_robust(
        host=rabbitmq_config['host'],
        port=rabbitmq_config['port'],
        login=rabbitmq_config['username'],
        password=rabbitmq_config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )
    
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=10)
        
        # Declare queues
        await channel.declare_queue(work_queue, durable=True)
        response_queue_obj = await channel.declare_queue(response_queue, durable=True)
        
        async def send_window_if_ready():
            """Check if we can send a new window and send it."""
            nonlocal windows_sent
            
            # Calculate which windows we can send based on buffer size
            # Window N requires segments [N*stride : N*stride + window_size]
            while True:
                start_idx = windows_sent * stride
                end_idx = start_idx + window_size
                
                if end_idx > len(segment_buffer):
                    # Not enough segments for this window yet
                    break
                
                # We have enough segments for this window
                window_segments = segment_buffer[start_idx:end_idx]
                
                # Format the window
                formatted_segments = [
                    format_segment_for_speaker_identification(seg) 
                    for seg in window_segments
                ]
                transcript_window = "\n\n".join(formatted_segments)
                
                job_message = {
                    'job_id': job_id,
                    'reply_to': response_queue,
                    'window_id': windows_sent,
                    'transcript_window': transcript_window,
                    'total_windows': None,  # Don't know total yet
                }
                
                await channel.default_exchange.publish(
                    aio_pika.Message(body=json.dumps(job_message).encode()),
                    routing_key=work_queue,
                )
                print(f"Sent speaker identification window {windows_sent} (segments {start_idx}-{end_idx-1})")
                windows_sent += 1
        
        async def receive_segments():
            """Receive cleaned segments and buffer them, sending windows as ready."""
            nonlocal stream_closed
            
            async with cleaned_segment_receive_stream:
                async for segment in cleaned_segment_receive_stream:
                    segment_buffer.append(segment)
                    all_segments.append(segment)
                    print(f"Received cleaned segment {segment.get('segment_count')} for speaker ID")
                    
                    # Try to send any windows that are now ready
                    await send_window_if_ready()
            
            stream_closed = True
            print(f"Segment stream closed. Total segments: {len(segment_buffer)}")
            
            # Send any remaining windows for partial coverage at the end
            # If we have fewer segments than window_size but haven't sent any windows, send what we have
            if windows_sent == 0 and len(segment_buffer) > 0:
                formatted_segments = [
                    format_segment_for_speaker_identification(seg) 
                    for seg in segment_buffer
                ]
                transcript_window = "\n\n".join(formatted_segments)
                
                job_message = {
                    'job_id': job_id,
                    'reply_to': response_queue,
                    'window_id': 0,
                    'transcript_window': transcript_window,
                    'total_windows': 1,
                }
                
                await channel.default_exchange.publish(
                    aio_pika.Message(body=json.dumps(job_message).encode()),
                    routing_key=work_queue,
                )
                print(f"Sent final partial window (all {len(segment_buffer)} segments)")
        
        async def consume_responses():
            """Consume speaker identification responses and merge results."""
            async with response_queue_obj.iterator() as queue_iter:
                async for message in queue_iter:
                    async with message.process():
                        response = json.loads(message.body.decode())
                        
                        if response.get("job_id") != job_id:
                            continue
                        
                        window_id = response.get("window_id")
                        if window_id is None:
                            continue
                        
                        print(f"Received speaker identification response for window {window_id}")
                        windows_received.add(window_id)
                        
                        if response.get("status") == "success":
                            parsed_result = response.get("result", {})
                            
                            # Merge results with confidence-based replacement
                            for speaker_id, speaker_info in parsed_result.items():
                                if isinstance(speaker_info, dict) and 'name' in speaker_info and 'confidence' in speaker_info:
                                    new_confidence = speaker_info['confidence']
                                    new_name = speaker_info['name']
                                    
                                    if speaker_id not in speaker_tally or new_confidence > speaker_tally[speaker_id]['confidence']:
                                        speaker_tally[speaker_id] = {
                                            'name': new_name,
                                            'confidence': new_confidence
                                        }
                                        print(f"  Updated {speaker_id} -> {new_name} (confidence: {new_confidence})")
                        else:
                            error = response.get("error", "Unknown error")
                            print(f"  Window {window_id} error: {error}")
                        
                        # Check if we've received all expected windows
                        # We're done when stream is closed and we've received all sent windows
                        if stream_closed and len(windows_received) >= windows_sent:
                            print(f"Received all {len(windows_received)} speaker identification responses")
                            break
        
        # Run segment receiving and response consuming concurrently
        async with anyio.create_task_group() as tg:
            tg.start_soon(receive_segments)
            tg.start_soon(consume_responses)
        
        # Delete the reply-to queue
        await channel.queue_delete(response_queue)
    
    # Apply speaker identifications
    print(f"Speaker identification complete. Identified {len(speaker_tally)} speakers:")
    for speaker_id, info in sorted(speaker_tally.items()):
        print(f"  {speaker_id}: {info['name']} (confidence: {info['confidence']})")
    
    if speaker_tally:
        final_segments = apply_speaker_identifications(all_segments, speaker_tally, min_confidence=2)
    else:
        final_segments = all_segments
    
    # Produce and send final transcript
    final_transcript = produce_transcript(final_segments)
    await transcript_send_stream.send(final_transcript)
    await transcript_send_stream.aclose()


async def journalistic_courtesy(
    segment_stream_receive: anyio.streams.memory.MemoryObjectReceiveStream[dict[str, Any]],
    transcript_send_stream: MemoryObjectSendStream[str],
    rabbitmq_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Clean up transcript using LLM via RabbitMQ worker, sending consolidated segments to RabbitMQ as they arrive.
    
    Consolidates consecutive segments from the same speaker before sending to cleanup.
    
    NOTE: This is the legacy version that returns all segments at once. 
    Use journalistic_courtesy_streaming for concurrent pipeline processing.
    
    Args:
        segment_stream_receive: Stream to receive raw transcript segments from
        transcript_send_stream: Stream for sending final transcript
        rabbitmq_config: RabbitMQ connection configuration
        
    Returns:
        Cleaned transcript segments
    """
    work_queue = "llm/cleanup"
    response_queue = f"{work_queue}-{uuid.uuid4()}"
    job_id = str(uuid.uuid4())
    
    # Create streams for consolidated segments and cleanup responses
    consolidated_send, consolidated_receive = anyio.create_memory_object_stream[dict[str, Any]](max_buffer_size=100)
    cleanup_response_send, cleanup_response_receive = anyio.create_memory_object_stream[dict[str, Any]](max_buffer_size=100)
    
    async def consolidate_segments_by_speaker():
        """Consolidate consecutive segments from the same speaker and forward to cleanup."""
        current_speaker = None
        accumulated_segment = None
        
        async with segment_stream_receive, consolidated_send:
            async for segment in segment_stream_receive:
                speaker = segment.get('speaker')
                print(f"Received segment for consolidation: {speaker} - {segment.get('text', '')[:50]}...")
                
                if current_speaker != speaker:
                    # Different speaker - send accumulated segment if it exists
                    if accumulated_segment is not None:
                        await consolidated_send.send(accumulated_segment)
                        print(f"Sent consolidated segment: {accumulated_segment.get('speaker')}")
                    
                    # Start new accumulated segment
                    accumulated_segment = {
                        'speaker': speaker,
                        'start': segment.get('start'),
                        'end': segment.get('end'),
                        'text': segment.get('text', ''),
                    }
                    current_speaker = speaker
                else:
                    # Same speaker - merge with current segment
                    if accumulated_segment is not None:
                        accumulated_segment['text'] += ' ' + segment.get('text', '')
                        accumulated_segment['end'] = segment.get('end')
            
            # Send the last accumulated segment
            if accumulated_segment is not None:
                await consolidated_send.send(accumulated_segment)
                print(f"Sent final consolidated segment: {accumulated_segment.get('speaker')}")
    
    async def send_to_rabbitmq():
        """Send consolidated segments to RabbitMQ as they arrive and collect responses."""
        from utils import create_ssl_context
        
        received_segments = set()
        cleaned_segments = []
        segment_count = 0
        total_segments = None
        
        # Connect to RabbitMQ
        ssl_context = create_ssl_context()
        connection = await aio_pika.connect_robust(
            host=rabbitmq_config['host'],
            port=rabbitmq_config['port'],
            login=rabbitmq_config['username'],
            password=rabbitmq_config['password'],
            ssl=True,
            ssl_context=ssl_context,
        )
        
        async with connection:
            channel = await connection.channel()
            await channel.set_qos(prefetch_count=10)
            
            # Declare queues
            await channel.declare_queue(work_queue, durable=True)
            response_queue_obj = await channel.declare_queue(response_queue, durable=True)
            
            # Start consuming responses in background task
            async def consume_responses():
                nonlocal total_segments
                async with response_queue_obj.iterator() as queue_iter:
                    async for message in queue_iter:
                        async with message.process():
                            response = json.loads(message.body.decode())
                            
                            if response.get("job_id") != job_id:
                                continue
                            
                            segment_num = response.get("segment_count")
                            print(f"Received cleanup response for segment {segment_num}")
                            
                            # Track this segment
                            if segment_num is not None:
                                received_segments.add(segment_num)
                            
                            if response.get("status") == "success":
                                cleaned_segments.append(response.get("segment"))
                            else:
                                # On error, keep original segment
                                print(f"Error cleaning segment {segment_num}: {response.get('error')}")
                                segment = response.get("segment")
                                cleaned_segments.append(segment)
                            
                            # Stop when all segments received
                            if total_segments is not None and len(received_segments) == total_segments:
                                print(f"Received all {total_segments} cleaned segments")
                                break
            
            # Start response consumer task
            async with anyio.create_task_group() as tg:
                tg.start_soon(consume_responses)
                
                # Send consolidated segments as they arrive
                async with consolidated_receive:
                    async for segment in consolidated_receive:
                        job_message = {
                            'job_id': job_id,
                            'reply_to': response_queue,
                            'segment': {
                                'speaker': segment.get('speaker'),
                                'start': segment.get('start'),
                                'end': segment.get('end'),
                                'text': segment.get('text'),
                                'segment_count': segment_count,
                            },
                            'segment_count': segment_count,
                            'total_segments': None,  # Don't know total yet
                        }
                        
                        # Send job to work queue immediately
                        await channel.default_exchange.publish(
                            aio_pika.Message(body=json.dumps(job_message).encode()),
                            routing_key=work_queue,
                        )
                        print(f"Sent consolidated segment {segment_count} to cleanup queue as it arrived")
                        segment_count += 1
                
                # Now we know the total
                total_segments = segment_count
                print(f"All {total_segments} consolidated segments sent to cleanup queue, waiting for responses...")
                
                # Wait for all responses to complete (consume_responses task will complete when done)
            
            # Delete the reply-to queue now that we're done consuming
            await channel.queue_delete(response_queue)
        
        # Sort and send results
        result_sorted = sorted(cleaned_segments, key=lambda x: x.get("segment_count", 0))
        
        # Send to output stream
        async with cleanup_response_send:
            await cleanup_response_send.send(result_sorted)
    
    # Run consolidation and RabbitMQ tasks concurrently
    async with anyio.create_task_group() as tg:
        tg.start_soon(consolidate_segments_by_speaker)
        tg.start_soon(send_to_rabbitmq)
    
    # Collect final result
    cleaned_segments_sorted = []
    async with cleanup_response_receive:
        async for segments in cleanup_response_receive:
            cleaned_segments_sorted = segments
            break
    
    await transcript_send_stream.send(produce_transcript(cleaned_segments_sorted))
    LOGGER.info(f"{cleaned_segments_sorted}")
    
    return cleaned_segments_sorted


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
    print(prompt)
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
