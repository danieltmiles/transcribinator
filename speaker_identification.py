"""
Speaker Identification Worker for RabbitMQ

This worker processes transcript windows to identify speaker names from context clues
using an LLM with a sliding window approach.

Usage:
    python speaker_identification.py speaker_identification_config.json

Configuration file format (JSON):
{
    "work_queue": "llm/speaker-identification",
    "model_path": "/path/to/model",
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
}
"""
import json
import re
import aio_pika
import argparse
import torch
import asyncio
import os

from aio_pika.abc import AbstractIncomingMessage
from aiormq import ChannelInvalidStateError, ChannelClosed, AMQPError
from pamqp.commands import Basic

from utils import load_config, create_ssl_context, load_quantized_llm_model, quantized_generate_from_prompt

# Device detection - prioritize CUDA over MPS
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
model_type = None
print(f"Detected device: {device}")


def parse_llm_json_output(raw_output: str) -> dict:
    """
    Parse messy LLM output to extract clean JSON dictionary.
    
    Handles common issues:
    - Extra "thinking" text before JSON
    - Multiple JSON blocks (sometimes duplicated)
    - Extra backticks and markdown formatting
    - Malformed JSON blocks
    
    Returns the best parsed JSON as a Python dictionary.
    """
    if not raw_output.strip():
        return {}
    
    # Remove common LLM "thinking" phrases and unnecessary text
    thinking_patterns = [
        r"Okay,?\s*let'?s?\s*try\s*to\s*figure\s*out.*?(?=\{|\[)",
        r"I'?ll\s*analyze.*?(?=\{|\[)",
        r"Let me\s*think.*?(?=\{|\[)",
        r"Here'?s?\s*(?:the\s*)?(?:my\s*)?(?:analysis|answer|response).*?(?=\{|\[)",
        r"Based\s*on.*?(?=\{|\[)",
    ]
    
    cleaned_output = raw_output
    for pattern in thinking_patterns:
        cleaned_output = re.sub(pattern, '', cleaned_output, flags=re.IGNORECASE | re.DOTALL)
    
    # Find all potential JSON blocks using multiple strategies
    json_candidates = []
    
    # Strategy 1: Find content between ```json and ``` markers
    json_blocks = re.findall(r'```json\s*\n?(.*?)\n?```', cleaned_output, re.DOTALL | re.IGNORECASE)
    json_candidates.extend(json_blocks)
    
    # Strategy 2: Find content between triple backticks (generic)
    generic_blocks = re.findall(r'```\s*\n?(.*?)\n?```', cleaned_output, re.DOTALL)
    json_candidates.extend(generic_blocks)
    
    # Strategy 3: Find JSON-like structures (starting with { and ending with })
    brace_blocks = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned_output, re.DOTALL)
    json_candidates.extend(brace_blocks)
    
    # Strategy 4: Extract everything that looks like JSON from the entire text
    potential_json = re.findall(r'\{.*?\}', cleaned_output, re.DOTALL)
    json_candidates.extend(potential_json)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in json_candidates:
        candidate_clean = candidate.strip()
        if candidate_clean and candidate_clean not in seen:
            seen.add(candidate_clean)
            unique_candidates.append(candidate_clean)
    
    # Try to parse each candidate
    best_result = {}
    best_score = 0
    
    for candidate in unique_candidates:
        candidate = candidate.strip()
        candidate = re.sub(r'^```+.*?\n?', '', candidate)
        candidate = re.sub(r'\n?```+$', '', candidate)
        candidate = candidate.strip()
        
        if not candidate.startswith('{') or not candidate.endswith('}'):
            continue
            
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                score = 0
                speaker_keys = [k for k in parsed.keys() if k.startswith('Speaker_')]
                score += len(speaker_keys) * 2
                
                for key, value in parsed.items():
                    if isinstance(value, dict):
                        if 'name' in value:
                            score += 1
                        if 'confidence' in value:
                            score += 1
                
                score += len(parsed)
                
                if score > best_score:
                    best_result = parsed
                    best_score = score
                    
        except json.JSONDecodeError:
            try:
                fixed_candidate = re.sub(r',\s*}', '}', candidate)
                fixed_candidate = re.sub(r',\s*]', ']', fixed_candidate)
                
                parsed = json.loads(fixed_candidate)
                if isinstance(parsed, dict) and len(parsed) > len(best_result):
                    best_result = parsed
            except json.JSONDecodeError:
                continue
    
    return best_result


def identify_speakers(transcript_window: str, model, tokenizer) -> dict:
    """
    Identify speakers from a transcript window using LLM.
    
    Args:
        transcript_window: A window of transcript text with speaker labels
        model: The loaded LLM model
        tokenizer: The tokenizer for the model (None for llama-cpp)
    
    Returns:
        dict: Speaker identification results with confidence scores
    """
    prompt_template = """Extract speaker identities from this auto-transcribed text. Speaker numbers may be inaccurate (one person = multiple numbers, or vice versa).
APPROACH:
First, reason through the clues step-by-step. Look for patterns where:
- Someone addresses a person by name, then that person speaks next
- Someone introduces themselves
- Context reveals roles (who opens meetings, who is thanked)

Then output your conclusions as JSON.

CRITICAL RULES:
1. Use full names when possible (e.g., "Joe Smith", "Mayor Jane Doe")
2. NEVER output bare titles like "Moderator", "Chair", "Professor" - always include the person's name
3. Output ONLY valid JSON, no markdown, no explanation
4. Results of, Unknown, should have a confidence score of 1

KEY CLUES TO ANALYZE:
- Direct Address: If Speaker A uses a person's name, like, "as my colleague John Smith will tell us," and Speaker B responds, Speaker B is likely John Smith.
- Introductions: If Speaker A introduces Speaker B, and Speaker B says, "Thank you, John Smith," speaker A is likely John Smith.
- Titles in address: "Mayor Jones" or "Councilmember Garcia" spoken TO someone identifies that person
- Self-introductions: "My name is..." statements
- Role references: Opening/chairing meetings suggests leadership role
- Chronological patterns: Who speaks immediately after being addressed?

Format:
{{"Speaker_01": {{"name": "Full Name or Unknown", "confidence": 1-10}}}}

Confidence scale:
10 = Direct self-identification ("my name is X")
9 = Addressed by name and responds immediately after
8 = Strong contextual evidence (multiple clues align)
5-7 = Moderate clues (role + context)
2-4 = Weak inference
1 = Pure guess

Transcript:
```
{transcript}
```
"""
    prompt = prompt_template.format(transcript=transcript_window.strip())
    
    try:
        # Use backend-specific generation
        generated_text = quantized_generate_from_prompt(prompt, model, tokenizer, model_type)
        
        if not generated_text:
            return {}
        
        # Parse the LLM output
        result = parse_llm_json_output(generated_text)
        
        return result
        
    except Exception as e:
        print(f"Error during speaker identification: {e}")
        import traceback
        traceback.print_exc()
        return {}


async def process_message(message: AbstractIncomingMessage, model, tokenizer):
    """
    Process a speaker identification job message from RabbitMQ.
    
    Expected message format:
    {
        'job_id': str,
        'reply_to': str (queue name for response),
        'window_id': int,
        'transcript_window': str,
        'total_windows': int
    }
    """
    try:
        print(f"Received message")
        body = json.loads(message.body.decode())
        
        job_id = body.get('job_id')
        reply_to = body.get('reply_to')
        window_id = body.get('window_id')
        transcript_window = body.get('transcript_window', '')
        total_windows = body.get('total_windows')
        
        print(f"Processing job {job_id}, window {window_id}/{total_windows}")
        print(f"Transcript window length: {len(transcript_window)} chars")
        
        # Run speaker identification in thread pool to prevent blocking
        loop = asyncio.get_event_loop()
        speaker_result = await loop.run_in_executor(
            None,
            identify_speakers,
            transcript_window,
            model,
            tokenizer
        )
        
        print(f"Window {window_id}: found {len(speaker_result)} speaker identifications")
        
        # Prepare response
        response = {
            'job_id': job_id,
            'status': 'success',
            'window_id': window_id,
            'result': speaker_result,
            'total_windows': total_windows,
        }
        
        # Send response with error handling for invalid state
        try:
            channel = message.channel
            await channel.basic_publish(
                body=json.dumps(response).encode(),
                exchange="",
                routing_key=reply_to,
                properties=Basic.Properties(
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
            )
        except (ChannelInvalidStateError, ChannelClosed) as channel_error:
            print(f"Channel error while sending response for job {job_id}: {channel_error}")
            print(f"Message will be re-queued for retry")
            # Nack the message so it gets requeued
            await message.nack(requeue=True)
            return

        print(f"Job {job_id} window {window_id} completed and response sent to {reply_to}")
        
        # Acknowledge successful processing
        await message.ack()
        
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to send error response if possible
        try:
            body = json.loads(message.body.decode())
            job_id = body.get('job_id', 'unknown')
            reply_to = body.get('reply_to')
            window_id = body.get('window_id')
            
            if reply_to:
                error_response = {
                    'job_id': job_id,
                    'status': 'error',
                    'error': str(e),
                    'window_id': window_id
                }
                
                try:
                    channel = message.channel
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(error_response).encode(),
                        ),
                        routing_key=reply_to,
                    )
                except (ChannelInvalidStateError, ChannelClosed):
                    print(f"Could not send error response due to channel error - message will be requeued")
        except Exception as error_e:
            print(f"Error sending error response: {error_e}")
        
        # Nack the message so it gets requeued (RabbitMQ will retry)
        try:
            await message.nack(requeue=True)
        except Exception as nack_error:
            print(f"Error nacking message: {nack_error}")


async def main(config):
    """
    Main function to start the speaker identification consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing speaker identification consumer...")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    # Load model once at startup
    print("Loading LLM model...")
    model_path = config.get('model_path')
    global model_type
    model, tokenizer, model_type = load_quantized_llm_model(device, model_path)
    
    ssl_context = create_ssl_context()
    
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ
            print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
            
            connection = await aio_pika.connect_robust(
                host=config['host'],
                port=config['port'],
                login=config['username'],
                password=config['password'],
                ssl=True,
                ssl_context=ssl_context,
            )
            
            # Reset retry count on successful connection
            retry_count = 0
            
            async with connection:
                channel = await connection.channel()
                await channel.set_qos(prefetch_count=1)
                
                work_queue = config['work_queue']
                queue = await channel.declare_queue(work_queue, durable=True)
                
                print(f"Successfully connected! Listening for speaker identification jobs on queue: {work_queue}")
                print("Waiting for jobs. To exit press CTRL+C")
                
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            await process_message(message, model, tokenizer)
                        except (ChannelInvalidStateError, ChannelClosed) as channel_err:
                            print(f"Channel error during message processing: {channel_err}")
                            print("Will attempt to reconnect...")
                            # Break out of the message loop to reconnect
                            raise
                        except Exception as e:
                            print(f"Unexpected error processing message: {e}")
                            import traceback
                            traceback.print_exc()
                            # Continue processing other messages
                            
        except (AMQPError, ChannelInvalidStateError, ChannelClosed, ConnectionError) as conn_error:
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise
            
            # Calculate exponential backoff delay
            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            print(f"Connection error: {conn_error}")
            print(f"Reconnection attempt {retry_count}/{max_retries} in {delay} seconds...")
            await asyncio.sleep(delay)
            
        except KeyboardInterrupt:
            print("\nShutting down gracefully...")
            break
        except Exception as e:
            print(f"Unexpected error in main loop: {e}")
            import traceback
            traceback.print_exc()
            
            retry_count += 1
            if retry_count > max_retries:
                print(f"Max retries ({max_retries}) exceeded. Giving up.")
                raise
            
            delay = min(base_retry_delay * (2 ** (retry_count - 1)), max_retry_delay)
            print(f"Retrying in {delay} seconds...")
            await asyncio.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='RabbitMQ consumer for LLM speaker identification jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "llm/speaker-identification",
    "model_path": "/path/to/model",
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest"
}
        '''
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file'
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"Model path: {config.get('model_path', 'default (Qwen/Qwen2.5-7B-Instruct)')}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")

    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
