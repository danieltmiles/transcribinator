import json
import numpy as np
import aio_pika
import argparse
import torch
import time
import asyncio

from pamqp.commands import Basic

from utils import load_config, create_ssl_context

# Set device for PyTorch
# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_whisper_model():
    """
    Load the Whisper model.
    
    Returns:
        Whisper model instance
    """
    try:
        import whisper
    except AttributeError:
        raise ImportError(
            "Please install the correct Whisper package using: pip install openai-whisper\n"
            "If you have the 'whisper' package installed, first uninstall it with: pip uninstall whisper"
        )
    
    print(f"Loading Whisper large model on device: {device}")
    start_time = time.time()
    model = whisper.load_model("large", device=device)
    end_time = time.time()
    print(f"Whisper model loaded in {end_time - start_time:.2f} seconds")
    
    return model


def perform_transcription(audio_data, whisper_model, temperature=0.2, language='en', 
                         initial_prompt=None, word_timestamps=True):
    """
    Perform transcription on the audio data.
    
    Args:
        audio_data: Numpy array containing the audio waveform
        whisper_model: Loaded Whisper model
        temperature: Temperature for sampling (default: 0.2)
        language: Language code (default: 'en')
        initial_prompt: Optional initial prompt for the model
        word_timestamps: Whether to include word-level timestamps (default: True)
    
    Returns:
        Transcription result dictionary
    """
    print(f"Starting transcription on audio with shape {audio_data.shape}")
    start_time = time.time()
    
    # Build transcription parameters
    transcribe_params = {
        'temperature': temperature,
        'word_timestamps': word_timestamps,
    }
    
    if language:
        transcribe_params['language'] = language
    
    if initial_prompt:
        transcribe_params['initial_prompt'] = initial_prompt
    
    # Perform transcription
    result = whisper_model.transcribe(audio_data, **transcribe_params)
    
    end_time = time.time()
    print(f"Transcription completed in {end_time - start_time:.2f} seconds")
    
    return result


async def process_message(message: aio_pika.IncomingMessage, whisper_model):
    """
    Process a transcription job message from RabbitMQ.
    
    Expected message format:
    {
        'job_id': str,
        'reply_to': str (queue name for response),
        'audio_segment': {
            'audio': list (audio data as list),
            'start': float,
            'end': float,
            'speaker': str
        },
        'speaker': str,
        'temperature': float,
        'language': str,
        'initial_prompt': str (optional),
        'word_timestamps': bool
    }
    """
    try:
        print(f"Received message")
        body = json.loads(message.body.decode())
        
        job_id = body.get('job_id')
        reply_to = body.get('reply_to')
        audio_segment = body.get('audio_segment', {})
        
        # Extract parameters
        temperature = body.get('temperature', 0.2)
        language = body.get('language', 'en')
        initial_prompt = body.get('initial_prompt')
        word_timestamps = body.get('word_timestamps', True)
        speaker = body.get('speaker', audio_segment.get('speaker', 'Unknown'))
        segment_count = body.get("segment_count")
        total_segments = body.get("total_segments")

        print(f"Processing job {job_id} for speaker {speaker}")
        
        # Convert audio data from list back to numpy array
        audio_data = np.array(audio_segment.get('audio', []), dtype=np.float32)
        
        if len(audio_data) == 0:
            raise ValueError("Empty audio data received")
        
        print(f"Audio data shape: {audio_data.shape}, duration: ~{len(audio_data) / 16000:.2f}s")
        
        # Run transcription in thread pool to prevent blocking the event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            perform_transcription,
            audio_data,
            whisper_model,
            temperature,
            language,
            initial_prompt,
            word_timestamps
        )
        
        # Prepare response
        response = {
            'job_id': job_id,
            'status': 'success',
            'transcription': result,
            'speaker': speaker,
            'audio_segment': {
                'start': audio_segment.get('start'),
                'end': audio_segment.get('end'),
                'speaker': audio_segment.get('speaker')
            },
            'segment_count': segment_count,
            'total_segments': total_segments,
        }
        
        # Get channel from message with error handling for invalid state
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
        except (aio_pika.ChannelInvalidStateError, aio_pika.ChannelClosed) as channel_error:
            print(f"Channel error while sending response for job {job_id}: {channel_error}")
            print(f"Message will be re-queued for retry")
            # Nack the message so it gets requeued
            await message.nack(requeue=True)
            return

        print(f"Job {job_id} completed and response sent to {reply_to}")
        
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
            
            if reply_to:
                error_response = {
                    'job_id': job_id,
                    'status': 'error',
                    'error': str(e)
                }
                
                try:
                    channel = message.channel
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(error_response).encode(),
                        ),
                        routing_key=reply_to,
                    )
                except (aio_pika.ChannelInvalidStateError, aio_pika.ChannelClosed):
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
    Main function to start the whisper transcription consumer with reconnection logic.
    Handles connection failures and automatically reconnects with exponential backoff.
    """
    print("Initializing Whisper transcription consumer...")
    
    # Retry configuration
    max_retries = 10
    base_retry_delay = 2  # seconds
    max_retry_delay = 60  # seconds
    
    # Load the Whisper model once at startup
    print("Loading Whisper model...")
    whisper_model = load_whisper_model()
    
    ssl_context = create_ssl_context()
    # If using self-signed certificates, uncomment:
    # ssl_context = create_ssl_context(verify=False)
    
    retry_count = 0
    
    while True:
        try:
            # Connect to RabbitMQ with TLS
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
                # Create channel
                channel = await connection.channel()
                
                # Set QoS to process one message at a time
                await channel.set_qos(prefetch_count=1)
                
                # Declare the work queue
                work_queue = config['work_queue']
                queue = await channel.declare_queue(work_queue, durable=True)
                
                print(f"Successfully connected! Listening for transcription jobs on queue: {work_queue}")
                print("Waiting for transcription jobs. To exit press CTRL+C")
                
                # Start consuming messages
                async with queue.iterator() as queue_iter:
                    async for message in queue_iter:
                        try:
                            await process_message(message, whisper_model)
                        except (aio_pika.ChannelInvalidStateError, aio_pika.ChannelClosed) as channel_err:
                            print(f"Channel error during message processing: {channel_err}")
                            print("Will attempt to reconnect...")
                            # Break out of the message loop to reconnect
                            raise
                        except Exception as e:
                            print(f"Unexpected error processing message: {e}")
                            import traceback
                            traceback.print_exc()
                            # Continue processing other messages
                            
        except (aio_pika.AMQPError, aio_pika.ChannelInvalidStateError, aio_pika.ChannelClosed, ConnectionError) as conn_error:
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='RabbitMQ consumer for Whisper transcription jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "whisper/large",
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
    
    # Load configuration from file
    config = load_config(args.config_file)
    
    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")
    
    # Run main with config
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
