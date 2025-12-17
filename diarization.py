import json
import pickle
import base64
import aio_pika
import argparse
import torch
import time
import asyncio

from pamqp.commands import Basic
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import DiarizeOutput

import shared_disks
from ai import load_hf_token
from utils import normalize_audio, load_config, create_ssl_context

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

def perform_diarization(waveform, sample_rate, pipeline):
    """
    Perform speaker diarization on the audio waveform.
    
    Args:
        waveform: PyTorch tensor containing the audio waveform (should be 2D: channel, time)
        sample_rate: Sample rate of the audio
        pipeline: Pyannote diarization pipeline
    
    Returns:
        DiarizeOutput object containing diarization results
    """
    print(f"Starting diarization on waveform with shape {waveform.shape}, sample_rate={sample_rate}")
    start_time = time.time()
    
    # Ensure waveform is 2D (channel, time) as required by pyannote
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Perform diarization
    diarization: DiarizeOutput = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    end_time = time.time()
    print(f"Diarization completed in {end_time - start_time:.2f} seconds")
    
    return diarization


def serialize_diarization(diarization):
    """
    Serialize DiarizeOutput object to a format that can be sent via RabbitMQ.
    
    Returns:
        Base64-encoded pickle of the diarization object
    """
    pickled = pickle.dumps(diarization)
    encoded = base64.b64encode(pickled).decode('utf-8')
    return encoded


async def process_message(message: aio_pika.IncomingMessage, pipeline):
    """
    Process a diarization job message from RabbitMQ.
    
    Expected message format:
    {
        'job_id': str,
        'waveform': base64-encoded pickle of PyTorch tensor,
        'sample_rate': int,
        'reply_to': str (queue name for response)
    }
    """
    async with message.process():
        try:
            print(f"Received message")
            body = json.loads(message.body.decode())
            
            job_id = body.get('job_id')
            remote_file_type = body.get("remote_file_type")
            info = body.get("remote_file_info")
            remote_storage: shared_disks.RemoteStorage = shared_disks.factory(remote_file_type, info)
            filename = info.get("filename")
            local_filename = f"/tmp/{filename}"
            print("retrieving remote file")
            remote_storage.retrieve(filename, local_filename)
            print("retrieved remote file")
            signal, sr = normalize_audio(local_filename)
            reply_to = body.get('reply_to')
            
            print(f"Processing job {job_id}")
            
            # Move waveform to appropriate device
            if isinstance(signal, torch.Tensor):
                signal = signal.to(torch.device(device))
            
            # Run diarization in thread pool to prevent blocking the event loop
            loop = asyncio.get_event_loop()
            diarization = await loop.run_in_executor(
                None, 
                perform_diarization, 
                signal, 
                sr, 
                pipeline
            )
            
            # Serialize result
            diarization_encoded = serialize_diarization(diarization)
            
            # Prepare response
            response = {
                'job_id': job_id,
                'status': 'success',
                'diarization': diarization_encoded
            }
            
            # Get channel from message
            channel = message.channel
            await channel.basic_publish(
                body=json.dumps(response).encode(),
                exchange="",
                routing_key=reply_to,
                properties=Basic.Properties(
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
            )

            print(f"Job {job_id} completed and response sent to {reply_to}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error response if possible
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
                    
                    channel = message.channel
                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(error_response).encode(),
                        ),
                        routing_key=reply_to,
                    )
            except Exception as error_e:
                print(f"Error sending error response: {error_e}")
            
            # Re-raise to reject the message
            raise


async def main(config):
    """
    Main function to start the diarization consumer.
    """
    print("Initializing diarization consumer...")
    print(f"Loading diarization pipeline...")
    
    # Load the diarization pipeline once
    start = time.time()
    pipeline = Pipeline.from_pretrained(
        checkpoint="pyannote/speaker-diarization-community-1",
        token=load_hf_token(),
    ).to(torch.device(device))
    end = time.time()
    print(f"Pipeline loaded in {end - start:.2f} seconds")
    
    # Connect to RabbitMQ with TLS
    print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
    
    ssl_context = create_ssl_context()
    # If using self-signed certificates, uncomment:
    # ssl_context = create_ssl_context(verify=False)
    
    connection = await aio_pika.connect_robust(
        host=config['host'],
        port=config['port'],
        login=config['username'],
        password=config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )
    
    async with connection:
        # Create channel
        channel = await connection.channel()
        
        # Set QoS to process one message at a time
        await channel.set_qos(prefetch_count=1)
        
        # Declare the work queue
        work_queue = config['work_queue']
        queue = await channel.declare_queue(work_queue, durable=True)
        
        print(f"Listening for diarization jobs on queue: {work_queue}")
        print("Waiting for diarization jobs. To exit press CTRL+C")
        
        # Start consuming messages
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_message(message, pipeline)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='RabbitMQ consumer for diarization jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "diarization_work",
    "response_queue": "diarization_response",
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
