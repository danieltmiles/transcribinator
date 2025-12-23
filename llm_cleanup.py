import json
import aio_pika
import argparse
import torch
import time
import asyncio

from aio_pika.abc import AbstractIncomingMessage
from pamqp.commands import Basic
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM

from utils import load_config, create_ssl_context
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


def load_llm_model():
    """
    Returns:
        tuple: (model_kit, tokenizer)
    """
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype = torch.float16,  # Force FP16
        device_map = "auto",
        trust_remote_code = True,
        max_memory = {0: "22GiB"},  # Reserve a bit less than total VRAM
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    return model, tokenizer


def llm_clean(text: str, model, tokenizer) -> str | None:
    """Clean transcribed text using LLM with mlx_engine."""
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
    try:
        outputs = model.generate(
            encoded.input_ids.to(device),
            attention_mask=attention_mask,
            max_new_tokens=prompt_length,  # Allow some buffer for expanded text
            temperature=0.2,  # Lower temperature for more consistent outputs
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2
        )
    except RuntimeError as rte:
        print(f"experienced rte: {rte}")
        return ""
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract the cleaned text between delimiters
    begin_delim = "BEGIN OUTPUT TEXT:"
    full_output = full_output[len(prompt) - len(begin_delim) - 1:]
    end_delimiter = "END OUTPUT TEXT"
    if end_delimiter in full_output:
        begin_indexes = [i for i in range(len(full_output)) if full_output.startswith(begin_delim, i)]
        end_delim = "END OUTPUT TEXT"
        end_indexes = [i for i in range(len(full_output)) if full_output.startswith(end_delim, i)]
        if len(end_indexes) > 0:
            # Get the text between the last BEGIN and first END after it
            if len(begin_indexes) > 0:
                chunk = full_output[begin_indexes[-1] + len(begin_delim):end_indexes[0]]
            else:
                # If no BEGIN found, just get text before END
                chunk = full_output[:end_indexes[0]]
            return chunk.strip()
    
    # If no proper delimiter found, return None or the full output
    print("Warning: END OUTPUT TEXT delimiter not found in output")
    return None


async def process_message(message: AbstractIncomingMessage, model: Qwen2ForCausalLM, tokenizer: AutoTokenizer):
    """
    Process an LLM cleanup job message from RabbitMQ.
    
    Expected message format:
    {
        'job_id': str,
        'reply_to': str (queue name for response),
        'segment': {
            'speaker': str,
            'start': float,
            'end': float,
            'text': str,
            'segment_count': int
        },
        'segment_count': int,
        'total_segments': int
    }
    """
    async with message.process():
        try:
            print(f"Received message")
            body = json.loads(message.body.decode())
            
            job_id = body.get('job_id')
            reply_to = body.get('reply_to')
            segment = body.get('segment', {})
            segment_count = body.get('segment_count')
            total_segments = body.get('total_segments')
            
            text = segment.get('text', '')
            
            print(f"Processing job {job_id}, segment {segment_count}/{total_segments}")
            
            # Run cleanup in thread pool to prevent blocking the event loop
            loop = asyncio.get_event_loop()
            cleaned_text = await loop.run_in_executor(
                None,
                llm_clean,
                text,
                model,
                tokenizer
            )
            
            # Prepare response
            response = {
                'job_id': job_id,
                'status': 'success',
                'segment': {
                    'speaker': segment.get('speaker'),
                    'start': segment.get('start'),
                    'end': segment.get('end'),
                    'text': cleaned_text,
                    'segment_count': segment.get('segment_count')
                },
                'segment_count': segment_count,
                'total_segments': total_segments,
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

            print(f"Job {job_id} segment {segment_count} completed and response sent to {reply_to}")
            
        except Exception as e:
            print(f"Error processing message: {e}")
            import traceback
            traceback.print_exc()
            
            # Send error response if possible
            try:
                body = json.loads(message.body.decode())
                job_id = body.get('job_id', 'unknown')
                reply_to = body.get('reply_to')
                segment_count = body.get('segment_count')
                
                if reply_to:
                    error_response = {
                        'job_id': job_id,
                        'status': 'error',
                        'error': str(e),
                        'segment_count': segment_count
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
    Main function to start the LLM cleanup consumer.
    """
    print("Initializing LLM cleanup consumer...")
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
        
        print(f"Listening for LLM cleanup jobs on queue: {work_queue}")
        print("Waiting for cleanup jobs. To exit press CTRL+C")

        model, tokenizer = load_llm_model()
        # Start consuming messages
        async with queue.iterator() as queue_iter:
            async for message in queue_iter:
                await process_message(message, model, tokenizer)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='RabbitMQ consumer for LLM text cleanup jobs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Configuration file format (JSON):
{
    "work_queue": "llm/cleanup",
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

    # Load configuration from file
    config = load_config(args.config_file)

    print(f"Loaded configuration from: {args.config_file}")
    print(f"Work queue: {config['work_queue']}")
    print(f"Model path: {config.get('model_path', '/Users/dmiles/.lmstudio/models/lmstudio-community/Qwen3-32B-MLX-4bit')}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")

    # Run main with config
    try:
        asyncio.run(main(config))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
