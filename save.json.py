#!/usr/bin/env python3
"""
Script to download all messages from a RabbitMQ queue and save them to a JSON file.
Messages are NOT acknowledged, so they remain in the queue for replay later.
"""
import json
import argparse
import asyncio
import aio_pika
from datetime import datetime
from utils import create_ssl_context


async def download_messages(config, output_file, queue_name=None, max_messages=None):
    """
    Download all messages from RabbitMQ queue and save to JSON file.
    
    Args:
        config: Configuration dictionary with RabbitMQ connection parameters
        output_file: Path to output JSON file
        queue_name: Optional queue name (overrides config['work_queue'])
        max_messages: Optional maximum number of messages to download
    """
    # Use provided queue_name or fall back to config
    queue_name = queue_name or config.get('work_queue')
    if not queue_name:
        raise ValueError("Queue name must be provided via --queue or in config file as 'work_queue'")
    
    print(f"Connecting to RabbitMQ at {config['host']}:{config['port']}...")
    
    # Create SSL context
    ssl_context = create_ssl_context()
    
    # Connect to RabbitMQ with TLS
    connection = await aio_pika.connect_robust(
        host=config['host'],
        port=config['port'],
        login=config['username'],
        password=config['password'],
        ssl=True,
        ssl_context=ssl_context,
    )
    
    messages = []
    
    async with connection:
        # Create channel
        channel = await connection.channel()
        
        # Declare the queue (passive=True means don't create if it doesn't exist)
        try:
            queue = await channel.declare_queue(queue_name, durable=True, passive=False)
        except Exception as e:
            print(f"Error declaring queue '{queue_name}': {e}")
            return
        
        message_count = queue.declaration_result.message_count
        print(f"Queue '{queue_name}' has {message_count} messages")
        
        if message_count == 0:
            print("No messages to download")
            return
        
        # Determine how many messages to download
        download_count = message_count
        if max_messages is not None:
            download_count = min(message_count, max_messages)
            print(f"Downloading up to {download_count} messages...")
        else:
            print(f"Downloading all {download_count} messages...")
        
        # Download messages without acknowledging them
        for i in range(download_count):
            try:
                # Get message with no_ack=False (we won't acknowledge it)
                message = await queue.get(timeout=5.0, no_ack=False)
                
                if message is None:
                    print(f"No more messages available after {i} messages")
                    break
                
                # Extract message data
                message_data = {
                    'message_id': message.message_id,
                    'correlation_id': message.correlation_id,
                    'content_type': message.content_type,
                    'content_encoding': message.content_encoding,
                    'delivery_mode': message.delivery_mode.value if message.delivery_mode else None,
                    'priority': message.priority,
                    'timestamp': message.timestamp.isoformat() if message.timestamp else None,
                    'type': message.type,
                    'user_id': message.user_id,
                    'app_id': message.app_id,
                    'reply_to': message.reply_to,
                    'expiration': message.expiration,
                    'headers': message.headers,
                    'body': message.body.decode('utf-8', errors='replace'),  # Decode body as string
                    'delivery_tag': message.delivery_tag,
                    'routing_key': message.routing_key,
                }
                
                messages.append(message_data)
                
                # IMPORTANT: Don't ack or reject - let RabbitMQ auto-requeue on connection close
                # This allows us to get different messages each iteration
                
                if (i + 1) % 10 == 0:
                    print(f"Downloaded {i + 1}/{download_count} messages...")
                    
            except asyncio.TimeoutError:
                print(f"Timeout waiting for message after {i} messages")
                break
            except Exception as e:
                print(f"Error getting message {i}: {e}")
                break
        
        print(f"Successfully downloaded {len(messages)} messages")
    
    # Save to JSON file
    output_data = {
        'metadata': {
            'queue_name': queue_name,
            'download_timestamp': datetime.now().isoformat(),
            'message_count': len(messages),
            'rabbitmq_host': config['host'],
            'rabbitmq_port': config['port'],
        },
        'messages': messages
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"Messages saved to: {output_file}")
    print(f"Total messages saved: {len(messages)}")


def main():
    parser = argparse.ArgumentParser(
        description='Download messages from RabbitMQ queue and save to JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download from queue specified in config file
  python save.json.py rabbitmq_config.json output.json
  
  # Download from specific queue
  python save.json.py rabbitmq_config.json output.json --queue my_queue
  
  # Download only first 100 messages
  python save.json.py rabbitmq_config.json output.json --max-messages 100

Configuration file format (JSON):
{
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest",
    "work_queue": "default_queue"  # optional, can be overridden with --queue
}
        '''
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file with RabbitMQ connection parameters'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='Path to output JSON file where messages will be saved'
    )
    parser.add_argument(
        '--queue',
        type=str,
        help='Queue name to download from (overrides config file)',
        default=None
    )
    parser.add_argument(
        '--max-messages',
        type=int,
        help='Maximum number of messages to download (default: all)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Load configuration from file
    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        return
    
    # Validate required connection fields
    required_fields = ['host', 'port', 'username', 'password']
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        print(f"Error: Missing required fields in config file: {', '.join(missing_fields)}")
        return
    
    # Ensure port is an integer
    config['port'] = int(config['port'])
    
    print(f"Loaded configuration from: {args.config_file}")
    print(f"RabbitMQ host: {config['host']}:{config['port']}")
    print(f"Username: {config['username']}")
    print(f"Output file: {args.output_file}")
    
    # Run async download
    try:
        asyncio.run(download_messages(config, args.output_file, args.queue, args.max_messages))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
