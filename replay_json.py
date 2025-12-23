#!/usr/bin/env python3
"""
Script to replay messages from a JSON file back to a RabbitMQ queue.
Reads messages saved by save.json.py and publishes them to the specified queue.
"""
import json
import argparse
import asyncio
import aio_pika
from datetime import datetime
from utils import create_ssl_context


async def replay_messages(config, input_file, queue_name=None):
    """
    Replay messages from JSON file to RabbitMQ queue.
    
    Args:
        config: Configuration dictionary with RabbitMQ connection parameters
        input_file: Path to input JSON file containing messages
        queue_name: Optional queue name (overrides queue from JSON metadata)
    """
    # Load messages from JSON file
    print(f"Loading messages from: {input_file}")
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        return
    
    # Extract metadata and messages
    metadata = data.get('metadata', {})
    messages = data.get('messages', [])
    
    if not messages:
        print("No messages found in input file")
        return
    
    # Determine target queue
    target_queue = queue_name or metadata.get('queue_name') or config.get('work_queue')
    if not target_queue:
        raise ValueError("Queue name must be provided via --queue, in JSON metadata, or in config file as 'work_queue'")
    
    print(f"Found {len(messages)} messages to replay")
    print(f"Original queue: {metadata.get('queue_name', 'unknown')}")
    print(f"Target queue: {target_queue}")
    print(f"Original download timestamp: {metadata.get('download_timestamp', 'unknown')}")
    
    print(f"\nConnecting to RabbitMQ at {config['host']}:{config['port']}...")
    
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
    
    async with connection:
        # Create channel
        channel = await connection.channel()
        
        # Declare the queue
        queue = await channel.declare_queue(target_queue, durable=True)
        
        print(f"Replaying messages to queue: {target_queue}")
        
        # Replay each message
        success_count = 0
        error_count = 0
        
        for i, msg_data in enumerate(messages):
            try:
                # Reconstruct message properties
                # Convert timestamp back to datetime if present
                timestamp = None
                if msg_data.get('timestamp'):
                    try:
                        timestamp = datetime.fromisoformat(msg_data['timestamp'])
                    except:
                        pass
                
                # Convert delivery_mode back to enum if present
                delivery_mode = None
                if msg_data.get('delivery_mode') is not None:
                    delivery_mode = aio_pika.DeliveryMode(msg_data['delivery_mode'])
                
                # Create message with original properties
                message = aio_pika.Message(
                    body=msg_data['body'].encode('utf-8'),
                    message_id=msg_data.get('message_id'),
                    correlation_id=msg_data.get('correlation_id'),
                    content_type=msg_data.get('content_type'),
                    content_encoding=msg_data.get('content_encoding'),
                    delivery_mode=delivery_mode,
                    priority=msg_data.get('priority'),
                    timestamp=timestamp,
                    type=msg_data.get('type'),
                    user_id=msg_data.get('user_id'),
                    app_id=msg_data.get('app_id'),
                    reply_to=msg_data.get('reply_to'),
                    expiration=msg_data.get('expiration'),
                    headers=msg_data.get('headers'),
                )
                
                # Publish message to the queue
                await channel.default_exchange.publish(
                    message,
                    routing_key=target_queue,
                )
                
                success_count += 1
                
                if (i + 1) % 10 == 0:
                    print(f"Replayed {i + 1}/{len(messages)} messages...")
                    
            except Exception as e:
                error_count += 1
                print(f"Error replaying message {i}: {e}")
                # Continue with next message
        
        print(f"\nReplay complete:")
        print(f"  Successfully replayed: {success_count} messages")
        if error_count > 0:
            print(f"  Errors: {error_count} messages")


def main():
    parser = argparse.ArgumentParser(
        description='Replay messages from JSON file to RabbitMQ queue',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Replay to original queue (from JSON metadata)
  python replay_json.py rabbitmq_config.json saved_messages.json
  
  # Replay to specific queue
  python replay_json.py rabbitmq_config.json saved_messages.json --queue my_queue
  
  # Replay to queue from config file
  python replay_json.py rabbitmq_config.json saved_messages.json

Configuration file format (JSON):
{
    "host": "localhost",
    "port": 5672,
    "username": "guest",
    "password": "guest",
    "work_queue": "default_queue"  # optional, can be overridden with --queue
}

Input JSON file format (created by save.json.py):
{
    "metadata": {
        "queue_name": "original_queue",
        "download_timestamp": "2024-01-01T12:00:00",
        "message_count": 10
    },
    "messages": [
        {
            "message_id": "...",
            "body": "...",
            "headers": {...},
            ...
        }
    ]
}
        '''
    )
    parser.add_argument(
        'config_file',
        type=str,
        help='Path to the JSON configuration file with RabbitMQ connection parameters'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input JSON file containing messages (created by save.json.py)'
    )
    parser.add_argument(
        '--queue',
        type=str,
        help='Queue name to replay to (overrides JSON metadata and config file)',
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
    print(f"Input file: {args.input_file}")
    
    # Run async replay
    try:
        asyncio.run(replay_messages(config, args.input_file, args.queue))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
