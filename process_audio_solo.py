import asyncio
from pathlib import Path

import anyio
from anyio import create_memory_object_stream
from anyio.streams.memory import MemoryObjectSendStream

from ai import process_audio

async def make_transcript(audio_file_path: str, result_send: MemoryObjectSendStream) -> str:
    print(f"make transcript for {audio_file_path}")
    transcript_send_stream, transcript_receive_stream = create_memory_object_stream[str](max_buffer_size=1000)
    
    total_text = ""
    
    async def process_audio_task():
        """Run the audio processing pipeline"""
        await process_audio(
            audio_file_path=audio_file_path,
            transcript_send_stream=transcript_send_stream,
        )
        print(f"finished process audio for {audio_file_path}")
    
    async def consume_transcript_stream():
        """Consume transcript parts as they arrive and print them immediately"""
        nonlocal total_text
        async with transcript_receive_stream:
            async for ts_part in transcript_receive_stream:
                print(f"\n{'='*80}")
                print(f"TRANSCRIPT UPDATE for {audio_file_path}:")
                print(f"{'='*80}")
                print(ts_part)
                print(f"{'='*80}\n")
                total_text += ts_part
        print(f"exited transcript_receive_stream for {audio_file_path}")
    
    # Run both tasks concurrently so transcript text is displayed as it arrives
    async with anyio.create_task_group() as tg:
        tg.start_soon(process_audio_task)
        tg.start_soon(consume_transcript_stream)
    
    async with result_send:  # closes when context exits
        print(f"sending complete transcript for {audio_file_path}")
        await result_send.send((audio_file_path, total_text))
    print("closed result send clone, exiting make_transcript")

async def limited_worker(semaphore, work_func, *args, **kwargs):
    async with semaphore:
        return await work_func(*args, **kwargs)

async def main():
    # Find all audio files in the target directory
    audio_dir = Path("/Users/dmiles/portland_city_council_audio")
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    audio_files = [
        str(f) for f in audio_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    semaphore = anyio.Semaphore(3)
    result_send, result_receive = create_memory_object_stream[str](max_buffer_size=10)
    async def save_transcript(result_receive):
        async with result_receive:
            print("waiting for result_receive")
            async for audio_file_path, transcript_text in result_receive:
                # Create output filename based on input filename
                output_filename = Path(audio_file_path).stem + "_transcript.txt"
                print(f"saving transcript for {audio_file_path} to {output_filename}")
                with open(output_filename, "w") as fl:
                    fl.write(transcript_text)
    async with anyio.create_task_group() as tg:
        tg.start_soon(save_transcript, result_receive)
        for audio_file in audio_files:
            # Start whisper jobs - will stream results to segment_send_stream
            tg.start_soon(
                limited_worker,
                semaphore,
                make_transcript,
                audio_file,
                result_send.clone(),
            )
    print("exited task group")
    print("closing result_send")
    await result_send.aclose()

if __name__ == "__main__":
    asyncio.run(main())
