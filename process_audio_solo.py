import asyncio
from pathlib import Path

import anyio
from anyio import create_memory_object_stream
from anyio.streams.memory import MemoryObjectSendStream

from ai import process_audio

async def make_transcript(audio_file_path: str, result_send: MemoryObjectSendStream) -> str:
    print(f"make transcript for {audio_file_path}")
    transcript_send_stream, transcript_receive_stream = create_memory_object_stream[str](max_buffer_size=1000)
    await process_audio(
        audio_file_path=audio_file_path,
        min_segment_length=1.0,
        transcript_send_stream=transcript_send_stream,
    )
    print(f"finished process audio for {audio_file_path}")
    total_text = ""
    async with transcript_receive_stream:
        async for ts_part in transcript_receive_stream:
            print("waiting for ts_part")
            total_text += ts_part
            print("got ts_part")
    print("exited transcript_receive_stream")
    async with result_send:  # closes when context exits
        print(f"sending transcript for {audio_file_path}")
        await result_send.send((audio_file_path, total_text))
    print("closed result send clone, exiting make_transcript")

async def limited_worker(semaphore, work_func, *args, **kwargs):
    async with semaphore:
        return await work_func(*args, **kwargs)

async def main():
    # Find all audio files in the target directory
    audio_dir = Path("/Volumes/SharedDrive/portland_city_council")
    audio_extensions = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    audio_files = [
        str(f) for f in audio_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    semaphore = anyio.Semaphore(3)
    result_send, result_receive = create_memory_object_stream[str](max_buffer_size=10)
    async with anyio.create_task_group() as tg:
        for audio_file in audio_files[:2]:
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
    async with result_receive:
        print("waiting for result_receive")
        async for audio_file_path, transcript_text in result_receive:
            # Create output filename based on input filename
            output_filename = Path(audio_file_path).stem + "_transcript.txt"
            print(f"saving transcript for {audio_file_path} to {output_filename}")
            with open(output_filename, "w") as fl:
                fl.write(transcript_text)

if __name__ == "__main__":
    asyncio.run(main())
