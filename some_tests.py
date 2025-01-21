import asyncio

from anyio import create_memory_object_stream, create_task_group
from ai import process_audio as ai_process_audio


async def amain():
   file_path = "uploads/short.wav"
   num_speakers = 1

   progress_send_stream, progress_receive_stream = create_memory_object_stream[int]()
   transcript_send_stream, transcript_receive_stream = create_memory_object_stream[str]()
   async with create_task_group() as tg:
       tg.start_soon(ai_process_audio, file_path, num_speakers, 1.0, progress_send_stream, transcript_send_stream)
       progress: int = await progress_receive_stream.receive()
       while progress < 100:
           # if job.websocket:
           #     await job.websocket.send_json({
           #         "type": "progress",
           #         "progress": progress,
           #         "stage": "Transcribing audio"
           #     })
           print(f"send progress {progress} to websocket")
           progress = await progress_receive_stream.receive()
       transcript = await transcript_receive_stream.receive()
   print(transcript)


if __name__ == "__main__":
    asyncio.run(amain())