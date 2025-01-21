from contextlib import asynccontextmanager
from logging import Logger

from anyio import create_task_group, create_memory_object_stream
from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio
import uuid
import aiofiles
import logging
from typing import Dict, Optional

from starlette.responses import FileResponse
from starlette.websockets import WebSocketState

from ai import process_audio as ai_process_audio
from dao import save_transcription, init_db, get_jobs, update_job_status, create_connection
from utils import TranscriptionJob

from starlette.websockets import WebSocketDisconnect

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Store active jobs
active_jobs: Dict[str, TranscriptionJob] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    init_db()
    active_jobs.update(get_jobs())
    yield


# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


async def process_audio(job_id: str, file_path: Path, num_speakers: int):
    """
    Process audio file and send progress updates via WebSocket.
    Replace the sleep calls with your actual AI model processing.
    """
    await asyncio.sleep(1)  # allow for websocket connection
    job = active_jobs[job_id]

    try:
        # Update status to processing
        job.status = "processing"
        if job.websocket:
            await job.websocket.send_json({
                "type": "progress",
                "progress": 0,
                "stage": "Initializing Transcription"
            })
        # Replace this with your actual transcription code
        # Example:
        # model = YourTranscriptionModel()
        # transcript = model.transcribe(str(file_path))
        progress_send_stream, progress_receive_stream = create_memory_object_stream[int]()
        transcript_send_stream, transcript_receive_stream = create_memory_object_stream[str]()
        async with create_task_group() as tg:
            tg.start_soon(ai_process_audio, file_path, num_speakers, 1.0, progress_send_stream, transcript_send_stream)
            #progress: int = await progress_receive_stream.receive()
            progress: int = 0
            while progress < 100:
                job = active_jobs[job_id]
                if job and job.websocket and job.websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await job.websocket.send_json({
                            "type": "progress",
                            "progress": progress,
                            "stage": "Transcribing audio"
                        })
                    except WebSocketDisconnect:
                        logger.error("web socket has disconnected")
                        active_jobs[job_id] = None
                else:
                    logger.info("no websocket, not sending progress")
                progress = await progress_receive_stream.receive()
            transcript = await transcript_receive_stream.receive()
            update_job_status(job_id, "completed", transcript)

        # Update job with transcript
        job.transcript = transcript
        job.status = "completed"
        job.progress = 100

        # Send final result
        if job.websocket:
            await job.websocket.send_json({
                "type": "progress",
                "progress": 100,
                "stage": "Complete"
            })
            await job.websocket.send_json({
                "type": "transcript",
                "text": transcript
            })

    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        job.status = "error"
        job.error = str(e)
        if job.websocket:
            await job.websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    finally:
        # Cleanup: delete the uploaded file
        try:
            pass
            # file_path.unlink()
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")

@app.get("/")
async def read_root():
    return FileResponse("static/transcribe.html")
@app.get("/transcribe.js")
async def read_root():
    return FileResponse("static/transcribe.js")

@app.post("/upload")
async def upload_file(
        file: UploadFile = File(...),
        num_speakers: int = Form(...),
        file_name: str = Form(...),
        background_tasks: BackgroundTasks = None):
    """Handle file upload and start transcription process."""
    try:
        # Generate unique job ID
        job_id = str(uuid.uuid4())

        # Save file
        file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
        logger.info(f"Uploading file: {file_path}")
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        # Create job
        job = TranscriptionJob(
            job_id=job_id,
            filename=file.filename,
            human_readable_filename=file_name,
            status="uploaded",
            progress=0,
        )
        active_jobs[job_id] = job
        save_transcription(job)

        # Start processing in background
        background_tasks.add_task(process_audio, job_id, file_path, num_speakers)

        return {"job_id": job_id}

    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return {"error": str(e)}, 500


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """Handle WebSocket connection for job updates."""
    await websocket.accept()

    if job_id not in active_jobs:
        await websocket.send_json({
            "type": "error",
            "message": "Job not found"
        })
        await websocket.close()
        return

    job = active_jobs[job_id]
    active_jobs[job_id].websocket = websocket

    try:
        # If job is already completed, send transcript immediately
        if job.status == "completed" and job.transcript:
            await websocket.send_json({
                "type": "progress",
                "progress": 100,
                "stage": "Complete"
            })
            await websocket.send_json({
                "type": "transcript",
                "text": job.transcript
            })

        # If job had an error, send error immediately
        elif job.status == "error" and job.error:
            await websocket.send_json({
                "type": "error",
                "message": job.error
            })

        # Otherwise, keep connection open for updates
        else:
            while job.status not in ["completed", "error"]:
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {str(e)}")
        if not websocket.client_state.DISCONNECTED:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })

    finally:
        job.websocket = None
        # Clean up old jobs
        if job.status in ["completed", "error"]:
            active_jobs.pop(job_id, None)


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    if job_id not in active_jobs:
        return {"error": "Job not found"}, 404

    job = active_jobs[job_id]
    return {
        "job_id": job.job_id,
        "status": job.status,
        "progress": job.progress,
        "filename": job.filename,
    }


# Optional: Cleanup endpoint for development
@app.post("/cleanup")
async def cleanup_jobs():
    """Remove all completed and error jobs."""
    removed = []
    for job_id in list(active_jobs.keys()):
        job = active_jobs[job_id]
        if job.status in ["completed", "error"]:
            active_jobs.pop(job_id)
            removed.append(job_id)
    return {"removed_jobs": removed}


@app.get("/jobs")
async def get_all_jobs():
    """Get list of all transcription jobs."""
    conn = create_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute('SELECT job_id, human_readable_filename, status, transcript FROM transcription_jobs')
            jobs = c.fetchall()
            return [{
                "job_id": job[0],
                "filename": job[1],
                "status": job[2],
                "transcript": job[3]
            } for job in jobs]
        finally:
            conn.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

