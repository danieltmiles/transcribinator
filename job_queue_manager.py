import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from anyio import create_task_group, create_memory_object_stream
from starlette.websockets import WebSocket, WebSocketState, WebSocketDisconnect

from dao import (
    get_next_queued_job, 
    update_queue_job_status, 
    update_job_status,
    cleanup_completed_queue_jobs
)
from ai import process_audio as ai_process_audio
from utils import TranscriptionJob

logger = logging.getLogger(__name__)


@dataclass
class QueuedJob:
    job_id: str
    file_path: str
    transcription_job: TranscriptionJob


class JobQueueManager:
    def __init__(self, active_jobs: Dict[str, TranscriptionJob]):
        self.active_jobs = active_jobs
        self.is_processing = False
        self.current_job_id: Optional[str] = None
        self.worker_task: Optional[asyncio.Task] = None
        self.should_stop = False
        
    async def start(self):
        """Start the queue processing worker."""
        if self.worker_task and not self.worker_task.done():
            logger.warning("Queue worker already running")
            return
            
        self.should_stop = False
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Job queue manager started")
        
    async def stop(self):
        """Stop the queue processing worker."""
        self.should_stop = True
        if self.worker_task:
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Job queue manager stopped")
        
    async def _worker_loop(self):
        """Main worker loop that processes jobs one at a time."""
        while not self.should_stop:
            try:
                if not self.is_processing:
                    next_job = get_next_queued_job()
                    if next_job:
                        job_id, file_path = next_job
                        await self._process_job(job_id, file_path)
                    else:
                        # Clean up old completed jobs periodically
                        cleanup_completed_queue_jobs()
                        
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in queue worker loop: {e}")
                await asyncio.sleep(5)  # Wait longer on error
                
    async def _process_job(self, job_id: str, file_path: str):
        """Process a single job."""
        if self.is_processing:
            logger.warning(f"Attempted to process job {job_id} while another job is processing")
            return
            
        self.is_processing = True
        self.current_job_id = job_id
        
        try:
            logger.info(f"Starting to process job {job_id}")
            
            # Update queue status to processing
            update_queue_job_status(job_id, 'processing')
            
            # Get the job from active_jobs
            job = self.active_jobs.get(job_id)
            if not job:
                logger.error(f"Job {job_id} not found in active_jobs")
                update_queue_job_status(job_id, 'error', 'Job not found in active jobs')
                return
                
            # Update job status to processing
            job.status = "processing"
            
            # Send initial progress if websocket is connected
            if job.websocket and job.websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await job.websocket.send_json({
                        "type": "progress",
                        "progress": 0,
                        "stage": "Initializing Transcription"
                    })
                except WebSocketDisconnect:
                    logger.warning(f"WebSocket disconnected for job {job_id}")
                    job.websocket = None
                    
            # Process the audio file
            await self._process_audio_file(job_id, Path(file_path), job)
            
            logger.info(f"Completed processing job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            update_queue_job_status(job_id, 'error', str(e))
            
            # Update job with error
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = "error"
                job.error = str(e)
                
                # Send error via websocket
                if job.websocket and job.websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await job.websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                    except WebSocketDisconnect:
                        pass
                        
        finally:
            self.is_processing = False
            self.current_job_id = None
            
    async def _process_audio_file(self, job_id: str, file_path: Path, job: TranscriptionJob):
        """Process the actual audio file using the AI module."""
        try:
            # Create streams for progress and transcript
            progress_send_stream, progress_receive_stream = create_memory_object_stream[dict]()
            transcript_send_stream, transcript_receive_stream = create_memory_object_stream[str]()
            
            async with create_task_group() as tg:
                # Start AI processing
                tg.start_soon(ai_process_audio, file_path, 1.0, progress_send_stream, transcript_send_stream)
                
                # Track progress stages
                stage_progress = {"diarization": 0, "transcription": 0, "cleanup": 0}
                cleanup_done = False
                
                # Process progress updates
                while not cleanup_done:
                    prog_msg = await progress_receive_stream.receive()
                    
                    if isinstance(prog_msg, dict):
                        stage = prog_msg.get("stage")
                        progress = int(prog_msg.get("progress", 0))
                        
                        if stage in stage_progress:
                            stage_progress[stage] = progress
                            
                        # Update job progress
                        job.progress = progress
                        
                        # Send progress via websocket
                        if job.websocket and job.websocket.client_state == WebSocketState.CONNECTED:
                            try:
                                await job.websocket.send_json({
                                    "type": "progress",
                                    "stage": stage,
                                    "progress": progress
                                })
                            except WebSocketDisconnect:
                                logger.warning(f"WebSocket disconnected for job {job_id}")
                                job.websocket = None
                                
                        if stage == "cleanup" and progress >= 100:
                            cleanup_done = True
                    else:
                        logger.warning(f"Unexpected progress message: {prog_msg}")
                        
                # Get the final transcript
                transcript = await transcript_receive_stream.receive()
                
                # Update job with results
                job.transcript = transcript
                job.status = "completed"
                job.progress = 100
                
                # Update database
                update_job_status(job_id, "completed", transcript)
                update_queue_job_status(job_id, 'completed')
                
                # Send final results via websocket
                if job.websocket and job.websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await job.websocket.send_json({
                            "type": "progress",
                            "progress": 100,
                            "stage": "Complete"
                        })
                        await job.websocket.send_json({
                            "type": "transcript",
                            "text": transcript
                        })
                    except WebSocketDisconnect:
                        pass
                        
        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")
            
        finally:
            # Cleanup: delete the uploaded file
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted uploaded file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
                
    def get_status(self):
        """Get current queue manager status."""
        return {
            "is_processing": self.is_processing,
            "current_job_id": self.current_job_id,
            "worker_running": self.worker_task and not self.worker_task.done() if self.worker_task else False
        }
        
    def is_job_being_processed(self, job_id: str) -> bool:
        """Check if a specific job is currently being processed."""
        return self.current_job_id == job_id
