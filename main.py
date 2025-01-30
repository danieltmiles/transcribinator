from contextlib import asynccontextmanager
from logging import Logger

from anyio import create_task_group, create_memory_object_stream
from fastapi import FastAPI, WebSocket, UploadFile, File, BackgroundTasks, Form, Depends, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import asyncio
import uuid
import aiofiles
import logging
from typing import Dict, Optional

from fastapi.params import Body
from fastapi.templating import Jinja2Templates
from pathlib import Path

templates = Jinja2Templates(directory="templates")

# Move your template files to a templates directory
template_dir = Path("templates")
template_dir.mkdir(exist_ok=True)

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from jose import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
import sqlite3

from starlette.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.websockets import WebSocketState

from ai import process_audio as ai_process_audio
from dao import save_transcription, init_db, get_jobs, update_job_status, create_connection
from utils import TranscriptionJob

import re

from starlette.websockets import WebSocketDisconnect

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.error("lifespan")
    # Load the ML model
    init_db()
    init_auth_db()
    yield


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

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


class CustomHTTPBearer(HTTPBearer):
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        logger.error("http bearer call")
        # Define paths that don't require authentication
        open_paths = {"/login", "/auth/login"}
        if request.url.path in open_paths:
            return None
        if re.match(r"^/ws.*", request.url.path):
            return None
        logger.error(request.url.path)

        # First check for cookie
        authorization = request.cookies.get("Authorization")

        # If no cookie, try header (for API calls)
        if not authorization:
            try:
                return await super().__call__(request)
            except HTTPException:
                raise HTTPException(
                    status_code=307,
                    detail="Not authenticated",
                    headers={"Location": "/login"}
                )
        return HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=authorization.replace("Bearer ", "")
        )

# Update the security instance
security = CustomHTTPBearer(auto_error=True)
SECRET_KEY = "31220446b6b9e0d7c663e212b33782b1f58ea19b58ccac50f3b7d38b7497fa0b"  # Change this to a secure secret key
ALGORITHM = "HS256"


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
        # def handle_tgerror(excgroup: ExceptionGroup) -> None:
        #     for exc in excgroup.exceptions:
        #         print(exc)
        #     raise exc
        try:
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
        except* Exception as excgroup:
            for exc in excgroup.exceptions:
                print(exc)
            raise exc

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

class UserCreate(BaseModel):
    email: str
    password: str

class User(BaseModel):
    email: str

def init_auth_db():
    conn = sqlite3.connect('transcriptions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def get_user_by_email(email: str) -> Optional[tuple]:
    conn = sqlite3.connect('transcriptions.db')
    c = conn.cursor()
    c.execute('SELECT email, password_hash FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return user

def create_user(email: str, password: str) -> bool:
    try:
        # Hash the password
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        
        conn = sqlite3.connect('transcriptions.db')
        c = conn.cursor()
        c.execute('INSERT INTO users (email, password_hash) VALUES (?, ?)',
                 (email, password_hash.decode('utf-8')))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)  # Token expires in 7 days
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    logger.error(f"get current user {credentials}")
    if credentials is None:  # For open paths
        return None
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        logger.error(f"{payload}")
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(
                status_code=307,  # Temporary redirect
                detail="Not authenticated",
                headers={"Location": "/login"}
            )
        return User(email=email)
    except (jwt.ExpiredSignatureError, jwt.JWTError):
        raise HTTPException(
            status_code=307,  # Temporary redirect
            detail="Not authenticated",
            headers={"Location": "/login"}
        )

@app.get("/login")
async def login_page(request: Request):
    return FileResponse("static/login.html")

@app.get("/")
async def read_root(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse(
        "transcribe.html.jinja2",
        {
            "request": request,  # Required by Jinja2Templates
            "email": user.email
        }
    )
@app.get("/auth.js")
async def read_auth():
    return FileResponse("static/auth.js")
@app.get("/transcribe.js")
async def read_transcribe():
    return FileResponse("static/transcribe.js")

@app.post("/upload")
async def upload_file(
        user: User = Depends(get_current_user),
        file: UploadFile = File(...),
        num_speakers: int = Form(...),
        file_name: str = Form(...),
        background_tasks: BackgroundTasks = None):
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
    save_transcription(job, user.email)

    # Start processing in background
    background_tasks.add_task(process_audio, job_id, file_path, num_speakers)

    return {"job_id": job_id}



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


@app.get("/jobs/{job_id}/speakers")
async def get_job_speakers(job_id: str):
    conn = create_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute('SELECT transcript FROM transcription_jobs WHERE job_id = ?', (job_id,))
            transcript = c.fetchone()[0]
            speakers = re.findall(r"\[.+\] (.+):", transcript)
            speakers_in_order = []
            for speaker in speakers:
                if speaker not in speakers_in_order:
                    speakers_in_order.append(speaker)
            return speakers_in_order
        finally:
            conn.close()


@app.post("/jobs/{job_id}/speakers")
async def update_job_speakers(job_id: str, updated_speakers: dict[str, str] = Body(...)):
    conn = create_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute('SELECT transcript FROM transcription_jobs WHERE job_id = ?', (job_id,))
            transcript = c.fetchone()[0]
            for old_speaker, new_speaker in updated_speakers.items():
                transcript = re.sub(old_speaker, new_speaker, transcript)
            c.execute('UPDATE transcription_jobs SET transcript = ? WHERE job_id = ?', (transcript, job_id))
            conn.commit()
            return {"transcript": transcript}
        finally:
            conn.close()
    else:
        raise HTTPException(status_code=404)


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
async def get_all_jobs(user: User = Depends(get_current_user)):
    """Get list of all transcription jobs."""
    conn = create_connection()
    if conn:
        try:
            c = conn.cursor()
            c.execute('SELECT job_id, human_readable_filename, status, transcript FROM transcription_jobs WHERE owner = ?', (user.email,))
            jobs = c.fetchall()
            return [{
                "job_id": job[0],
                "filename": job[1],
                "status": job[2],
                "transcript": job[3]
            } for job in jobs]
        finally:
            conn.close()



# Add these routes to your FastAPI app
@app.post("/auth/signup")
async def signup(user: UserCreate):
    if len(user.password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long"
        )
    
    if get_user_by_email(user.email):
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    success = create_user(user.email, user.password)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Failed to create user"
        )
    
    return {"message": "User created successfully"}

@app.post("/auth/login")
async def login(user: UserCreate):
    db_user = get_user_by_email(user.email)
    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    stored_password_hash = db_user[1].encode('utf-8')
    if not bcrypt.checkpw(user.password.encode('utf-8'), stored_password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    access_token = create_access_token({"sub": user.email})

    # response = RedirectResponse(url="/", status_code=303)
    response = JSONResponse(
        content={"redirect": "/"},
        status_code=200
    )
    response.set_cookie(
        key="Authorization",
        value=f"Bearer {access_token}",
        httponly=True,
        max_age=1800,  # 30 minutes
        secure=True,  # For HTTPS
        samesite="lax"
    )
    return response

# Modify your existing endpoints to require authentication
@app.get("/")
async def read_root(user: User = Depends(get_current_user)):
    return FileResponse("static/transcribe.html")

@app.post("/upload")
async def upload_file(
        file: UploadFile = File(...),
        num_speakers: int = Form(...),
        file_name: str = Form(...),
        background_tasks: BackgroundTasks = None,
        user: User = Depends(get_current_user)):
    # Your existing upload code here
    pass




if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


