from dataclasses import dataclass
from typing import Optional

from starlette.websockets import WebSocket


@dataclass
class TranscriptionJob:
    job_id: str
    filename: str
    human_readable_filename: str
    status: str
    progress: int
    websocket: Optional[WebSocket] = None
    transcript: Optional[str] = None
    error: Optional[str] = None
