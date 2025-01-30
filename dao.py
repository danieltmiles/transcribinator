import sqlite3
import logging
from typing import Optional

from utils import TranscriptionJob

LOGGER = logging.getLogger(__name__)


def create_connection():
    try:
        return sqlite3.connect("transcriptions.db")
    except sqlite3.Error as e:
        LOGGER.error(f"Error connecting to database: {e}")
        return None
def init_db():
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS transcription_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    human_readable_filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    transcript TEXT,
                    FOREIGN KEY (owner) REFERENCES users(email)
                );
            """)
            conn.commit()
        finally:
            conn.close()

def save_transcription(job: TranscriptionJob, user_email: str):
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO transcription_jobs (job_id, owner, filename, human_readable_filename, status, transcript)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (job.job_id, user_email, job.filename, job.human_readable_filename, job.status, job.transcript))
            conn.commit()
        finally:
            conn.close()

def get_jobs(user_email: str):
    jobs: dict[str, TranscriptionJob] = {}
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("SELECT job_id, filename, human_readable_filename, status FROM transcription_jobs WHERE owner = ?", user_email)
            for row in cur.fetchall():
                status = row[3] if row[3] == "completed" else "restarted"
                job_obj = TranscriptionJob(job_id=row[0], filename=row[1], human_readable_filename=row[2], status=status)
                jobs[row[0]] = job_obj
        finally:
            conn.close()
    return jobs

def update_job_status(job_id: str, status: str, transcript: Optional[str] = None):
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("UPDATE transcription_jobs SET status = ? WHERE job_id = ?", (status, job_id))
            if transcript:
                cur.execute("UPDATE transcription_jobs SET transcript = ? WHERE job_id = ?", (transcript, job_id))
            conn.commit()
        finally:
            conn.close()
