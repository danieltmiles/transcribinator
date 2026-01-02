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
            
            # Create job queue table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS job_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    priority INTEGER DEFAULT 0
                );
            """)
            
            # Create index for efficient queue processing
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_status_priority 
                ON job_queue(status, priority DESC, created_at ASC);
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


# Job Queue Operations
def enqueue_job(job_id: str, file_path: str, priority: int = 0):
    """Add a job to the processing queue."""
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO job_queue (job_id, file_path, priority)
                VALUES (?, ?, ?)
            """, (job_id, file_path, priority))
            conn.commit()
        finally:
            conn.close()


def get_next_queued_job():
    """Get the next job from the queue (highest priority, oldest first)."""
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT job_id, file_path FROM job_queue 
                WHERE status = 'queued' 
                ORDER BY priority DESC, created_at ASC 
                LIMIT 1
            """)
            result = cur.fetchone()
            return result if result else None
        finally:
            conn.close()
    return None


def update_queue_job_status(job_id: str, status: str, error_message: Optional[str] = None):
    """Update the status of a job in the queue."""
    if conn := create_connection():
        try:
            cur = conn.cursor()
            timestamp_field = None
            if status == 'processing':
                timestamp_field = 'started_at'
            elif status in ['completed', 'error']:
                timestamp_field = 'completed_at'
            
            if timestamp_field:
                cur.execute(f"""
                    UPDATE job_queue 
                    SET status = ?, {timestamp_field} = CURRENT_TIMESTAMP, error_message = ?
                    WHERE job_id = ?
                """, (status, error_message, job_id))
            else:
                cur.execute("""
                    UPDATE job_queue 
                    SET status = ?, error_message = ?
                    WHERE job_id = ?
                """, (status, error_message, job_id))
            conn.commit()
        finally:
            conn.close()


def get_queue_status():
    """Get the current queue status."""
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM job_queue 
                GROUP BY status
            """)
            status_counts = dict(cur.fetchall())
            
            # Get position of queued jobs
            cur.execute("""
                SELECT job_id, 
                       ROW_NUMBER() OVER (ORDER BY priority DESC, created_at ASC) as position
                FROM job_queue 
                WHERE status = 'queued'
            """)
            queue_positions = dict(cur.fetchall())
            
            return {
                'status_counts': status_counts,
                'queue_positions': queue_positions
            }
        finally:
            conn.close()
    return {'status_counts': {}, 'queue_positions': {}}


def get_job_queue_position(job_id: str):
    """Get the position of a specific job in the queue."""
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT position FROM (
                    SELECT job_id, 
                           ROW_NUMBER() OVER (ORDER BY priority DESC, created_at ASC) as position
                    FROM job_queue 
                    WHERE status = 'queued'
                ) WHERE job_id = ?
            """, (job_id,))
            result = cur.fetchone()
            return result[0] if result else None
        finally:
            conn.close()
    return None


def cleanup_completed_queue_jobs(older_than_hours: int = 24):
    """Clean up completed/error jobs older than specified hours."""
    if conn := create_connection():
        try:
            cur = conn.cursor()
            cur.execute("""
                DELETE FROM job_queue 
                WHERE status IN ('completed', 'error') 
                AND completed_at < datetime('now', '-{} hours')
            """.format(older_than_hours))
            deleted_count = cur.rowcount
            conn.commit()
            return deleted_count
        finally:
            conn.close()
    return 0
