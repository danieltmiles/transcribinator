const uploadArea = document.getElementById('uploadArea');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const errorMessage = document.getElementById('errorMessage');
const transcriptContainer = document.getElementById('transcriptContainer');
const transcriptText = document.getElementById('transcriptText');
const copyButton = document.getElementById('copyButton');

const API_URL = 'https://transcribe.doodledome.org';  // Replace with your API URL

// Drag and drop handlers
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.add('dragover');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, () => {
        uploadArea.classList.remove('dragover');
    });
});

// Handle file drop
uploadArea.addEventListener('drop', handleDrop);
uploadArea.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'audio/*';
    input.onchange = e => {
        const file = e.target.files[0];
        handleFile(file);
    };
    input.click();
});

function handleDrop(e) {
    const file = e.dataTransfer.files[0];
    handleFile(file);
}

async function handleFile(file) {
    if (!file.type.startsWith('audio/')) {
        showError('Please upload an audio file.');
        return;
    }

    // Store the file and show dialog
    currentFile = file;
    speakerDialog.style.display = 'flex';
}

async function uploadFile(file, numSpeakers, fileName) {
    // Reset UI
    errorMessage.style.display = 'none';
    transcriptContainer.style.display = 'none';
    progressContainer.style.display = 'block';

    try {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('num_speakers', numSpeakers);
        formData.append('file_name', fileName);

        const response = await fetch('https://transcribe.doodledome.org/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Upload failed');
        }

        const { job_id } = await response.json();
        connectWebSocket(job_id);

    } catch (error) {
        showError('Failed to upload file: ' + error.message);
    }
}

async function loadPreviousTranscripts() {
    try {
        const response = await fetch('/jobs');
        const jobs = await response.json();
        const fileList = document.getElementById('fileList');
        fileList.innerHTML = '';

        jobs.forEach(job => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <h3>${job.filename}</h3>
                <p>Status: ${job.status}</p>
            `;

            if (job.transcript) {
                fileItem.addEventListener('click', () => {
                    document.getElementById('transcriptText').textContent = job.transcript;
                    document.getElementById('transcriptContainer').style.display = 'block';
                });
            }

            fileList.appendChild(fileItem);
        });
    } catch (error) {
        console.error('Error loading previous transcripts:', error);
    }
}

// Handle dialog buttons
document.getElementById('confirmUpload').addEventListener('click', () => {
    const speakers = parseInt(numSpeakers.value);
    if (speakers < 1 || speakers > 10) {
        alert('Please enter a number between 1 and 10');
        return;
    }
    const fileName = document.getElementById('fileName').value.trim();
    speakerDialog.style.display = 'none';
    if (currentFile) {
        uploadFile(currentFile, speakers, fileName);
        currentFile = null;
    }
});

document.getElementById('cancelUpload').addEventListener('click', () => {
    speakerDialog.style.display = 'none';
    currentFile = null;
});

function connectWebSocket(jobId, repeat_count = 0) {
    // // Close existing connection if any
    // if (websocket) {
    //     websocket.close();
    // }
    if (repeat_count > 10) {
        return;
    }

    websocket = new WebSocket(`wss://transcribe.doodledome.org/ws/${jobId}`);

    websocket.onmessage = function(event) {
        const data = JSON.parse(event.data);

        switch(data.type) {
            case 'progress':
                updateProgress(data.progress);
                updateStage(data.stage);
                break;

            case 'transcript':
                showTranscript(data.text);
                websocket.close();
                break;

            case 'error':
                showError(data.message);
                websocket.close();
                break;
        }
    };

    websocket.onerror = function(error) {
        showError('WebSocket error occurred', error);
        websocket.close();
        connectWebSocket(jobId, repeat_count + 1);
    };

    websocket.onclose = function() {
        websocket = null;
        showError('WebSocket connection closed unexpectedly');
	connectWebSocket(jobId, repeat_count + 1);
    };
}

function updateStage(stage) {
    document.getElementById('stageText').textContent = stage;
}

function updateProgress(percent) {
    progressBar.style.width = `${percent}%`;
    progressText.textContent = `${percent}%`;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    progressContainer.style.display = 'none';
}

function showTranscript(text) {
    transcriptText.textContent = text;
    transcriptContainer.style.display = 'block';
    progressContainer.style.display = 'none';
}

// Copy to clipboard functionality
copyButton.addEventListener('click', () => {
    navigator.clipboard.writeText(transcriptText.textContent)
        .then(() => {
            const originalText = copyButton.textContent;
            copyButton.textContent = 'Copied!';
            setTimeout(() => {
                copyButton.textContent = originalText;
            }, 2000);
        })
        .catch(err => {
            showError('Failed to copy to clipboard');
        });
});

document.addEventListener('DOMContentLoaded', loadPreviousTranscripts);
