// Declare variables at the top level so they're accessible everywhere
let uploadArea, progressContainer, progressBar, progressText, errorMessage,
    transcriptContainer, transcriptText, copyButton, fileNameDisplay;
let speakerRenameDialog, speakerInputs, cancelRename, confirmRename, renameSpeakersButton;

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const errorMessage = document.getElementById('errorMessage');
    const transcriptContainer = document.getElementById('transcriptContainer');
    const transcriptText = document.getElementById('transcriptText');
    const copyButton = document.getElementById('copyButton');
    const fileNameDisplay = document.getElementById('fileNameDisplay');

    const speakerRenameDialog = document.getElementById('speakerRenameDialog');
    const speakerInputs = document.getElementById('speakerInputs');
    const cancelRename = document.getElementById('cancelRename');
    const confirmRename = document.getElementById('confirmRename');
    const renameSpeakersButton = document.getElementById('renameSpeakersButton');
    
    const API_URL = 'https://transcribe.doodledome.org';  // Replace with your API URL
    
    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    
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
    	if (file) {
		uploadArea.querySelectorAll('p').forEach(p => p.style.display = 'none');
                fileNameDisplay.textContent = file.name;
                fileNameDisplay.style.display = 'block';
            }
            handleFile(file);
        };
        input.click();
    });
    
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

    document.getElementById('cancelUpload').addEventListener('click', () => {
        speakerDialog.style.display = 'none';
        currentFile = null;
    });

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
            resetUploadArea();
            connectWebSocket(job_id);
    
        } catch (error) {
            showError('Failed to upload file: ' + error.message);
        }
    }
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function handleDrop(e) {
        const file = e.dataTransfer.files[0];
        if (file) {
    	uploadArea.querySelectorAll('p').forEach(p => p.style.display = 'none');
            fileNameDisplay.textContent = file.name;
            fileNameDisplay.style.display = 'block';
        }
        handleFile(file);
    }
    
    // Add a reset function to clear the filename display
    function resetUploadArea() {
        fileNameDisplay.textContent = '';
        fileNameDisplay.style.display = 'none';
        uploadArea.querySelectorAll('p').forEach(p => p.style.display = 'none');
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
    		fileItem.style.cursor = 'pointer';
    		fileItem.classList.add('clickable');
    		fileItem.style.color = 'blue';
                    fileItem.addEventListener('click', () => {
                        document.getElementById('transcriptText').textContent = job.transcript;
                        document.getElementById('transcriptContainer').style.display = 'block';
    		        document.getElementById('transcriptContainer').scrollIntoView({ behavior: 'smooth' });
			document.getElementById('transcriptTextJobID').setAttribute('data-job-id', job.job_id);
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
        resetUploadArea();
    }
    
    function showTranscript(text) {
        transcriptText.textContent = text;
        transcriptContainer.style.display = 'block';
        progressContainer.style.display = 'none';
    }


    async function fetchSpeakers(jobId) {
        try {
            const response = await fetch(`${API_URL}/jobs/${jobId}/speakers`);
            if (!response.ok) {
                throw new Error('Failed to fetch speakers');
            }
            return await response.json();
        } catch (error) {
            console.error('Error fetching speakers:', error);
            throw error;
        }
    }
    
    function createSpeakerInputs(speakers) {
        speakerInputs.innerHTML = '';
        speakers.forEach(speaker => {
            const inputGroup = document.createElement('div');
            inputGroup.className = 'speaker-input-group';
    
            const label = document.createElement('label');
            label.textContent = `Rename "${speaker}" to:`;
    
            const input = document.createElement('input');
            input.type = 'text';
            input.placeholder = 'Enter name';
            input.dataset.originalName = speaker;
    
            inputGroup.appendChild(label);
            inputGroup.appendChild(input);
            speakerInputs.appendChild(inputGroup);
        });
    }
    
    async function submitSpeakerNames() {
        const inputs = speakerInputs.querySelectorAll('input');
        const speakerMap = {};
    
        inputs.forEach(input => {
            if (input.value.trim()) {
                speakerMap[input.dataset.originalName] = input.value.trim();
            }
        });
    
        try {
            confirmRename.classList.add('loading');
            let currentJobId = document.getElementById('transcriptTextJobID').dataset.jobId;
            const response = await fetch(`${API_URL}/jobs/${currentJobId}/speakers`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(speakerMap)
            });
    
            if (!response.ok) {
                throw new Error('Failed to update speaker names');
            }
    
            // Get the updated transcript
            const newTranscriptText = await response.json()
            transcriptText.textContent = newTranscriptText.transcript;
            speakerRenameDialog.style.display = 'none';
    
        } catch (error) {
            console.error('Error updating speakers:', error);
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-text';
            errorDiv.textContent = error.message;
            speakerInputs.appendChild(errorDiv);
        } finally {
            confirmRename.classList.remove('loading');
        }
    }
    
    // Modify your showTranscript function to store the job ID and show the rename button
    function showTranscript(text, jobId) {
        transcriptText.textContent = text;
        transcriptContainer.style.display = 'block';
        progressContainer.style.display = 'none';
        renameSpeakersButton.style.display = 'inline-block';
    }
    
    // Add these event listeners
    renameSpeakersButton.addEventListener('click', async () => {
        let currentJobId = document.getElementById('transcriptTextJobID').dataset.jobId;
        try {
            renameSpeakersButton.classList.add('loading');
            const speakers = await fetchSpeakers(currentJobId);
            if (speakers.length === 0) {
                alert('No speakers found in the transcript.');
                return;
            }
            createSpeakerInputs(speakers);
            speakerRenameDialog.style.display = 'flex';
        } catch (error) {
            alert('Failed to load speakers. Please try again.');
        } finally {
            renameSpeakersButton.classList.remove('loading');
        }
    });
    
    cancelRename.addEventListener('click', () => {
        speakerRenameDialog.style.display = 'none';
    });
    
    confirmRename.addEventListener('click', submitSpeakerNames);
    
    //// Modify your WebSocket message handler to include the job ID when showing the transcript
    //websocket.onmessage = function(event) {
    //    const data = JSON.parse(event.data);
    //
    //    switch(data.type) {
    //        case 'progress':
    //            updateProgress(data.progress);
    //            updateStage(data.stage);
    //            break;
    //
    //        case 'transcript':
    //            showTranscript(data.text, data.job_id); // Pass the job ID
    //            websocket.close();
    //            break;
    //
    //        case 'error':
    //            showError(data.message);
    //            websocket.close();
    //            break;
    //    }
    //};
    loadPreviousTranscripts();
});

