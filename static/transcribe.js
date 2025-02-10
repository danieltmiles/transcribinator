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
        input.accept = 'audio/*,video/*';
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
        if (file.type.startsWith('audio/') || file.type.startsWith('video/')) {
            // Store the file and show dialog
            currentFile = file;
            speakerDialog.style.display = 'flex';
        } else {
            showError('Please upload an audio or video file.');
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
    document.getElementById('confirmUpload').addEventListener('click', async () => {
        const speakers = parseInt(numSpeakers.value);
        if (speakers < 1 || speakers > 10) {
            alert('Please enter a number between 1 and 10');
            return;
        }
        const fileName = document.getElementById('fileName').value.trim();
        speakerDialog.style.display = 'none';
        if (currentFile) {
            if (currentFile.type.startsWith('video/')) {
              const audioBlob = await extractAudioFromVideo(currentFile);
              // Create a File object from the blob
              // Use original filename but change extension to .wav
              const originalName = currentFile.name.replace(/\.[^/.]+$/, "");
              const audioFile = new File(
                [audioBlob], 
                `${originalName}.mkv`,
                { type: 'audio/wav' }
              );
              currentFile = audioFile;
            }
            uploadFile(currentFile, speakers, fileName);
            currentFile = null;
        }
    });


	/**
 * Extracts audio from a video file as fast as possible and outputs WAV format
 * @param {File} videoFile - The video file to extract audio from
 * @returns {Promise<Blob>} - A promise that resolves with the audio blob in WAV format
 */
async function extractAudioFromVideo(videoFile) {
  console.log('Starting high-speed audio extraction from:', videoFile.name);
  console.log('Video file size:', (videoFile.size / (1024 * 1024)).toFixed(2), 'MB');

  // Create video element
  const video = document.createElement('video');
  video.src = URL.createObjectURL(videoFile);

  console.log('Loading video...');
  await video.load();
  const originalDuration = video.duration;
  console.log('Video loaded. Original duration:', originalDuration, 'seconds');

  // Set maximum playback speed
  video.playbackRate = 16.0;
  console.log('Set playback speed to:', video.playbackRate, 'x');

  // Create audio context and connections
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  console.log('Audio context created. Sample rate:', audioContext.sampleRate, 'Hz');

  const source = audioContext.createMediaElementSource(video);
  const destination = audioContext.createMediaStreamDestination();
  source.connect(destination);
  console.log('Audio connections established');

  // Create promise to handle MediaRecorder
  return new Promise((resolve, reject) => {
    try {
      // Try to use WAV encoding if available
      const options = {
        mimeType: 'audio/wav'
      };

      // Fall back to default if WAV not supported
      if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        console.log('WAV not directly supported, will convert final output');
        options.mimeType = 'audio/webm;codecs=pcm';
      }

      const mediaRecorder = new MediaRecorder(destination.stream, options);
      console.log('MediaRecorder created with MIME type:', mediaRecorder.mimeType);

      const startTime = Date.now();
      const chunks = [];

      mediaRecorder.ondataavailable = (e) => {
        chunks.push(e.data);
        const chunkSizeMB = (e.data.size / (1024 * 1024)).toFixed(2);
        console.log(`Chunk received: ${chunkSizeMB} MB`);
      };

      // Log progress every 500ms
      let lastTime = 0;
      const progressInterval = setInterval(() => {
        const currentTime = video.currentTime;
        const progress = (currentTime / originalDuration * 100).toFixed(1);
        const timeElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        const extractionSpeed = (currentTime - lastTime) * 2;
        lastTime = currentTime;

        console.log(`Progress: ${progress}% (${timeElapsed}s elapsed, processing at ${extractionSpeed.toFixed(1)}x speed)`);
      }, 500);

      mediaRecorder.onstop = async () => {
        clearInterval(progressInterval);

        // Always create final blob as WAV
        const blob = new Blob(chunks, { type: 'audio/wav' });
        const totalTime = (Date.now() - startTime) / 1000;
        const speedFactor = originalDuration / totalTime;

        console.log('Audio extraction complete!');
        console.log(`Original duration: ${originalDuration.toFixed(1)}s`);
        console.log(`Processing time: ${totalTime.toFixed(1)}s`);
        console.log(`Average speed: ${speedFactor.toFixed(1)}x`);
        console.log('Final audio size:', (blob.size / (1024 * 1024)).toFixed(2), 'MB');
        console.log('Final audio format:', blob.type);

        URL.revokeObjectURL(video.src);
        audioContext.close();
        resolve(blob);
      };

      // Start recording and playing
      video.muted = true;
      mediaRecorder.start(500);
      console.log('Starting high-speed playback and recording...');
      video.play();

      // Stop when video ends
      video.onended = () => {
        console.log('Video playback complete, stopping recorder...');
        mediaRecorder.stop();
      };

      // Handle errors
      video.onerror = () => {
        clearInterval(progressInterval);
        console.error('Video loading failed:', video.error);
        reject(new Error('Video loading failed'));
      };

      mediaRecorder.onerror = () => {
        clearInterval(progressInterval);
        console.error('Recording failed:', mediaRecorder.error);
        reject(new Error('Audio recording failed'));
      };

    } catch (error) {
      console.error('Extraction error:', error);
      reject(error);
    }
  });
}
    
    function connectWebSocket(jobId, repeat_count = 0) {
        // // Close existing connection if any
        // if (websocket) {
        //     websocket.close();
        // }
        if (repeat_count > 10) {
            return;
        }
    
	let complete = false;
        websocket = new WebSocket(`wss://transcribe.doodledome.org/ws/${jobId}`);
    
        websocket.onmessage = function(event) {
            const data = JSON.parse(event.data);
    
            switch(data.type) {
                case 'progress':
                    updateProgress(data.progress);
                    updateStage(data.stage);
                    break;
    
                case 'transcript':
                    showTranscript(data.text, jobId);
                    complete = true;
                    websocket.close();
                    break;
    
                case 'error':
                    showError(data.message);
                    websocket.close();
                    break;
            }
        };
    
        websocket.onerror = function(error) {
            if(!complete) {
                showError('WebSocket error occurred', error);
                websocket.close();
                connectWebSocket(jobId, repeat_count + 1);
            }
        };
    
        websocket.onclose = function() {
            if(!complete) {
                websocket = null;
                showError('WebSocket connection closed unexpectedly');
    	        connectWebSocket(jobId, repeat_count + 1);
            }
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
    
    function showTranscript(text, job_id) {
        transcriptText.textContent = text;
        transcriptContainer.style.display = 'block';
        progressContainer.style.display = 'none';
        document.getElementById('transcriptTextJobID').setAttribute('data-job-id', job_id);
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
    
    loadPreviousTranscripts();
});

