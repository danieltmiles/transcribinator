import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
import tqdm
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from multiprocessing import Pool, cpu_count
import warnings

warnings.filterwarnings("ignore")

def compute_metrics_for_n_speakers(args):
    """
    Worker function to compute clustering metrics for a specific number of speakers.
    This function will be executed in parallel across multiple CPUs.
    """
    embeddings_normalized, n_speakers = args
    
    # Run Agglomerative Clustering
    agg_clustering = AgglomerativeClustering(n_clusters=n_speakers, linkage='average')
    agg_labels = agg_clustering.fit_predict(embeddings_normalized)
    
    # Calculate multiple metrics
    silhouette = silhouette_score(embeddings_normalized, agg_labels)
    calinski_harabasz = calinski_harabasz_score(embeddings_normalized, agg_labels)
    davies_bouldin = davies_bouldin_score(embeddings_normalized, agg_labels)
    
    return n_speakers, {
        'silhouette': silhouette,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin,
        'labels': agg_labels
    }

def find_optimal_speakers_multi_metric_parallel(embeddings, max_speakers=30, min_speakers=2, n_processes=None):
    """
    Find optimal number of speakers using multiple clustering metrics and methods.
    This version uses multiprocessing to parallelize the computation across multiple CPUs.
    
    Args:
        embeddings: Speaker embeddings array
        max_speakers: Maximum number of speakers to test
        min_speakers: Minimum number of speakers to test
        n_processes: Number of CPU processes to use (None = auto-detect)
    """
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)
    
    # Determine number of processes to use
    if n_processes is None:
        n_processes = min(cpu_count(), max_speakers - min_speakers + 1)
    
    print(f"Evaluating speaker counts with multiple metrics using {n_processes} CPU processes...")
    
    # Test range of speaker counts
    speaker_range = range(min_speakers, min(max_speakers + 1, len(embeddings)))
    
    # Prepare arguments for parallel processing
    # Note: We need to pass embeddings_normalized to each worker
    args_list = [(embeddings_normalized, n_speakers) for n_speakers in speaker_range]
    
    # Use multiprocessing to compute metrics in parallel
    metrics_results = {}
    
    with Pool(processes=n_processes) as pool:
        # Use tqdm to show progress
        results = list(tqdm.tqdm(
            pool.imap(compute_metrics_for_n_speakers, args_list),
            total=len(args_list),
            desc="Computing metrics"
        ))
    
    # Organize results
    for n_speakers, metrics in results:
        metrics_results[n_speakers] = metrics
        print(f"  {n_speakers} speakers: sil={metrics['silhouette']:.3f}, "
              f"ch={metrics['calinski_harabasz']:.1f}, db={metrics['davies_bouldin']:.3f}")
    
    # Combine metrics with weighted scoring
    best_score = -float('inf')
    best_n_speakers = min_speakers
    
    for n_speakers, metrics in metrics_results.items():
        # Normalize and weight the metrics
        # Silhouette: higher is better (weight: 0.3)
        # Calinski-Harabasz: higher is better (weight: 0.4) 
        # Davies-Bouldin: lower is better (weight: 0.3)
        
        silhouette_norm = metrics['silhouette']  # Already -1 to 1
        ch_norm = metrics['calinski_harabasz'] / 1000  # Normalize CH index
        db_norm = 1 / (1 + metrics['davies_bouldin'])  # Invert DB (lower is better)
        
        combined_score = (0.3 * silhouette_norm + 
                         0.4 * ch_norm + 
                         0.3 * db_norm)
        
        if combined_score > best_score:
            best_score = combined_score
            best_n_speakers = n_speakers
    
    print(f"Multi-metric optimal speakers: {best_n_speakers} (combined score: {best_score:.3f})")
    print(f"Processed using {n_processes} CPU processes")
    
    return best_n_speakers, metrics_results[best_n_speakers]['labels']

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
    audio_file_path = 'council_recording.mp3'
    signal, sr = torchaudio.load(audio_file_path)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    signal = signal.squeeze()
    # whisper needs a sample rate of 16000
    if sr != 16000:
        signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        sr = 16000
    speaker_recognition = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    # Parameters for segmentation
    min_segment_length = 1.0
    window_size = int(sr * min_segment_length * 3)  # 3x min_segment_length windows
    stride = int(sr * min_segment_length * 2)  # 2x min_segment_length stride

    # Separate segments for diarization (speaker identification) and transcription
    diarization_segments = []
    all_segments = []
    embeddings = []

    print(f"Processing audio segments from 0 through {len(signal)} with stride {stride}")
    diar_last_progress = 0
    diar_total_iters = max(1, len(range(0, len(signal), stride)))
    for idx, start in enumerate(tqdm.tqdm(range(0, len(signal), stride), total=diar_total_iters)):
        end = min(start + window_size, len(signal))
        segment = signal[start:end]

        # Always add to transcription segments (covers all audio)
        all_segments.append({
            'start': start / sr,
            'end': end / sr,
            'audio': segment
        })

        # Only add to diarization if segment is long enough for good speaker embeddings
        if len(segment) >= sr * min_segment_length:
            embedding = speaker_recognition.encode_batch(segment.unsqueeze(0))
            embeddings.append(embedding.squeeze().cpu().numpy())

            diarization_segments.append({
                'start': start / sr,
                'end': end / sr,
                'audio': segment,
                'segment_index': len(all_segments) - 1  # Link back to all_segments
            })
    from sklearn.preprocessing import normalize
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    # Use parallel optimal number of speakers detection
    print(f"Available CPU cores: {cpu_count()}")
    num_speakers = find_optimal_speakers_multi_metric_parallel(
        embeddings_normalized, 
        max_speakers=150,
        n_processes=None  # Auto-detect optimal number of processes
    )
    print(f"Estimated number of speakers: {num_speakers}")


if __name__ == "__main__":
    main()
