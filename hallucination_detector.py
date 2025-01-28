import numpy as np
from typing import List, Dict, Tuple
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import webrtcvad
import wave
from dataclasses import dataclass
from typing import Optional


@dataclass
class SuspiciousSegment:
    text: str
    confidence: float
    start_time: float
    end_time: float
    reason: str
    audio_energy: Optional[float] = None


class HallucinationDetector:
    def __init__(self, vad_mode: int = 3):
        """
        Initialize the hallucination detector.

        Args:
            vad_mode: WebRTC VAD aggressiveness (0-3, 3 being most aggressive)
        """
        self.vad = webrtcvad.Vad(vad_mode)

    def analyze_audio_energy(self, audio_path: str, segments: List[dict]) -> List[float]:
        """
        Calculate audio energy for each segment to detect unusually quiet parts
        that shouldn't contain speech.
        """
        # Load audio file
        y, sr = librosa.load(audio_path)

        segment_energies = []
        for segment in segments:
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)

            # Get audio segment
            segment_audio = y[start_sample:end_sample]

            # Calculate RMS energy
            energy = np.sqrt(np.mean(segment_audio ** 2))
            segment_energies.append(energy)

        return segment_energies

    def detect_vad_mismatch(self, audio_path: str, segments: List[dict]) -> List[bool]:
        """
        Use WebRTC VAD to detect segments where Whisper claims there's speech
        but VAD disagrees.
        """
        # Read audio file
        with wave.open(audio_path, 'rb') as wf:
            frame_duration = 30  # ms
            frames = []
            while True:
                frame = wf.readframes(int(wf.getframerate() * frame_duration / 1000))
                if not frame:
                    break
                frames.append(frame)

        vad_results = []
        for segment in segments:
            start_frame = int(segment['start'] * 1000 / frame_duration)
            end_frame = int(segment['end'] * 1000 / frame_duration)

            # Check what portion of frames in this segment contain speech
            speech_frames = 0
            total_frames = end_frame - start_frame

            for i in range(start_frame, end_frame):
                if i < len(frames):
                    try:
                        is_speech = self.vad.is_speech(frames[i], 16000)
                        if is_speech:
                            speech_frames += 1
                    except:
                        continue

            # If less than 30% of frames contain speech, mark as suspicious
            vad_results.append(speech_frames / max(1, total_frames) < 0.3)

        return vad_results

    def detect_repeating_phrases(self, segments: List[dict],
                                 max_words: int = 4) -> List[bool]:
        """
        Detect when the same phrase appears multiple times in close proximity,
        which might indicate hallucination.
        """
        results = []

        for i, segment in enumerate(segments):
            text = segment['text'].lower()
            words = text.split()

            # Look for repeating phrases up to max_words in length
            has_repeat = False
            for phrase_len in range(2, min(len(words), max_words + 1)):
                for j in range(len(words) - phrase_len + 1):
                    phrase = ' '.join(words[j:j + phrase_len])

                    # Check nearby segments for the same phrase
                    window = 3  # Check 3 segments before and after
                    start_idx = max(0, i - window)
                    end_idx = min(len(segments), i + window + 1)

                    repeat_count = 0
                    for k in range(start_idx, end_idx):
                        if k != i and phrase in segments[k]['text'].lower():
                            repeat_count += 1

                    if repeat_count > 0:
                        has_repeat = True
                        break

            results.append(has_repeat)

        return results

    def detect_timing_anomalies(self, segments: List[dict]) -> List[bool]:
        """
        Detect segments with unusual timing patterns that might indicate
        hallucination.
        """
        # Calculate average words per second across all segments
        total_words = sum(len(s['text'].split()) for s in segments)
        total_duration = sum(s['end'] - s['start'] for s in segments)
        avg_words_per_second = total_words / total_duration if total_duration > 0 else 0

        results = []
        for segment in segments:
            duration = segment['end'] - segment['start']
            word_count = len(segment['text'].split())

            if duration > 0:
                segment_wps = word_count / duration
                # Mark as suspicious if significantly faster than average
                # or impossibly fast for human speech
                is_suspicious = (segment_wps > avg_words_per_second * 2) or (segment_wps > 7)
                results.append(is_suspicious)
            else:
                results.append(True)  # Zero duration is suspicious

        return results

    def analyze_transcript(self, audio_path: str,
                           segments: List[dict]) -> List[SuspiciousSegment]:
        """
        Analyze a transcript for potential hallucinations using multiple detection methods.

        Args:
            audio_path: Path to the audio file
            segments: List of segment dictionaries from Whisper

        Returns:
            List of SuspiciousSegment objects for segments that might contain hallucinations
        """
        # Run all detection methods
        energies = self.analyze_audio_energy(audio_path, segments)
        vad_mismatches = self.detect_vad_mismatch(audio_path, segments)
        repeating_phrases = self.detect_repeating_phrases(segments)
        timing_anomalies = self.detect_timing_anomalies(segments)

        suspicious_segments = []

        for i, segment in enumerate(segments):
            reasons = []

            # Check energy (using relative threshold)
            mean_energy = np.mean(energies)
            if energies[i] < mean_energy * 0.3:
                reasons.append("unusually_quiet")

            if vad_mismatches[i]:
                reasons.append("no_speech_detected")

            if repeating_phrases[i]:
                reasons.append("repeating_phrase")

            if timing_anomalies[i]:
                reasons.append("unusual_timing")

            # Check for language anomalies
            language_anomalies = self.detect_language_anomalies([segment])[0]
            if language_anomalies:
                reasons.append("unexpected_language")

            if reasons:
                suspicious_segments.append(SuspiciousSegment(
                    text=segment['text'],
                    confidence=segment.get('confidence', 0.0),
                    start_time=segment['start'],
                    end_time=segment['end'],
                    reason=", ".join(reasons),
                    audio_energy=energies[i]
                ))

        return suspicious_segments


# Example usage
if __name__ == "__main__":
    # Example segments (would come from Whisper)
    example_segments = [
        {"text": "And... Ruth, I believe, joined.", "start": 0.0, "end": 2.0},
        {"text": "a year ago.", "start": 2.0, "end": 2.5},
        {"text": "A year or two after that.", "start": 2.5, "end": 3.5}
    ]

    detector = HallucinationDetector()
    suspicious = detector.analyze_transcript("audio_file.wav", example_segments)

    for segment in suspicious:
        print(f"Suspicious segment: {segment.text}")
        print(f"Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
        print(f"Reason: {segment.reason}")
        print(f"Audio energy: {segment.audio_energy:.3f}")
        print("---")