"""
Voice Activity Detection (VAD) Service.
Uses Silero VAD locally on CPU to segment speech and ignore background silence.
"""

import time
import numpy as np
import torch
from typing import Tuple, Optional


class VADService:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5, silence_duration_sec: float = 0.8):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_duration_sec = silence_duration_sec
        self.model = None
        self._load_model()
        
        # State tracking for streaming audio
        self.is_speaking = False
        self.silence_start_time: Optional[float] = None
        self.speech_start_time: Optional[float] = None
        self.speech_end_time: Optional[float] = None
        self.buffer = []

    def _load_model(self):
        """Load Silero VAD locally."""
        from silero_vad import load_silero_vad
        self.model = load_silero_vad()
        # Set to evaluation mode on CPU
        self.model.eval()

    def reset_state(self):
        """Reset internal streaming speech state."""
        self.is_speaking = False
        self.silence_start_time = None
        self.speech_start_time = None
        self.speech_end_time = None
        self.buffer = []
        if hasattr(self.model, "reset_states"):
            self.model.reset_states()

    def process_chunk(self, chunk: np.ndarray) -> Tuple[bool, bool, Optional[float]]:
        """
        Processes a single audio chunk (16kHz, float32, mono).
        Returns:
            (is_voice_active_in_chunk, speech_segment_finished, speech_end_timestamp)
        """
        # Ensure 1D float32
        if chunk.ndim > 1:
            chunk = chunk.flatten()
        if chunk.dtype != np.float32:
            chunk = chunk.astype(np.float32)

        tensor_chunk = torch.from_numpy(chunk)
        
        # Silero VAD requires chunks of 512 samples for 16kHz
        chunk_len = len(tensor_chunk)
        if chunk_len < 512:
            # Pad to 512
            padded = torch.zeros(512, dtype=torch.float32)
            padded[:chunk_len] = tensor_chunk
            speech_prob = self.model(padded, self.sample_rate).item()
        elif chunk_len > 512:
            # Evaluate on the first 512 or average
            speech_prob = self.model(tensor_chunk[:512], self.sample_rate).item()
        else:
            speech_prob = self.model(tensor_chunk, self.sample_rate).item()

        now = time.time()
        is_speech = speech_prob >= self.threshold
        speech_finished = False
        speech_end_ts = None

        if is_speech:
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = now
            self.silence_start_time = None
            self.buffer.append(chunk)
        else:
            if self.is_speaking:
                self.buffer.append(chunk)
                if self.silence_start_time is None:
                    self.silence_start_time = now
                elif (now - self.silence_start_time) >= self.silence_duration_sec:
                    # Speech turn complete!
                    self.is_speaking = False
                    speech_finished = True
                    self.speech_end_time = self.silence_start_time
                    speech_end_ts = self.speech_end_time
                    self.silence_start_time = None

        return is_speech, speech_finished, speech_end_ts

    def get_collected_speech(self) -> np.ndarray:
        """Return and clear buffered audio of the speech segment."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(self.buffer)
        self.buffer = []
        return audio
