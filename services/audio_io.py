"""
Audio Input / Output Service.
Handles microphone streaming, speaker playback, and audio formatting via sounddevice.
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
from typing import Optional, Callable


class AudioIOService:
    def __init__(self, sample_rate: int = 16000, block_size: int = 512, input_device: Optional[int] = None, output_device: Optional[int] = None):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.input_device = input_device
        self.output_device = output_device
        
        self.audio_queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False
        self._lock = threading.Lock()

    def _audio_callback(self, indata, frames, time_info, status):
        """Internal sounddevice stream callback."""
        if status:
            pass  # Overflow / underflow handling
        # Copy chunk to prevent buffer overwrite
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self.audio_queue.put(chunk)

    def start_recording(self):
        """Start microphone input stream."""
        with self._lock:
            if self.is_recording:
                return
            self.audio_queue = queue.Queue()
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.input_device,
                channels=1,
                dtype="float32",
                callback=self._audio_callback
            )
            self.stream.start()
            self.is_recording = True

    def stop_recording(self):
        """Stop microphone input stream."""
        with self._lock:
            if not self.is_recording:
                return
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            self.is_recording = False

    def get_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Fetch next recorded chunk from queue."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def play_audio(self, audio: np.ndarray, sample_rate: int = 16000, blocking: bool = False):
        """Play audio out through system speaker."""
        sd.play(audio, samplerate=sample_rate, device=self.output_device)
        if blocking:
            sd.wait()

    def get_devices(self):
        """Query and return available audio input and output devices."""
        return sd.query_devices()
