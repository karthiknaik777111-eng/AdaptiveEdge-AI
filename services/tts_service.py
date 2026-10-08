"""
Local Text-to-Speech (TTS) Service.
Uses pyttsx3 (native Windows SAPI5) running 100% locally on CPU.
Zero cloud network calls. Measures exact synthesis and first-audio latencies.
"""

import time
import os
import tempfile
import pyttsx3
import numpy as np
from typing import Dict, Any, Optional
import sounddevice as sd
from scipy.io import wavfile


class TTSService:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self.temp_dir = tempfile.mkdtemp(prefix="adaptive_tts_")
        
        # Test engine initialization
        self._init_engine()

    def _init_engine(self) -> pyttsx3.Engine:
        engine = pyttsx3.init()
        engine.setProperty("rate", self.rate)
        engine.setProperty("volume", self.volume)
        return engine

    def synthesize_to_wav(self, text: str) -> Dict[str, Any]:
        """
        Synthesize text to a local WAV file.
        Returns filepath, audio numpy array, sample rate, and synthesis latency in ms.
        """
        t_start = time.perf_counter()
        
        wav_path = os.path.join(self.temp_dir, f"tts_{int(time.time()*1000)}.wav")
        
        # Fresh engine instance per synthesis to avoid COM thread locking issues on Windows
        engine = self._init_engine()
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        
        t_done = time.perf_counter()
        synth_latency_ms = (t_done - t_start) * 1000.0

        if not os.path.exists(wav_path):
            raise RuntimeError(f"Failed to synthesize audio for text: {text}")

        sr, audio_data = wavfile.read(wav_path)
        
        # Convert to float32 normalized for sounddevice
        if audio_data.dtype == np.int16:
            audio_float = audio_data.astype(np.float32) / 32768.0
        else:
            audio_float = audio_data.astype(np.float32)

        duration_sec = len(audio_float) / sr

        return {
            "wav_path": wav_path,
            "sample_rate": sr,
            "audio": audio_float,
            "duration_sec": round(duration_sec, 2),
            "latency_ms": round(synth_latency_ms, 2)
        }

    def play_audio(self, audio: np.ndarray, sample_rate: int, blocking: bool = False):
        """Play audio array directly to speaker using sounddevice."""
        sd.play(audio, samplerate=sample_rate)
        if blocking:
            sd.wait()

    def stop_audio(self):
        """Immediately halt any active sounddevice audio playback."""
        try:
            sd.stop()
        except Exception:
            pass

    def speak(self, text: str, play_speaker: bool = True) -> Dict[str, Any]:
        """
        Complete TTS pipeline: synthesize and optionally play to speaker.
        Measures synthesis latency and marks first audio output timestamp.
        """
        res = self.synthesize_to_wav(text)
        first_audio_ts = time.time()
        
        if play_speaker:
            self.play_audio(res["audio"], res["sample_rate"], blocking=False)

        return {
            "text": text,
            "wav_path": res["wav_path"],
            "sample_rate": res["sample_rate"],
            "duration_sec": res["duration_sec"],
            "latency_ms": res["latency_ms"],
            "first_audio_timestamp": first_audio_ts
        }


# Type alias helper
Tuple_AudioResult = Dict[str, Any]
