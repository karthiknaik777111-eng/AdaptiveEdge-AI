"""
Multilingual Local Automatic Speech Recognition (ASR) Service.
Evaluates and supports:
  1. whisper.cpp (pywhispercpp, ggml-tiny.bin, ~77 MB CPU C++ engine)
  2. faster-whisper (CTranslate2 int8 CPU)

Features:
  - Automatic language identification with confidence score
  - Native-language transcription
  - Optional translation to English
  - Real measured latency tracking
"""

import time
import os
import threading
import numpy as np
from typing import Dict, Any, Optional, List


class MultilingualASRService:
    def __init__(
        self,
        backend: str = "whisper.cpp",  # "whisper.cpp" | "faster-whisper"
        model_name: str = "tiny",
        models_dir: str = "models",
        num_threads: int = 4
    ):
        self.backend_choice = backend
        self.model_name = model_name
        self.models_dir = models_dir
        self.num_threads = num_threads
        self._lock = threading.Lock()
        
        self.active_backend = None
        self.cpp_model = None
        self.fw_model = None
        
        self._init_backend()

    def set_threads(self, n_threads: int):
        """Dynamically update CPU ASR thread allocation."""
        if n_threads > 0:
            self.num_threads = n_threads

    def _init_backend(self):
        """Initialize selected ASR backend with graceful fallback."""
        os.makedirs(os.path.join(self.models_dir, "whispercpp"), exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "whisper"), exist_ok=True)

        if self.backend_choice == "whisper.cpp":
            try:
                from pywhispercpp.model import Model
                cpp_dir = os.path.join(self.models_dir, "whispercpp")
                self.cpp_model = Model(self.model_name, models_dir=cpp_dir)
                self.active_backend = "whisper.cpp"
                print(f"[ASR] whisper.cpp ({self.model_name}) initialized successfully on CPU.")
                return
            except Exception as e:
                print(f"[ASR] Failed to initialize whisper.cpp ({e}). Falling back to faster-whisper.")

        # Fallback or explicit faster-whisper
        try:
            from faster_whisper import WhisperModel
            fw_dir = os.path.join(self.models_dir, "whisper")
            self.fw_model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.num_threads,
                download_root=fw_dir
            )
            self.active_backend = "faster-whisper"
            print(f"[ASR] faster-whisper ({self.model_name} int8) initialized on CPU.")
        except Exception as e:
            raise RuntimeError(f"Could not initialize any local ASR backend: {e}")

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,  # None/'auto' = auto detect, or 'en', 'es', 'fr', 'de', 'hi', etc.
        translate: bool = False
    ) -> Dict[str, Any]:
        """
        Transcribe audio buffer locally on CPU.
        Returns transcript, detected language, confidence, and real measured latency in ms.
        """
        t_start = time.perf_counter()

        if audio.ndim > 1:
            audio = audio.flatten()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            return {
                "text": "",
                "latency_ms": 0.0,
                "language": "en",
                "language_prob": 1.0,
                "translated": False,
                "backend": self.active_backend
            }

        detected_lang = "en"
        lang_prob = 1.0
        full_text = ""

        with self._lock:
            if self.active_backend == "whisper.cpp":
                try:
                    whisper_lang = language if language and language != "auto" else "auto"
                    # pywhispercpp transcribe
                    segs = self.cpp_model.transcribe(
                        audio,
                        n_threads=self.num_threads,
                        language=whisper_lang,
                        translate=translate
                    )
                    text_parts = [s.text.strip() for s in segs if s.text.strip()]
                    full_text = " ".join(text_parts).strip()

                    # Extract language if available
                    if segs and hasattr(segs[0], "language") and segs[0].language:
                        detected_lang = segs[0].language
                    elif language and language != "auto":
                        detected_lang = language
                    else:
                        detected_lang = "auto"
                except Exception as e:
                    print(f"[ASR] whisper.cpp error ({e}), trying fallback...")
                    if self.fw_model is None:
                        from faster_whisper import WhisperModel
                        self.fw_model = WhisperModel(self.model_name, device="cpu", compute_type="int8", download_root=os.path.join(self.models_dir, "whisper"))
                    return self._transcribe_faster_whisper(audio, t_start, sample_rate, language, translate)
            else:
                return self._transcribe_faster_whisper(audio, t_start, sample_rate, language, translate)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "text": full_text,
            "latency_ms": round(latency_ms, 2),
            "language": detected_lang,
            "language_prob": round(lang_prob, 2),
            "translated": translate,
            "duration_sec": round(len(audio) / sample_rate, 2),
            "backend": self.active_backend
        }

    def _transcribe_faster_whisper(
        self,
        audio: np.ndarray,
        t_start: float,
        sample_rate: int,
        language: Optional[str],
        translate: bool
    ) -> Dict[str, Any]:
        task = "translate" if translate else "transcribe"
        fw_lang = language if language and language != "auto" else None

        segments, info = self.fw_model.transcribe(
            audio,
            beam_size=1,
            language=fw_lang,
            task=task,
            vad_filter=False,
            temperature=0.0
        )
        transcript_parts = [s.text.strip() for s in segments]
        full_text = " ".join(transcript_parts).strip()
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return {
            "text": full_text,
            "latency_ms": round(latency_ms, 2),
            "language": info.language,
            "language_prob": round(info.language_probability, 2),
            "translated": translate,
            "duration_sec": round(len(audio) / sample_rate, 2),
            "backend": "faster-whisper"
        }

    def compare_backends(self, audio: np.ndarray) -> Dict[str, Any]:
        """Direct side-by-side benchmark comparison between whisper.cpp and faster-whisper."""
        # Test whisper.cpp
        cpp_res = None
        if self.cpp_model:
            t0 = time.perf_counter()
            segs = self.cpp_model.transcribe(audio, n_threads=self.num_threads)
            text_cpp = " ".join(s.text.strip() for s in segs)
            cpp_lat = (time.perf_counter() - t0) * 1000.0
            cpp_res = {"latency_ms": round(cpp_lat, 2), "text": text_cpp, "model_size_mb": 77.1}

        # Test faster-whisper
        if self.fw_model is None:
            from faster_whisper import WhisperModel
            self.fw_model = WhisperModel(self.model_name, device="cpu", compute_type="int8", download_root=os.path.join(self.models_dir, "whisper"))
        
        t1 = time.perf_counter()
        segs_fw, info = self.fw_model.transcribe(audio, beam_size=1)
        text_fw = " ".join(s.text.strip() for s in segs_fw)
        fw_lat = (time.perf_counter() - t1) * 1000.0
        fw_res = {"latency_ms": round(fw_lat, 2), "text": text_fw, "lang": info.language, "model_size_mb": 75.0}

        return {
            "whisper_cpp": cpp_res,
            "faster_whisper": fw_res
        }


# Aliases for backward compatibility
ASRService = MultilingualASRService
