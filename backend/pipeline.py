"""
Core Conversational Pipeline for AdaptiveEdge On-Device Conversational AI.
Features:
  - Multilingual Local ASR (whisper.cpp / faster-whisper) with auto language detection
  - Tiered Local Semantic Routing (Exact, Rule, and Dense Vector Similarity)
  - Incremental Sentence-Streamed LLM -> TTS pipeline for minimal Time-to-First-Audio (TTFA)
  - Sub-millisecond Instrumentation: T0 (Speech End), T1 (ASR Done), T2 (Router Done),
                                     T3 (First Token), T4 (First Audio), T5 (Final Audio)
  - Real Dynamic CPU-Core / Thread Adaptation across all inferences
  - Full browser & speaker audio payload packaging
All execution is strictly local on CPU without external AI APIs.
"""

import time
import os
import re
import base64
import threading
import numpy as np
from typing import Dict, Any, Optional, Callable, List

from services.audio_io import AudioIOService
from services.vad_service import VADService
from services.asr_service import MultilingualASRService
from services.llm_service import LLMService
from services.tts_service import TTSService
from backend.router import router
from monitoring.resource_controller import resource_controller
from monitoring.resource_monitor import monitor


class VoicePipeline:
    def __init__(
        self,
        sample_rate: int = 16000,
        enable_speaker: bool = False,  # False by default: browser is primary client player
        enable_streaming_tts: bool = True
    ):
        self.sample_rate = sample_rate
        self.enable_speaker = enable_speaker
        self.enable_streaming_tts = enable_streaming_tts
        
        print("[Pipeline] Initializing AdaptiveEdge On-Device AI stack...")
        self.audio_io = AudioIOService(sample_rate=sample_rate)
        self.vad = VADService(sample_rate=sample_rate)
        self.asr = MultilingualASRService()
        self.llm = LLMService()
        self.tts = TTSService()
        self.router = router
        self.resource_controller = resource_controller
        print("[Pipeline] AdaptiveEdge On-Device AI stack initialized successfully.")

        self.is_listening = False
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._processing_lock = threading.Lock()
        self.conversation_history: List[Dict[str, Any]] = []
        self.on_turn_completed: Optional[Callable[[Dict[str, Any]], None]] = None
        self.current_listening_state = "IDLE"  # "IDLE" | "LISTENING" | "PROCESSING" | "SPEAKING"
        self._last_e2e_latency: Optional[float] = None

    def _apply_active_threads(self):
        """Dynamically propagate active resource controller thread budget to AI engines."""
        policy = self.resource_controller.get_active_policy()
        if hasattr(self.llm, "set_threads"):
            self.llm.set_threads(policy.llm_threads)
        if hasattr(self.asr, "set_threads"):
            self.asr.set_threads(policy.llm_threads)

    def _create_audio_payload(self, final_wav_path: Optional[str]) -> Dict[str, Any]:
        """Convert synthesized WAV file to client-ready endpoints and base64 data URL."""
        audio_filename = None
        audio_url = None
        audio_data_url = None
        if final_wav_path and os.path.exists(final_wav_path):
            audio_filename = os.path.basename(final_wav_path)
            audio_url = f"/api/audio/{audio_filename}"
            try:
                with open(final_wav_path, "rb") as f:
                    wav_bytes = f.read()
                    if wav_bytes:
                        audio_data_url = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('ascii')}"
            except Exception as e:
                print(f"[Pipeline] Audio payload packaging notice: {e}")
        return {
            "wav_path": final_wav_path,
            "audio_filename": audio_filename,
            "audio_url": audio_url,
            "audio_data_url": audio_data_url
        }

    def process_speech_audio(
        self,
        audio: np.ndarray,
        speech_end_ts: Optional[float] = None,
        language: Optional[str] = None,
        translate: bool = False
    ) -> Dict[str, Any]:
        """
        Process a speech segment end-to-end with detailed T0..T5 timestamping.
        T0: Speech End (VAD endpoint)
        T1: ASR Done
        T2: Router Done
        T3: First LLM Output
        T4: First Audio Ready (TTFA)
        T5: Final Audio Stream Complete
        """
        # Apply active CPU thread allocation based on system load
        self._apply_active_threads()

        t0 = speech_end_ts if speech_end_ts is not None else time.time()
        self.current_listening_state = "PROCESSING"

        # 1. Local Multilingual ASR
        t_asr_start = time.perf_counter()
        asr_res = self.asr.transcribe(
            audio,
            sample_rate=self.sample_rate,
            language=language,
            translate=translate
        )
        user_text = asr_res["text"].strip()
        asr_latency_ms = asr_res["latency_ms"]
        t1 = time.time()  # T1: ASR result available

        if not user_text:
            self.current_listening_state = "IDLE" if not self.is_listening else "LISTENING"
            turn_data = {
                "timestamp": time.time(),
                "user_text": "",
                "response_text": "I didn't catch that. Could you please repeat?",
                "routing_decision": "EMPTY_AUDIO",
                "match_key": None,
                "match_type": "NONE",
                "similarity_score": 0.0,
                "threshold": 0.0,
                "operating_mode": self.resource_controller.current_mode,
                "processing_path": "VAD -> ASR (EMPTY)",
                "detected_language": asr_res.get("language", "en"),
                "asr_latency_ms": asr_latency_ms,
                "router_latency_ms": 0.0,
                "llm_latency_ms": 0.0,
                "ttft_ms": 0.0,
                "tts_latency_ms": 0.0,
                "ttfa_ms": round((t1 - t0) * 1000.0, 2),
                "e2e_latency_ms": round((t1 - t0) * 1000.0, 2),
                "wav_path": None,
                "audio_filename": None,
                "audio_url": None,
                "audio_data_url": None
            }
            return turn_data

        # 2. Tiered Local Semantic Router
        route_res = self.router.route(user_text, recent_latency_ms=self._last_e2e_latency)
        router_latency_ms = route_res.latency_ms
        t2 = time.time()  # T2: Router decision available

        policy = self.resource_controller.get_active_policy()
        t3 = None
        t4 = None
        t5 = None
        ttft_ms = 0.0
        llm_latency_ms = 0.0
        tokens_generated = 0
        tok_per_sec = 0.0
        final_wav_path = None

        # 3. Tier 1 Cache Hit OR Tier 2 Local LLM
        if route_res.decision == "CACHE HIT":
            response_text = route_res.response
            llm_latency_ms = 0.0
            ttft_ms = 0.0
            t3 = t2
            processing_path = f"CACHE HIT ({route_res.match_type}) -> LOCAL TTS"

            # Local TTS
            self.current_listening_state = "SPEAKING"
            tts_res = self.tts.speak(response_text, play_speaker=self.enable_speaker)
            tts_latency_ms = tts_res["latency_ms"]
            t4 = tts_res["first_audio_timestamp"]  # T4: First audio ready
            t5 = t4 + tts_res["duration_sec"]      # T5: Final audio
            final_wav_path = tts_res["wav_path"]
        else:
            # Tier 2 Local Quantized LLM with Sentence-Level Streaming to TTS
            processing_path = f"CACHE MISS -> LOCAL QUANTIZED LLM ({policy.name.upper()}) -> STREAMING TTS"
            t_llm_start = time.perf_counter()

            if self.enable_streaming_tts:
                # Stream tokens and detect first sentence boundary for low TTFA
                stream = self.llm.generate_stream(user_text, max_tokens=policy.llm_max_tokens)
                full_tokens = []
                first_sentence = ""
                first_sentence_synthesized = False
                tts_total_lat = 0.0

                for chunk in stream:
                    tok = chunk["token"]
                    full_tokens.append(tok)
                    if chunk["first_token"] and t3 is None:
                        t3 = time.time()
                        ttft_ms = chunk["ttft_ms"] or round((t3 - t2) * 1000.0, 2)

                    # Sentence boundary detection
                    current_text = "".join(full_tokens)
                    if not first_sentence_synthesized and len(current_text) > 25:
                        match = re.search(r'([.?!;\n]+)\s*', current_text)
                        if match:
                            split_pos = match.end()
                            first_sentence = current_text[:split_pos].strip()
                            first_sentence_synthesized = True
                            # Immediately synthesize and play opening sentence!
                            self.current_listening_state = "SPEAKING"
                            s1_res = self.tts.speak(first_sentence, play_speaker=self.enable_speaker)
                            tts_total_lat += s1_res["latency_ms"]
                            t4 = s1_res["first_audio_timestamp"]  # T4: First audio starts NOW
                            final_wav_path = s1_res["wav_path"]

                response_text = "".join(full_tokens).strip()
                t_llm_done = time.perf_counter()
                llm_latency_ms = round((t_llm_done - t_llm_start) * 1000.0, 2)
                tokens_generated = len(full_tokens)
                tok_per_sec = round((tokens_generated / (llm_latency_ms / 1000.0)), 2) if llm_latency_ms > 0 else 0.0

                # Synthesize complete coherent response for client playback
                full_tts = self.tts.speak(response_text, play_speaker=self.enable_speaker if not first_sentence_synthesized else False)
                tts_total_lat += full_tts["latency_ms"]
                final_wav_path = full_tts["wav_path"]
                if t4 is None:
                    t4 = full_tts["first_audio_timestamp"]
                t5 = time.time()
                tts_latency_ms = round(tts_total_lat, 2)
            else:
                # Monolithic non-streaming fallback
                llm_res = self.llm.generate(user_text, max_tokens=policy.llm_max_tokens)
                response_text = llm_res["text"]
                llm_latency_ms = llm_res["latency_ms"]
                tokens_generated = llm_res["tokens"]
                tok_per_sec = llm_res["tokens_per_second"]
                t3 = time.time()
                ttft_ms = llm_latency_ms

                self.current_listening_state = "SPEAKING"
                tts_res = self.tts.speak(response_text, play_speaker=self.enable_speaker)
                tts_latency_ms = tts_res["latency_ms"]
                t4 = tts_res["first_audio_timestamp"]
                t5 = t4 + tts_res["duration_sec"]
                final_wav_path = tts_res["wav_path"]

        # Default fallback timestamps if anything was missed
        if t4 is None:
            t4 = time.time()
        if t5 is None:
            t5 = t4

        # Primary Latency Metrics
        ttfa_ms = round((t4 - t0) * 1000.0, 2)            # Time to First Audio (T4 - T0)
        e2e_total_ms = round((t5 - t0) * 1000.0, 2)       # Total Response Latency (T5 - T0)
        self._last_e2e_latency = ttfa_ms

        # Closed-loop adaptation based on real TTFA
        self.resource_controller.evaluate_operating_mode(recent_latency_ms=ttfa_ms)

        # Prepare audio payloads for browser and external speakers
        audio_payload = self._create_audio_payload(final_wav_path)

        turn_data = {
            "timestamp": time.time(),
            "user_text": user_text,
            "response_text": response_text,
            "routing_decision": route_res.decision,
            "match_key": route_res.match_key,
            "match_type": route_res.match_type,
            "similarity_score": route_res.similarity_score,
            "threshold": route_res.threshold,
            "operating_mode": route_res.operating_mode,
            "processing_path": processing_path,
            "detected_language": asr_res.get("language", "en"),
            "language_prob": asr_res.get("language_prob", 1.0),
            "asr_backend": asr_res.get("backend", "local"),
            "audio_duration_sec": asr_res.get("duration_sec", 0.0),
            # Detailed Breakdown:
            "asr_latency_ms": asr_latency_ms,
            "router_latency_ms": router_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "ttft_ms": ttft_ms,
            "tts_latency_ms": tts_latency_ms,
            # Primary Metrics:
            "ttfa_ms": ttfa_ms,
            "time_to_first_audio_ms": ttfa_ms,
            "e2e_latency_ms": e2e_total_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tok_per_sec,
            # Audio Output Deliverables:
            "wav_path": audio_payload["wav_path"],
            "audio_filename": audio_payload["audio_filename"],
            "audio_url": audio_payload["audio_url"],
            "audio_data_url": audio_payload["audio_data_url"]
        }

        monitor.record_pipeline_turn(turn_data)
        self.conversation_history.append(turn_data)

        self.current_listening_state = "IDLE" if not self.is_listening else "LISTENING"

        if self.on_turn_completed:
            try:
                self.on_turn_completed(turn_data)
            except Exception as e:
                print(f"[Pipeline] Callback error: {e}")

        return turn_data

    def process_text_query(self, user_text: str) -> Dict[str, Any]:
        """
        Process text query through Router -> (Cache | LLM) -> TTS with detailed timestamping.
        """
        # Apply active CPU thread allocation
        self._apply_active_threads()

        t0 = time.time()
        
        # 1. Router
        route_res = self.router.route(user_text, recent_latency_ms=self._last_e2e_latency)
        router_latency_ms = route_res.latency_ms
        t2 = time.time()

        policy = self.resource_controller.get_active_policy()
        t3 = None
        t4 = None
        t5 = None
        ttft_ms = 0.0
        final_wav_path = None

        if route_res.decision == "CACHE HIT":
            response_text = route_res.response
            llm_latency_ms = 0.0
            tokens_generated = 0
            tok_per_sec = 0.0
            processing_path = f"CACHE HIT ({route_res.match_type}) -> LOCAL TTS"

            tts_res = self.tts.speak(response_text, play_speaker=self.enable_speaker)
            tts_latency_ms = tts_res["latency_ms"]
            t4 = tts_res["first_audio_timestamp"]
            t5 = t4 + tts_res["duration_sec"]
            final_wav_path = tts_res["wav_path"]
        else:
            processing_path = f"CACHE MISS -> LOCAL QUANTIZED LLM ({policy.name.upper()}) -> STREAMING TTS"
            t_llm_start = time.perf_counter()

            if self.enable_streaming_tts:
                stream = self.llm.generate_stream(user_text, max_tokens=policy.llm_max_tokens)
                full_tokens = []
                first_sentence = ""
                first_sentence_synthesized = False
                tts_total_lat = 0.0

                for chunk in stream:
                    tok = chunk["token"]
                    full_tokens.append(tok)
                    if chunk["first_token"] and t3 is None:
                        t3 = time.time()
                        ttft_ms = chunk["ttft_ms"] or round((t3 - t2) * 1000.0, 2)

                    current_text = "".join(full_tokens)
                    if not first_sentence_synthesized and len(current_text) > 25:
                        match = re.search(r'([.?!;\n]+)\s*', current_text)
                        if match:
                            first_sentence = current_text[:match.end()].strip()
                            first_sentence_synthesized = True
                            s1_res = self.tts.speak(first_sentence, play_speaker=self.enable_speaker)
                            tts_total_lat += s1_res["latency_ms"]
                            t4 = s1_res["first_audio_timestamp"]
                            final_wav_path = s1_res["wav_path"]

                response_text = "".join(full_tokens).strip()
                llm_latency_ms = round((time.perf_counter() - t_llm_start) * 1000.0, 2)
                tokens_generated = len(full_tokens)
                tok_per_sec = round((tokens_generated / (llm_latency_ms / 1000.0)), 2) if llm_latency_ms > 0 else 0.0

                rem = response_text[len(first_sentence):].strip() if first_sentence else response_text
                if rem:
                    rem_res = self.tts.speak(rem, play_speaker=self.enable_speaker)
                    tts_total_lat += rem_res["latency_ms"]
                    final_wav_path = rem_res["wav_path"]
                    if t4 is None:
                        t4 = rem_res["first_audio_timestamp"]
                    t5 = time.time()
                else:
                    t5 = time.time()
                tts_latency_ms = round(tts_total_lat, 2)
            else:
                llm_res = self.llm.generate(user_text, max_tokens=policy.llm_max_tokens)
                response_text = llm_res["text"]
                llm_latency_ms = llm_res["latency_ms"]
                tokens_generated = llm_res["tokens"]
                tok_per_sec = llm_res["tokens_per_second"]
                t3 = time.time()
                ttft_ms = llm_latency_ms

                tts_res = self.tts.speak(response_text, play_speaker=self.enable_speaker)
                tts_latency_ms = tts_res["latency_ms"]
                t4 = tts_res["first_audio_timestamp"]
                t5 = t4 + tts_res["duration_sec"]
                final_wav_path = tts_res["wav_path"]

        if t4 is None:
            t4 = time.time()
        if t5 is None:
            t5 = t4

        ttfa_ms = round((t4 - t0) * 1000.0, 2)
        e2e_total_ms = round((t5 - t0) * 1000.0, 2)
        self._last_e2e_latency = ttfa_ms

        self.resource_controller.evaluate_operating_mode(recent_latency_ms=ttfa_ms)

        # Prepare audio payloads for browser and external speakers
        audio_payload = self._create_audio_payload(final_wav_path)

        turn_data = {
            "timestamp": time.time(),
            "user_text": user_text,
            "response_text": response_text,
            "routing_decision": route_res.decision,
            "match_key": route_res.match_key,
            "match_type": route_res.match_type,
            "similarity_score": route_res.similarity_score,
            "threshold": route_res.threshold,
            "operating_mode": route_res.operating_mode,
            "processing_path": processing_path,
            "detected_language": "en",
            "language_prob": 1.0,
            "asr_backend": "none (direct text input)",
            "audio_duration_sec": 0.0,
            "asr_latency_ms": 0.0,
            "router_latency_ms": router_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "ttft_ms": ttft_ms,
            "tts_latency_ms": tts_latency_ms,
            "ttfa_ms": ttfa_ms,
            "time_to_first_audio_ms": ttfa_ms,
            "e2e_latency_ms": e2e_total_ms,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tok_per_sec,
            # Audio Output Deliverables:
            "wav_path": audio_payload["wav_path"],
            "audio_filename": audio_payload["audio_filename"],
            "audio_url": audio_payload["audio_url"],
            "audio_data_url": audio_payload["audio_data_url"]
        }

        monitor.record_pipeline_turn(turn_data)
        self.conversation_history.append(turn_data)

        if self.on_turn_completed:
            try:
                self.on_turn_completed(turn_data)
            except Exception as e:
                print(f"[Pipeline] Callback error: {e}")

        return turn_data

    def start_listening(self):
        """Start microphone input stream and background VAD loop."""
        if self.is_listening:
            return
        if self._listener_thread and self._listener_thread.is_alive():
            self._stop_event.set()
            self._listener_thread.join(timeout=2.0)
        self._stop_event.clear()
        self.vad.reset_state()
        self.audio_io.start_recording()
        self.is_listening = True
        self.current_listening_state = "LISTENING"

        self._listener_thread = threading.Thread(target=self._listening_loop, daemon=True)
        self._listener_thread.start()
        print("[Pipeline] Live microphone listening loop started.")

    def stop_listening(self):
        """Stop listening loop and audio stream."""
        if not self.is_listening:
            return
        self._stop_event.set()
        self.audio_io.stop_recording()
        self.is_listening = False
        self.current_listening_state = "IDLE"
        if hasattr(self.tts, "stop_audio"):
            self.tts.stop_audio()
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=3.0)
        self._listener_thread = None
        print("[Pipeline] Live microphone listening loop stopped.")

    def _listening_loop(self):
        """Continuous background thread evaluating audio through VAD."""
        while not self._stop_event.is_set():
            chunk = self.audio_io.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            # Mute microphone processing while system is actively speaking to prevent acoustic feedback
            if self.current_listening_state in ("SPEAKING", "PROCESSING"):
                self.vad.reset_state()
                continue

            is_active, speech_done, speech_end_ts = self.vad.process_chunk(chunk)

            if speech_done:
                audio_segment = self.vad.get_collected_speech()
                self.vad.reset_state()
                if len(audio_segment) >= self.sample_rate * 0.4:  # At least 400ms
                    # Ensure single concurrent execution through processing lock
                    if not self._processing_lock.acquire(blocking=False):
                        print("[Pipeline] Speech processing already in progress; skipping overlapping buffer.")
                        continue
                    try:
                        print(f"[Pipeline] Speech detected ({len(audio_segment)/self.sample_rate:.2f}s). Processing...")
                        self.process_speech_audio(audio_segment, speech_end_ts=speech_end_ts)
                    except Exception as ex:
                        print(f"[Pipeline] Speech processing notice: {ex}")
                    finally:
                        self._processing_lock.release()

    def get_state(self) -> Dict[str, Any]:
        return {
            "is_listening": self.is_listening,
            "current_state": self.current_listening_state,
            "speaker_enabled": self.enable_speaker,
            "streaming_tts_enabled": self.enable_streaming_tts,
            "asr_backend": self.asr.active_backend,
            "turns_count": len(self.conversation_history),
            "last_turn": self.conversation_history[-1] if self.conversation_history else None,
            "controller": self.resource_controller.get_controller_state()
        }
