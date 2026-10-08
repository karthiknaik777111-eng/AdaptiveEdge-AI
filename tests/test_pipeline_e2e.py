"""
End-to-End Pipeline Verification Test.
Tests the full loop:
Microphone / Audio In → VAD → Local ASR → Local Router → Cache Hit OR Local Quantized LLM → Local TTS → Speaker.
All components run 100% locally on CPU with zero cloud AI APIs.
Measures real latencies and resource telemetry.
"""

import unittest
import time
import numpy as np
from scipy import signal

from backend.pipeline import VoicePipeline
from monitoring.resource_monitor import monitor


class TestEndToEndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize pipeline without playing audio to speaker during fast automated tests
        cls.pipeline = VoicePipeline(enable_speaker=False)

    def _create_synthetic_speech_input(self, text: str) -> np.ndarray:
        """Helper to create a valid 16kHz speech waveform using local TTS."""
        tts_res = self.pipeline.tts.synthesize_to_wav(text)
        orig_sr = tts_res["sample_rate"]
        orig_audio = tts_res["audio"]
        target_len = int(len(orig_audio) * 16000 / orig_sr)
        resampled_16k = signal.resample(orig_audio, target_len).astype(np.float32)
        return resampled_16k

    def test_e2e_cache_hit_path(self):
        """
        Verify: Audio In → ASR → Router → CACHE HIT → Local TTS (LLM Bypassed).
        """
        print("\n--- [TEST] Starting End-to-End Cache Hit Path ---")
        audio_in = self._create_synthetic_speech_input("What is your name?")
        speech_end_ts = time.time()

        turn = self.pipeline.process_speech_audio(audio_in, speech_end_ts=speech_end_ts)

        print(f"Transcribed Text : '{turn['user_text']}'")
        print(f"Router Decision  : {turn['routing_decision']}")
        print(f"Response Text    : '{turn['response_text']}'")
        print(f"ASR Latency      : {turn['asr_latency_ms']} ms")
        print(f"Router Latency   : {turn['router_latency_ms']} ms")
        print(f"LLM Latency      : {turn['llm_latency_ms']} ms (Bypassed)")
        print(f"TTS Latency      : {turn['tts_latency_ms']} ms")
        print(f"E2E Latency      : {turn['e2e_latency_ms']} ms")

        self.assertEqual(turn["routing_decision"], "CACHE HIT")
        self.assertEqual(turn["llm_latency_ms"], 0.0)
        self.assertIn("AdaptiveEdge", turn["response_text"])
        self.assertGreater(turn["asr_latency_ms"], 0.0)
        self.assertGreater(turn["tts_latency_ms"], 0.0)
        self.assertGreater(turn["e2e_latency_ms"], 0.0)

    def test_e2e_cache_miss_llm_path(self):
        """
        Verify: Audio In → ASR → Router → CACHE MISS → Local Quantized LLM → Local TTS.
        """
        print("\n--- [TEST] Starting End-to-End Cache Miss / Local LLM Path ---")
        audio_in = self._create_synthetic_speech_input("Explain what a computer is.")
        speech_end_ts = time.time()

        turn = self.pipeline.process_speech_audio(audio_in, speech_end_ts=speech_end_ts)

        print(f"Transcribed Text : '{turn['user_text']}'")
        print(f"Router Decision  : {turn['routing_decision']}")
        print(f"Response Text    : '{turn['response_text']}'")
        print(f"ASR Latency      : {turn['asr_latency_ms']} ms")
        print(f"Router Latency   : {turn['router_latency_ms']} ms")
        print(f"LLM Latency      : {turn['llm_latency_ms']} ms")
        print(f"TTS Latency      : {turn['tts_latency_ms']} ms")
        print(f"E2E Latency      : {turn['e2e_latency_ms']} ms")

        self.assertEqual(turn["routing_decision"], "CACHE MISS")
        self.assertGreater(turn["llm_latency_ms"], 0.0)
        self.assertGreater(turn["tokens_generated"], 0)
        self.assertGreater(turn["tokens_per_second"], 0.0)
        self.assertGreater(turn["tts_latency_ms"], 0.0)
        self.assertGreater(turn["e2e_latency_ms"], 0.0)

    def test_resource_and_telemetry_monitoring(self):
        """
        Verify real CPU, RAM, and cache telemetry tracking.
        """
        sys_metrics = monitor.get_system_metrics()
        cache_stats = monitor.get_cache_stats()

        print("\n--- [TELEMETRY] Real System Resource Snapshot ---")
        print(f"CPU Utilization  : {sys_metrics['cpu_percent']}%")
        print(f"Logical Cores    : {sys_metrics['logical_cores']}")
        print(f"RAM Total        : {sys_metrics['ram_total_mb']} MB")
        print(f"RAM Used         : {sys_metrics['ram_used_mb']} MB ({sys_metrics['ram_percent']}%)")
        print(f"Process RSS      : {sys_metrics['process_rss_mb']} MB")
        print(f"Cloud Status     : {sys_metrics['cloud_status']}")
        print(f"Execution Mode   : {sys_metrics['execution_mode']}")
        print(f"Total Cache Qs   : {cache_stats['total_queries']} (Hit rate: {cache_stats['hit_rate_percent']}%)")

        self.assertEqual(sys_metrics["cloud_status"], "OFF")
        self.assertEqual(sys_metrics["execution_mode"], "CPU-only")
        self.assertGreater(sys_metrics["logical_cores"], 0)
        self.assertGreater(sys_metrics["ram_total_mb"], 0)
        self.assertGreater(cache_stats["total_queries"], 0)


if __name__ == "__main__":
    unittest.main()
