"""
Comprehensive Multilingual, Streaming & Empirical Benchmark Test Suite.
Strictly implements the verification Test Plan:
  TEST 1: Normal voice conversation
  TEST 2: Cache HIT
  TEST 3: Cache MISS
  TEST 4: Semantic match
  TEST 5: Multilingual ASR (whisper.cpp & language detection)
  TEST 6: Performance mode
  TEST 7: Balanced mode
  TEST 8: Survival mode
  TEST 9: Resource-constrained execution (Constrained Device Simulation)
  TEST 10: Streaming / first-audio latency (TTFA breakdown)
  TEST 11: Baseline vs AdaptiveEdge (BenchmarkEngine live execution)
  TEST 12: Offline execution (Zero cloud AI, CPU-only verification)
"""

import unittest
import time
import json
import numpy as np
from scipy import signal

from backend.pipeline import VoicePipeline
from monitoring.benchmark_engine import BenchmarkEngine
from monitoring.resource_monitor import monitor
from monitoring.resource_controller import resource_controller


class TestPhase3MultilingualBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = VoicePipeline(enable_speaker=False)
        cls.benchmark_engine = BenchmarkEngine()

    def _create_synthetic_speech(self, text: str) -> np.ndarray:
        """Helper to synthesize clean WAV audio using local TTS engine."""
        tts_res = self.pipeline.tts.synthesize_to_wav(text)
        orig_sr = tts_res["sample_rate"]
        orig_audio = tts_res["audio"]
        target_len = int(len(orig_audio) * 16000 / orig_sr)
        return signal.resample(orig_audio, target_len).astype(np.float32)

    def test_01_normal_voice_conversation(self):
        """TEST 1: Normal voice conversation (Speech -> Whisper ASR -> Router -> TTS)."""
        print("\n--- TEST 1: Normal Voice Conversation ---")
        audio_in = self._create_synthetic_speech("What is your name?")
        speech_end_ts = time.time()
        turn = self.pipeline.process_speech_audio(audio_in, speech_end_ts=speech_end_ts)

        self.assertIsNotNone(turn["user_text"])
        self.assertIn("response_text", turn)
        self.assertGreater(turn["asr_latency_ms"], 0.0)
        self.assertGreater(turn["ttfa_ms"], 0.0)
        print(f"User: '{turn['user_text']}' -> AI: '{turn['response_text']}'")
        print(f"ASR Latency: {turn['asr_latency_ms']} ms | TTFA: {turn['ttfa_ms']} ms")

    def test_02_cache_hit(self):
        """TEST 2: Cache HIT (Exact match)."""
        print("\n--- TEST 2: Cache HIT (Exact) ---")
        turn = self.pipeline.process_text_query("What is your name?")
        self.assertEqual(turn["routing_decision"], "CACHE HIT")
        self.assertEqual(turn["match_type"], "EXACT")
        self.assertEqual(turn["llm_latency_ms"], 0.0)
        print(f"Decision: {turn['routing_decision']} ({turn['match_type']}) | TTFA: {turn['ttfa_ms']} ms")

    def test_03_cache_miss(self):
        """TEST 3: Cache MISS (Uncached complex query -> Local Quantized LLM)."""
        print("\n--- TEST 3: Cache MISS (Quantized LLM) ---")
        self.pipeline.resource_controller.set_mode("Performance", manual=True)
        turn = self.pipeline.process_text_query("Explain gravity in 10 words.")
        self.assertEqual(turn["routing_decision"], "CACHE MISS")
        self.assertGreater(turn["llm_latency_ms"], 0.0)
        self.assertGreater(turn["tokens_generated"], 0)
        print(f"Decision: {turn['routing_decision']} | LLM Latency: {turn['llm_latency_ms']} ms | Tokens: {turn['tokens_generated']}")

    def test_04_semantic_match(self):
        """TEST 4: Semantic match (Paraphrased query -> Semantic vector search)."""
        print("\n--- TEST 4: Semantic Match ---")
        self.pipeline.resource_controller.set_mode("Balanced", manual=True)
        # Paraphrase of "what are your working hours"
        turn = self.pipeline.process_text_query("Can you tell me your office timing?")
        self.assertEqual(turn["routing_decision"], "CACHE HIT")
        self.assertEqual(turn["match_type"], "SEMANTIC")
        self.assertGreaterEqual(turn["similarity_score"], turn["threshold"])
        self.assertEqual(turn["llm_latency_ms"], 0.0)
        print(f"Decision: {turn['routing_decision']} | Similarity: {turn['similarity_score']} >= {turn['threshold']}")

    def test_05_multilingual_asr(self):
        """TEST 5: Multilingual ASR (whisper.cpp engine transcription & language handling)."""
        print("\n--- TEST 5: Multilingual ASR (whisper.cpp) ---")
        self.pipeline.asr.backend_choice = "whisper.cpp"
        self.pipeline.asr._init_backend()

        self.assertEqual(self.pipeline.asr.active_backend, "whisper.cpp")
        print(f"Active ASR Backend: {self.pipeline.asr.active_backend}")

        test_speech = self._create_synthetic_speech("Hello world, this is a local voice test.")
        res = self.pipeline.asr.transcribe(test_speech)
        self.assertTrue(len(res["text"].strip()) > 0)
        self.assertIn("language", res)
        self.assertGreater(res["duration_sec"], 0.0)
        self.assertLess(res["latency_ms"], 2000.0)
        print(f"Transcribed: '{res['text']}' | Language: {res['language']} | Latency: {res['latency_ms']} ms")

    def test_06_performance_mode(self):
        """TEST 6: Performance mode policy verification."""
        print("\n--- TEST 6: Performance Mode ---")
        self.pipeline.resource_controller.set_mode("Performance", manual=True)
        policy = self.pipeline.resource_controller.get_active_policy()
        self.assertEqual(policy.name, "Performance")
        self.assertEqual(policy.llm_max_tokens, 80)
        self.assertEqual(policy.semantic_threshold, 0.65)
        self.assertEqual(policy.context_window, 1024)

    def test_07_balanced_mode(self):
        """TEST 7: Balanced mode policy verification."""
        print("\n--- TEST 7: Balanced Mode ---")
        self.pipeline.resource_controller.set_mode("Balanced", manual=True)
        policy = self.pipeline.resource_controller.get_active_policy()
        self.assertEqual(policy.name, "Balanced")
        self.assertEqual(policy.llm_max_tokens, 40)
        self.assertEqual(policy.semantic_threshold, 0.44)
        self.assertEqual(policy.context_window, 512)

    def test_08_survival_mode(self):
        """TEST 8: Survival mode policy verification."""
        print("\n--- TEST 8: Survival Mode ---")
        self.pipeline.resource_controller.set_mode("Survival", manual=True)
        policy = self.pipeline.resource_controller.get_active_policy()
        self.assertEqual(policy.name, "Survival")
        self.assertEqual(policy.llm_max_tokens, 18)
        self.assertEqual(policy.semantic_threshold, 0.38)
        self.assertEqual(policy.context_window, 256)

    def test_09_resource_constrained_execution(self):
        """TEST 9: Resource-constrained execution (Constrained Device Simulation mode)."""
        print("\n--- TEST 9: Constrained Device Simulation ---")
        self.pipeline.resource_controller.set_mode("Constrained", manual=True)
        policy = self.pipeline.resource_controller.get_active_policy()
        self.assertEqual(policy.name, "Constrained Device")
        self.assertEqual(policy.llm_max_tokens, 15)
        self.assertEqual(policy.semantic_threshold, 0.35)
        self.assertEqual(policy.context_window, 256)
        self.assertEqual(policy.llm_threads, 2)

        # Test generation constrained to 15 tokens
        turn = self.pipeline.process_text_query("What is quantum computing?")
        self.assertLessEqual(turn["tokens_generated"], 15)
        print(f"Constrained Policy Budget: 2 Threads | Max 15 tokens | Generated: {turn['tokens_generated']} tokens")

    def test_10_streaming_first_audio_latency(self):
        """TEST 10: Streaming / first-audio latency (TTFA breakdown T0..T5)."""
        print("\n--- TEST 10: Streaming & TTFA Breakdown ---")
        self.pipeline.resource_controller.set_mode("Performance", manual=True)
        
        # Test sentence-level streaming enabled
        self.pipeline.enable_streaming_tts = True
        turn_stream = self.pipeline.process_text_query("Tell me three interesting facts about space.")
        self.assertIn("ttfa_ms", turn_stream)
        self.assertIn("e2e_latency_ms", turn_stream)
        self.assertIn("ttft_ms", turn_stream)
        
        # Time to first audio should be less than or equal to total response latency
        self.assertLessEqual(turn_stream["ttfa_ms"], turn_stream["e2e_latency_ms"])
        print(f"Streaming TTFA: {turn_stream['ttfa_ms']} ms | Total E2E: {turn_stream['e2e_latency_ms']} ms")
        print(f"TTFT: {turn_stream['ttft_ms']} ms | LLM Latency: {turn_stream['llm_latency_ms']} ms")

    def test_11_baseline_vs_adaptive_edge(self):
        """TEST 11: Baseline vs AdaptiveEdge (Empirical BenchmarkEngine comparison)."""
        print("\n--- TEST 11: Live Empirical Benchmark Comparison ---")
        results = self.benchmark_engine.run_comprehensive_benchmark(self.pipeline)
        
        self.assertIn("baseline", results)
        self.assertIn("adaptive_edge", results)
        self.assertIn("ablation_findings", results)
        
        base = results["baseline"]
        edge = results["adaptive_edge"]
        abl = results["ablation_findings"]

        print(f"Baseline Avg TTFA:      {base['avg_ttfa_ms']} ms (100% LLM Invocations)")
        print(f"AdaptiveEdge Avg TTFA:  {edge['avg_ttfa_ms']} ms")
        print(f"Cache-Hit Avg Latency:  {edge['cache_hit_avg_latency_ms']} ms")
        print(f"LLM Invocations Saved:  {abl['llm_invocations_avoided_percent']}%")
        print(f"TTFA Reduction:         {abl['ttfa_reduction_percent']}%")

        self.assertLess(edge["avg_ttfa_ms"], base["avg_ttfa_ms"])
        self.assertGreater(abl["llm_invocations_avoided_percent"], 0.0)

        # Test export formats
        csv_out = self.benchmark_engine.export_csv()
        self.assertIn("Metric,Baseline,AdaptiveEdge,Difference", csv_out)
        self.assertIn("Time to First Audio (TTFA)", csv_out)

    def test_12_offline_execution(self):
        """TEST 12: Offline execution verification (Zero cloud AI, CPU-only)."""
        print("\n--- TEST 12: Offline Execution & Telemetry Verification ---")
        telem = monitor.get_system_telemetry()
        
        self.assertEqual(telem["cloud_status"], "OFF")
        self.assertEqual(telem["execution_mode"], "CPU-only")
        self.assertIn("battery", telem)
        self.assertTrue(telem["logical_cores"] > 0)
        self.assertTrue(telem["ram_total_mb"] > 0)
        print(f"Cloud Status: {telem['cloud_status']} | Arch: {telem['execution_mode']} | Cores: {telem['logical_cores']}")
        print(f"Energy telemetry: {telem['battery']['energy_status']}")


if __name__ == "__main__":
    unittest.main()
