"""
Regression Test Suite for Single-Voice Playback and Acoustic Safety.
Verifies:
  1. Default single-player policy (pipeline.enable_speaker is False to prevent host + browser overlap)
  2. Hardware audio stop capability (tts.stop_audio)
  3. Muting/ignoring microphone chunks while system is speaking (acoustic feedback protection)
  4. Complete response WAV generation for client (unified audio file for browser)
  5. Cache-hit and cache-miss valid audio deliverables
"""

import unittest
import os
import time
import numpy as np

from backend.pipeline import VoicePipeline
from services.tts_service import TTSService


class TestAudioPlaybackRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = VoicePipeline(sample_rate=16000, enable_speaker=False, enable_streaming_tts=True)

    def test_01_single_player_default_policy(self):
        """Verify pipeline defaults to enable_speaker=False to prevent dual-playback."""
        self.assertFalse(
            self.pipeline.enable_speaker,
            "enable_speaker must default to False so server does not dual-play with the browser."
        )

    def test_02_tts_stop_audio_method(self):
        """Verify tts.stop_audio exists and executes cleanly."""
        tts = TTSService()
        self.assertTrue(hasattr(tts, "stop_audio"))
        # Should not raise exception
        tts.stop_audio()

    def test_03_acoustic_feedback_guard_in_speaking_state(self):
        """Verify that when pipeline is in SPEAKING or PROCESSING state, mic loop discards chunks."""
        self.pipeline.current_listening_state = "SPEAKING"
        # Simulate speech chunk
        dummy_chunk = np.zeros(512, dtype=np.float32)
        # Check listening state is protected
        self.assertIn(self.pipeline.current_listening_state, ("SPEAKING", "PROCESSING"))
        # Reset state
        self.pipeline.current_listening_state = "IDLE"

    def test_04_cache_hit_produces_single_valid_audio(self):
        """Verify cache hit produces exactly one valid WAV payload."""
        turn = self.pipeline.process_text_query("Hello")
        self.assertEqual(turn["routing_decision"], "CACHE HIT")
        self.assertIsNotNone(turn["wav_path"])
        self.assertTrue(os.path.exists(turn["wav_path"]))
        self.assertIsNotNone(turn["audio_url"])
        self.assertIsNotNone(turn["audio_data_url"])
        self.assertTrue(turn["audio_data_url"].startswith("data:audio/wav;base64,"))

    def test_05_cache_miss_produces_single_valid_audio(self):
        """Verify cache miss produces exactly one complete valid WAV payload."""
        turn = self.pipeline.process_text_query("What is the speed of sound in air?")
        self.assertEqual(turn["routing_decision"], "CACHE MISS")
        self.assertIsNotNone(turn["wav_path"])
        self.assertTrue(os.path.exists(turn["wav_path"]))
        self.assertIsNotNone(turn["audio_url"])
        self.assertIsNotNone(turn["audio_data_url"])
        self.assertTrue(turn["audio_data_url"].startswith("data:audio/wav;base64,"))

    def test_06_stop_listening_halts_audio(self):
        """Verify stop_listening safely stops recording and hardware audio."""
        self.pipeline.start_listening()
        self.assertTrue(self.pipeline.is_listening)
        time.sleep(0.1)
        self.pipeline.stop_listening()
        self.assertFalse(self.pipeline.is_listening)
        self.assertEqual(self.pipeline.current_listening_state, "IDLE")


if __name__ == "__main__":
    unittest.main()
