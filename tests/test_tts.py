"""
Tests for Local TTS Service.
Verifies local CPU synthesis using Windows SAPI5 without network calls.
"""

import unittest
import os
from services.tts_service import TTSService


class TestLocalTTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tts = TTSService()

    def test_synthesize_wav(self):
        text = "Testing local voice synthesis on CPU."
        res = self.tts.synthesize_to_wav(text)
        
        self.assertIn("wav_path", res)
        self.assertTrue(os.path.exists(res["wav_path"]))
        self.assertGreater(res["duration_sec"], 0.5)
        self.assertGreater(res["latency_ms"], 0.0)
        self.assertGreater(len(res["audio"]), 0)
        print(f"[TTS Test] Latency: {res['latency_ms']}ms | Duration: {res['duration_sec']}s | Sample Rate: {res['sample_rate']}")


if __name__ == "__main__":
    unittest.main()
