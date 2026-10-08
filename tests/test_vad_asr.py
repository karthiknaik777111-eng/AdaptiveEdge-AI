"""
Tests for Local VAD & ASR Services.
Verifies voice detection and transcription locally on CPU.
"""

import unittest
import numpy as np
from scipy import signal
from services.vad_service import VADService
from services.asr_service import ASRService
from services.tts_service import TTSService


class TestVADAndASR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vad = VADService(sample_rate=16000)
        cls.asr = ASRService()
        cls.tts = TTSService()

    def test_vad_silence(self):
        # 1 second of pure silence
        silence = np.zeros(16000, dtype=np.float32)
        is_speech, finished, _ = self.vad.process_chunk(silence[:512])
        self.assertFalse(is_speech)
        self.assertFalse(finished)

    def test_asr_with_synthesized_speech(self):
        # Generate clean local test audio using our local TTS
        tts_res = self.tts.synthesize_to_wav("What is your name?")
        orig_sr = tts_res["sample_rate"]
        orig_audio = tts_res["audio"]

        # Resample to 16kHz for Whisper and VAD
        target_len = int(len(orig_audio) * 16000 / orig_sr)
        resampled_16k = signal.resample(orig_audio, target_len).astype(np.float32)

        # 1. Test VAD across chunks of spoken audio
        speech_detected_in_any = False
        for i in range(0, target_len - 512, 512):
            chunk = resampled_16k[i : i + 512]
            if np.max(np.abs(chunk)) > 0.05:  # Audio has voice power
                is_speech, _, _ = self.vad.process_chunk(chunk)
                if is_speech:
                    speech_detected_in_any = True
                    break

        print(f"[VAD Test] Speech detected in spoken audio: {speech_detected_in_any}")
        self.assertTrue(speech_detected_in_any)

        # 2. Test ASR transcription
        asr_res = self.asr.transcribe(resampled_16k, sample_rate=16000)
        print(f"[ASR Test] Transcript: '{asr_res['text']}' | Latency: {asr_res['latency_ms']}ms")

        self.assertGreater(asr_res["latency_ms"], 0.0)
        # Check transcript matches query
        clean_text = asr_res["text"].lower()
        self.assertTrue("name" in clean_text or "what" in clean_text or "your" in clean_text)


if __name__ == "__main__":
    unittest.main()
