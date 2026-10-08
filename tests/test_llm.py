"""
Tests for Local Quantized LLM Service.
Verifies CPU-only inference, speed, and streaming capability.
"""

import unittest
from services.llm_service import LLMService


class TestLocalLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = LLMService()

    def test_llm_generation(self):
        result = self.llm.generate("State the color of the sky in two words.")
        self.assertIsNotNone(result["text"])
        self.assertTrue(len(result["text"]) > 0)
        self.assertGreater(result["latency_ms"], 0.0)
        self.assertGreater(result["tokens"], 0)
        self.assertGreater(result["tokens_per_second"], 0.0)
        print(f"[LLM Test] Output: '{result['text']}' | Latency: {result['latency_ms']}ms | Speed: {result['tokens_per_second']} tok/s")

    def test_llm_streaming(self):
        chunks = []
        first_token_ttft = None
        for chunk in self.llm.generate_stream("What is 10 plus 5? Answer briefly."):
            chunks.append(chunk["token"])
            if chunk["first_token"]:
                first_token_ttft = chunk["ttft_ms"]

        full_stream_text = "".join(chunks)
        self.assertTrue(len(full_stream_text) > 0)
        self.assertIsNotNone(first_token_ttft)
        print(f"[LLM Stream Test] Output: '{full_stream_text}' | TTFT: {first_token_ttft}ms")


if __name__ == "__main__":
    unittest.main()
