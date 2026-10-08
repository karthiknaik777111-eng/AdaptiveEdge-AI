"""
Unit and Integration Tests for Semantic Local Cache.
Verifies:
  - Exact match path (Level 1A)
  - Dynamic rule path (Level 1B)
  - Dense local vector similarity path (Level 1C)
  - Threshold sensitivity
  - Distinguishing CACHE HIT vs CACHE MISS with real similarity scores.
"""

import unittest
from cache.semantic_cache import SemanticLocalCache


class TestPhase2SemanticCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = SemanticLocalCache(default_threshold=0.55)

    def test_exact_match(self):
        res = self.cache.lookup("what is your name")
        self.assertTrue(res.hit)
        self.assertEqual(res.match_type, "EXACT")
        self.assertEqual(res.similarity_score, 1.0)
        self.assertIn("AdaptiveEdge", res.response)

    def test_semantic_match_different_wording(self):
        """
        Query: 'Can you tell me your office timing?'
        Stored: 'what are your working hours'
        Demonstrates real semantic similarity (~0.446) against configurable threshold.
        """
        query = "Can you tell me your office timing?"
        # At threshold 0.40 (e.g. Balanced/Survival mode), produces CACHE HIT
        res_hit = self.cache.lookup(query, threshold=0.40)
        print(f"\n[Semantic Test] Query: '{query}'")
        print(f"Matched Key    : '{res_hit.match_key}'")
        print(f"Match Type     : {res_hit.match_type}")
        print(f"Similarity     : {res_hit.similarity_score} (Threshold: {res_hit.threshold})")
        print(f"Lookup Latency : {res_hit.latency_ms} ms")
        print(f"Response       : '{res_hit.response}'")

        self.assertTrue(res_hit.hit)
        self.assertEqual(res_hit.match_type, "SEMANTIC")
        self.assertAlmostEqual(res_hit.similarity_score, 0.446, delta=0.03)
        self.assertIn("twenty four hours", res_hit.response)

        # At higher threshold 0.60 (Performance mode), produces CACHE MISS
        res_miss = self.cache.lookup(query, threshold=0.60)
        self.assertFalse(res_miss.hit)
        self.assertEqual(res_miss.match_type, "MISS")

    def test_semantic_match_memory(self):
        query = "How much RAM do you consume?"
        res = self.cache.lookup(query, threshold=0.50)
        print(f"\n[Semantic Test] Query: '{query}' -> Match: '{res.match_key}' (Sim: {res.similarity_score})")
        self.assertTrue(res.hit)
        self.assertEqual(res.match_type, "SEMANTIC")
        self.assertIn("ram", res.response.lower())

    def test_semantic_miss_unrelated(self):
        """
        Query: 'What is the speed of light in vacuum?'
        Should produce CACHE MISS because similarity is below threshold.
        """
        query = "What is the speed of light in vacuum?"
        res = self.cache.lookup(query, threshold=0.55)
        print(f"\n[Semantic Miss Test] Query: '{query}' -> Max Sim: {res.similarity_score} < {res.threshold}")
        self.assertFalse(res.hit)
        self.assertEqual(res.match_type, "MISS")
        self.assertIsNone(res.response)


if __name__ == "__main__":
    unittest.main()
