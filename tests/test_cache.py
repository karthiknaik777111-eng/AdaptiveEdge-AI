"""
Unit tests for Tier 1 Local Cache and Router.
Verifies hit/miss logic, normalized pattern matching, and latency tracking.
"""

import unittest
from cache.local_cache import LocalKnowledgeCache
from backend.router import LocalRouter


class TestLocalCacheAndRouter(unittest.TestCase):
    def setUp(self):
        self.cache = LocalKnowledgeCache()
        self.router = LocalRouter()

    def test_cache_hit_exact(self):
        res = self.cache.lookup("what is your name")
        self.assertTrue(res.hit)
        self.assertIn("AdaptiveEdge", res.response)
        self.assertGreater(res.latency_ms, 0.0)

    def test_cache_hit_normalized_punctuation(self):
        # Query with caps and punctuation
        res = self.cache.lookup("What is your name???")
        self.assertTrue(res.hit)
        self.assertIn("AdaptiveEdge", res.response)

    def test_cache_hit_tagline(self):
        res = self.cache.lookup("what is your tagline")
        self.assertTrue(res.hit)
        self.assertIn("adapts to the limits", res.response)

    def test_cache_hit_dynamic_time(self):
        res = self.cache.lookup("What time is it now?")
        self.assertTrue(res.hit)
        self.assertIn("current local time is", res.response)

    def test_cache_miss_unseen(self):
        res = self.cache.lookup("Explain quantum entanglement and quantum teleportation.")
        self.assertFalse(res.hit)
        self.assertIsNone(res.response)
        self.assertGreater(res.latency_ms, 0.0)

    def test_router_routing_decisions(self):
        hit_route = self.router.route("hello")
        self.assertEqual(hit_route.decision, "CACHE HIT")
        self.assertIsNotNone(hit_route.response)

        miss_route = self.router.route("What is the capital of France?")
        self.assertEqual(miss_route.decision, "CACHE MISS")
        self.assertIsNone(miss_route.response)


if __name__ == "__main__":
    unittest.main()
