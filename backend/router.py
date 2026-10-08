"""
Tiered Local Semantic Router for AdaptiveEdge On-Device Conversational AI.
Evaluates queries hierarchically:
  Level 1A: Exact / Normalized Match (Instant)
  Level 1B: Dynamic Rules
  Level 1C: Local Semantic Vector Match (via sentence-transformers)
Governed dynamically by the ResourceController operating mode.
"""

import time
from typing import NamedTuple, Optional, Dict, Any
from cache.semantic_cache import semantic_cache, SemanticLookupResult
from monitoring.resource_controller import resource_controller
from monitoring.resource_monitor import monitor


class RouteResult(NamedTuple):
    decision: str           # "CACHE HIT" | "CACHE MISS"
    response: Optional[str]
    match_key: Optional[str]
    match_type: str         # "EXACT" | "RULE" | "SEMANTIC" | "MISS"
    similarity_score: float
    threshold: float
    operating_mode: str
    latency_ms: float


class TieredLocalRouter:
    def __init__(self):
        self.cache = semantic_cache
        self.controller = resource_controller

    def route(self, query: str, recent_latency_ms: Optional[float] = None) -> RouteResult:
        """
        Evaluate query against Tier 1 local knowledge cache using the active resource mode.
        """
        t_start = time.perf_counter()

        # Update / evaluate operating mode based on current telemetry
        active_mode = self.controller.evaluate_operating_mode(recent_latency_ms=recent_latency_ms)
        policy = self.controller.get_active_policy()

        # Query semantic cache with dynamic threshold determined by resource mode
        lookup = self.cache.lookup(query, threshold=policy.semantic_threshold)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if lookup.hit:
            decision = "CACHE HIT"
            monitor.record_cache_event(hit=True)
            return RouteResult(
                decision=decision,
                response=lookup.response,
                match_key=lookup.match_key,
                match_type=lookup.match_type,
                similarity_score=lookup.similarity_score,
                threshold=lookup.threshold,
                operating_mode=active_mode,
                latency_ms=round(latency_ms, 3)
            )
        else:
            decision = "CACHE MISS"
            monitor.record_cache_event(hit=False)
            return RouteResult(
                decision=decision,
                response=None,
                match_key=lookup.match_key,
                match_type="MISS",
                similarity_score=lookup.similarity_score,
                threshold=lookup.threshold,
                operating_mode=active_mode,
                latency_ms=round(latency_ms, 3)
            )


# Global instance
router = TieredLocalRouter()
LocalRouter = TieredLocalRouter  # Backward compatibility alias
