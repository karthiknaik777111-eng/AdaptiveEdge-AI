"""
Dynamic Resource-Aware Controller for AdaptiveEdge On-Device Conversational AI.
Monitors real CPU workload, RAM, and pipeline latency to dynamically govern:
  - Operating Mode: PERFORMANCE | BALANCED | RESOURCE-SAFE | SURVIVAL | CONSTRAINED
  - Dynamic CPU Thread / Core Allocation for AI execution
  - Semantic Cache Similarity Thresholds (preferring local cache under pressure)
  - LLM Execution Constraints (context truncation, max tokens, thread throttling)
  - Graceful Degradation policies

All telemetry is measured from the actual OS environment via psutil and real timestamps.
"""

import time
import os
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from monitoring.resource_monitor import monitor


@dataclass
class ModePolicy:
    name: str
    semantic_threshold: float
    llm_max_tokens: int
    llm_threads: int
    context_window: int
    cache_preference: str  # "STANDARD" | "HIGH" | "AGGRESSIVE"
    description: str


class ResourceController:
    def __init__(
        self,
        # Configurable thresholds
        cpu_balanced_threshold: float = 50.0,
        cpu_resource_safe_threshold: float = 70.0,
        cpu_survival_threshold: float = 85.0,
        ram_balanced_threshold: float = 82.0,
        ram_resource_safe_threshold: float = 88.0,
        ram_survival_threshold: float = 93.0,
        latency_balanced_threshold_ms: float = 1200.0,
        latency_resource_safe_threshold_ms: float = 2000.0,
        latency_survival_threshold_ms: float = 3000.0,
    ):
        self.cpu_balanced_threshold = cpu_balanced_threshold
        self.cpu_resource_safe_threshold = cpu_resource_safe_threshold
        self.cpu_survival_threshold = cpu_survival_threshold
        self.ram_balanced_threshold = ram_balanced_threshold
        self.ram_resource_safe_threshold = ram_resource_safe_threshold
        self.ram_survival_threshold = ram_survival_threshold
        self.latency_balanced_threshold_ms = latency_balanced_threshold_ms
        self.latency_resource_safe_threshold_ms = latency_resource_safe_threshold_ms
        self.latency_survival_threshold_ms = latency_survival_threshold_ms

        total_cores = psutil.cpu_count(logical=True) or 4
        # Allocate up to 8 threads on high-core machines, or half cores on smaller machines
        perf_threads = min(8, max(4, total_cores // 2))

        # Policies
        self.policies: Dict[str, ModePolicy] = {
            "Performance": ModePolicy(
                name="Performance",
                semantic_threshold=0.65,
                llm_max_tokens=80,
                llm_threads=perf_threads,
                context_window=1024,
                cache_preference="STANDARD",
                description=f"Low system load. Peak AI allocation ({perf_threads} CPU threads) for rapid local inference."
            ),
            "Balanced": ModePolicy(
                name="Balanced",
                semantic_threshold=0.44,
                llm_max_tokens=40,
                llm_threads=4,
                context_window=512,
                cache_preference="HIGH",
                description="Moderate system load. Balanced thread budget (4 CPU threads) preserving machine responsiveness."
            ),
            "Resource-Safe": ModePolicy(
                name="Resource-Safe",
                semantic_threshold=0.40,
                llm_max_tokens=25,
                llm_threads=2,
                context_window=256,
                cache_preference="AGGRESSIVE",
                description="Elevated system load. Throttling AI allocation to 2 CPU threads to eliminate resource contention."
            ),
            "Survival": ModePolicy(
                name="Survival",
                semantic_threshold=0.38,
                llm_max_tokens=18,
                llm_threads=2,
                context_window=256,
                cache_preference="AGGRESSIVE",
                description="Severe resource constraint. Minimal thread allocation keeping voice loop responsive."
            ),
            "Constrained": ModePolicy(
                name="Constrained Device",
                semantic_threshold=0.35,
                llm_max_tokens=15,
                llm_threads=2,
                context_window=256,
                cache_preference="AGGRESSIVE",
                description="Constrained Device Simulation (2-core budget, 15 max tokens, aggressive cache preference)."
            )
        }

        self.current_mode: str = "Performance"
        self.manual_override: bool = False
        self.last_evaluation_time: float = time.time()
        self.mode_change_history: List[Dict[str, Any]] = []

    def evaluate_operating_mode(self, recent_latency_ms: Optional[float] = None) -> str:
        """
        Evaluate real hardware and pipeline telemetry to set the operating mode.
        If manual_override is set, returns current_mode directly.
        """
        if self.manual_override:
            return self.current_mode

        vm = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_pct = vm.percent

        # Determine target mode based on actual pressure
        target_mode = "Performance"
        reason = "System load within baseline (< 50% CPU)"

        # Check Survival conditions
        if (
            cpu_pct >= self.cpu_survival_threshold
            or ram_pct >= self.ram_survival_threshold
            or (recent_latency_ms is not None and recent_latency_ms >= self.latency_survival_threshold_ms)
        ):
            target_mode = "Survival"
            reason = f"Critical resource pressure (CPU: {cpu_pct}%, RAM: {ram_pct}%, Latency: {recent_latency_ms}ms)"
        # Check Resource-Safe conditions
        elif (
            cpu_pct >= self.cpu_resource_safe_threshold
            or ram_pct >= self.ram_resource_safe_threshold
            or (recent_latency_ms is not None and recent_latency_ms >= self.latency_resource_safe_threshold_ms)
        ):
            target_mode = "Resource-Safe"
            reason = f"High system load (CPU: {cpu_pct}%, RAM: {ram_pct}%, Latency: {recent_latency_ms}ms)"
        # Check Balanced conditions
        elif (
            cpu_pct >= self.cpu_balanced_threshold
            or ram_pct >= self.ram_balanced_threshold
            or (recent_latency_ms is not None and recent_latency_ms >= self.latency_balanced_threshold_ms)
        ):
            target_mode = "Balanced"
            reason = f"Moderate system load (CPU: {cpu_pct}%, RAM: {ram_pct}%, Latency: {recent_latency_ms}ms)"

        if target_mode != self.current_mode:
            self._record_mode_transition(self.current_mode, target_mode, reason)
            self.current_mode = target_mode

        return self.current_mode

    def set_mode(self, mode_name: str, manual: bool = True):
        """Set mode manually (useful for testing or dashboard selection)."""
        # Handle case variations gracefully
        lookup = {k.lower(): k for k in self.policies.keys()}
        target = lookup.get(mode_name.lower())
        if not target:
            raise ValueError(f"Unknown mode: {mode_name}")
        old_mode = self.current_mode
        self.current_mode = target
        self.manual_override = manual
        self._record_mode_transition(old_mode, target, "Manual user or test override")

    def set_auto_mode(self):
        """Re-enable automated resource monitoring."""
        self.manual_override = False
        self.evaluate_operating_mode()

    def get_active_policy(self) -> ModePolicy:
        """Get the current operational policy."""
        return self.policies.get(self.current_mode, self.policies["Performance"])

    def get_current_policy(self) -> ModePolicy:
        """Alias for get_active_policy."""
        return self.get_active_policy()

    def _record_mode_transition(self, from_mode: str, to_mode: str, reason: str):
        event = {
            "timestamp": time.time(),
            "from_mode": from_mode,
            "to_mode": to_mode,
            "reason": reason
        }
        self.mode_change_history.append(event)
        print(f"[ResourceController] Mode transitioned: {from_mode} -> {to_mode} ({reason})")

    def get_controller_state(self) -> Dict[str, Any]:
        policy = self.get_active_policy()
        available_cores = psutil.cpu_count(logical=True) or 1
        sys_load = psutil.cpu_percent(interval=None)

        return {
            "current_mode": self.current_mode,
            "manual_override": self.manual_override,
            "adaptation_status": f"Auto-Adapting to Load ({self.current_mode})" if not self.manual_override else f"Manual Override ({self.current_mode})",
            "system_load_percent": round(sys_load, 1),
            "available_threads": available_cores,
            "ai_threads": policy.llm_threads,
            "semantic_threshold": policy.semantic_threshold,
            "llm_max_tokens": policy.llm_max_tokens,
            "llm_threads": policy.llm_threads,
            "context_window": policy.context_window,
            "cache_preference": policy.cache_preference,
            "description": policy.description,
            "thresholds": {
                "cpu_balanced": self.cpu_balanced_threshold,
                "cpu_resource_safe": self.cpu_resource_safe_threshold,
                "cpu_survival": self.cpu_survival_threshold,
                "ram_balanced": self.ram_balanced_threshold,
                "ram_resource_safe": self.ram_resource_safe_threshold,
                "ram_survival": self.ram_survival_threshold,
                "latency_balanced_ms": self.latency_balanced_threshold_ms,
                "latency_resource_safe_ms": self.latency_resource_safe_threshold_ms,
                "latency_survival_ms": self.latency_survival_threshold_ms
            },
            "recent_transitions": self.mode_change_history[-5:]
        }


# Global instance
resource_controller = ResourceController()
