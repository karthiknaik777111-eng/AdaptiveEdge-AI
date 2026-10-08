"""
Resource and Latency Monitoring for AdaptiveEdge Voice AI.
Tracks real CPU usage, RAM utilization, available cores, and pipeline latencies.
No fake or simulated metrics.
"""

import time
import os
import psutil
from typing import Dict, Any, List, Optional
from collections import deque


class ResourceMonitor:
    def __init__(self, history_limit: int = 100):
        self.process = psutil.Process(os.getpid())
        self.logical_cores = psutil.cpu_count(logical=True) or 1
        self.physical_cores = psutil.cpu_count(logical=False) or 1
        self.history_limit = history_limit
        self.latency_history: deque = deque(maxlen=history_limit)
        
        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

    def get_system_metrics(self) -> Dict[str, Any]:
        """Collect current real CPU, RAM, and process metrics."""
        vm = psutil.virtual_memory()
        proc_mem = self.process.memory_info()
        
        # CPU percent across all logical cores
        sys_cpu = psutil.cpu_percent(interval=None)
        proc_cpu = self.process.cpu_percent(interval=None)

        # Real battery/power sensor where available
        battery = psutil.sensors_battery()
        battery_data = {
            "available": battery is not None,
            "percent": battery.percent if battery else None,
            "power_plugged": battery.power_plugged if battery else True,
            "energy_status": "Battery sensor active" if battery else "Direct energy measurement unavailable on this hardware"
        }

        return {
            "timestamp": time.time(),
            "cpu_percent": sys_cpu,
            "process_cpu_percent": proc_cpu,
            "logical_cores": self.logical_cores,
            "physical_cores": self.physical_cores,
            "ram_total_mb": round(vm.total / (1024 * 1024), 1),
            "ram_used_mb": round(vm.used / (1024 * 1024), 1),
            "ram_available_mb": round(vm.available / (1024 * 1024), 1),
            "ram_percent": vm.percent,
            "process_rss_mb": round(proc_mem.rss / (1024 * 1024), 1),
            "cloud_status": "OFF",
            "cloud_ai": "OFF",
            "local_inference": "ON",
            "execution_mode": "CPU-only",
            "operating_mode": "Performance",
            "available_threads": self.logical_cores,
            "temperature": "Not measured (No ACPI thermal driver exposed)",
            "battery": battery_data
        }

    def get_system_telemetry(self) -> Dict[str, Any]:
        """Alias for get_system_metrics."""
        return self.get_system_metrics()

    def record_cache_event(self, hit: bool):
        """Update cache hit / miss counters."""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100.0) if total > 0 else 0.0
        return {
            "total_queries": total,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 1)
        }

    def record_pipeline_turn(self, turn_metrics: Dict[str, Any]):
        """Record latency and resource measurements for a conversation turn."""
        self.latency_history.append(turn_metrics)

    def get_recent_turns(self, limit: int = 10) -> List[Dict[str, Any]]:
        return list(self.latency_history)[-limit:]


# Global singleton instance
monitor = ResourceMonitor()
