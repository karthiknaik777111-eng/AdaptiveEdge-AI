"""
Safe, bounded CPU resource stress test for demonstration of dynamic adaptation.
Runs a controlled workload in a background thread for at most 7 seconds.
Automatically yields with small sleeps to never freeze or harm the host machine.
"""
import time
import threading
from typing import Dict, Any


class SafeStressTester:
    def __init__(self, duration_sec: float = 7.0):
        self.duration_sec = duration_sec
        self.is_active = False
        self._stop_event = threading.Event()
        self._thread = None
        self._start_time = 0.0

    def start(self) -> Dict[str, Any]:
        if self.is_active:
            return {"status": "already_running", "is_active": True}
        self.is_active = True
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_safe_load, daemon=True)
        self._thread.start()
        return {"status": "started", "duration_sec": self.duration_sec, "is_active": True}

    def stop(self) -> Dict[str, Any]:
        if not self.is_active:
            return {"status": "not_running", "is_active": False}
        self._stop_event.set()
        self.is_active = False
        return {"status": "stopped", "is_active": False}

    def get_status(self) -> Dict[str, Any]:
        elapsed = round(time.time() - self._start_time, 1) if self.is_active else 0.0
        return {
            "is_active": self.is_active,
            "elapsed_sec": elapsed,
            "duration_sec": self.duration_sec
        }

    def _run_safe_load(self):
        print(f"[StressTester] Bounded stress test started (max {self.duration_sec}s)...")
        t_end = time.time() + self.duration_sec
        while not self._stop_event.is_set() and time.time() < t_end:
            # Safe floating math chunk
            total = 0.0
            for i in range(120000):
                total += (i ** 0.5)
            # Yield to OS scheduler so user's PC stays responsive
            time.sleep(0.005)
        self.is_active = False
        print("[StressTester] Bounded stress test completed safely.")


stress_tester = SafeStressTester(duration_sec=7.0)
