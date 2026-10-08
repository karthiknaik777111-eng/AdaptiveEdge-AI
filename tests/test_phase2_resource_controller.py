"""
Unit Tests for Resource-Aware Controller.
Verifies:
  - Evaluation of Performance, Balanced, and Survival modes
  - Policy differences (semantic threshold, max tokens, threads, context)
  - Controlled resource pressure and latency transitions
  - Graceful degradation parameters
"""

import unittest
from monitoring.resource_controller import ResourceController


class TestPhase2ResourceController(unittest.TestCase):
    def setUp(self):
        # Create a test controller instance with known test thresholds
        self.controller = ResourceController(
            cpu_balanced_threshold=65.0,
            cpu_survival_threshold=85.0,
            ram_balanced_threshold=90.0,
            ram_survival_threshold=96.0,
            latency_balanced_threshold_ms=1200.0,
            latency_survival_threshold_ms=2500.0
        )

    def test_default_mode_and_policies(self):
        state = self.controller.get_controller_state()
        self.assertEqual(state["current_mode"], "Performance")
        
        perf = self.controller.policies["Performance"]
        bal = self.controller.policies["Balanced"]
        surv = self.controller.policies["Survival"]

        # Performance vs Balanced vs Survival constraints
        self.assertGreater(perf.llm_max_tokens, bal.llm_max_tokens)
        self.assertGreater(bal.llm_max_tokens, surv.llm_max_tokens)
        
        # Cache preference: Survival lowers threshold to aggressively prefer cache
        self.assertGreater(perf.semantic_threshold, bal.semantic_threshold)
        self.assertGreater(bal.semantic_threshold, surv.semantic_threshold)

        # Context reduction
        self.assertGreater(perf.context_window, bal.context_window)
        self.assertGreater(bal.context_window, surv.context_window)

    def test_latency_driven_adaptation(self):
        """
        Verify that latency measurements trigger automatic adaptation.
        """
        # Set high CPU/RAM thresholds in this test to isolate latency trigger
        self.controller.cpu_balanced_threshold = 99.0
        self.controller.cpu_resource_safe_threshold = 99.0
        self.controller.cpu_survival_threshold = 99.0
        self.controller.ram_balanced_threshold = 99.0
        self.controller.ram_resource_safe_threshold = 99.0
        self.controller.ram_survival_threshold = 99.0

        # Normal latency -> Performance
        mode1 = self.controller.evaluate_operating_mode(recent_latency_ms=450.0)
        self.assertEqual(mode1, "Performance")

        # Spiked latency above balanced threshold (1200ms) -> Balanced
        mode2 = self.controller.evaluate_operating_mode(recent_latency_ms=1400.0)
        self.assertEqual(mode2, "Balanced")
        self.assertEqual(self.controller.get_active_policy().llm_max_tokens, 40)

        # Severe latency above survival threshold (2500ms) -> Survival
        mode3 = self.controller.evaluate_operating_mode(recent_latency_ms=2800.0)
        self.assertEqual(mode3, "Survival")
        self.assertEqual(self.controller.get_active_policy().llm_max_tokens, 18)

    def test_manual_override_and_return_to_auto(self):
        self.controller.set_mode("Survival", manual=True)
        self.assertEqual(self.controller.current_mode, "Survival")
        self.assertTrue(self.controller.manual_override)

        # In manual override, low latency will not force it back to Performance
        self.controller.evaluate_operating_mode(recent_latency_ms=200.0)
        self.assertEqual(self.controller.current_mode, "Survival")

        # Re-enabling auto mode
        self.controller.set_auto_mode()
        self.assertFalse(self.controller.manual_override)

    def test_mode_transition_logging(self):
        initial_history_len = len(self.controller.mode_change_history)
        self.controller.set_mode("Balanced", manual=True)
        self.controller.set_mode("Survival", manual=True)
        self.assertEqual(len(self.controller.mode_change_history), initial_history_len + 2)


if __name__ == "__main__":
    unittest.main()
