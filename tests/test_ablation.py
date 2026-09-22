#!/usr/bin/env python3
"""
Unit & Integration Tests for Ablation Study (Raw VLA vs. VLA + Memory)
"""
import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from vla_wrapper import MockVLAWrapper, OctoWrapper
from web_server import RoboticWebServer
from simpler_env_adapter import SimplerEnvAdapter
from chronicler import Chronicler
from spatial_tracker import SpatialTracker
from painter import Painter
from run_ablation_study import run_single_episode


class TestAblationStudy(unittest.TestCase):
    def setUp(self):
        self.mock_vla = MockVLAWrapper(action_chunk_size=4)
        self.dummy_image = np.zeros((256, 256, 3), dtype=np.uint8)
        self.proprio = {
            "tcp_pos": [0.45, 0.0, 0.52],
            "target_pos": [0.52, 0.0, 0.38],
            "gripper_width": 1.0,
            "is_occluded": False
        }

    def test_vla_ablation_mode_switching(self):
        """Verifies policy isolates prompt in Raw mode and respects memory in Memory mode."""
        # 1. Memory Mode (Default)
        self.mock_vla.set_ablation_mode("memory")
        self.assertEqual(self.mock_vla.ablation_mode, "memory")
        
        dynamic_prompt = "[Task]: pick coke can | [Status]: GRASP_LOCKED: Object secured. | [Memory]: Step 3: LIFTING"
        parsed = self.mock_vla.parse_prompt(dynamic_prompt)
        self.assertTrue(parsed["is_grasped"])

        # 2. Raw Mode
        self.mock_vla.set_ablation_mode("raw")
        self.assertEqual(self.mock_vla.ablation_mode, "raw")
        # In raw mode, buffer is cleared
        self.assertEqual(len(self.mock_vla.chunk_buffer), 0)

    def test_web_server_ablation_stats_api(self):
        """Verifies RoboticWebServer ablation stats calculation and toggle."""
        server = RoboticWebServer(host="127.0.0.1", port=8899)
        self.assertEqual(server.ablation_mode, "memory")

        # Toggle to raw
        ok = server.set_ablation_mode("raw")
        self.assertTrue(ok)
        self.assertEqual(server.ablation_mode, "raw")

        # Fetch stats
        stats = server.get_ablation_stats()
        self.assertIn("raw", stats)
        self.assertIn("memory", stats)
        self.assertIn("delta", stats)
        self.assertIn("current_mode", stats)
        self.assertEqual(stats["current_mode"], "raw")
        self.assertIn("success_rate", stats["delta"])

    def test_run_single_episode_execution(self):
        """Verifies episode rollout in both Raw and Memory ablation conditions."""
        env = SimplerEnvAdapter(task_name="google_robot_pick_coke_can", max_steps=35, seed=101)
        vla = MockVLAWrapper(action_chunk_size=4)
        chronicler = Chronicler(use_real_vlm=False)
        tracker = SpatialTracker(tracker_type="opencv_fallback")
        painter = Painter()

        # Run Memory Condition
        res_mem = run_single_episode(
            env=env, vla=vla, chronicler=chronicler,
            tracker=tracker, painter=painter,
            ablation_mode="memory", test_perturbation=False, seed=101
        )
        self.assertIn("success", res_mem)
        self.assertIn("steps", res_mem)
        self.assertIn("jerk", res_mem)
        self.assertTrue(res_mem["success"])

        # Run Raw Condition
        res_raw = run_single_episode(
            env=env, vla=vla, chronicler=chronicler,
            tracker=tracker, painter=painter,
            ablation_mode="raw", test_perturbation=False, seed=101
        )
        self.assertIn("success", res_raw)
        self.assertIn("steps", res_raw)

    def test_idle_state_and_model_switching(self):
        """Verifies robot stays in IDLE until command is sent and set_model works."""
        server = RoboticWebServer(host="127.0.0.1", port=8897)
        self.assertEqual(server.task_status, "IDLE")

        # In IDLE, run_simulation_step should not advance step_idx
        payload = server.run_simulation_step()
        self.assertEqual(payload["status"], "IDLE")
        self.assertEqual(server.step_idx, 0)

        # Repositioning target should remain IDLE and not burn steps
        new_pos = server.reposition_target()
        self.assertEqual(server.task_status, "IDLE")
        self.assertEqual(server.step_idx, 0)
        self.assertEqual(len(new_pos), 3)

        # Model switching with readiness check
        # openvla without API should be rejected
        ok, msg = server.set_model("openvla", api_url="")
        self.assertFalse(ok)
        self.assertEqual(server.vla_name, "Octo-Small")

        # openvla with API should succeed
        ok, msg = server.set_model("openvla", api_url="http://localhost:8000/act")
        self.assertTrue(ok)
        self.assertEqual(server.vla_name, "OpenVLA-7B")
        self.assertEqual(server.task_status, "IDLE")

        # rt1 is ready offline
        ok, msg = server.set_model("rt1")
        self.assertTrue(ok)
        self.assertEqual(server.vla_name, "Google RT-1")

        # pi0 with API should succeed
        ok, msg = server.set_model("pi0", api_url="http://localhost:8000/act")
        self.assertTrue(ok)
        self.assertEqual(server.vla_name, "Physical Intelligence pi0")


if __name__ == "__main__":
    unittest.main()

