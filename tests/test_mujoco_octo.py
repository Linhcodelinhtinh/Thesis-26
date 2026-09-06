#!/usr/bin/env python3
"""
Unit & Integration Tests for Octo VLA + MuJoCo Physical Simulation
"""
import sys
import os
import unittest
import numpy as np

# Add src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vla_wrapper import OctoWrapper
from src.mujoco_env import MujocoRoboticEnv


class TestMujocoOctoIntegration(unittest.TestCase):
    def setUp(self):
        self.scene_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "franka_table_scene.xml"))

    def test_octo_wrapper_7dof_action(self):
        """
        Verifies that OctoWrapper outputs valid 7-DoF Cartesian delta action.
        """
        octo = OctoWrapper(checkpoint_type="octo-small")
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        prompt = "[Task]: Pick up the cube | [Status]: Approaching target. Object is visible. Gripper is OPEN."
        proprioception = {
            "tcp_pos": [0.45, 0.0, 0.50],
            "gripper_pos": [0.45, 0.0],
            "gripper_width": 1.0,
            "joint_positions": [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.785]
        }

        action = octo.predict_action(dummy_img, prompt, proprioception)
        self.assertEqual(len(action), 7, "Action vector must contain exactly 7 elements.")
        self.assertIsInstance(action, (list, np.ndarray))
        # Verify gripper command is within [0.0, 1.0]
        self.assertGreaterEqual(action[6], 0.0)
        self.assertLessEqual(action[6], 1.0)

    def test_mujoco_env_lifecycle(self):
        """
        Tests MuJoCo environment initialization, reset, and step execution.
        """
        env = MujocoRoboticEnv(scene_path=self.scene_path, render_width=320, render_height=240)
        
        # Test Reset
        obs = env.reset()
        self.assertIn("gripper_pos", obs)
        self.assertIn("gripper_width", obs)
        self.assertIn("tcp_pos", obs)

        # Test Step forward
        action = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        step_obs = env.step(action)

        self.assertIn("proprioception", step_obs)
        self.assertIn("metrics", step_obs)
        self.assertIsInstance(step_obs["metrics"]["lift_height_meters"], float)
        self.assertIsInstance(step_obs["metrics"]["max_contact_force_n"], float)
        self.assertIsInstance(step_obs["metrics"]["is_grasped"], bool)

    def test_mujoco_camera_offscreen_render(self):
        """
        Verifies that MuJoCo offscreen rendering produces valid RGB frames.
        """
        env = MujocoRoboticEnv(scene_path=self.scene_path, render_width=320, render_height=240)
        frame = env.render_camera("overhead_cam")

        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        # Verify image is not all black
        self.assertGreater(np.mean(frame), 5.0)

    def test_end_to_end_octo_mujoco_closed_loop(self):
        """
        Verifies end-to-end multi-step closed-loop interaction.
        """
        env = MujocoRoboticEnv(scene_path=self.scene_path, render_width=320, render_height=240)
        octo = OctoWrapper(checkpoint_type="octo-small")

        env.reset()
        prompt = "[Task]: Grasp cube | [Status]: Approaching target. Object is visible. Gripper is OPEN."

        for _ in range(15):
            raw_rgb = env.render_camera("overhead_cam")
            proprio = env.get_proprioception()
            action = octo.predict_action(raw_rgb, prompt, proprio)
            step_obs = env.step(action)
            metrics = step_obs["metrics"]

            self.assertIn("lift_height_meters", metrics)
            self.assertIn("max_contact_force_n", metrics)
            self.assertIn("is_grasped", metrics)


if __name__ == "__main__":
    unittest.main()
