#!/usr/bin/env python3
"""
Unit and Integration Tests for 7-DoF Action Format, Kinematics Execution, and Workspace Setup.
"""
import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from mujoco_env import MujocoRoboticEnv
from vla_wrapper import create_vla_model, get_all_models_status


class Test7DoFActionAndKinematics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = MujocoRoboticEnv()

    def setUp(self):
        self.env.reset()

    def test_tcp_site_is_transparent(self):
        """Verify the panda_tcp green tracking dot was rendered transparent."""
        site_id = self.env.tcp_site_id
        rgba = self.env.model.site_rgba[site_id]
        # rgba should have alpha == 0.0 so no green dot appears on camera
        self.assertAlmostEqual(rgba[3], 0.0, places=3, msg="panda_tcp site alpha must be 0 (transparent)")

    def test_table_dimensions_and_workspace_reach(self):
        """Verify the table is widened and all 5 objects and basket are comfortably within reach."""
        table_geom_id = self.env.model.geom("table_top").id
        table_size = self.env.model.geom_size[table_geom_id]
        # Half-sizes: [0.50, 0.65, 0.02] -> Table is 1.00m x 1.30m
        self.assertGreaterEqual(table_size[0], 0.48)
        self.assertGreaterEqual(table_size[1], 0.60)

        all_objs = self.env.get_all_objects_info()
        self.assertEqual(len(all_objs), 6, "Must have 5 objects + 1 basket")

        for k, info in all_objs.items():
            pos = np.array(info["pos"][:2])
            radius = np.linalg.norm(pos)
            # Franka arm maximum reach is ~0.85m, all objects should be comfortably in [0.40m, 0.65m]
            self.assertGreater(radius, 0.35, f"{k} is too close to robot base: {radius:.3f}m")
            self.assertLess(radius, 0.72, f"{k} exceeds comfortable reach: {radius:.3f}m")

    def test_vla_action_format_and_types(self):
        """Verify all VLA wrappers generate strictly valid 7-DoF actions."""
        test_models = ["octo", "rt1", "gemini_vla", "mock", "smolvla"]
        proprio = self.env.get_proprioception()
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)

        for m_name in test_models:
            vla = create_vla_model(m_name, chunk_size=4)
            action = vla.predict_action(dummy_img, "pick up the can and put into basket", proprio)

            # Strict 7-DoF check
            self.assertIsInstance(action, (list, np.ndarray), f"Model {m_name} must return list or array")
            self.assertEqual(len(action), 7, f"Model {m_name} action must have exactly 7 elements: {action}")

            for i, val in enumerate(action):
                self.assertIsInstance(float(val), float, f"Element {i} of {m_name} action must be float convertible")
                self.assertFalse(np.isnan(val), f"Element {i} of {m_name} cannot be NaN")
                self.assertFalse(np.isinf(val), f"Element {i} of {m_name} cannot be Inf")

            # Gripper cmd must be in [0, 1]
            self.assertGreaterEqual(action[6], 0.0)
            self.assertLessEqual(action[6], 1.0)

    def test_wrist_rotation_physics_execution(self):
        """Verify commanding dyaw physically rotates Joint 7 without displacing Cartesian position."""
        self.env.reset()
        init_tcp = self.env.get_tcp_pos()
        init_q7 = float(self.env.data.qpos[6])

        # Command pure yaw rotation step of +0.05 rad
        action_rotate = [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 1.0]
        obs = self.env.step(action_rotate)

        new_tcp = self.env.get_tcp_pos()
        new_q7 = float(self.env.data.qpos[6])

        # Joint 7 must have moved in the commanded direction
        q7_delta = new_q7 - init_q7
        self.assertGreater(q7_delta, 0.015, f"Joint 7 did not rotate as commanded: delta={q7_delta:.4f} rad")

        # TCP Cartesian position must remain essentially stationary (< 6 mm physical compliance)
        tcp_shift = np.linalg.norm(new_tcp - init_tcp)
        self.assertLess(tcp_shift, 0.006, f"Pure yaw command caused unwanted Cartesian translation: {tcp_shift*1000:.2f} mm")

    def test_full_trajectory_wrist_alignment_and_reach(self):
        """Verify that rolling out VLA policy steps rotates Joint 7 and converges to object."""
        self.env.reset()
        vla = create_vla_model("octo", chunk_size=4)
        target_pos = self.env.get_target_pos()

        # Expected target heading for active can object
        expected_yaw = float(np.arctan2(target_pos[1], target_pos[0])) - np.pi / 4.0
        initial_q7 = float(self.env.data.qpos[6])

        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        for _ in range(30):
            proprio = self.env.get_proprioception()
            act = vla.step_policy(dummy_img, "pick up soup can", proprio)
            self.env.step(act)

        final_q7 = float(self.env.data.qpos[6])
        # Joint 7 should have actively rotated from initial home (-0.7853) towards expected_yaw
        diff_initial = abs(expected_yaw - initial_q7)
        diff_final = abs(expected_yaw - final_q7)
        self.assertLess(diff_final, diff_initial, f"Wrist Joint 7 should rotate towards target heading. Initial diff: {diff_initial:.3f}, Final diff: {diff_final:.3f}")


if __name__ == "__main__":
    unittest.main()
