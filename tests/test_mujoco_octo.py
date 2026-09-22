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

    def test_mujoco_reposition_target_and_basket(self):
        """
        Verifies target repositioning on table and basket containment geometry.
        """
        env = MujocoRoboticEnv(scene_path=self.scene_path, render_width=320, render_height=240)
        env.reset()
        
        # Test basket location
        basket_pos = env.get_basket_pos()
        self.assertEqual(len(basket_pos), 3)
        self.assertAlmostEqual(basket_pos[0], 0.45, places=2)
        self.assertAlmostEqual(basket_pos[1], -0.28, places=2)

        # Initially object is on table, not in basket
        self.assertFalse(env.check_basket_containment())

        # Test repositioning target
        old_pos = np.copy(env.get_target_pos())
        new_pos = env.reposition_target(x=0.52, y=0.04, z=0.40)
        self.assertAlmostEqual(new_pos[0], 0.52, places=2)
        self.assertAlmostEqual(new_pos[1], 0.04, places=2)

        # Place object inside basket coordinates and verify containment
        env.reposition_target(x=basket_pos[0], y=basket_pos[1], z=basket_pos[2] + 0.06)
        self.assertTrue(env.check_basket_containment())

    def test_interactive_mujoco_runner_lifecycle(self):
        """
        Verifies InteractiveMujocoSim initialization, grounding, and closed-loop stepping.
        """
        from scripts.run_interactive_mujoco import InteractiveMujocoSim
        sim = InteractiveMujocoSim(model_name="octo", task_instruction="put can in basket", headless=True)
        self.assertIsNotNone(sim.env)
        self.assertIsNotNone(sim.vla)

        sim.start_policy_execution()
        self.assertTrue(sim.is_running_policy)
        self.assertGreaterEqual(len(sim.tracker.current_points), 1)

        # Execute 3 control steps
        for _ in range(3):
            raw_frame = sim.env.render_camera(sim.camera_list[sim.current_cam_idx])
            prop = sim.env.get_proprioception()
            pts, flags = sim.tracker.track_points(raw_frame)
            dynamic_prompt = sim.chronicler.update_state(raw_frame, sim.task_instruction, prop, {"target_pos": prop["target_pos"]})
            action = sim.vla.step_policy(raw_frame, dynamic_prompt, prop)
            step_obs = sim.env.step(action)
            self.assertEqual(len(action), 7)
            self.assertIn("metrics", step_obs)

    def test_mujoco_runner_model_readiness_filtering(self):
        """
        Verifies that requesting an unconfigured model (openvla without API) falls back to ready model (octo).
        """
        from src.mujoco_runner import InteractiveMujocoSim
        # Without API: requested openvla -> falls back to octo
        sim_fallback = InteractiveMujocoSim(model_name="openvla", api_url="", headless=True)
        self.assertEqual(sim_fallback.model_name, "octo")

        # With API: requested openvla -> accepted
        sim_ready = InteractiveMujocoSim(model_name="openvla", api_url="http://localhost:8000/act", headless=True)
        self.assertEqual(sim_ready.model_name, "openvla")


if __name__ == "__main__":
    unittest.main()
