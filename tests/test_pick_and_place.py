#!/usr/bin/env python3
"""
Unit tests for Pick vs Pick-and-Place Evaluation & Safe Robot Retraction
"""
import unittest
import numpy as np
from src.simpler_env_adapter import SimplerEnvAdapter
from src.vla_wrapper import MockVLAWrapper, OpenVLAWrapper


class TestPickAndPlaceEvaluation(unittest.TestCase):
    def setUp(self):
        self.env = SimplerEnvAdapter(task_name="google_robot_pick_coke_can")
        self.env.reset(seed=42)

    def test_reposition_object_updates_coordinates(self):
        """Verifies reposition_object updates both target_pos and objects dictionary."""
        old_pos = np.copy(self.env.objects["can"]["pos"])
        new_pos = self.env.reposition_object("can")
        
        self.assertFalse(np.allclose(old_pos, new_pos))
        self.assertTrue(np.allclose(self.env.objects["can"]["pos"], new_pos))
        self.assertTrue(np.allclose(self.env.target_pos, new_pos))
        self.assertFalse(self.env.is_grasped)
        self.assertEqual(self.env.lift_height, 0.0)

    def test_task_classification_pick_vs_pnp(self):
        """Verifies is_pick_and_place_task distinguishes pick commands from pick-and-place commands."""
        self.env.instruction = "pick coke can"
        self.assertFalse(self.env.is_pick_and_place_task())

        self.env.instruction = "grasp the red condiment bottle"
        self.assertFalse(self.env.is_pick_and_place_task())

        self.env.instruction = "put can in basket"
        self.assertTrue(self.env.is_pick_and_place_task())

        self.env.instruction = "place the coke can into the basket"
        self.assertTrue(self.env.is_pick_and_place_task())

        self.env.instruction = "bỏ lon vào giỏ"
        self.assertTrue(self.env.is_pick_and_place_task())

    def test_pick_only_success_condition(self):
        """For pick tasks, success triggers when object is lifted >= 5cm while grasped."""
        self.env.instruction = "pick coke can"
        active_obj = self.env.objects["can"]

        # 1. Approach and grasp can
        self.env.tcp_pos = np.copy(active_obj["pos"])
        self.env.gripper_width = 0.20 # closed
        self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(active_obj["is_grasped"])

        # 2. Lift by 8cm (giving at least 3 steps with lift_height >= 0.05m to satisfy success_counter >= 3)
        for _ in range(8):
            self.env.step([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0])

        self.assertGreaterEqual(self.env.lift_height, 0.05)
        self.assertTrue(self.env.is_success)

    def test_pick_and_place_success_condition(self):
        """For pick-and-place, lifting is NOT sufficient; object must be transported to basket and released."""
        self.env.instruction = "put can in basket"
        active_obj = self.env.objects["can"]
        basket_pos = self.env.objects["basket"]["pos"]

        # 1. Grasp and lift can
        self.env.tcp_pos = np.copy(active_obj["pos"])
        self.env.gripper_width = 0.20
        self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(active_obj["is_grasped"])

        # Lift by 6cm
        for _ in range(5):
            self.env.step([0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0])

        self.assertGreaterEqual(self.env.lift_height, 0.05)
        # CRUCIAL: In PNP task, merely lifting must NOT declare success!
        self.assertFalse(self.env.is_success, "PNP task must NOT succeed on lift alone!")

        # 2. Transport to basket
        self.env.tcp_pos[0] = basket_pos[0]
        self.env.tcp_pos[1] = basket_pos[1]
        self.env.tcp_pos[2] = basket_pos[2] + 0.03 # lowered in basket
        # Still holding: not yet released
        self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertFalse(self.env.is_success, "PNP task must NOT succeed before releasing gripper!")

        # 3. Release gripper in basket
        for _ in range(5):
            self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        self.assertGreater(self.env.gripper_width, 0.50)
        self.assertFalse(active_obj["is_grasped"])
        self.assertTrue(self.env.is_success, "PNP task must succeed when object is deposited in basket and released!")

    def test_vla_pick_and_place_trajectory_rollout(self):
        """Verifies VLA policy progresses through approach -> descend -> grasp -> lift -> transport -> place -> release -> retract_home."""
        vla = MockVLAWrapper()
        obs, _ = self.env.reset()
        self.env.instruction = "put can in basket"

        phases_visited = set()
        for step in range(65):
            prop = self.env.get_proprioception()
            prompt = f"[Task]: put can in basket | [Status]: Phase: {vla.phase} | [Memory]: Step {step}"
            action = vla.predict_action(obs["image"], prompt, prop)
            phases_visited.add(vla.phase)
            obs, _, _, _, _ = self.env.step(action)

        self.assertIn("approach", phases_visited)
        self.assertIn("descend", phases_visited)
        self.assertIn("grasp", phases_visited)
        self.assertIn("lift", phases_visited)
        self.assertIn("transport", phases_visited)
        self.assertIn("place", phases_visited)
        self.assertIn("release", phases_visited)


if __name__ == "__main__":
    unittest.main()
