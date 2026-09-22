#!/usr/bin/env python3
"""
Unit & Integration Tests for Visual Grounding, TAPIR Spatial Tracking, and Multi-Object Manipulation
"""

import os
import sys
import unittest
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from visual_grounding import VisualGroundingDetector
from spatial_tracker import SpatialTracker
from simpler_env_adapter import SimplerEnvAdapter


class TestVisualGroundingAndMultiObject(unittest.TestCase):
    def setUp(self):
        self.env = SimplerEnvAdapter(task_name="google_robot_pick_coke_can", max_steps=40)
        self.grounding = VisualGroundingDetector()
        self.tracker = SpatialTracker(tracker_type="tapir")

    def test_instruction_semantic_parsing(self):
        """Verifies natural language keywords (English & Vietnamese) map to correct object IDs."""
        self.assertEqual(self.grounding.parse_target_id("pick the red bottle"), "red_bottle")
        self.assertEqual(self.grounding.parse_target_id("gắp chai tương ớt đỏ"), "red_bottle")
        self.assertEqual(self.grounding.parse_target_id("pick the blue box"), "blue_box")
        self.assertEqual(self.grounding.parse_target_id("gắp cái hộp màu xanh"), "blue_box")
        self.assertEqual(self.grounding.parse_target_id("pick dark syrup bottle"), "dark_bottle")
        self.assertEqual(self.grounding.parse_target_id("gắp chai màu đen"), "dark_bottle")
        self.assertEqual(self.grounding.parse_target_id("pick coke can"), "can")
        self.assertEqual(self.grounding.parse_target_id("put can in basket"), "can")
        self.assertEqual(self.grounding.parse_target_id("pick the basket"), "basket")

    def test_visual_detection_on_camera_frame(self):
        """Verifies visual detection finds object centroids and bounding boxes from raw camera pixels."""
        frame = self.env.render()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (300, 300, 3))

        # Test Red Bottle Grounding
        res_red = self.grounding.ground_instruction(frame, "pick red bottle")
        self.assertEqual(res_red["target_id"], "red_bottle")
        self.assertIn("centroid", res_red)
        cx, cy = res_red["centroid"]
        self.assertTrue(100 <= cx <= 200, f"Red bottle X expected in [100, 200], got {cx}")
        self.assertTrue(100 <= cy <= 200, f"Red bottle Y expected in [100, 200], got {cy}")
        self.assertGreater(res_red["confidence"], 0.6)

        # Test Blue Box Grounding
        res_blue = self.grounding.ground_instruction(frame, "pick blue box")
        self.assertEqual(res_blue["target_id"], "blue_box")
        bx, by = res_blue["centroid"]
        self.assertTrue(150 <= bx <= 230, f"Blue box X expected in [150, 230], got {bx}")

        # Test Can Grounding
        res_can = self.grounding.ground_instruction(frame, "pick coke can")
        self.assertEqual(res_can["target_id"], "can")
        kx, ky = res_can["centroid"]
        self.assertTrue(120 <= kx <= 220, f"Can X expected in [120, 220], got {kx}")
        self.assertTrue(150 <= ky <= 240, f"Can Y expected in [150, 240], got {ky}")

    def test_spatial_tracker_on_demand_lifecycle(self):
        """Verifies tracker starts clean (IDLE) and only creates points upon command initialization."""
        self.assertFalse(self.tracker.has_points())
        pts, occ = self.tracker.track(self.env.render())
        self.assertEqual(len(pts), 0)
        self.assertEqual(len(occ), 0)

        # Spawn points via visual grounding result
        frame = self.env.render()
        res = self.grounding.ground_instruction(frame, "pick red bottle")
        self.tracker.initialize_tracking_points(frame, res["centroid"], num_points=9, bbox=res.get("bbox"))
        self.assertTrue(self.tracker.has_points())
        self.assertEqual(len(self.tracker.points), 9)

        # Track with gripper away -> points visible (False)
        pts_track, flags = self.tracker.track(frame, gripper_pos=[50.0, 50.0])
        self.assertEqual(len(pts_track), 9)
        self.assertFalse(any(flags), "Points should be visible when gripper is far")

        # Track with gripper close -> points occluded (True)
        pts_track_occ, flags_occ = self.tracker.track(frame, gripper_pos=res["centroid"])
        self.assertTrue(any(flags_occ), "Points should be occluded when gripper is directly over them")

        # Reset clears tracker
        self.tracker.clear()
        self.assertFalse(self.tracker.has_points())

    def test_multi_object_physics_grasping(self):
        """Verifies any chosen object (e.g. red bottle) can be targeted, grasped, and lifted."""
        # Target red bottle
        success = self.env.set_target_object("red_bottle")
        self.assertTrue(success)
        self.assertEqual(self.env.active_target_id, "red_bottle")

        # Move robot directly to red bottle coordinates
        target_xyz = self.env.objects["red_bottle"]["pos"]
        self.env.tcp_pos = np.copy(target_xyz)
        self.env.tcp_pos[2] += 0.01  # just above
        self.env.gripper_width = 0.5  # partially closed

        # Step with closing gripper command (run 2 steps for actuator smoothing)
        self.env.step([0.0, 0.0, -0.01, 0, 0, 0, 0.0])
        self.env.step([0.0, 0.0, 0.0, 0, 0, 0, 0.0])
        self.assertTrue(self.env.objects["red_bottle"]["is_grasped"])

        # Step lifting up (2 steps)
        self.env.step([0.0, 0.0, 0.03, 0, 0, 0, 0.0])
        self.env.step([0.0, 0.0, 0.03, 0, 0, 0, 0.0])
        self.assertGreater(self.env.objects["red_bottle"]["lift_height"], 0.01)
        self.assertGreater(self.env.objects["red_bottle"]["pos"][2], 0.38)


if __name__ == "__main__":
    unittest.main()
