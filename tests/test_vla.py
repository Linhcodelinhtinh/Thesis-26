#!/usr/bin/env python3
"""
Unit and Integration Tests for VLA Multi-Modal Memory Architecture
-----------------------------------------------------------------
Verifies Chronicler, Spatial Tracker, Painter, Sync Layer, and VLA Wrappers.
"""
import sys
import os
import unittest
import numpy as np

# Add src folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from chronicler import Chronicler
from spatial_tracker import SpatialTracker
from painter import Painter
from sync_layer import SyncLayer
from vla_wrapper import MockVLAWrapper, OctoWrapper, OpenVLAWrapper, Pi0Wrapper, RTXWrapper


class TestVLAMiddleware(unittest.TestCase):
    def setUp(self):
        self.dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_vla_wrappers_api(self):
        """
        Verifies that all VLA wrappers conform to the 7-DoF API.
        """
        wrappers = [
            MockVLAWrapper(),
            OctoWrapper(),
            OpenVLAWrapper(),
            Pi0Wrapper(),
            RTXWrapper()
        ]
        
        proprioception = {"tcp_pos": [0.4, 0.0, 0.5], "gripper_pos": [100.0, 200.0], "gripper_width": 1.0}
        prompt = "[Task]: Grasp | [Status]: APPROACHING: Object visible. Gripper is OPEN. | [Memory]: Step 1: APPROACHING"
        
        for wrapper in wrappers:
            action = wrapper.predict_action(self.dummy_image, prompt, proprioception)
            self.assertEqual(len(action), 7, f"Wrapper {type(wrapper).__name__} did not return 7-DoF action vector.")
            self.assertIsInstance(action, (list, np.ndarray))

    def test_prompt_parsing(self):
        """
        Verifies prompt parsing helper in VLA base wrapper.
        """
        vla = MockVLAWrapper()
        test_prompt = "[Task]: Pick up the cup | [Status]: OCCLUSION_RECOVERY: Object is OCCLUDED. Gripper is CLOSED. | [Memory]: Step 1: APPROACHING -> Step 2: OCCLUDED_APPROACH"
        parsed = vla.parse_prompt(test_prompt)
        
        self.assertEqual(parsed["task"], "Pick up the cup")
        self.assertTrue(parsed["is_occluded"])
        self.assertTrue("Step 2: OCCLUDED_APPROACH" in parsed["memory"])

    def test_chronicler_temporal_state_machine(self):
        """
        Verifies state transitions and temporal memory logging in Chronicler.
        """
        chronicler = Chronicler(use_real_vlm=False)
        
        # Step 1: Far from target
        prop_1 = {"gripper_pos": [100.0, 100.0], "gripper_width": 1.0}
        track_1 = {"target_pos": [200.0, 200.0], "is_occluded": False}
        prompt_1 = chronicler.update_state(self.dummy_image, "Pick up the cup", prop_1, track_1)
        self.assertIn("[Task]: Pick up the cup", prompt_1)
        self.assertIn("APPROACHING", prompt_1)
        self.assertIn("Gripper is OPEN", prompt_1)
        
        # Step 2: Occlusion occurs
        track_2 = {"target_pos": [200.0, 200.0], "is_occluded": True}
        prompt_2 = chronicler.update_state(self.dummy_image, "Pick up the cup", prop_1, track_2)
        self.assertIn("OCCLUDED", prompt_2)
        self.assertTrue(chronicler.was_occluded)
        
        # Step 3: Grasp reached and closed
        prop_3 = {"gripper_pos": [200.0, 200.0], "gripper_width": 0.05}
        track_3 = {"target_pos": [200.0, 200.0], "is_occluded": True}
        prompt_3 = chronicler.update_state(self.dummy_image, "Pick up the cup", prop_3, track_3)
        self.assertIn("GRASP_LOCKED", prompt_3)
        self.assertIn("LIFTING", prompt_3)
        
        # Check history buffer
        state_dict = chronicler.get_state_dict()
        self.assertEqual(state_dict["current_phase"], "LIFTING")
        self.assertGreater(len(state_dict["history"]), 1)

    def test_spatial_tracker_persistent_memory(self):
        """
        Verifies tracking points initialization and persistent memory under occlusion.
        """
        tracker = SpatialTracker(tracker_type="opencv_fallback")
        tracker.initialize_tracking_points(self.dummy_image, [320, 240], num_points=9)
        
        self.assertEqual(len(tracker.points), 9)
        
        # Run track with gripper nearby (within occlusion radius)
        pts, flags = tracker.track(self.dummy_image, gripper_pos=[325, 245])
        self.assertEqual(len(pts), 9)
        self.assertEqual(len(flags), 9)
        self.assertTrue(all(flags))

    def test_octo_prompt_conditioned_action(self):
        """
        Verifies that OctoWrapper adjusts its 3D action trajectory based on Chronicler dynamic prompt.
        """
        octo = OctoWrapper()
        proprioception = {"tcp_pos": [0.55, 0.0, 0.45], "gripper_width": 0.1, "target_pos": [0.55, 0.0, 0.395]}
        
        # Prompt indicating object is grasped & lifting
        lifting_prompt = "[Task]: Grasp object | [Status]: GRASP_LOCKED: Object secured, executing lift. Object is VISIBLE. Gripper is CLOSED. | [Memory]: Step 3: LIFTING"
        action = octo.predict_action(self.dummy_image, lifting_prompt, proprioception)
        dx, dy, dz, _, _, _, gripper_cmd = action
        
        # dz should be positive (moving upwards) and gripper commanded closed (0.0)
        self.assertGreater(dz, 0.0)
        self.assertEqual(gripper_cmd, 0.0)

    def test_mock_vla_servoing(self):
        """
        Verifies visual-servoing behavior of the Mock VLA model with painted image and prompt.
        """
        vla = MockVLAWrapper()
        
        painted_image = self.dummy_image.copy()
        # Paint green dot at (200, 200)
        painted_image[200, 200] = [0, 255, 0]
        
        proprioception = {"gripper_pos": [100.0, 200.0], "gripper_width": 1.0}
        prompt = "[Task]: Grasp | [Status]: APPROACHING: Navigating. Object is VISIBLE. Gripper is OPEN."
        action = vla.predict_action(painted_image, prompt, proprioception)
        dx, dy, _, _, _, _, gripper_cmd = action
        
        self.assertGreater(dx, 0.0)
        self.assertAlmostEqual(dy, 0.0, places=3)
        self.assertEqual(gripper_cmd, 1.0)

    def test_sync_layer_concurrency(self):
        """
        Verifies thread-safe variable updates and retrievals.
        """
        sync = SyncLayer()
        sync.update_image("test_image_data")
        sync.update_prompt("test_prompt_data")
        
        img, prompt = sync.get_latest_state()
        self.assertEqual(img, "test_image_data")
        self.assertEqual(prompt, "test_prompt_data")

    def test_painter_rendering(self):
        """
        Verifies rendering output shape and HUD banner rendering.
        """
        painter = Painter()
        pts = [[100.0, 100.0], [200.0, 200.0]]
        flags = [False, True]
        prompt = "[Task]: Grasp mug | [Status]: Approaching. Object visible. | [Memory]: Step 1: APPROACHING"
        
        painted = painter.draw_overlay(self.dummy_image, pts, flags, prompt)
        self.assertEqual(painted.shape, self.dummy_image.shape)


if __name__ == "__main__":
    unittest.main()
