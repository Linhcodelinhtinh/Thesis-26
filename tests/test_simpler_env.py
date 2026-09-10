#!/usr/bin/env python3
"""
Unit & Integration Tests for SimplerEnv Evaluation Platform
------------------------------------------------------------
Verifies:
1. SimplerEnv Adapter lifecycle across Google Robot & WidowX tasks
2. Observation schema: RGB, Depth, Camera Intrinsics/Extrinsics, Proprioception
3. 7-DoF Cartesian action execution & Action Chunking
4. Success condition triggers (Pick Coke Can & Open Drawer)
5. Trajectory logging & Deterministic Replay verification
"""
import sys
import os
import unittest
import numpy as np

# Add src and scripts to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

from simpler_env_adapter import SimplerEnvAdapter
from trajectory_recorder import TrajectoryRecorder
from teleop_controller import TeleopController, ScriptedOracleExpert
from vla_wrapper import OctoWrapper, MockVLAWrapper
from replay_trajectory import replay_trajectory


class TestSimplerEnvPlatform(unittest.TestCase):
    def test_adapter_google_robot_pick_coke_can(self):
        """Tests Google Robot Pick Coke Can task initialization and step execution."""
        env = SimplerEnvAdapter("google_robot_pick_coke_can", max_steps=20, seed=42)
        obs, info = env.reset()

        # Observation assertions
        self.assertIn("image", obs)
        self.assertIn("depth", obs)
        self.assertIn("proprioception", obs)
        self.assertIn("instruction", obs)
        self.assertIn("camera_calibration", obs)

        # Image shape (300, 300, 3)
        self.assertEqual(obs["image"].shape, (300, 300, 3))
        self.assertEqual(obs["image"].dtype, np.uint8)

        # Proprioception keys
        proprio = obs["proprioception"]
        self.assertIn("tcp_pos", proprio)
        self.assertIn("gripper_width", proprio)
        self.assertIn("joint_positions", proprio)
        self.assertEqual(len(proprio["tcp_pos"]), 3)
        self.assertEqual(len(proprio["joint_positions"]), 7)

        # Camera calibration
        calib = obs["camera_calibration"]
        self.assertIn("intrinsics", calib)
        self.assertIn("extrinsics", calib)

        # Step execution
        action = [0.01, -0.01, -0.01, 0.0, 0.0, 0.0, 1.0]
        step_obs, r, term, trunc, step_info = env.step(action)
        self.assertIsInstance(r, float)
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        self.assertIn("success", step_info)
        self.assertIn("lift_height", step_info)
        env.close()

    def test_adapter_drawer_task(self):
        """Tests Google Robot Open Drawer task."""
        env = SimplerEnvAdapter("google_robot_open_drawer", max_steps=20, seed=42)
        obs, info = env.reset()
        self.assertEqual(obs["instruction"], "open top drawer")

        # Step
        step_obs, r, term, trunc, step_info = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertIn("drawer_displacement", step_info)
        env.close()

    def test_adapter_widowx_task(self):
        """Tests WidowX Bridge task format."""
        env = SimplerEnvAdapter("widowx_put_eggplant_in_basket", max_steps=20, seed=42)
        obs, info = env.reset()
        # WidowX uses 256x256 resolution
        self.assertEqual(obs["image"].shape, (256, 256, 3))
        self.assertEqual(len(obs["proprioception"]["joint_positions"]), 6)
        env.close()

    def test_action_chunking(self):
        """Tests action chunking prediction and buffer consumption in VLA wrappers."""
        octo = OctoWrapper(action_chunk_size=4)
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        prompt = "[Task]: pick coke can | [Status]: Approaching"
        proprio = {"tcp_pos": [0.45, 0.0, 0.52], "gripper_width": 1.0, "target_pos": [0.52, 0.0, 0.38]}

        # Predict chunk
        chunk = octo.predict_action_chunk(dummy_img, prompt, proprio, chunk_size=4)
        self.assertEqual(len(chunk), 4)
        for act in chunk:
            self.assertEqual(len(act), 7)

        # Step policy via buffer
        act1 = octo.step_policy(dummy_img, prompt, proprio)
        self.assertEqual(len(act1), 7)
        self.assertEqual(len(octo.chunk_buffer), 3)

    def test_scripted_oracle_expert_success(self):
        """Tests that the Scripted Oracle Expert successfully completes the pick task."""
        env = SimplerEnvAdapter("google_robot_pick_coke_can", max_steps=40, seed=100)
        expert = ScriptedOracleExpert("google_robot_pick_coke_can")
        obs, _ = env.reset(seed=100)
        success = False

        for _ in range(35):
            act = expert.get_action(obs)
            obs, r, term, trunc, info = env.step(act)
            if info.get("success", False):
                success = True
                break

        self.assertTrue(success, "Scripted oracle expert must achieve success on pick task.")
        env.close()

    def test_trajectory_recording_and_deterministic_replay(self):
        """Tests end-to-end trajectory logging and zero-drift deterministic replay."""
        save_dir = "data/test_replay_dir"
        env = SimplerEnvAdapter("google_robot_pick_coke_can", max_steps=35, seed=777)
        recorder = TrajectoryRecorder(save_dir=save_dir, task_name="google_robot_pick_coke_can")
        expert = ScriptedOracleExpert("google_robot_pick_coke_can")

        obs, _ = env.reset(seed=777)
        recorder.start_trajectory(task_name="google_robot_pick_coke_can", seed=777, source="test_replay")

        for step_i in range(30):
            act = expert.get_action(obs)
            obs, r, term, trunc, info = env.step(act)
            recorder.record_step(step_i, obs, act, r, term, trunc, info)
            if info["success"]:
                break

        res = recorder.save()
        env.close()

        self.assertIsNotNone(res["json_path"])
        self.assertTrue(os.path.exists(res["json_path"]))

        # Execute deterministic replay
        replay_res = replay_trajectory(res["json_path"], visualize=False, tolerance=0.005)
        self.assertTrue(replay_res["passed"], "Deterministic replay must pass.")
        self.assertLess(replay_res["max_deviation_mm"], 1.0, "Cartesian deviation must be < 1.0mm.")
        self.assertTrue(replay_res["matched_success"], "Replayed success status must match original.")

    def test_teleop_key_mappings(self):
        """Tests keyboard teleop command generation."""
        teleop = TeleopController(step_size=0.02)
        # Move forward (W)
        act_w = teleop.process_key(ord('w'))
        self.assertAlmostEqual(act_w[0], 0.02)
        # Move up (R)
        act_r = teleop.process_key(ord('r'))
        self.assertAlmostEqual(act_r[2], 0.02)
        # Gripper toggle (Space)
        init_grp = teleop.gripper_state
        act_space = teleop.process_key(32)
        self.assertNotEqual(teleop.gripper_state, init_grp)


if __name__ == "__main__":
    unittest.main()
