#!/usr/bin/env python3
"""
Teleoperation & Expert Demonstration Controller for SimplerEnv Tasks
---------------------------------------------------------------------
Provides:
1. Keyboard Teleoperation: Interactive manual robot control via standard keys
2. Scripted Oracle Expert: Automated heuristic policy generating guaranteed
   successful demonstration trajectories for dataset collection & replay verification.
"""
import time
import math
import numpy as np


class TeleopController:
    """
    Translates user keyboard inputs or external teleop devices into
    7-DoF delta Cartesian robot commands: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """
    def __init__(self, step_size=0.02, rot_step=0.1):
        self.step_size = step_size
        self.rot_step = rot_step
        self.gripper_state = 1.0  # 1.0 = Open, 0.0 = Closed

    def process_key(self, key_code):
        """
        Maps standard key codes to 7-DoF actions.
        Key mappings:
        - W / S: +X (Forward) / -X (Backward)
        - A / D: +Y (Left) / -Y (Right)
        - R / F: +Z (Up) / -Z (Down)
        - Q / E: -Yaw / +Yaw
        - Space: Toggle Gripper Open/Close
        """
        dx, dy, dz = 0.0, 0.0, 0.0
        dyaw = 0.0

        if key_code in [ord('w'), ord('W')]:
            dx = self.step_size
        elif key_code in [ord('s'), ord('S')]:
            dx = -self.step_size
        elif key_code in [ord('a'), ord('A')]:
            dy = self.step_size
        elif key_code in [ord('d'), ord('D')]:
            dy = -self.step_size
        elif key_code in [ord('r'), ord('R')]:
            dz = self.step_size
        elif key_code in [ord('f'), ord('F')]:
            dz = -self.step_size
        elif key_code in [ord('q'), ord('Q')]:
            dyaw = -self.rot_step
        elif key_code in [ord('e'), ord('E')]:
            dyaw = self.rot_step
        elif key_code == 32:  # Spacebar
            self.gripper_state = 0.0 if self.gripper_state > 0.5 else 1.0
            print(f"[Teleop] Gripper toggled to: {'OPEN' if self.gripper_state > 0.5 else 'CLOSED'}")

        return [dx, dy, dz, 0.0, 0.0, dyaw, self.gripper_state]


class ScriptedOracleExpert:
    """
    Heuristic waypoint motion planner for generating ground-truth expert demonstrations
    across SimplerEnv benchmark tasks.
    """
    def __init__(self, task_name="google_robot_pick_coke_can"):
        self.task_name = task_name
        self.phase = "INIT"
        self.timer = 0

    def reset(self):
        self.phase = "INIT"
        self.timer = 0

    def get_action(self, obs):
        """
        Computes the next optimal Cartesian delta action based on observation proprioception.
        """
        proprio = obs.get("proprioception", {})
        tcp_pos = np.array(proprio.get("tcp_pos", [0.45, 0.0, 0.52]), dtype=np.float32)
        target_pos = np.array(proprio.get("target_pos", [0.52, 0.0, 0.38]), dtype=np.float32)
        gripper_w = float(proprio.get("gripper_width", 1.0))

        dx, dy, dz = 0.0, 0.0, 0.0
        gripper_cmd = 1.0
        step_speed = 0.025

        if "pick_coke_can" in self.task_name:
            dist_xy = float(np.linalg.norm(tcp_pos[:2] - target_pos[:2]))
            target_hover_z = target_pos[2] + 0.12
            target_grasp_z = target_pos[2] - 0.005

            if self.phase == "INIT":
                self.phase = "ALIGN_XY"

            if self.phase == "ALIGN_XY":
                gripper_cmd = 1.0
                if dist_xy > 0.012:
                    dx = float((target_pos[0] - tcp_pos[0]) / dist_xy) * min(step_speed, dist_xy)
                    dy = float((target_pos[1] - tcp_pos[1]) / dist_xy) * min(step_speed, dist_xy)
                else:
                    self.phase = "DESCEND"

            elif self.phase == "DESCEND":
                gripper_cmd = 1.0
                if tcp_pos[2] > target_grasp_z + 0.008:
                    dz = -0.018
                    # Fine-tune XY while descending
                    if dist_xy > 0.005:
                        dx = float((target_pos[0] - tcp_pos[0]) / dist_xy) * 0.005
                        dy = float((target_pos[1] - tcp_pos[1]) / dist_xy) * 0.005
                else:
                    self.phase = "GRASP"
                    self.timer = 0

            elif self.phase == "GRASP":
                gripper_cmd = 0.0
                self.timer += 1
                if self.timer > 8 or gripper_w < 0.35:
                    self.phase = "LIFT"

            elif self.phase == "LIFT":
                gripper_cmd = 0.0
                if tcp_pos[2] < target_hover_z + 0.06:
                    dz = 0.022
                else:
                    # Hold steadily in the air to sustain success condition
                    dz = 0.0

        elif "open_drawer" in self.task_name:
            handle_pos = target_pos
            dist_xy = float(np.linalg.norm(tcp_pos[:2] - handle_pos[:2]))

            if self.phase == "INIT":
                self.phase = "APPROACH_HANDLE"

            if self.phase == "APPROACH_HANDLE":
                gripper_cmd = 1.0
                if dist_xy > 0.01:
                    dx = float((handle_pos[0] - tcp_pos[0]) / dist_xy) * min(step_speed, dist_xy)
                    dy = float((handle_pos[1] - tcp_pos[1]) / dist_xy) * min(step_speed, dist_xy)
                else:
                    self.phase = "DESCEND_HANDLE"

            elif self.phase == "DESCEND_HANDLE":
                gripper_cmd = 1.0
                if abs(tcp_pos[2] - handle_pos[2]) > 0.01:
                    dz = -0.015 if tcp_pos[2] > handle_pos[2] else 0.015
                else:
                    self.phase = "GRASP_HANDLE"
                    self.timer = 0

            elif self.phase == "GRASP_HANDLE":
                gripper_cmd = 0.0
                self.timer += 1
                if self.timer > 6:
                    self.phase = "PULL_DRAWER"

            elif self.phase == "PULL_DRAWER":
                gripper_cmd = 0.0
                # Pull backwards (-X)
                dx = -0.020

        elif "basket" in self.task_name:
            # WidowX Eggplant task
            dist_xy = float(np.linalg.norm(tcp_pos[:2] - target_pos[:2]))
            if self.phase == "INIT":
                self.phase = "APPROACH"

            if self.phase == "APPROACH":
                gripper_cmd = 1.0
                if dist_xy > 0.01:
                    dx = float((target_pos[0] - tcp_pos[0]) / dist_xy) * min(step_speed, dist_xy)
                    dy = float((target_pos[1] - tcp_pos[1]) / dist_xy) * min(step_speed, dist_xy)
                else:
                    self.phase = "DESCEND"

            elif self.phase == "DESCEND":
                gripper_cmd = 1.0
                if tcp_pos[2] > 0.14:
                    dz = -0.015
                else:
                    self.phase = "GRASP"
                    self.timer = 0

            elif self.phase == "GRASP":
                gripper_cmd = 0.0
                self.timer += 1
                if self.timer > 6:
                    self.phase = "LIFT"

            elif self.phase == "LIFT":
                gripper_cmd = 0.0
                if tcp_pos[2] < 0.25:
                    dz = 0.02
                else:
                    self.phase = "MOVE_TO_BASKET"

            elif self.phase == "MOVE_TO_BASKET":
                gripper_cmd = 0.0
                # Basket target ~ [0.28, 0.10]
                basket_x, basket_y = 0.28, 0.10
                dist_b = float(np.sqrt((tcp_pos[0] - basket_x)**2 + (tcp_pos[1] - basket_y)**2))
                if dist_b > 0.015:
                    dx = float((basket_x - tcp_pos[0]) / dist_b) * 0.02
                    dy = float((basket_y - tcp_pos[1]) / dist_b) * 0.02
                else:
                    self.phase = "RELEASE"

            elif self.phase == "RELEASE":
                gripper_cmd = 1.0

        return [dx, dy, dz, 0.0, 0.0, 0.0, gripper_cmd]


if __name__ == "__main__":
    print("[TeleopController] Testing ScriptedOracleExpert on 'google_robot_pick_coke_can'...")
    from simpler_env_adapter import SimplerEnvAdapter

    env = SimplerEnvAdapter("google_robot_pick_coke_can", max_steps=40)
    expert = ScriptedOracleExpert("google_robot_pick_coke_can")

    obs, _ = env.reset(seed=42)
    success = False

    for step_num in range(35):
        act = expert.get_action(obs)
        obs, r, term, trunc, info = env.step(act)
        if info["success"]:
            print(f"  Expert achieved SUCCESS at step {step_num + 1}! (Lift height: {info['lift_height']:.3f}m)")
            success = True
            break

    assert success, "Oracle expert must achieve success on pick_coke_can!"
    print("[TeleopController] Oracle expert verification passed successfully!")
