#!/usr/bin/env python3
"""
Native MuJoCo Interactive 3D Simulation & Evaluation Suite
==========================================================
Provides real, physical closed-loop evaluation between:
- Franka Emika Panda (DeepMind CAD model with 7-DoF position actuators + parallel gripper)
- Real YCB object (Campbell's Tomato Soup Can) + Placement Basket
- Vision-Language-Action (VLA) Policies (Octo, RT-1, OpenVLA, Pi0 with readiness filtering)
- Visual Grounding + TAPIR Persistent Spatial Tracking
- Chronicler Dynamic Prompt (Memory-as-a-Prompt)
- Native 3D OpenGL Viewer (mujoco.viewer) with real-time mouse perturbation (Ctrl+Right Click Drag)
- Real-time OpenCV Camera Perception & Visual Memory HUD window
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Add src folder to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mujoco_env import MujocoRoboticEnv
from vla_wrapper import create_vla_model, check_model_readiness, get_all_models_status
from spatial_tracker import SpatialTracker
from visual_grounding import VisualGroundingDetector
from chronicler import Chronicler
from painter import Painter

try:
    import mujoco
    import mujoco.viewer as mj_viewer
except ImportError:
    mujoco = None
    mj_viewer = None


class InteractiveMujocoSim:
    """
    Manages the interactive closed-loop simulation loop with dual visualization:
    1. Native MuJoCo 3D OpenGL Viewer (high FPS, interactive 3D camera & physical perturbation).
    2. OpenCV Window displaying the active robot camera feed with TAPIR tracking dots and Dynamic Prompt HUD.
    """
    def __init__(
        self,
        scene_path="assets/franka_table_scene.xml",
        model_name="octo",
        task_instruction="put can in basket",
        ablation_mode="memory",
        camera_name="overhead_cam",
        api_url="",
        headless=False
    ):
        if not os.path.isabs(scene_path):
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate = os.path.join(project_root, scene_path)
            if os.path.exists(candidate):
                scene_path = candidate

        self.scene_path = scene_path
        self.task_instruction = task_instruction
        self.ablation_mode = ablation_mode
        self.camera_name = camera_name
        self.api_url = api_url.strip() if api_url else os.environ.get("VLA_API_URL", "").strip()
        self.headless = headless

        # 1. Initialize MuJoCo Physics Environment
        print(f"[InteractiveSim] Initializing MuJoCo physics from '{scene_path}'...")
        self.env = MujocoRoboticEnv(scene_path=scene_path, render_width=640, render_height=480)

        # 2. Check and Filter Ready VLA Models
        self.all_models = ["octo", "rt1", "openvla", "pi0"]
        self.ready_models = [m for m in self.all_models if check_model_readiness(m, self.api_url)["ready"]]
        if not self.ready_models:
            self.ready_models = ["octo"]

        # Validate requested model
        req = str(model_name).lower().strip()
        readiness = check_model_readiness(req, self.api_url)
        if not readiness["ready"]:
            print(f"[InteractiveSim] [WARNING]: Model '{req}' is NOT READY ({readiness['badge']}).")
            print(f"[InteractiveSim] Ghosting '{req}' and falling back to ready model '{self.ready_models[0]}'.")
            print(f"[InteractiveSim] To enable '{req}', provide --api-url http://<host>:<port>/act")
            self.model_name = self.ready_models[0]
        else:
            self.model_name = req

        self.current_model_idx = self.ready_models.index(self.model_name) if self.model_name in self.ready_models else 0
        self._init_vla(self.ready_models[self.current_model_idx])

        # 3. Initialize Visual Perception & Memory Components
        self.grounding_detector = VisualGroundingDetector()
        self.tracker = SpatialTracker(tracker_type="tapir")
        self.chronicler = Chronicler(use_real_vlm=False)
        self.painter = Painter()

        # 4. State Flags
        self.is_running_policy = False
        self.is_teleop = False
        self.is_paused = False
        self.step_count = 0
        self.success = False
        self.teleop_delta = np.zeros(3, dtype=np.float32)
        self.teleop_gripper = 1.0

        # Cameras available
        self.camera_list = ["overhead_cam", "wrist_cam", "side_cam"]
        self.current_cam_idx = self.camera_list.index(camera_name) if camera_name in self.camera_list else 0

        self.reset_simulation()

    def _init_vla(self, model_name):
        readiness = check_model_readiness(model_name, self.api_url)
        print(f"[InteractiveSim] Loading VLA policy: {readiness['name']} [Ready: {readiness['badge']}] (Mode: {self.ablation_mode})...")
        self.vla = create_vla_model(
            model_name,
            chunk_size=4,
            mode=self.ablation_mode,
            api_url=self.api_url
        )
        self.vla_name = readiness["name"]

    def reset_simulation(self, randomize_target=False):
        """Resets robot to home configuration and initializes tracking."""
        self.env.reset()
        if randomize_target:
            self.env.reposition_target()

        self.chronicler.reset()
        self.tracker.clear()
        if hasattr(self.vla, "reset"):
            self.vla.reset()

        self.step_count = 0
        self.success = False
        self.is_running_policy = False
        print(f"[InteractiveSim] Reset completed. Target object at: {np.round(self.env.get_target_pos(), 3)}")

    def start_policy_execution(self):
        """Initializes Visual Grounding and starts on-demand execution."""
        raw_frame = self.env.render_camera(self.camera_list[self.current_cam_idx])
        
        # Authentic visual grounding from raw camera pixels
        grounding_res = self.grounding_detector.ground_instruction(raw_frame, self.task_instruction)
        centroid = grounding_res.get("centroid", [320.0, 240.0])
        bbox = grounding_res.get("bbox")

        # Spawn persistent TAPIR tracking points
        self.tracker.initialize_tracking_points(raw_frame, centroid, num_points=9, bbox=bbox)
        
        self.step_count = 0
        self.success = False
        self.is_running_policy = True
        self.is_teleop = False
        if hasattr(self.vla, "reset"):
            self.vla.reset()
        print(f"[InteractiveSim] Executing instruction: '{self.task_instruction}' with {self.vla_name}...")

    def draw_hud_overlay(self, frame, prompt_str, metrics, proprio):
        """Overlays dynamic prompt, telemetry badges, and hotkey guide onto OpenCV frame."""
        h, w, _ = frame.shape
        hud = frame.copy()

        # 1. Top header banner
        cv2.rectangle(hud, (0, 0), (w, 52), (18, 22, 28), -1)
        cv2.line(hud, (0, 52), (w, 52), (60, 75, 95), 1)

        model_badge = f"Model: {self.vla_name} [{self.ablation_mode.upper()}]"
        cv2.putText(hud, "MUJOCO NATIVE SIMULATION", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 225, 255), 2)
        cv2.putText(hud, model_badge, (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1)

        cam_text = f"Cam: {self.camera_list[self.current_cam_idx]}"
        cv2.putText(hud, cam_text, (w - 200, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 210, 160), 1)

        status_color = (0, 255, 120) if self.is_running_policy else ((0, 165, 255) if self.is_teleop else (180, 180, 180))
        status_text = "STATUS: RUNNING" if self.is_running_policy else ("STATUS: TELEOP" if self.is_teleop else "STATUS: IDLE")
        cv2.putText(hud, status_text, (w - 200, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 2)

        # 2. Bottom Dynamic Prompt Strip
        strip_h = 75
        cv2.rectangle(hud, (0, h - strip_h), (w, h), (14, 18, 24), -1)
        cv2.line(hud, (0, h - strip_h), (w, h - strip_h), (50, 65, 80), 1)

        # Dynamic Prompt Text
        cv2.putText(hud, "DYNAMIC PROMPT (Chronicler Memory):", (14, h - strip_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 255), 1)

        short_prompt = prompt_str[:90] + "..." if len(prompt_str) > 90 else prompt_str
        cv2.putText(hud, short_prompt, (14, h - strip_h + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 235, 245), 1)

        # Metrics bar
        lift_m = metrics.get("lift_height_meters", 0.0)
        force_n = metrics.get("max_contact_force_n", 0.0)
        grp_state = "OPEN" if proprio.get("gripper_width", 1.0) > 0.5 else "CLOSED"
        tele_text = f"Lift: {lift_m*100:.1f}cm | Force: {force_n:.1f}N | Gripper: {grp_state} | Step: {self.step_count}"
        cv2.putText(hud, tele_text, (14, h - strip_h + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 240, 160), 1)

        # 3. Hotkey Legend in Top Right
        legend_lines = [
            "[P] Run VLA Policy  [R] Randomize Target",
            "[M] Switch Model    [C] Switch Camera",
            "[T] Manual Teleop   [Space] Pause  [Q] Quit"
        ]
        for i, line in enumerate(legend_lines):
            cv2.putText(hud, line, (w - 380, h - strip_h + 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 160, 180), 1)

        # 4. Success Overlay Banner
        if self.success:
            cv2.rectangle(hud, (w//4, h//2 - 40), (3*w//4, h//2 + 40), (0, 180, 40), -1)
            cv2.rectangle(hud, (w//4, h//2 - 40), (3*w//4, h//2 + 40), (255, 255, 255), 2)
            cv2.putText(hud, "TASK SUCCESSFUL! (Goal Reached)", (w//4 + 20, h//2 + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return hud

    def run(self):
        """
        Main simulation loop. 
        When GUI interaction is requested, routes directly to the Web Dashboard 
        (eliminating native popup windows).
        In headless mode, executes a headless evaluation loop.
        """
        if not self.headless:
            print("\n=======================================================")
            print("  FRANKA PANDA MUJOCO VLA WORKSPACE")
            print("=======================================================")
            print("  Native desktop popup windows have been deprecated.")
            print("  Starting unified Web Dashboard at: http://localhost:8080")
            print("=======================================================\n")
            from web_server import RoboticWebServer
            server = RoboticWebServer(host="0.0.0.0", port=8080)
            server.task_name = self.task_instruction
            server.task_instruction = self.task_instruction
            server.set_model(self.model_name, api_url=self.api_url)
            server.start()
            return

        # Headless evaluation loop
        self.start_policy_execution()
        for _ in range(30):
            active_cam = self.camera_list[self.current_cam_idx]
            raw_frame = self.env.render_camera(active_cam)
            prop = self.env.get_proprioception()
            pts, flags = self.tracker.track_points(raw_frame)
            dynamic_prompt = self.chronicler.update_state(
                raw_frame, self.task_instruction, prop, {"target_pos": prop["target_pos"]}
            )
            action = self.vla.step_policy(raw_frame, dynamic_prompt, prop)
            obs = self.env.step(action)
            if self.env.check_basket_containment():
                self.success = True
                break
        print("[InteractiveSim] Headless execution completed.")


def main():
    parser = argparse.ArgumentParser(description="Franka Panda MuJoCo Robotic Simulation")
    parser.add_argument("--model", type=str, default="octo", choices=["octo", "openvla", "rt1", "pi0", "mock"], help="VLA model choice")
    parser.add_argument("--task", type=str, default="put can in basket", help="Task natural language instruction")
    parser.add_argument("--mode", type=str, default="memory", choices=["memory", "raw"], help="Ablation mode")
    parser.add_argument("--camera", type=str, default="overhead_cam", choices=["overhead_cam", "wrist_cam", "side_cam"], help="Initial camera")
    parser.add_argument("--api-url", type=str, default="", help="Optional remote GPU VLA API URL")
    parser.add_argument("--headless", action="store_true", help="Run in headless evaluation mode")
    args = parser.parse_args()

    sim = InteractiveMujocoSim(
        model_name=args.model,
        task_instruction=args.task,
        ablation_mode=args.mode,
        camera_name=args.camera,
        api_url=args.api_url,
        headless=args.headless
    )
    sim.run()


if __name__ == "__main__":
    main()
