#!/usr/bin/env python3
"""
Main Entrypoint — VLA Embodied Memory & SimplerEnv Evaluation Platform
-----------------------------------------------------------------------
Orchestrates:
1. SimplerEnv Simulation: Zero-shot real-world robot evaluation (Google Robot & WidowX)
2. The Chronicler: Memory-as-a-Prompt state management
3. The Spatial Tracker & Painter: Persistent tracking under occlusion
4. VLA Policy Inference: Action Chunking execution (Octo / OpenVLA / Mock)
5. Trajectory Recording: Observation -> Action -> Reward/Success with calibration & timestamps
6. UI Notifications: Prominent real-time TASK SUCCESS / TASK FAILED overlays
"""
import sys
import os
import time
import argparse
import threading
import yaml
import numpy as np
import cv2

# Import project modules
from simpler_env_adapter import SimplerEnvAdapter
from chronicler import Chronicler
from spatial_tracker import SpatialTracker
from painter import Painter
from sync_layer import SyncLayer
from trajectory_recorder import TrajectoryRecorder
from teleop_controller import TeleopController, ScriptedOracleExpert
from vla_wrapper import MockVLAWrapper, OctoWrapper, OpenVLAWrapper, Pi0Wrapper, RTXWrapper


class SimplerVLAPlatform:
    def __init__(
        self,
        config_path="configs/config.yaml",
        task_name=None,
        control_mode="policy",
        action_chunk_size=4,
        record=True,
        seed=42,
        max_steps=60
    ):
        self.config = self.load_config(config_path)
        sim_cfg = self.config.get("simpler_env", {})

        self.task_name = task_name or sim_cfg.get("task_name", "google_robot_pick_coke_can")
        self.control_mode = control_mode  # "policy", "expert", "teleop"
        self.action_chunk_size = int(action_chunk_size or self.config.get("action_chunk_size", 4))
        self.record_enabled = record
        self.seed = int(seed)
        self.max_steps = int(max_steps or sim_cfg.get("max_steps", 60))

        # 1. Initialize SimplerEnv Adapter
        print(f"[Platform] Initializing SimplerEnv for task '{self.task_name}'...")
        self.env = SimplerEnvAdapter(
            task_name=self.task_name,
            max_steps=self.max_steps,
            seed=self.seed
        )

        # 2. Initialize Middleware components
        self.sync_layer = SyncLayer()
        self.painter = Painter()
        self.tracker = SpatialTracker(tracker_type=self.config.get("tracker_type", "opencv_fallback"))

        chronicler_cfg = self.config.get("chronicler", {})
        self.chronicler = Chronicler(
            use_real_vlm=chronicler_cfg.get("use_real_vlm", False),
            model_id=chronicler_cfg.get("vlm_model", "HuggingFaceTB/SmolVLM-Instruct"),
            device=chronicler_cfg.get("device", "auto"),
            max_history_length=chronicler_cfg.get("max_history_length", 5)
        )
        self.chronicler_tick_rate = float(chronicler_cfg.get("tick_rate_hz", 7.0))

        # 3. Initialize VLA Policy
        vla_choice = self.config.get("vla_model", "octo")
        model_paths = self.config.get("model_paths", {})
        if vla_choice == "octo":
            self.vla = OctoWrapper(
                model_id=model_paths.get("octo_small", "hf://rail-berkeley/octo-small-1.5"),
                action_chunk_size=self.action_chunk_size
            )
        elif vla_choice == "openvla":
            self.vla = OpenVLAWrapper(
                model_id=model_paths.get("openvla", "openvla/openvla-7b"),
                action_chunk_size=self.action_chunk_size
            )
        else:
            self.vla = MockVLAWrapper(action_chunk_size=self.action_chunk_size)

        # 4. Controllers & Recorders
        self.teleop = TeleopController()
        self.expert = ScriptedOracleExpert(self.task_name)
        save_dir = sim_cfg.get("trajectory_save_dir", "data/trajectories")
        self.recorder = TrajectoryRecorder(save_dir=save_dir, task_name=self.task_name)

        # State flags
        self.running = False
        self.threads = []
        self.current_obs = None
        self.last_action = [0.0] * 7
        self.last_info = {}
        self.task_status = "RUNNING"  # "RUNNING", "SUCCESS", "FAILED"
        self.status_banner_timer = 0

    def load_config(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {"vla_model": "octo", "action_chunk_size": 4}

    def reset_episode(self):
        """Resets the task, memory modules, and trajectory recorder."""
        self.current_obs, reset_info = self.env.reset(seed=self.seed)
        self.chronicler.reset()
        self.vla.clear_chunk_buffer()
        self.expert.reset()
        self.task_status = "RUNNING"
        self.status_banner_timer = 0

        # Initialize tracking points on target
        raw_rgb = self.current_obs["image"]
        target_pos_3d = self.current_obs["proprioception"]["target_pos"]
        # Approximate 2D target projection
        h, w, _ = raw_rgb.shape
        approx_2d = [w * 0.5, h * 0.6]
        self.tracker.initialize_tracking_points(raw_rgb, approx_2d)

        if self.record_enabled:
            self.recorder.start_trajectory(
                task_name=self.task_name,
                seed=self.seed,
                camera_calibration=self.current_obs.get("camera_calibration"),
                robot_type=self.env.robot_type,
                source=f"{self.control_mode}_{type(self.vla).__name__}",
                config_dict=self.config,
                action_chunk_size=self.action_chunk_size
            )
        print(f"[Platform] Episode reset for task '{self.task_name}'. Ready.")

    def tracker_loop(self):
        """High-speed tracking thread (15-20 Hz)."""
        rate = 1.0 / 15.0
        while self.running:
            start_t = time.time()
            if self.current_obs is not None:
                raw_rgb = self.current_obs["image"]
                pts, flags = self.tracker.track(raw_rgb)
                _, latest_prompt = self.sync_layer.get_latest_state()
                painted = self.painter.draw_overlay(raw_rgb, pts, flags, latest_prompt or "")
                self.sync_layer.update_image(painted)
            elapsed = time.time() - start_t
            time.sleep(max(0.001, rate - elapsed))

    def chronicler_loop(self):
        """Temporal logic & state chronicler thread (5-10 Hz)."""
        rate = 1.0 / max(1.0, self.chronicler_tick_rate)
        while self.running:
            start_t = time.time()
            if self.current_obs is not None:
                raw_rgb = self.current_obs["image"]
                proprio = self.current_obs["proprioception"]
                tracking_info = {
                    "target_pos": proprio.get("target_pos", []),
                    "is_occluded": self.last_info.get("lift_height", 0.0) > 0.02
                }
                dynamic_prompt = self.chronicler.update_state(
                    raw_rgb, self.env.instruction, proprio, tracking_info
                )
                self.sync_layer.update_prompt(dynamic_prompt)
            elapsed = time.time() - start_t
            time.sleep(max(0.001, rate - elapsed))

    def draw_ui_dashboard(self, base_image, step_idx):
        """
        Draws the comprehensive SimplerEnv evaluation dashboard, including
        highly visible TASK SUCCESS / TASK FAILED alert banners.
        """
        dash = base_image.copy()
        h, w, _ = dash.shape

        # 1. Top Header Banner: Task & Model Info
        header_bg = np.zeros((45, w, 3), dtype=np.uint8)
        dash[:45, :] = cv2.addWeighted(dash[:45, :], 0.3, header_bg, 0.7, 0)
        cv2.putText(dash, f"SIMPLER-ENV: {self.task_name.upper()}", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(dash, f"Instruction: \"{self.env.instruction}\"", (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)

        # Mode Badge
        mode_text = f"MODE: {self.control_mode.upper()} (Chunk: {self.action_chunk_size})"
        cv2.putText(dash, mode_text, (w - 240, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(dash, f"Step: {step_idx}/{self.max_steps}", (w - 120, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Bottom Telemetry Bar
        bar_y = h - 35
        tele_bg = np.zeros((35, w, 3), dtype=np.uint8)
        dash[bar_y:, :] = cv2.addWeighted(dash[bar_y:, :], 0.3, tele_bg, 0.7, 0)
        proprio = self.current_obs.get("proprioception", {})
        tcp = proprio.get("tcp_pos", [0, 0, 0])
        grp = proprio.get("gripper_width", 1.0)
        tele_str = f"TCP: [{tcp[0]:.2f}, {tcp[1]:.2f}, {tcp[2]:.2f}] | Grp: {'OPEN' if grp > 0.5 else 'CLOSED'} ({grp:.2f})"
        cv2.putText(dash, tele_str, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

        # 3. PROMINENT SUCCESS / FAILED NOTIFICATION BANNER
        if self.task_status == "SUCCESS":
            # Bright Emerald Green Glowing Banner
            overlay = dash.copy()
            banner_y1 = int(h * 0.38)
            banner_y2 = int(h * 0.62)
            cv2.rectangle(overlay, (20, banner_y1), (w - 20, banner_y2), (0, 180, 40), -1)
            cv2.rectangle(overlay, (20, banner_y1), (w - 20, banner_y2), (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.85, dash, 0.15, 0, dash)

            cv2.putText(dash, "TASK SUCCESSFUL!", (int(w * 0.15), int(h * 0.49)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dash, f"Goal Condition Satisfied at Step {step_idx}!", (int(w * 0.12), int(h * 0.57)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 255, 240), 1, cv2.LINE_AA)

        elif self.task_status == "FAILED":
            # Bold Crimson Red Banner
            overlay = dash.copy()
            banner_y1 = int(h * 0.38)
            banner_y2 = int(h * 0.62)
            cv2.rectangle(overlay, (20, banner_y1), (w - 20, banner_y2), (0, 0, 200), -1)
            cv2.rectangle(overlay, (20, banner_y1), (w - 20, banner_y2), (255, 255, 255), 2)
            cv2.addWeighted(overlay, 0.85, dash, 0.15, 0, dash)

            cv2.putText(dash, "TASK FAILED", (int(w * 0.25), int(h * 0.49)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(dash, "Max steps reached without meeting success criteria", (int(w * 0.05), int(h * 0.57)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 255), 1, cv2.LINE_AA)

        return dash

    def run_main_loop(self):
        """Primary SimplerEnv evaluation loop running at 10 Hz."""
        print(f"[Platform] Starting main execution loop (Mode: {self.control_mode})...")
        self.reset_episode()
        step_idx = 0
        rate = 1.0 / 10.0

        cv2.namedWindow("SimplerEnv VLA Platform", cv2.WINDOW_AUTOSIZE)

        while self.running:
            loop_start = time.time()

            # 1. Fetch visual & dynamic prompt states
            painted_img, dynamic_prompt = self.sync_layer.get_latest_state()
            active_img = painted_img if painted_img is not None else self.current_obs["image"]
            proprio = self.current_obs["proprioception"]

            # 2. Obtain action based on control mode
            if self.task_status == "RUNNING":
                step_idx += 1
                if self.control_mode == "expert":
                    action = self.expert.get_action(self.current_obs)
                elif self.control_mode == "teleop":
                    # Keypress handled in cv2.waitKey
                    action = self.last_action
                else:
                    # VLA Policy with Action Chunking
                    action = self.vla.step_policy(active_img, dynamic_prompt or "", proprio)

                self.last_action = action

                # 3. Step SimplerEnv simulation forward
                next_obs, reward, term, trunc, info = self.env.step(action)
                self.last_info = info

                # 4. Trajectory recording
                if self.record_enabled:
                    self.recorder.record_step(
                        step_idx, self.current_obs, action, reward, term, trunc, info,
                        dynamic_prompt=dynamic_prompt
                    )

                self.current_obs = next_obs

                # 5. Check Success Condition
                if info.get("success", False):
                    self.task_status = "SUCCESS"
                    print(f"\n🏆 [SimplerEnv] TASK SUCCESS! Met goal condition at step {step_idx}!")
                    if self.record_enabled:
                        self.recorder.save()
                elif trunc:
                    self.task_status = "FAILED"
                    print(f"\n❌ [SimplerEnv] TASK FAILED: Reached max steps ({self.max_steps}).")
                    if self.record_enabled:
                        self.recorder.save()

            # 6. Render Dashboard
            dash_frame = self.draw_ui_dashboard(active_img, step_idx)
            cv2.imshow("SimplerEnv VLA Platform", dash_frame)

            # 7. Keyboard events
            key = cv2.waitKey(20) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                self.running = False
                break
            elif key in [ord('r'), ord('R')]:
                step_idx = 0
                self.reset_episode()
            elif self.control_mode == "teleop" and key != 255:
                self.last_action = self.teleop.process_key(key)

            elapsed = time.time() - loop_start
            time.sleep(max(0.001, rate - elapsed))

        cv2.destroyAllWindows()
        self.env.close()

    def start(self):
        self.running = True

        # Thread 1: Persistent Point Tracker
        t1 = threading.Thread(target=self.tracker_loop, daemon=True)
        t1.start()
        self.threads.append(t1)

        # Thread 2: Chronicler State VLM
        t2 = threading.Thread(target=self.chronicler_loop, daemon=True)
        t2.start()
        self.threads.append(t2)

        # Run main loop on primary thread
        self.run_main_loop()

        for t in self.threads:
            t.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SimplerEnv VLA Evaluation Platform")
    parser.add_argument("--task", type=str, default=None, help="SimplerEnv task name")
    parser.add_argument("--mode", type=str, choices=["policy", "expert", "teleop"], default="policy",
                        help="Control source: policy (VLA), expert (oracle script), teleop (keyboard)")
    parser.add_argument("--chunk", type=int, default=4, help="Action chunking horizon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--no-record", action="store_true", help="Disable trajectory recording")
    parser.add_argument("--web", action="store_true", help="Start Web Visualization Server")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    args = parser.parse_args()

    if args.web:
        from web_server import RoboticWebServer
        server = RoboticWebServer(host="0.0.0.0", port=args.port)
        server.start()
    else:
        platform = SimplerVLAPlatform(
            task_name=args.task,
            control_mode=args.mode,
            action_chunk_size=args.chunk,
            record=not args.no_record,
            seed=args.seed,
            max_steps=args.max_steps
        )
        platform.start()
