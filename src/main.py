#!/usr/bin/env python3
"""
Main Entrypoint — Robotic Control & Memory Architecture
------------------------------------------------------
Orchestrates Chronicler, Spatial Tracker, Painter, Sync Layer, and VLA Model inference.
Runs an interactive OpenCV simulation demonstrating full multi-modal closed loop.
"""
import sys
import os
import time
import threading
import yaml
import numpy as np
import cv2

# Import custom components
from chronicler import Chronicler
from spatial_tracker import SpatialTracker
from painter import Painter
from sync_layer import SyncLayer
from vla_wrapper import MockVLAWrapper, OctoWrapper, OpenVLAWrapper, Pi0Wrapper, RTXWrapper


class RoboticVLADemo:
    def __init__(self, config_path="configs/config.yaml"):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize modules
        self.sync_layer = SyncLayer()
        self.painter = Painter()
        
        # Select VLA wrapper
        vla_choice = self.config.get("vla_model", "mock")
        model_paths = self.config.get("model_paths", {})
        
        if vla_choice == "octo":
            checkpoint = self.config.get("octo_checkpoint", "octo-small")
            path = model_paths.get("octo_small" if "small" in checkpoint else "octo_base", "hf://rail-berkeley/octo-small-1.5")
            self.vla = OctoWrapper(model_id=path, checkpoint_type=checkpoint)
        elif vla_choice == "openvla":
            self.vla = OpenVLAWrapper(model_paths.get("openvla", "openvla/openvla-7b"))
        elif vla_choice == "pi0":
            self.vla = Pi0Wrapper(model_paths.get("pi0", "physical-intelligence/pi0"))
        elif vla_choice == "rtx":
            self.vla = RTXWrapper(model_paths.get("rtx", "google/rt-x"))
        else:
            self.vla = MockVLAWrapper()
            
        # Select Tracker
        self.tracker = SpatialTracker(
            tracker_type=self.config.get("tracker_type", "opencv_fallback"),
            config=self.config
        )
        
        # Initialize Chronicler with config
        chronicler_cfg = self.config.get("chronicler", {})
        self.chronicler = Chronicler(
            use_real_vlm=chronicler_cfg.get("use_real_vlm", False),
            model_id=chronicler_cfg.get("vlm_model", "HuggingFaceTB/SmolVLM-Instruct"),
            device=chronicler_cfg.get("device", "auto"),
            max_history_length=chronicler_cfg.get("max_history_length", 5)
        )
        self.chronicler_tick_rate = float(chronicler_cfg.get("tick_rate_hz", 7.0))
        
        # Simulation States (Physical simulation)
        self.width, self.height = 640, 480
        self.gripper_pos = [100.0, 240.0]
        self.gripper_width = 1.0  # 1.0 = Fully Open, 0.0 = Closed
        self.target_pos = [480.0, 240.0]
        self.human_command = "Grasp the orange object"
        
        # Obstacle defining an occlusion zone
        self.obstacle_box = [260, 120, 380, 360]  # [x_min, y_min, x_max, y_max]
        
        # Control flags
        self.running = False
        self.threads = []

    def load_config(self, path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "vla_model": "mock",
            "tracker_type": "opencv_fallback",
            "chronicler": {"enabled": True, "use_real_vlm": False, "tick_rate_hz": 7.0},
            "simulation": {"enabled": True, "fps": 15, "arm_speed": 0.1, "occlusion_radius": 40}
        }

    def generate_raw_frame(self):
        """
        Simulates the raw camera feed. Draws the target object and the obstacle.
        """
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 45  # Dark gray background
        
        # Draw Workspace Grid Lines
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (60, 60, 60), 1)
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (60, 60, 60), 1)
            
        # 1. Draw Target Object (Orange Circle)
        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        cv2.circle(frame, (tx, ty), 15, (0, 140, 255), -1)
        cv2.circle(frame, (tx, ty), 15, (0, 200, 255), 2)
        
        # 2. Draw Occlusion Obstacle (Darker Grey semi-opaque panel)
        ox1, oy1, ox2, oy2 = self.obstacle_box
        sub_img = frame[oy1:oy2, ox1:ox2]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 80
        res = cv2.addWeighted(sub_img, 0.4, white_rect, 0.6, 1.0)
        frame[oy1:oy2, ox1:ox2] = res
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (100, 100, 100), 2)
        cv2.putText(
            frame, "OCCLUSION WALL", (ox1 + 10, oy1 + 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA
        )
        
        # 3. Draw Robot Arm Gripper
        gx, gy = int(self.gripper_pos[0]), int(self.gripper_pos[1])
        # Draw Wrist Connection
        cv2.line(frame, (0, gy), (gx - 20, gy), (180, 180, 180), 6)
        cv2.circle(frame, (gx - 20, gy), 10, (100, 100, 100), -1)
        # Draw Fingers (Open/Close spacing based on self.gripper_width)
        offset = int(self.gripper_width * 15) + 5
        cv2.line(frame, (gx - 20, gy - offset), (gx + 5, gy - offset), (220, 220, 220), 4)
        cv2.line(frame, (gx - 20, gy + offset), (gx + 5, gy + offset), (220, 220, 220), 4)
        cv2.line(frame, (gx + 5, gy - offset), (gx + 5, gy - 2), (200, 200, 200), 4)
        cv2.line(frame, (gx + 5, gy + offset), (gx + 5, gy + 2), (200, 200, 200), 4)
        
        # Draw tool center point
        cv2.circle(frame, (gx, gy), 3, (0, 0, 255), -1)
        
        return frame

    def tracker_loop(self):
        """
        Tracker thread runs at 15-20 Hz.
        Tracks coordinates and paints visual indicators.
        """
        print("[Tracker Thread] Started.")
        rate = 1.0 / 15.0
        
        # Initialize points around target
        raw_frame = self.generate_raw_frame()
        self.tracker.initialize_tracking_points(raw_frame, self.target_pos)
        
        while self.running:
            start_time = time.time()
            
            # Fetch latest physical state
            raw_frame = self.generate_raw_frame()
            
            # Estimate occlusion geometrically or via vision
            pts, flags = self.tracker.track(raw_frame, gripper_pos=self.gripper_pos)
            
            # Check if centroid is behind the obstacle box
            tx, ty = self.target_pos
            ox1, oy1, ox2, oy2 = self.obstacle_box
            is_behind_wall = (ox1 <= tx <= ox2) and (oy1 <= ty <= oy2)
            
            if is_behind_wall:
                flags = [True] * len(flags)
                
            # Render visual overlays with latest dynamic prompt
            _, latest_prompt = self.sync_layer.get_latest_state()
            prompt_str = latest_prompt or ""
            
            painted_image = self.painter.draw_overlay(raw_frame, pts, flags, prompt_str)
            
            # Update synchronization layer
            self.sync_layer.update_image(painted_image)
            
            elapsed = time.time() - start_time
            time.sleep(max(0.001, rate - elapsed))

    def chronicler_loop(self):
        """
        Chronicler thread runs at 5-10 Hz.
        Updates logical states and temporal memory.
        """
        print("[Chronicler Thread] Started.")
        rate = 1.0 / max(1.0, self.chronicler_tick_rate)
        
        while self.running:
            start_time = time.time()
            
            # Fetch tracking details for state analysis
            tx, ty = self.target_pos
            ox1, oy1, ox2, oy2 = self.obstacle_box
            is_behind_wall = (ox1 <= tx <= ox2) and (oy1 <= ty <= oy2)
            
            tracking_info = {
                "target_pos": self.target_pos,
                "is_occluded": is_behind_wall or (np.linalg.norm(np.array(self.target_pos) - np.array(self.gripper_pos)) < 40)
            }
            
            proprioception = {
                "gripper_pos": self.gripper_pos,
                "gripper_width": self.gripper_width
            }
            
            # Run state analysis
            raw_frame = self.generate_raw_frame()
            dynamic_prompt = self.chronicler.update_state(
                raw_frame, self.human_command, proprioception, tracking_info
            )
            
            # Push prompt to Sync Layer
            self.sync_layer.update_prompt(dynamic_prompt)
            
            elapsed = time.time() - start_time
            time.sleep(max(0.001, rate - elapsed))

    def main_control_loop(self):
        """
        Simulated VLA control loop running at 10 Hz.
        Retrieves inputs from Sync Layer, queries VLA, updates physics, and shows dashboard.
        """
        print("[VLA Control Loop] Started.")
        rate = 1.0 / 10.0
        
        cv2.namedWindow("Robotic VLA Multi-Modal Memory Demo", cv2.WINDOW_AUTOSIZE)
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.target_pos = [float(x), float(y)]
                raw_f = self.generate_raw_frame()
                self.tracker.initialize_tracking_points(raw_f, self.target_pos)
                self.chronicler.reset()
                print(f"[Demo] Repositioned target to ({x}, {y})")
                
        cv2.setMouseCallback("Robotic VLA Multi-Modal Memory Demo", on_mouse)

        while self.running:
            start_time = time.time()
            
            # 1. Fetch latest synced states
            painted_image, dynamic_prompt = self.sync_layer.get_latest_state()
            
            if painted_image is not None and dynamic_prompt is not None:
                # 2. Query VLA Model with both Painted Image and Dynamic Prompt
                proprioception = {
                    "gripper_pos": self.gripper_pos,
                    "gripper_width": self.gripper_width
                }
                action = self.vla.predict_action(painted_image, dynamic_prompt, proprioception)
                
                # 3. Apply Action to Physics simulation
                dx, dy, _, _, _, _, gripper_cmd = action
                self.gripper_pos[0] += dx
                self.gripper_pos[1] += dy
                
                # Smoothly close/open gripper
                target_w = gripper_cmd
                self.gripper_width += (target_w - self.gripper_width) * 0.3
                
                # If target is grasped, move it with the gripper
                if self.gripper_width < 0.15:
                    dist_to_obj = np.linalg.norm(np.array(self.target_pos) - np.array(self.gripper_pos))
                    if dist_to_obj < 22.0:
                        self.target_pos[0] = self.gripper_pos[0]
                        self.target_pos[1] = self.gripper_pos[1]
                
                # Render the dashboard
                dash_image = painted_image.copy()
                h, w, _ = dash_image.shape
                
                # UI Info Box at bottom right
                cv2.rectangle(dash_image, (w - 240, h - 90), (w - 10, h - 10), (0, 0, 0), -1)
                cv2.rectangle(dash_image, (w - 240, h - 90), (w - 10, h - 10), (0, 255, 0), 1)
                cv2.putText(dash_image, "DEMO DASHBOARD", (w - 230, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(dash_image, f"VLA Model: {type(self.vla).__name__}", (w - 230, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dash_image, "[Click to move target]", (w - 230, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.putText(dash_image, "[Press 'R' to reset]", (w - 230, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
                
                cv2.imshow("Robotic VLA Multi-Modal Memory Demo", dash_image)
            
            # Poll keyboard events
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # Esc
                self.running = False
                break
            elif key == ord('r'):
                self.gripper_pos = [100.0, 240.0]
                self.gripper_width = 1.0
                self.target_pos = [480.0, 240.0]
                self.chronicler.reset()
                raw_f = self.generate_raw_frame()
                self.tracker.initialize_tracking_points(raw_f, self.target_pos)
                print("[Demo] Reset workspace.")
                
            elapsed = time.time() - start_time
            time.sleep(max(0.001, rate - elapsed))
            
        cv2.destroyAllWindows()

    def start(self):
        self.running = True
        
        # Thread 1: Tracker
        t1 = threading.Thread(target=self.tracker_loop)
        t1.daemon = True
        t1.start()
        self.threads.append(t1)
        
        # Thread 2: Chronicler
        t2 = threading.Thread(target=self.chronicler_loop)
        t2.daemon = True
        t2.start()
        self.threads.append(t2)
        
        # Run main control loop on primary thread for UI / GUI window
        self.main_control_loop()
        
        # Wait for threads
        for t in self.threads:
            t.join()


if __name__ == "__main__":
    if "--web" in sys.argv or "-w" in sys.argv:
        from web_server import RoboticWebServer
        port = 8080
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        server = RoboticWebServer(host="0.0.0.0", port=port)
        server.start()
    else:
        demo = RoboticVLADemo()
        demo.start()

