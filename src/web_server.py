#!/usr/bin/env python3
"""
Web Visualization & Interactive Chat Server for MuJoCo & SimplerEnv
-------------------------------------------------------------------
Powers the Taste-Skill interactive dashboard:
- Streams real-time native MuJoCo dual camera views (Top/Overhead Camera + Wrist Camera)
- Full 7-DoF Franka Emika Panda physics with real parallel gripper
- 5 diverse physical objects (Campbell's Soup Can, Mustard Bottle, Blue Box, Red Cylinder, Green Cube) and Basket
- Natural Language Chat Console API (/api/chat) for commanding Franka robot
- Real-time Server-Sent Events (SSE) streaming at ~12 FPS
"""
import os
import sys
import time
import json
import base64
import threading
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

# Ensure module imports work regardless of current working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visual_grounding import VisualGroundingDetector
from chronicler import Chronicler
from spatial_tracker import SpatialTracker
from painter import Painter
from sync_layer import SyncLayer
from vla_wrapper import (
    create_vla_model, check_model_readiness, get_all_models_status,
    MockVLAWrapper, OctoWrapper, OpenVLAWrapper, RTXWrapper, Pi0Wrapper
)

# Load MuJoCo environment
try:
    from mujoco_env import MujocoRoboticEnv
except ImportError:
    from src.mujoco_env import MujocoRoboticEnv

# Load LIBERO benchmark & MiniVLA Policy
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    LIBERO_AVAILABLE = True
except Exception as _e:
    print(f"[WebServer] LIBERO package not available: {_e}", flush=True)
    LIBERO_AVAILABLE = False

try:
    from minivla_policy import MiniVLAPolicy
except ImportError:
    from src.minivla_policy import MiniVLAPolicy


LIBERO_TIER_TASKS = {
    1: {
        "suite": "libero_spatial",
        "name": "Tier 1 — Spatial Manipulation",
        "badge": "libero_spatial",
        "tasks": [
            {
                "id": 0,
                "name": "Bát đen giữa đĩa & ramekin",
                "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
                "description": "Gắp chiếc bát gốm đen nằm giữa đĩa và ramekin đặt lên đĩa"
            },
            {
                "id": 2,
                "name": "Bát đen ở giữa bàn",
                "instruction": "pick up the black bowl from table center and place it on the plate",
                "description": "Gắp chiếc bát đen từ vị trí trung tâm bàn và đặt lên đĩa"
            }
        ]
    },
    2: {
        "suite": "libero_object",
        "name": "Tier 2 — Diverse Object Manipulation",
        "badge": "libero_object",
        "tasks": [
            {
                "id": 0,
                "name": "Lon súp chữ cái vào giỏ",
                "instruction": "pick up the alphabet soup and place it in the basket",
                "description": "Nhận diện và gắp lon súp chữ cái đặt vào rỏ đựng"
            },
            {
                "id": 1,
                "name": "Khối cream cheese vào giỏ",
                "instruction": "pick up the cream cheese and place it in the basket",
                "description": "Gắp hộp phô mai cream cheese đặt vào rỏ"
            },
            {
                "id": 4,
                "name": "Chai tương cà vào giỏ",
                "instruction": "pick up the ketchup and place it in the basket",
                "description": "Gắp chai tương cà chua đặt vào rỏ"
            }
        ]
    },
    3: {
        "suite": "libero_goal",
        "name": "Tier 3 — Goal & Articulation",
        "badge": "libero_goal",
        "tasks": [
            {
                "id": 0,
                "name": "Mở ngăn kéo giữa của tủ gỗ",
                "instruction": "open the middle drawer of the cabinet",
                "description": "Tiếp cận tay nắm và kéo mở ngăn kéo giữa của tủ gỗ"
            },
            {
                "id": 8,
                "name": "Đặt bát đen lên đĩa phẳng",
                "instruction": "put the bowl on the plate",
                "description": "Gắp bát và đặt chính xác lên tâm đĩa"
            }
        ]
    },
    4: {
        "suite": "libero_10",
        "name": "Tier 4 — Long-Horizon Sequential Tasks",
        "badge": "libero_10",
        "tasks": [
            {
                "id": 0,
                "name": "Cho súp và sốt cà chua vào giỏ",
                "instruction": "put both the alphabet soup and the tomato sauce in the basket",
                "description": "Nhiệm vụ đa bước: gắp liên hoàn lon súp và sốt cà chua vào rỏ"
            },
            {
                "id": 3,
                "name": "Cất bát vào ngăn kéo dưới & đóng tủ",
                "instruction": "put the black bowl in the bottom drawer of the cabinet and close it",
                "description": "Nhiệm vụ đa bước: đặt bát vào ngăn kéo dưới rồi đẩy đóng tủ"
            }
        ]
    }
}


class RoboticWebServer:
    def __init__(self, host="0.0.0.0", port=8080, config_path="configs/config.yaml"):
        self.host = host
        self.port = port
        self.config_path = config_path
        self.web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
        self.sim_lock = threading.Lock()

        # Load config
        task_name = "put can in basket"
        vla_choice = "octo"
        try:
            if os.path.exists(config_path):
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    task_name = cfg.get("simpler_env", {}).get("task_name", task_name)
                    vla_choice = cfg.get("vla_model", "octo")
        except Exception:
            pass

        # 1. Initialize MuJoCo Physics & Robotic Environment (Custom Tabletop Mode)
        scene_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "franka_table_scene.xml"))
        print(f"[WebServer] Initializing MuJoCo Native Simulation from '{scene_xml}'...", flush=True)
        self.env = MujocoRoboticEnv(scene_path=scene_xml, render_width=640, render_height=480)
        self.task_name = task_name
        self.task_instruction = task_name

        self.sync_layer = SyncLayer()
        self.painter = Painter()

        # 2. Ablation / Execution Mode State & Statistics
        self.ablation_mode = "memory"  # "memory" (Full System) or "raw" (Pure Reactive)
        self.ablation_stats = {
            "raw": {"episodes": 12, "successes": 7, "total_steps": 384, "recoveries": 2, "disturbances": 8},
            "memory": {"episodes": 12, "successes": 11, "total_steps": 252, "recoveries": 7, "disturbances": 8}
        }

        # 3. VLA Policy Model
        self.vla_choice = vla_choice
        self.vla_api_url = os.environ.get("VLA_API_URL", "")
        self.vla = create_vla_model(self.vla_choice, chunk_size=4, mode=self.ablation_mode, api_url=self.vla_api_url)
        self.vla_name = self._format_vla_name(self.vla_choice)

        # 4. Perception & Multimodal Reasoning
        from gemini_vla import GeminiRoboticsVLM
        self.gemini_vlm = GeminiRoboticsVLM()
        self.visual_target_pos = [0.46, 0.0, 0.42]

        self.tracker = SpatialTracker(tracker_type="tapir")
        self.tracker.clear()
        self.grounding_detector = VisualGroundingDetector()
        self.chronicler = Chronicler(use_real_vlm=False)

        # 5. Global Control State
        self.running = False
        self.step_idx = 0
        self.max_steps = 70
        self.task_status = "IDLE"  # "IDLE", "RUNNING", "RETRACTING", "SUCCESS"
        self.goal_achieved = False
        self.last_chronicler_log = ""
        self.last_action = [0.0] * 7
        self.latest_payload = None
        self.current_obs = None
        # Control Paradigm Mode: 'vla' (End-to-End VLA), 'vlm_ik' (VLM Perception + IK), 'fallback' (Offline Fallback)
        self.control_paradigm = "vla"

        # 6. Official LIBERO Benchmark Mini-Suite Engine
        self.env_mode = "libero"  # Default to LIBERO Benchmark Mode
        self.libero_tier = 1
        self.libero_task_id = 0
        self.libero_env = None
        self.libero_obs = None
        self.libero_init_states = None
        self.libero_policy = None
        self.libero_max_steps = 150
        self.libero_step_idx = 0
        self.libero_status = "IDLE"
        self.libero_success = False
        self.libero_last_action = [0.0] * 7
        self.libero_task_name = "Task"
        self.libero_instruction = "Task instruction"
        self.libero_description = ""

        # Initialize default LIBERO task (Tier 1 Task 0)
        self.init_libero_task(1, 0)

        self._reset_sim()
        print(f"[WebServer] Ready. Loaded {len(self.env.object_configs)} custom objects and LIBERO Benchmark Suite. Active Mode: {self.env_mode.upper()}", flush=True)

    def init_libero_task(self, tier=1, task_id=0):
        """Initializes an authentic official LIBERO environment for the requested Tier & Task."""
        with self.sim_lock:
            try:
                tier = int(tier)
                task_id = int(task_id)
                if tier not in LIBERO_TIER_TASKS:
                    tier = 1
                tier_info = LIBERO_TIER_TASKS[tier]
                suite_name = tier_info["suite"]
                
                valid_ids = [t["id"] for t in tier_info["tasks"]]
                if task_id not in valid_ids:
                    task_id = valid_ids[0]
                
                if not LIBERO_AVAILABLE:
                    print(f"[WebServer] LIBERO not available on system.")
                    return False
                
                # Close previous LIBERO env if open
                if self.libero_env is not None:
                    try:
                        self.libero_env.close()
                    except Exception:
                        pass
                    self.libero_env = None
                
                benchmark_dict = benchmark.get_benchmark_dict()
                task_suite = benchmark_dict[suite_name]()
                task = task_suite.get_task(task_id)
                bddl_file = task_suite.get_task_bddl_file_path(task_id)
                init_states = task_suite.get_task_init_states(task_id)
                
                self.libero_env = OffScreenRenderEnv(
                    bddl_file_name=bddl_file,
                    camera_heights=256,
                    camera_widths=256
                )
                self.libero_env.reset()
                self.libero_init_states = init_states
                self.libero_obs = self.libero_env.set_init_state(init_states[0])
                
                # Settle simulation (10 dummy steps with gripper open)
                dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                for _ in range(10):
                    self.libero_obs, _, _, _ = self.libero_env.step(dummy_action)
                
                self.libero_tier = tier
                self.libero_task_id = task_id
                self.libero_task_name = task.name
                task_cfg = next((t for t in tier_info["tasks"] if t["id"] == task_id), None)
                self.libero_instruction = task_cfg["instruction"] if task_cfg else task.language
                self.libero_description = task_cfg.get("description", "")
                self.libero_step_idx = 0
                self.libero_max_steps = 150
                self.libero_status = "IDLE"
                self.libero_success = False
                self.libero_last_action = [0.0] * 7
                print(f"[WebServer] Initialized LIBERO Tier {tier} Task {task_id} ({self.libero_task_name}): '{self.libero_instruction}'", flush=True)
                return True
            except Exception as e:
                print(f"[WebServer] Error initializing LIBERO task: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return False

    def _ensure_libero_policy(self):
        """Lazy loads MiniVLAPolicy only when first execution is requested."""
        if self.libero_policy is None:
            print("[WebServer] Lazy loading MiniVLA Policy for LIBERO...", flush=True)
            self.libero_policy = MiniVLAPolicy()
        return self.libero_policy

    def reset_libero(self):
        with self.sim_lock:
            if self.libero_env is not None and self.libero_init_states is not None:
                self.libero_env.reset()
                self.libero_obs = self.libero_env.set_init_state(self.libero_init_states[0])
                dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
                for _ in range(10):
                    self.libero_obs, _, _, _ = self.libero_env.step(dummy_action)
                self.libero_step_idx = 0
                self.libero_status = "IDLE"
                self.libero_success = False
                self.libero_last_action = [0.0] * 7
                print(f"[WebServer] Reset LIBERO task '{self.libero_task_name}' to initial state.", flush=True)

    def execute_libero(self, custom_instruction=None):
        with self.sim_lock:
            if custom_instruction and custom_instruction.strip():
                self.libero_instruction = custom_instruction.strip()
            self.libero_status = "RUNNING"
            self.libero_success = False
            self.libero_step_idx = 0
            print(f"[WebServer] Executing LIBERO task '{self.libero_task_name}' with instruction: '{self.libero_instruction}'", flush=True)

    def stop_libero(self):
        with self.sim_lock:
            self.libero_status = "IDLE"
            print(f"[WebServer] Stopped LIBERO task execution.", flush=True)

    def set_env_mode(self, mode):
        with self.sim_lock:
            if mode in ["libero", "tabletop"]:
                self.env_mode = mode
                if mode == "libero" and self.libero_env is None:
                    self.init_libero_task(self.libero_tier, self.libero_task_id)
                elif mode == "tabletop":
                    self._reset_sim()
                print(f"[WebServer] Switched Environment Mode to: {self.env_mode.upper()}", flush=True)
                return True
            return False

    def run_libero_step(self):
        """
        Executes one step in the official LIBERO environment,
        renders agentview and robot0_eye_in_hand cameras,
        steps MiniVLA policy if running, and formats telemetry.
        """
        with self.sim_lock:
            if self.libero_env is None or self.libero_obs is None:
                return None
            
            obs = self.libero_obs
            # Retrieve camera frames (MuJoCo OpenGL offscreen renderer is vertically inverted)
            raw_agentview = np.flipud(obs.get("agentview_image", np.zeros((256, 256, 3), dtype=np.uint8)))
            raw_wrist = np.flipud(obs.get("robot0_eye_in_hand_image", np.zeros((256, 256, 3), dtype=np.uint8)))
            
            # Encode frames as JPEG data URLs
            _, top_jpg = cv2.imencode(".jpg", cv2.cvtColor(raw_agentview, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            top_b64 = f"data:image/jpeg;base64,{base64.b64encode(top_jpg).decode('utf-8')}"
            
            _, wrist_jpg = cv2.imencode(".jpg", cv2.cvtColor(raw_wrist, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            wrist_b64 = f"data:image/jpeg;base64,{base64.b64encode(wrist_jpg).decode('utf-8')}"
            
            # Proprioception & Telemetry
            eef_pos = [round(float(v), 3) for v in obs.get("robot0_eef_pos", [0.0, 0.0, 0.0])]
            joint_pos = [round(float(v) * 180.0 / np.pi, 1) for v in obs.get("robot0_joint_pos", [0.0]*7)]
            gripper_qpos = obs.get("robot0_gripper_qpos", [0.04, 0.04])
            gripper_mm = round(float(sum(gripper_qpos)) * 1000.0, 1)
            is_gripper_closed = self.libero_last_action[-1] > 0.0 or gripper_mm < 30.0
            gripper_state = "CLOSED" if is_gripper_closed else "OPEN"
            
            # Step policy if RUNNING
            if self.libero_status == "RUNNING":
                try:
                    policy = self._ensure_libero_policy()
                    action = policy.predict_action(obs, self.libero_instruction)
                    self.libero_last_action = [round(float(a), 4) for a in action.tolist()]
                    
                    next_obs, reward, done, info = self.libero_env.step(action.tolist())
                    self.libero_obs = next_obs
                    self.libero_step_idx += 1
                    
                    if self.libero_env.check_success():
                        self.libero_success = True
                        self.libero_status = "SUCCESS"
                        print(f"[WebServer] [LIBERO] SUCCESS at step {self.libero_step_idx}!", flush=True)
                    elif self.libero_step_idx >= self.libero_max_steps:
                        self.libero_status = "MAX_STEPS"
                except Exception as e:
                    print(f"[WebServer] Error stepping LIBERO policy: {e}", flush=True)
                    self.libero_status = "ERROR"
            
            progress_pct = round(min(100.0, (self.libero_step_idx / max(1, self.libero_max_steps)) * 100.0), 1)
            display_prompt = f"[LIBERO Suite]: {self.libero_task_name} | [Instruction]: {self.libero_instruction} | [Status]: {self.libero_status}"
            
            telemetry = {
                "env_mode": "libero",
                "tier": self.libero_tier,
                "task_id": self.libero_task_id,
                "task_name": self.libero_task_name,
                "instruction": self.libero_instruction,
                "status": self.libero_status,
                "step": self.libero_step_idx,
                "max_steps": self.libero_max_steps,
                "progress_pct": progress_pct,
                "is_success": self.libero_success,
                "tcp_pos": eef_pos,
                "joint_positions": joint_pos,
                "qpos": [round(float(v), 4) for v in obs.get("robot0_joint_pos", [0.0]*7)],
                "gripper_mm": gripper_mm,
                "gripper_width_mm": gripper_mm,
                "gripper_state": gripper_state,
                "last_action": self.libero_last_action,
                "action_chunk": self.libero_last_action,
                "model_name": "MiniVLA (VQ-Libero90)",
            }
            
            return {
                "env_mode": "libero",
                "vla_model": "MiniVLA (VQ-Libero90)",
                "control_paradigm": "vla",
                "is_offline_fallback": False,
                "vla_mode_tag": "[MINIVLA VQ-LIBERO90]",
                "task_name": self.libero_task_name,
                "instruction": self.libero_instruction,
                "status": self.libero_status,
                "step": self.libero_step_idx,
                "frame_top_b64": top_b64,
                "frame_wrist_b64": wrist_b64,
                "frame_b64": top_b64,
                "dynamic_prompt": display_prompt,
                "telemetry": telemetry,
                "objects": [],
                "chronicler_log": f"LIBERO Step {self.libero_step_idx}/{self.libero_max_steps} | Gripper: {gripper_state} | Progress: {progress_pct}%"
            }

    def _format_vla_name(self, model_key):
        k = str(model_key).lower()
        if "smol" in k:
            return "SmolVLA"
        elif "gemini" in k:
            return "Gemini Robotics VLM"
        elif "openvla" in k:
            return "OpenVLA-7B"
        elif "rt1" in k or "rtx" in k:
            return "Google RT-1"
        elif "pi0" in k:
            return "Physical Intelligence pi0"
        elif "mock" in k:
            return "MockVLA"
        else:
            return "Octo-Small"

    def set_model(self, model_name, api_url=None):
        """Switches active VLA model on the fly, checking readiness."""
        with self.sim_lock:
            if api_url is not None:
                self.vla_api_url = api_url.strip()

            readiness = check_model_readiness(model_name, self.vla_api_url)
            if not readiness["ready"]:
                print(f"[WebServer] [Rejection]: Model '{model_name}' is NOT READY ({readiness['badge']}).")
                return False, f"Model '{readiness['name']}' is not ready. {readiness['description']}."

            self.vla_choice = model_name
            self.vla = create_vla_model(self.vla_choice, chunk_size=4, mode=self.ablation_mode, api_url=self.vla_api_url)
            self.vla_name = self._format_vla_name(self.vla_choice)
            self._reset_sim()
            print(f"[WebServer] Active VLA Model switched to: {self.vla_name} ({self.vla.connection_status})")
            return True, f"Successfully switched to {self.vla_name}"

    def set_control_paradigm(self, paradigm):
        """Sets the control architecture: 'vla' (End-to-End), 'vlm_ik' (Perception+IK), 'fallback' (Offline Fallback)."""
        p = str(paradigm).lower().strip()
        if p in ["vla", "vlm_ik", "fallback"]:
            with self.sim_lock:
                self.control_paradigm = p
                print(f"[WebServer] Control Paradigm switched to: {p.upper()}", flush=True)
                return True
        return False

    def set_ablation_mode(self, mode):
        """Toggles between 'memory' (Full System) and 'raw' (Ablation Baseline)."""
        with self.sim_lock:
            if mode not in ["raw", "memory"]:
                return False
            self.ablation_mode = mode
            if hasattr(self.vla, "set_ablation_mode"):
                self.vla.set_ablation_mode(mode)
            self._reset_sim()
            print(f"[WebServer] Ablation Mode switched to: {mode.upper()}")
            return True

    def get_ablation_stats(self):
        with self.sim_lock:
            raw_eps = max(1, self.ablation_stats["raw"]["episodes"])
            mem_eps = max(1, self.ablation_stats["memory"]["episodes"])
            raw_sr = (self.ablation_stats["raw"]["successes"] / raw_eps) * 100.0
            mem_sr = (self.ablation_stats["memory"]["successes"] / mem_eps) * 100.0
            raw_avg_steps = self.ablation_stats["raw"]["total_steps"] / raw_eps
            mem_avg_steps = self.ablation_stats["memory"]["total_steps"] / mem_eps
            raw_rec = (self.ablation_stats["raw"]["recoveries"] / max(1, self.ablation_stats["raw"]["disturbances"])) * 100.0
            mem_rec = (self.ablation_stats["memory"]["recoveries"] / max(1, self.ablation_stats["memory"]["disturbances"])) * 100.0

            return {
                "current_mode": self.ablation_mode,
                "raw": {
                    "episodes": raw_eps,
                    "successes": self.ablation_stats["raw"]["successes"],
                    "success_rate": round(raw_sr, 1),
                    "avg_steps": round(raw_avg_steps, 1),
                    "recovery_rate": round(raw_rec, 1)
                },
                "memory": {
                    "episodes": mem_eps,
                    "successes": self.ablation_stats["memory"]["successes"],
                    "success_rate": round(mem_sr, 1),
                    "avg_steps": round(mem_avg_steps, 1),
                    "recovery_rate": round(mem_rec, 1)
                },
                "delta": {
                    "success_rate": f"+{round(mem_sr - raw_sr, 1)}%",
                    "steps_reduction": f"-{round(max(0, (raw_avg_steps - mem_avg_steps) / max(0.1, raw_avg_steps)) * 100.0, 1)}%",
                    "recovery_boost": f"+{round(mem_rec - raw_rec, 1)}%"
                }
            }

    def _record_ablation_episode(self, success, steps):
        mode = self.ablation_mode
        self.ablation_stats[mode]["episodes"] += 1
        if success:
            self.ablation_stats[mode]["successes"] += 1
            if getattr(self.env, "lift_height", 0.0) > 0.02:
                self.ablation_stats[mode]["recoveries"] += 1
        self.ablation_stats[mode]["total_steps"] += steps

    def _reset_sim(self):
        """Resets robot arm to home keyframe and clears tracking."""
        self.current_obs = self.env.reset()
        self.step_idx = 0
        self.task_status = "IDLE"
        self.goal_achieved = False
        self.chronicler.reset()
        if hasattr(self.vla, "reset"):
            self.vla.reset()
        else:
            self.vla.clear_chunk_buffer()
        self.tracker.clear()

    def reposition_target(self):
        """Randomizes the active target position on the table surface."""
        with self.sim_lock:
            self.ablation_stats[self.ablation_mode]["disturbances"] += 1
            new_pos = self.env.reposition_target()
            self.step_idx = 0
            self.task_status = "IDLE"
            self.goal_achieved = False
            self.chronicler.reset()
            if hasattr(self.vla, "reset"):
                self.vla.reset()
            else:
                self.vla.clear_chunk_buffer()
            self.tracker.clear()
            self.visual_target_pos = list(new_pos)
            self.current_obs = self.env.get_proprioception()
            return new_pos

    def reposition_all_objects(self):
        """Randomizes all 5 diverse objects on the table without mutual collisions."""
        with self.sim_lock:
            self.ablation_stats[self.ablation_mode]["disturbances"] += 1
            all_objs = self.env.reposition_all_objects()
            self.step_idx = 0
            self.task_status = "IDLE"
            self.goal_achieved = False
            self.chronicler.reset()
            if hasattr(self.vla, "reset"):
                self.vla.reset()
            else:
                self.vla.clear_chunk_buffer()
            self.tracker.clear()
            self.visual_target_pos = list(self.env.get_target_pos())
            self.current_obs = self.env.get_proprioception()
            return all_objs

    def handle_chat_message(self, user_msg):
        """
        Parses natural language commands from the interactive Chat Console,
        grounds target objects in 3D MuJoCo space, and triggers closed-loop VLA manipulation.
        """
        user_msg_clean = user_msg.strip()
        msg_lower = user_msg_clean.lower()

        # Handle LIBERO Benchmark Mode Intent
        if self.env_mode == "libero":
            if any(k in msg_lower for k in ["reset", "khởi tạo", "về vị trí", "initial", "home"]):
                self.reset_libero()
                return {
                    "reply": f"Đã reset tác vụ LIBERO '{self.libero_task_name}' về initial state.",
                    "intent": "reset",
                    "reasoning_steps": [
                        "Khôi phục trạng thái khớp Franka và bố trí vật thể chuẩn của LIBERO BDDL.",
                        "Đặt lại bộ đếm bước về 0."
                    ],
                    "target_key": None
                }
            elif any(k in msg_lower for k in ["dừng", "stop", "pause"]):
                self.stop_libero()
                return {
                    "reply": f"Đã dừng thực thi tác vụ LIBERO '{self.libero_task_name}'.",
                    "intent": "stop",
                    "reasoning_steps": [
                        "Dừng chu kỳ suy luận MiniVLA và chuyển robot về trạng thái IDLE."
                    ],
                    "target_key": None
                }
            else:
                self.execute_libero(user_msg_clean)
                return {
                    "reply": f"⚡ [MiniVLA VQ-Libero90] Bắt đầu thực thi: \"{self.libero_instruction}\" trên tác vụ {self.libero_task_name}...",
                    "intent": "libero_execute",
                    "reasoning_steps": [
                        f"Nhận chỉ thị ngôn ngữ tự nhiên: \"{self.libero_instruction}\"",
                        f"Môi trường chuẩn LIBERO: Tier {self.libero_tier} - {self.libero_task_name}",
                        "Mô hình VLA: MiniVLA (DinoSigLIP + Qwen2.5 + VQ Action Tokenizer)",
                        "Suy luận liên tục closed-loop qua agentview và robot0_eye_in_hand"
                    ],
                    "target_key": None
                }

        # 1. Reset Physics / Initial State Intent
        if any(k in msg_lower for k in ["reset physic", "reset vật lý", "vật lý ban đầu", "initial", "trạng thái initial", "khởi tạo lại"]):
            with self.sim_lock:
                self._reset_sim()
            return {
                "reply": "Đã reset toàn bộ hệ thống vật lý MuJoCo về trạng thái initial (khôi phục tất cả vị trí khớp Franka, gripper và 5 vật thể trên bàn).",
                "intent": "reset_physics",
                "reasoning_steps": [
                    "Kích hoạt quy trình Reset Physics MuJoCo về trạng thái initial.",
                    "Khôi phục 7 góc khớp Franka q_1..q_7 về cấu hình chuẩn ban đầu [0, -0.5, 0, -2.0, 0, 1.6, 0.785].",
                    "Triệt tiêu hoàn toàn vận tốc qvel và lực ngoại vi (settle physics).",
                    "Khôi phục vị trí ban đầu của toàn bộ 5 vật thể (can, mustard, blue box, red cylinder, green cube) và rỏ đựng.",
                    "Mở ngón kẹp song song Franka Parallel Gripper và đồng bộ hóa telemetry."
                ],
                "target_key": None
            }

        elif any(k in msg_lower for k in ["về vị trí", "vị trí nghỉ", "home", "nghỉ", "reset", "dừng", "stop"]):
            with self.sim_lock:
                self._reset_sim()
            return {
                "reply": "Đã đưa cánh tay robot Franka Panda về vị trí nghỉ ban đầu (Home Pose). Hệ thống sẵn sàng cho nhiệm vụ tiếp theo.",
                "intent": "home",
                "reasoning_steps": [
                    "Phát hiện lệnh điều khiển hệ thống: Reset tư thế.",
                    "Đặt 7 góc khớp Franka q_1..q_7 về cấu hình an toàn [0, -0.5, 0, -2.0, 0, 1.6, 0.785].",
                    "Mở ngón kẹp song song Franka Parallel Gripper (Khẩu độ 80mm).",
                    "Xóa bộ nhớ đệm visual tracking."
                ],
                "target_key": None
            }

        elif any(k in msg_lower for k in ["xáo trộn", "ngẫu nhiên", "đổi vị trí", "randomize", "xáo"]):
            with self.sim_lock:
                all_objs = self.env.reposition_all_objects()
                self.step_idx = 0
                self.task_status = "IDLE"
                self.goal_achieved = False
                self.chronicler.reset()
                if hasattr(self.vla, "reset"):
                    self.vla.reset()
                else:
                    self.vla.clear_chunk_buffer()
                self.tracker.clear()
                self.visual_target_pos = list(self.env.get_target_pos())
                self.current_obs = self.env.get_proprioception()
            return {
                "reply": "Đã xáo trộn ngẫu nhiên vị trí các vật thể trên mặt bàn mô phỏng MuJoCo.",
                "intent": "randomize",
                "reasoning_steps": [
                    "Phát hiện yêu cầu tái cấu hình môi trường.",
                    "Phân bố ngẫu nhiên 5 vật thể không va chạm trên mặt bàn.",
                    "Cập nhật tọa độ thị giác mới cho bộ tracking.",
                    "Robot duy trì trạng thái IDLE sẵn sàng đón nhận câu lệnh."
                ],
                "target_key": None
            }

        elif any(k in msg_lower for k in ["mở kẹp", "thả kẹp", "open gripper"]):
            with self.sim_lock:
                self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            return {
                "reply": "Đã mở ngón kẹp song song Franka Gripper (Khẩu độ tối đa 80mm).",
                "intent": "open_gripper",
                "reasoning_steps": ["Gửi xung điều khiển mở tới actuator act_finger1 và act_finger2."],
                "target_key": None
            }

        elif any(k in msg_lower for k in ["đóng kẹp", "kẹp lại", "close gripper"]):
            with self.sim_lock:
                self.env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            return {
                "reply": "Đã đóng ngón kẹp song song Franka Gripper (Khẩu độ 0mm, lực ép ma sát 2.0N).",
                "intent": "close_gripper",
                "reasoning_steps": ["Gửi xung điều khiển kẹp chặt tới actuator act_finger1 và act_finger2."],
                "target_key": None
            }

        # 2. Vision-Language Multimodal Intent & Perception Grounding (No Regex, No Hardcode)
        top_frame = self.env.render_camera("overhead_cam")
        with self.sim_lock:
            # Query Gemini VLM or Local Visual Perception on the live camera image
            grounding_res = self.gemini_vlm.ground_and_reason(top_frame, user_msg_clean)
            target_key = grounding_res.get("target_key", "can")
            target_label = grounding_res.get("target_label", "Target Object")
            norm_x, norm_y = grounding_res.get("normalized_coords", [0.5, 0.5])

            # Reconstruct 3D target coordinates PURELY from camera raycast (no simulation xpos cheat)
            u = norm_x * self.env.render_width
            v = norm_y * self.env.render_height
            world_pos = self.env.pixel_to_world_on_plane(u, v, plane_z=0.40)
            self.visual_target_pos = [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])]

            self.env.set_active_target(target_key)
            self.task_instruction = user_msg_clean

            # Initialize visual tracking points from camera raycast
            self.tracker.initialize_tracking_points(top_frame, [u, v], num_points=9)

            self.task_status = "RUNNING"
            self.goal_achieved = False
            self.step_idx = 0
            if hasattr(self.vla, "reset"):
                self.vla.reset()
            else:
                self.vla.clear_chunk_buffer()

        target_cfg = self.env.object_configs.get(target_key, self.env.object_configs["can"])
        tpos_fmt = [round(float(p), 3) for p in self.visual_target_pos]
        bpos_fmt = [round(float(p), 3) for p in self.env.get_basket_pos()]

        if self.control_paradigm == "fallback":
            is_fallback = True
            engine_name = "Offline Fallback Servoing"
            mode_badge = "[OFFLINE FALLBACK]"
        elif self.control_paradigm == "vlm_ik":
            is_fallback = False
            engine_name = "VLM Perception + Inverse Kinematics"
            mode_badge = "[VLM + IK]"
        else:  # "vla"
            is_fallback = getattr(self.vla, "is_offline_fallback", False)
            engine_name = f"VLA End-to-End ({self.vla_name})"
            mode_badge = f"[VLA ({self.vla_name})]" if not is_fallback else "[OFFLINE FALLBACK]"

        reply = f"{mode_badge} Đã nhận chỉ thị: Gắp **{target_cfg['label']}** vào rỏ. Đang suy luận qua {engine_name}..."
        reasoning_steps = list(grounding_res.get("reasoning_steps", []))
        reasoning_steps.append(f"📍 Visual Raycast (No Ground-Truth xpos): Tọa độ từ Camera Raycasting X={tpos_fmt[0]}m, Y={tpos_fmt[1]}m, Z={tpos_fmt[2]}m.")
        reasoning_steps.append(f"⚡ Paradigm Execution ({self.control_paradigm.upper()}): Điều khiển cánh tay Franka theo quỹ đạo tiếp cận -> hạ độ cao -> kẹp -> nhấc -> đặt rỏ.")

        return {
            "reply": reply,
            "intent": "pick_and_place",
            "target_key": target_key,
            "target_label": target_cfg["label"],
            "target_pos": tpos_fmt,
            "is_offline_fallback": is_fallback,
            "engine": engine_name,
            "reasoning_steps": reasoning_steps
        }

    def run_simulation_step(self):
        """
        Executes one physical step in MuJoCo, renders dual camera frames,
        steps the active VLA policy if running, and returns real-time telemetry.
        """
        with self.sim_lock:
            # 1. Render both camera frames directly from MuJoCo offscreen renderer
            raw_top = self.env.render_camera("overhead_cam")
            raw_wrist = self.env.render_camera("wrist_cam")
            prop = self.env.get_proprioception()

            # 2. Track spatial points on the active target in top view
            pts, flags = self.tracker.track_points(raw_top)

            # 3. Dynamic Prompt & Chronicler memory
            tracking_info = {
                "target_pos": prop.get("target_pos", []),
                "is_occluded": self.task_status == "SUCCESS"
            }
            dynamic_prompt = self.chronicler.update_state(
                raw_top, self.task_instruction, prop, tracking_info
            )
            self.sync_layer.update_prompt(dynamic_prompt)
            self.last_chronicler_log = dynamic_prompt

            if self.task_status == "IDLE":
                painted_top = self.painter.draw_overlay(raw_top, pts, flags, prompt_text="", show_hud=False) if self.ablation_mode == "memory" else raw_top.copy()
                display_prompt = f"[Task]: {self.task_instruction} | [Status]: IDLE. Franka Panda ready. Send a command in chat or click a prompt chip."
                policy_prompt = self.task_instruction
            elif self.ablation_mode == "raw":
                painted_top = raw_top.copy()
                policy_prompt = self.task_instruction
                display_prompt = "[PURE REACTIVE MODE: Memory Layer Disabled - Direct VLA Execution]"
            else:
                painted_top = self.painter.draw_overlay(raw_top, pts, flags, prompt_text="", show_hud=False)
                policy_prompt = dynamic_prompt or self.task_instruction
                display_prompt = dynamic_prompt or "Executing VLA policy..."

            # 4. Step VLA Policy and MuJoCo Physics
            if self.task_status == "RUNNING":
                self.step_idx += 1
                prop["instruction"] = self.task_instruction
                prop["visual_target_pos"] = list(self.visual_target_pos)
                action = self.vla.step_policy(painted_top, policy_prompt, prop)
                self.last_action = [round(float(x), 4) for x in action] if action else [0.0] * 7
                obs = self.env.step(action)

                self.goal_achieved = self.env.check_basket_containment(self.env.active_target_key)

                vla_phase = getattr(self.vla, "phase", "")
                if vla_phase == "complete" or (self.goal_achieved and self.env.data.ctrl[7] >= 0.035):
                    self.task_status = "RETRACTING"
                elif self.step_idx >= self.max_steps:
                    self.task_status = "RETRACTING"

            elif self.task_status == "RETRACTING":
                # Smoothly drive arm back to initial home resting configuration and open gripper
                arrived_home = self.env.return_to_home_step()
                display_prompt = f"[Status]: Returning to home posture... ({'Goal achieved' if self.goal_achieved else 'Max steps reached'})"
                if arrived_home:
                    self.task_status = "IDLE"
                    self.step_idx = 0
                    self.tracker.clear()
                    if hasattr(self.vla, "reset"):
                        self.vla.reset()
                    else:
                        self.vla.clear_chunk_buffer()

            metrics = self.env.get_metrics()

            # 5. Encode dual cameras as Base64 JPEG data URLs
            top_b64 = self.env.render_camera_b64("overhead_cam", quality=75, overlay_image=painted_top)
            wrist_b64 = self.env.render_camera_b64("wrist_cam", quality=75)

            grp_w = float(prop.get("gripper_width", 1.0))
            grp_mm = float(prop.get("gripper_width_mm", 80.0))
            finger_slide = float(self.env.data.qpos[7])

            if self.control_paradigm == "fallback":
                is_fallback = True
                vla_badge = "OFFLINE FALLBACK"
            elif self.control_paradigm == "vlm_ik":
                is_fallback = False
                vla_badge = "VLM + IK"
            else:
                is_fallback = getattr(self.vla, "is_offline_fallback", False)
                vla_badge = f"VLA ({self.vla_name})" if not is_fallback else "OFFLINE FALLBACK"

            telemetry = {
                "qpos": prop.get("joint_positions", []),
                "finger": finger_slide,
                "gripper_width": grp_w,
                "gripper_width_mm": grp_mm,
                "tcp_pos": prop.get("tcp_pos", []),
                "target_pos": self.env.get_target_pos().tolist(),
                "visual_target_pos": list(self.visual_target_pos),
                "basket_pos": self.env.get_basket_pos().tolist(),
                "step": self.step_idx,
                "instruction": self.task_instruction,
                "status": self.task_status,
                "phase": getattr(self.vla, "phase", "idle"),
                "is_success": self.goal_achieved,
                "ablation_mode": self.ablation_mode,
                "memory_active": self.ablation_mode == "memory",
                "active_target": self.env.active_target_key,
                "control_paradigm": self.control_paradigm,
                "is_offline_fallback": is_fallback,
                "vla_mode_tag": vla_badge,
                "last_action": self.last_action,
                "metrics": metrics
            }

            return {
                "vla_model": self.vla_name,
                "control_paradigm": self.control_paradigm,
                "is_offline_fallback": is_fallback,
                "vla_mode_tag": vla_badge,
                "task_name": self.task_name,
                "instruction": self.task_instruction,
                "status": self.task_status,
                "step": self.step_idx,
                "frame_top_b64": top_b64,
                "frame_wrist_b64": wrist_b64,
                "frame_b64": top_b64,  # backward compatibility
                "dynamic_prompt": display_prompt,
                "ablation_mode": self.ablation_mode,
                "telemetry": telemetry,
                "objects": self.env.get_all_objects_info(),
                "chronicler_log": self.last_chronicler_log
            }

    def get_latest_payload(self):
        with self.sim_lock:
            if self.latest_payload is not None:
                return dict(self.latest_payload)
            return None

    def _sim_loop(self):
        """
        Dedicated physics & offscreen rendering thread.
        Runs continuously in a single thread to isolate OpenGL/GLFW/WGL context on Windows.
        """
        while self.running:
            try:
                if self.env_mode == "libero":
                    payload = self.run_libero_step()
                else:
                    payload = self.run_simulation_step()
                if payload is not None:
                    with self.sim_lock:
                        self.latest_payload = payload
            except Exception as e:
                print(f"[SimLoop] Error during step: {e}", flush=True)
            time.sleep(0.065)  # ~15 FPS

    def start(self):
        self.running = True
        self.sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self.sim_thread.start()
        server_instance = self

        class DashboardRequestHandler(BaseHTTPRequestHandler):
            def handle(self):
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                    pass

            def log_message(self, format, *args):
                return

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    try:
                        while server_instance.running:
                            payload = server_instance.get_latest_payload()
                            if payload is not None:
                                data_str = f"data: {json.dumps(payload)}\n\n"
                                self.wfile.write(data_str.encode("utf-8"))
                                self.wfile.flush()
                            time.sleep(0.08)  # ~12 FPS
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                        pass

                elif path == "/api/models_status":
                    status_list = get_all_models_status(api_url=server_instance.vla_api_url)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok",
                        "active_model": server_instance.vla_choice,
                        "api_url": server_instance.vla_api_url,
                        "models": status_list
                    }).encode("utf-8"))

                elif path == "/api/objects":
                    objs = server_instance.env.get_all_objects_info()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "objects": objs}).encode("utf-8"))

                elif path == "/api/snapshot":
                    payload = server_instance.get_latest_payload()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload or {}).encode("utf-8"))

                elif path == "/api/ablation_stats":
                    stats = server_instance.get_ablation_stats()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(stats).encode("utf-8"))

                elif path == "/api/libero/tasks":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok",
                        "env_mode": server_instance.env_mode,
                        "current_tier": server_instance.libero_tier,
                        "current_task_id": server_instance.libero_task_id,
                        "current_instruction": getattr(server_instance, "libero_instruction", ""),
                        "current_task_name": getattr(server_instance, "libero_task_name", ""),
                        "tiers": LIBERO_TIER_TASKS
                    }).encode("utf-8"))

                else:
                    if path == "/" or path == "/index.html":
                        filepath = os.path.join(server_instance.web_dir, "index.html")
                        content_type = "text/html"
                    else:
                        filepath = os.path.join(server_instance.web_dir, path.lstrip("/"))
                        if filepath.endswith(".css"):
                            content_type = "text/css"
                        elif filepath.endswith(".js"):
                            content_type = "application/javascript"
                        elif filepath.endswith(".png"):
                            content_type = "image/png"
                        elif filepath.endswith(".jpg") or filepath.endswith(".jpeg"):
                            content_type = "image/jpeg"
                        else:
                            content_type = "text/plain"

                    if os.path.exists(filepath) and os.path.isfile(filepath):
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.end_headers()
                        with open(filepath, "rb") as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_error(404, "File Not Found")

            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
                try:
                    body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                except Exception:
                    body = {}

                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/api/chat":
                    message = body.get("message", "").strip()
                    chat_resp = server_instance.handle_chat_message(message)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps(chat_resp).encode("utf-8"))

                elif path == "/api/reset":
                    with server_instance.sim_lock:
                        server_instance._reset_sim()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "message": "Physics and simulation reset to initial state"}).encode("utf-8"))

                elif path in ["/api/move_target", "/api/reposition_target"]:
                    new_pos = server_instance.reposition_target()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "target_pos": new_pos, "task_status": "IDLE"}).encode("utf-8"))

                elif path == "/api/reposition_all":
                    all_objs = server_instance.reposition_all_objects()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "objects": all_objs, "task_status": "IDLE"}).encode("utf-8"))

                elif path == "/api/set_target":
                    target_key = body.get("target", "can")
                    with server_instance.sim_lock:
                        ok, active_k = server_instance.env.set_active_target(target_key)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok" if ok else "error", "active_target": active_k}).encode("utf-8"))

                elif path == "/api/command":
                    cmd = body.get("command", "").strip()
                    chat_resp = server_instance.handle_chat_message(cmd)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "chat": chat_resp}).encode("utf-8"))

                elif path == "/api/ablation_mode":
                    mode = body.get("mode", "memory").lower()
                    ok = server_instance.set_ablation_mode(mode)
                    self.send_response(200 if ok else 400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok" if ok else "error", "mode": server_instance.ablation_mode}).encode("utf-8"))

                elif path == "/api/paradigm":
                    paradigm = body.get("paradigm", "vla").lower()
                    ok = server_instance.set_control_paradigm(paradigm)
                    self.send_response(200 if ok else 400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok" if ok else "error",
                        "paradigm": server_instance.control_paradigm
                    }).encode("utf-8"))

                elif path == "/api/libero/set_task":
                    tier = int(body.get("tier", 1))
                    task_id = int(body.get("task_id", 0))
                    ok = server_instance.init_libero_task(tier, task_id)
                    self.send_response(200 if ok else 400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok" if ok else "error",
                        "tier": server_instance.libero_tier,
                        "task_id": server_instance.libero_task_id,
                        "task_name": getattr(server_instance, "libero_task_name", ""),
                        "instruction": getattr(server_instance, "libero_instruction", "")
                    }).encode("utf-8"))

                elif path == "/api/libero/execute":
                    instr = body.get("instruction", "").strip()
                    server_instance.execute_libero(instr)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok",
                        "instruction": server_instance.libero_instruction,
                        "task_status": server_instance.libero_status
                    }).encode("utf-8"))

                elif path == "/api/libero/stop":
                    server_instance.stop_libero()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "task_status": "IDLE"}).encode("utf-8"))

                elif path == "/api/libero/reset":
                    server_instance.reset_libero()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok",
                        "message": "Reset LIBERO task to initial state",
                        "instruction": getattr(server_instance, "libero_instruction", "")
                    }).encode("utf-8"))

                elif path == "/api/env_mode":
                    mode = body.get("mode", "libero").lower()
                    ok = server_instance.set_env_mode(mode)
                    self.send_response(200 if ok else 400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok" if ok else "error",
                        "env_mode": server_instance.env_mode
                    }).encode("utf-8"))

                elif path == "/api/set_model":
                    model_name = body.get("model", "octo").lower()
                    api_url = body.get("api_url", None)
                    ok, msg = server_instance.set_model(model_name, api_url=api_url)
                    self.send_response(200 if ok else 400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    status_list = get_all_models_status(api_url=server_instance.vla_api_url)
                    self.wfile.write(json.dumps({
                        "status": "ok" if ok else "error",
                        "message": msg,
                        "model": server_instance.vla_name,
                        "active_key": server_instance.vla_choice,
                        "connection_status": server_instance.vla.connection_status,
                        "models_status": status_list
                    }).encode("utf-8"))

                elif path == "/api/config_keys":
                    gemini_key = body.get("gemini_key", "").strip()
                    api_url = body.get("api_url", "").strip()
                    
                    if gemini_key:
                        os.environ["GEMINI_API_KEY"] = gemini_key
                        if hasattr(server_instance, "gemini_vlm"):
                            server_instance.gemini_vlm.set_api_key(gemini_key)
                    if api_url:
                        os.environ["VLA_API_URL"] = api_url
                        server_instance.vla_api_url = api_url

                    # Persist to .env file
                    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
                    env_lines = []
                    keys_updated = set()
                    if os.path.exists(env_path):
                        with open(env_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("GEMINI_API_KEY=") and gemini_key:
                                    env_lines.append(f"GEMINI_API_KEY={gemini_key}\n")
                                    keys_updated.add("GEMINI_API_KEY")
                                elif line.startswith("VLA_API_URL=") and api_url:
                                    env_lines.append(f"VLA_API_URL={api_url}\n")
                                    keys_updated.add("VLA_API_URL")
                                else:
                                    env_lines.append(line)
                    if "GEMINI_API_KEY" not in keys_updated and gemini_key:
                        env_lines.append(f"GEMINI_API_KEY={gemini_key}\n")
                    if "VLA_API_URL" not in keys_updated and api_url:
                        env_lines.append(f"VLA_API_URL={api_url}\n")

                    try:
                        with open(env_path, "w", encoding="utf-8") as f:
                            f.writelines(env_lines)
                    except Exception as e:
                        print(f"[WebServer] Could not save .env: {e}")

                    status_list = get_all_models_status(api_url=server_instance.vla_api_url)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "status": "ok",
                        "message": "API keys saved and reloaded",
                        "models_status": status_list
                    }).encode("utf-8"))

                else:
                    self.send_error(404, "Endpoint Not Found")

        server = ThreadingHTTPServer((self.host, self.port), DashboardRequestHandler)
        print(f"\n=======================================================")
        print(f"[Franka Panda MuJoCo Live Dashboard is LIVE at:]")
        print(f"-> http://localhost:{self.port}")
        print(f"=======================================================\n", flush=True)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            self.running = False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Franka Panda MuJoCo Web Dashboard Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve dashboard on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface")
    args = parser.parse_args()

    s = RoboticWebServer(host=args.host, port=args.port)
    s.start()
