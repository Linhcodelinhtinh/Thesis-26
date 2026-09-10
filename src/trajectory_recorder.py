#!/usr/bin/env python3
"""
Trajectory Recorder — Multi-Modal Dataset Logging for SimplerEnv & VLA Policies
--------------------------------------------------------------------------------
Captures high-fidelity trajectories containing:
- Observation (RGB, Depth, Camera Intrinsics/Extrinsics, Proprioception, Language Instruction)
- Action (7-DoF Delta Cartesian pose or joint action, with Action Chunking support)
- Reward, Success condition, Subgoals, and Termination flags
- Metadata: Precise timestamps, camera calibration, environment seed, version & config hash
"""
import os
import time
import json
import hashlib
import datetime
import numpy as np
import cv2


class TrajectoryRecorder:
    def __init__(self, save_dir="data/trajectories", task_name="google_robot_pick_coke_can"):
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.task_name = task_name
        self.is_recording = False
        self.metadata = {}
        self.steps = []
        self.frames = []
        self.start_time = 0.0

    def start_trajectory(
        self,
        task_name=None,
        seed=42,
        task_version="1.0.0",
        simpler_env_version="0.1.0",
        camera_calibration=None,
        robot_type="google_robot",
        source="vla_policy",
        config_dict=None,
        action_chunk_size=1
    ):
        """
        Initializes a new recording episode with comprehensive metadata.
        """
        self.task_name = task_name or self.task_name
        self.is_recording = True
        self.steps = []
        self.frames = []
        self.start_time = time.time()
        start_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Compute deterministic hash of config
        cfg_str = json.dumps(config_dict or {}, sort_keys=True)
        cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:12]

        self.metadata = {
            "task_name": self.task_name,
            "task_version": task_version,
            "simpler_env_version": simpler_env_version,
            "seed": int(seed),
            "timestamp_start_iso": start_iso,
            "timestamp_start_epoch": self.start_time,
            "config_hash": cfg_hash,
            "robot_type": robot_type,
            "source": source,
            "action_chunk_size": int(action_chunk_size),
            "control_frequency_hz": 10.0,
            "camera_calibration": camera_calibration or {}
        }
        print(f"[TrajectoryRecorder] Started recording trajectory for '{self.task_name}' (Seed: {seed}, Source: {source})")

    def record_step(
        self,
        step_idx,
        obs,
        action,
        reward,
        terminated,
        truncated,
        info,
        dynamic_prompt=None,
        action_chunk=None
    ):
        """
        Records a single transition step in the trajectory.
        """
        if not self.is_recording:
            return

        current_time = time.time()
        elapsed_ms = (current_time - self.start_time) * 1000.0

        # Extract RGB frame for video rollout
        raw_rgb = obs.get("image", None)
        if raw_rgb is not None and isinstance(raw_rgb, np.ndarray):
            self.frames.append(raw_rgb.copy())

        # Extract proprioception
        proprio = obs.get("proprioception", {})
        tcp_pos = proprio.get("tcp_pos", [])
        tcp_euler = proprio.get("tcp_rot_euler", [])
        gripper_w = proprio.get("gripper_width", 1.0)
        joint_pos = proprio.get("joint_positions", [])
        joint_vel = proprio.get("joint_velocities", [])

        action_list = action.tolist() if isinstance(action, np.ndarray) else list(action)
        chunk_list = None
        if action_chunk is not None:
            chunk_list = action_chunk.tolist() if isinstance(action_chunk, np.ndarray) else [
                a.tolist() if isinstance(a, np.ndarray) else list(a) for a in action_chunk
            ]

        step_record = {
            "step_index": int(step_idx),
            "timestamp_elapsed_ms": round(elapsed_ms, 2),
            "timestamp_epoch": current_time,
            "observation": {
                "instruction": obs.get("instruction", ""),
                "dynamic_prompt": dynamic_prompt or "",
                "tcp_pos": [round(float(v), 5) for v in tcp_pos],
                "tcp_rot_euler": [round(float(v), 5) for v in tcp_euler],
                "gripper_width": round(float(gripper_w), 4),
                "joint_positions": [round(float(v), 5) for v in joint_pos],
                "joint_velocities": [round(float(v), 5) for v in joint_vel],
                "target_pos": proprio.get("target_pos", [])
            },
            "action": [round(float(v), 5) for v in action_list],
            "action_chunk": chunk_list,
            "reward": round(float(reward), 4),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(info.get("success", False)),
            "info": {
                k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                for k, v in info.items() if k != "subgoals"
            },
            "subgoals": info.get("subgoals", {})
        }
        self.steps.append(step_record)

    def save(self, filename=None, save_video=True, save_npz=True):
        """
        Saves recorded trajectory data to disk (JSON + NPZ + MP4 Video).
        """
        if not self.steps:
            print("[TrajectoryRecorder] No steps recorded to save.")
            return None

        self.is_recording = False
        duration_sec = time.time() - self.start_time
        final_success = any(s.get("success", False) for s in self.steps)

        # File naming convention: task_seed_timestamp_status
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        status_tag = "SUCCESS" if final_success else "FAIL"
        base_name = filename or f"{self.task_name}_s{self.metadata.get('seed', 0)}_{timestamp_str}_{status_tag}"

        json_path = os.path.join(self.save_dir, f"{base_name}.json")
        video_path = os.path.join(self.save_dir, f"{base_name}.mp4")
        npz_path = os.path.join(self.save_dir, f"{base_name}.npz")

        # Update metadata summary
        self.metadata["total_steps"] = len(self.steps)
        self.metadata["final_success"] = final_success
        self.metadata["duration_seconds"] = round(duration_sec, 2)
        self.metadata["saved_files"] = {
            "json": os.path.basename(json_path),
            "npz": os.path.basename(npz_path) if save_npz else None,
            "video": os.path.basename(video_path) if save_video else None
        }

        # 1. Save Full JSON Log
        full_payload = {
            "metadata": self.metadata,
            "steps": self.steps
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_payload, f, indent=2)

        # 2. Save NumPy Compressed Arrays (.npz) for high-performance ML ingestion
        if save_npz:
            actions = np.array([s["action"] for s in self.steps], dtype=np.float32)
            rewards = np.array([s["reward"] for s in self.steps], dtype=np.float32)
            successes = np.array([s["success"] for s in self.steps], dtype=bool)
            tcp_poses = np.array([s["observation"]["tcp_pos"] for s in self.steps], dtype=np.float32)
            gripper_widths = np.array([s["observation"]["gripper_width"] for s in self.steps], dtype=np.float32)

            npz_dict = {
                "actions": actions,
                "rewards": rewards,
                "successes": successes,
                "tcp_poses": tcp_poses,
                "gripper_widths": gripper_widths,
                "metadata_json": json.dumps(self.metadata)
            }
            if self.frames:
                npz_dict["images"] = np.stack(self.frames, axis=0)

            np.savez_compressed(npz_path, **npz_dict)

        # 3. Save MP4 Video Rollout
        if save_video and self.frames:
            h, w, _ = self.frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))
            for frame in self.frames:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr)
            out.release()

        print(f"[TrajectoryRecorder] Saved episode ({len(self.steps)} steps, Success={final_success}) to: {json_path}")
        return {
            "json_path": json_path,
            "video_path": video_path if save_video else None,
            "npz_path": npz_path if save_npz else None,
            "success": final_success,
            "total_steps": len(self.steps)
        }

    @staticmethod
    def load_trajectory(file_path):
        """
        Loads a previously saved trajectory from JSON or NPZ.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trajectory file not found: {file_path}")

        if file_path.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif file_path.endswith(".npz"):
            data = np.load(file_path, allow_pickle=True)
            meta = json.loads(str(data["metadata_json"])) if "metadata_json" in data else {}
            return {
                "metadata": meta,
                "actions": data["actions"],
                "rewards": data["rewards"],
                "successes": data["successes"],
                "tcp_poses": data["tcp_poses"],
                "gripper_widths": data["gripper_widths"],
                "images": data["images"] if "images" in data else None
            }
        else:
            raise ValueError("Unsupported format. Use .json or .npz")


if __name__ == "__main__":
    print("[TrajectoryRecorder] Testing trajectory recording lifecycle...")
    from simpler_env_adapter import SimplerEnvAdapter

    env = SimplerEnvAdapter("google_robot_pick_coke_can", max_steps=10)
    recorder = TrajectoryRecorder(save_dir="data/test_trajectories", task_name="google_robot_pick_coke_can")

    obs, reset_info = env.reset(seed=123)
    recorder.start_trajectory(
        task_name="google_robot_pick_coke_can",
        seed=123,
        camera_calibration=obs.get("camera_calibration"),
        source="unit_test"
    )

    for step_i in range(5):
        act = [0.01, 0.0, -0.005, 0.0, 0.0, 0.0, 1.0]
        obs, r, term, trunc, info = env.step(act)
        recorder.record_step(step_i, obs, act, r, term, trunc, info)

    res = recorder.save()
    print("Recorded output:", res)
    loaded = TrajectoryRecorder.load_trajectory(res["json_path"])
    print(f"Loaded {len(loaded['steps'])} steps. Metadata seed: {loaded['metadata']['seed']}")
    print("[TrajectoryRecorder] Lifecycle test passed!")
