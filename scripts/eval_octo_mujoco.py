#!/usr/bin/env python3
"""
Octo VLA + MuJoCo Physical Simulation & Evaluation Script
---------------------------------------------------------
Executes closed-loop evaluation between Octo VLA, Chronicler (Memory-as-a-Prompt),
Spatial Tracker (Persistent Visual Overlay), and 3D MuJoCo Franka Panda environment.
Logs Grasp Success Rate (GSR), lift height, contact forces, and occlusion tracking robustness.
"""
import os
import sys
import time
import argparse
import yaml
import numpy as np

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from src.mujoco_env import MujocoRoboticEnv
from src.vla_wrapper import OctoWrapper
from src.spatial_tracker import SpatialTracker
from src.painter import Painter
from src.chronicler import Chronicler
from src.sync_layer import SyncLayer

try:
    import mujoco
    import mujoco.viewer as mj_viewer
except ImportError:
    mujoco = None
    mj_viewer = None


def load_config(config_path="configs/config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def run_evaluation(num_episodes=5, max_steps_per_episode=120, use_viewer=False, camera_name="overhead_cam", config_path="configs/config.yaml"):
    config = load_config(config_path)
    
    print("=" * 70)
    print("  OCTO VLA + MUJOCO 3D ROBOTIC MULTI-MODAL MEMORY EVALUATION")
    print("=" * 70)

    # 1. Initialize Environment
    scene_path = config.get("mujoco", {}).get("scene_path", "assets/franka_table_scene.xml")
    env = MujocoRoboticEnv(scene_path=scene_path)
    
    # 2. Initialize VLA & Memory Components from Config
    octo_checkpoint = config.get("octo_checkpoint", "octo-small")
    model_paths = config.get("model_paths", {})
    model_path = model_paths.get("octo_small" if "small" in octo_checkpoint else "octo_base", "hf://rail-berkeley/octo-small-1.5")
    
    octo_vla = OctoWrapper(model_id=model_path, checkpoint_type=octo_checkpoint)
    spatial_tracker = SpatialTracker(tracker_type=config.get("tracker_type", "opencv_fallback"), config=config)
    painter = Painter()
    
    chronicler_cfg = config.get("chronicler", {})
    chronicler = Chronicler(
        use_real_vlm=chronicler_cfg.get("use_real_vlm", False),
        model_id=chronicler_cfg.get("vlm_model", "HuggingFaceTB/SmolVLM-Instruct"),
        device=chronicler_cfg.get("device", "auto"),
        max_history_length=chronicler_cfg.get("max_history_length", 5)
    )
    sync_layer = SyncLayer()

    episode_results = []

    # Optional 3D Passive Viewer
    viewer = None
    if use_viewer and mj_viewer is not None:
        try:
            viewer = mj_viewer.launch_passive(env.model, env.data)
            print("[Evaluation] MuJoCo 3D Passive Viewer launched.")
        except Exception as e:
            print(f"[Evaluation] Passive viewer launch notice ({e}). Continuing in headless offscreen mode.")
            viewer = None

    for ep in range(1, num_episodes + 1):
        print(f"\n--- Episode {ep}/{num_episodes} ---")
        # Reset environment with randomized target cube position
        random_offset_x = np.random.uniform(-0.03, 0.03)
        random_offset_y = np.random.uniform(-0.04, 0.04)
        target_pos = [0.55 + random_offset_x, 0.0 + random_offset_y, 0.395]
        
        obs = env.reset(target_pos=target_pos)
        octo_vla.phase = "approach"
        octo_vla.grasp_counter = 0
        chronicler.reset()

        # Initial tracker calibration
        raw_frame = env.render_camera(camera_name)
        spatial_tracker.initialize_tracking_points(raw_frame, [320, 240], num_points=9)

        ep_max_force = 0.0
        ep_max_lift = 0.0
        ep_occlusion_steps = 0
        ep_latencies = []

        human_command = "Grasp the target object on the table"

        for step in range(max_steps_per_episode):
            t_start = time.time()

            # 1. Offscreen Render Camera Frame
            raw_rgb = env.render_camera(camera_name)

            # 2. Spatial Tracking & Occlusion Detection
            gripper_xy = obs["gripper_pos"] if isinstance(obs, dict) and "gripper_pos" in obs else [320, 240]
            pts, flags = spatial_tracker.track(raw_rgb, gripper_pos=gripper_xy)
            
            # Check physical occlusion in MuJoCo
            is_physically_occluded = env.check_occlusion()
            if is_physically_occluded:
                flags = [True] * len(flags)
                ep_occlusion_steps += 1

            # 3. Dynamic Prompt Generation via Chronicler
            proprioception = env.get_proprioception()
            proprioception["target_pos"] = env.get_target_pos().tolist()
            tracking_info = {
                "target_pos": env.get_target_pos()[:2].tolist(),
                "is_occluded": is_physically_occluded
            }
            dynamic_prompt = chronicler.update_state(
                raw_rgb, human_command, proprioception, tracking_info
            )

            # 4. Painter: Render visual cues with dynamic prompt
            painted_image = painter.draw_overlay(raw_rgb, pts, flags, dynamic_prompt)

            # 5. Octo VLA Inference (conditioned on both painted image & dynamic prompt)
            t_infer_start = time.time()
            action_7dof = octo_vla.predict_action(painted_image, dynamic_prompt, proprioception)
            ep_latencies.append((time.time() - t_infer_start) * 1000)

            # 6. Apply Action to MuJoCo Physics
            step_result = env.step(action_7dof)
            metrics = step_result["metrics"]

            # Log metrics
            ep_max_force = max(ep_max_force, metrics["max_contact_force_n"])
            ep_max_lift = max(ep_max_lift, metrics["lift_height_meters"])

            # 7. Synchronize 3D Viewer if active
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.01)

            # Check early termination if successful grasp and lift achieved
            if metrics["lift_height_meters"] > 0.015:
                print(f"  [Step {step+1}] Target successfully lifted to {metrics['lift_height_meters']:.4f}m!")
                break

        final_metrics = env.get_metrics()
        success = bool(final_metrics["lift_height_meters"] > 0.015 or final_metrics["is_grasped"])

        result_summary = {
            "episode": ep,
            "success": success,
            "max_lift_height_m": round(ep_max_lift, 4),
            "max_contact_force_n": round(ep_max_force, 2),
            "occlusion_ratio": round(ep_occlusion_steps / max_steps_per_episode, 3),
            "avg_latency_ms": round(float(np.mean(ep_latencies)), 2)
        }
        episode_results.append(result_summary)

        print(f"  Result: {'[SUCCESS]' if success else '[FAILED]'}")
        print(f"  - Max Lift Height: {result_summary['max_lift_height_m']} m")
        print(f"  - Max Contact Force: {result_summary['max_contact_force_n']} N")
        print(f"  - Occlusion Ratio: {result_summary['occlusion_ratio'] * 100:.1f}%")
        print(f"  - Mean Inference Latency: {result_summary['avg_latency_ms']} ms")
        print(f"  - Final Chronicler State: {chronicler.get_state_dict()['current_phase']}")

    if viewer is not None:
        viewer.close()

    # Final Benchmark Report
    total_episodes = len(episode_results)
    successes = sum(1 for r in episode_results if r["success"])
    gsr = (successes / total_episodes) * 100.0
    avg_lift = np.mean([r["max_lift_height_m"] for r in episode_results])
    avg_force = np.mean([r["max_contact_force_n"] for r in episode_results])
    avg_latency = np.mean([r["avg_latency_ms"] for r in episode_results])

    print("\n" + "=" * 70)
    print("  FINAL EVALUATION REPORT SUMMARY")
    print("=" * 70)
    print(f"  Total Episodes Tested: {total_episodes}")
    print(f"  Grasp Success Rate (GSR): {gsr:.1f}% ({successes}/{total_episodes})")
    print(f"  Average Lift Height: {avg_lift:.4f} m")
    print(f"  Average Peak Contact Force: {avg_force:.2f} N")
    print(f"  Average Step Latency: {avg_latency:.2f} ms")
    print("=" * 70)

    return episode_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Octo VLA in MuJoCo 3D Environment")
    parser.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    parser.add_argument("--steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D viewer")
    parser.add_argument("--camera", type=str, default="overhead_cam", help="Camera name to render")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()

    run_evaluation(
        num_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        use_viewer=args.viewer,
        camera_name=args.camera,
        config_path=args.config
    )
