#!/usr/bin/env python3
"""
Comprehensive Performance Comparison: Raw VLA vs. VLA + Embodied Memory on LIBERO
================================================================================
Evaluates empirical performance of ready models on authentic official LIBERO benchmarks.

Ablation Conditions:
1. raw: Clean Visual Feed + Raw Natural Instruction (MiniVLA Baseline)
2. visual_memory: SpatialTracker Keypoints + Painter Overlay + Raw Natural Instruction
3. temporal_memory: Clean Visual Feed + Chronicler Dynamic Prompt
4. full_memory: Painter Visual Overlay + Chronicler Dynamic Prompt

Measures:
- Success Rate (%) via env.check_success()
- Mean Steps to Completion
- Trajectory Jerk / Action Variance (Cartesian delta smoothness)
- Gripper Action Variance (chattering / premature grasping)
- Real inference latency (FPS / step time)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("external/openvla-mini"))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from src.minivla_policy import MiniVLAPolicy
from src.chronicler import Chronicler
from src.spatial_tracker import SpatialTracker
from src.painter import Painter


def run_episode(
    env: OffScreenRenderEnv,
    policy: MiniVLAPolicy,
    chronicler: Chronicler,
    tracker: SpatialTracker,
    painter: Painter,
    instruction: str,
    condition: str = "raw",  # 'raw', 'visual_memory', 'temporal_memory', 'full_memory'
    init_state=None,
    max_steps: int = 150,
    save_frames: bool = False,
):
    """Executes a single evaluation episode under a specific ablation condition."""
    env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
    else:
        obs = env.reset()

    chronicler.reset()
    tracker.reset()

    # Let objects settle in simulation (standard LIBERO 10 wait steps with gripper open)
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)

    # Initialize tracker on right-side-up image
    raw_img = np.flipud(obs["agentview_image"])
    h, w = raw_img.shape[:2]
    # Initialize tracking points at table center (approximate target area [128, 145])
    tracker.initialize_tracking_points(raw_img, [int(w * 0.50), int(h * 0.58)], num_points=9)

    success = False
    step_count = 0
    actions_taken = []
    frames = []
    latencies = []
    t0 = time.time()

    for step in range(max_steps):
        step_count += 1
        t_step_start = time.time()

        # Extract proprioception info from LIBERO observation
        eef_pos = obs.get("robot0_eef_pos", [0.0, 0.0, 0.0])
        gripper_qpos = obs.get("robot0_gripper_qpos", [0.0, 0.0])
        gripper_width = float(gripper_qpos[0] + gripper_qpos[1]) if len(gripper_qpos) >= 2 else 1.0

        proprioception = {
            "tcp_pos": eef_pos.tolist() if hasattr(eef_pos, "tolist") else list(eef_pos),
            "gripper_pos": eef_pos.tolist() if hasattr(eef_pos, "tolist") else list(eef_pos),
            "gripper_width": gripper_width,
        }

        # Spatial Tracker step
        current_upright_frame = np.flipud(obs["agentview_image"])
        pts, occ_flags = tracker.track_points(current_upright_frame)

        # Chronicler update
        tracking_info = {
            "target_pos": pts[0] if (pts is not None and len(pts) > 0) else [128, 145],
            "is_occluded": any(occ_flags) if occ_flags else False,
        }
        dynamic_prompt = chronicler.update_state(
            current_upright_frame, instruction, proprioception, tracking_info
        )

        # Formulate observation & instruction based on ablation condition
        if condition == "raw":
            policy_img = obs["agentview_image"]  # original MuJoCo frame (flipped internally by preprocess_image)
            policy_prompt = instruction
        elif condition == "visual_memory":
            # Paint on upright frame, then flip back to MuJoCo convention
            painted_upright = painter.draw_overlay(current_upright_frame, pts, occ_flags, show_hud=False)
            policy_img = np.flipud(painted_upright)
            policy_prompt = instruction
        elif condition == "temporal_memory":
            policy_img = obs["agentview_image"]
            policy_prompt = dynamic_prompt
        elif condition == "full_memory":
            painted_upright = painter.draw_overlay(current_upright_frame, pts, occ_flags, show_hud=False)
            policy_img = np.flipud(painted_upright)
            policy_prompt = dynamic_prompt
        else:
            raise ValueError(f"Unknown condition: {condition}")

        if save_frames:
            # Save upright frame for visualization
            if "visual" in condition or "full" in condition:
                frames.append(painted_upright)
            else:
                frames.append(current_upright_frame)

        # Policy inference
        obs_input = {"agentview_image": policy_img}
        action = policy.predict_action(obs_input, policy_prompt)

        latencies.append(time.time() - t_step_start)
        actions_taken.append(action)

        # Step environment
        obs, reward, done, info = env.step(action.tolist())

        # LIBERO success check
        if env.check_success():
            success = True
            break

    total_duration = time.time() - t0

    # Compute trajectory smoothness (jerk)
    actions_arr = np.array(actions_taken)
    if len(actions_arr) > 2:
        cartesian_deltas = actions_arr[:, :3]
        jerk = float(np.mean(np.linalg.norm(np.diff(cartesian_deltas, n=2, axis=0), axis=1)))
        gripper_variance = float(np.var(actions_arr[:, -1]))
    else:
        jerk = 0.0
        gripper_variance = 0.0

    return {
        "condition": condition,
        "success": success,
        "steps": step_count,
        "duration": total_duration,
        "fps": step_count / max(total_duration, 0.001),
        "avg_step_latency": float(np.mean(latencies)) if latencies else 0.0,
        "jerk": jerk,
        "gripper_variance": gripper_variance,
        "frames": frames if save_frames else None,
    }


def evaluate_comparison(
    tier: int = 1,
    task_id: int = 0,
    episodes_per_condition: int = 2,
    max_steps: int = 120,
    conditions: Optional[List[str]] = None,
    output_dir: str = "eval_results",
):
    """
    Evaluates MiniVLA across conditions on a specific LIBERO task.
    """
    if conditions is None:
        conditions = ["raw", "visual_memory", "temporal_memory", "full_memory"]

    from scripts.eval_libero_minisuite import TIER_TASKS

    cfg = TIER_TASKS[tier]
    suite_name = cfg["suite"]
    tier_name = cfg["name"]

    print("\n" + "=" * 78)
    print(f"  LIBERO PERFORMANCE BENCHMARK: RAW VLA vs. EMBODIED MEMORY")
    print(f"  Suite: {suite_name} ({tier_name}) | Task ID: {task_id}")
    print(f"  Conditions: {', '.join(conditions)}")
    print(f"  Episodes per condition: {episodes_per_condition} | Max steps: {max_steps}")
    print("=" * 78 + "\n")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl = task_suite.get_task_bddl_file_path(task_id)
    instruction = task.language

    try:
        initial_states = task_suite.get_task_init_states(task_id)
    except Exception:
        initial_states = None

    print(f"Task Name: '{task.name}'")
    print(f"Instruction: '{instruction}'")
    print(f"BDDL: {task_bddl}")
    print(f"Init states available: {len(initial_states) if initial_states is not None else 0}\n")

    print("[Policy] Initializing MiniVLAPolicy (minivla_vq_libero90)...")
    policy = MiniVLAPolicy(device="cpu")
    chronicler = Chronicler(use_real_vlm=False)
    tracker = SpatialTracker(tracker_type="opencv_fallback")
    painter = Painter()

    results_by_condition = {cond: [] for cond in conditions}
    os.makedirs(output_dir, exist_ok=True)

    env_args = {
        "bddl_file_name": task_bddl,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)

    for cond in conditions:
        cond_label = {
            "raw": "1. Raw Baseline (Clean Image + Raw Text)",
            "visual_memory": "2. Visual Memory Only (Painter Dots + Raw Text)",
            "temporal_memory": "3. Temporal Memory Only (Clean Image + Chronicler Prompt)",
            "full_memory": "4. Full Memory (Painter Dots + Chronicler Prompt)",
        }.get(cond, cond)

        print(f"\n--- Condition: {cond_label} ---")
        for ep in range(episodes_per_condition):
            init_st = initial_states[ep % len(initial_states)] if initial_states is not None else None
            print(f"  [Ep {ep + 1}/{episodes_per_condition}] Running...", end="", flush=True)
            res = run_episode(
                env=env,
                policy=policy,
                chronicler=chronicler,
                tracker=tracker,
                painter=painter,
                instruction=instruction,
                condition=cond,
                init_state=init_st,
                max_steps=max_steps,
                save_frames=False,
            )
            results_by_condition[cond].append(res)
            st = "SUCCESS" if res["success"] else "FAIL"
            print(f" {st} | Steps: {res['steps']:3d} | Jerk: {res['jerk']:.4f} | Latency: {res['avg_step_latency']:.2f}s")

    env.close()

    # Aggregate Metrics
    summary = {}
    print("\n" + "=" * 85)
    print("                 LIBERO BENCHMARK EMPIRICAL COMPARISON RESULTS")
    print("=" * 85)
    print(f"{'Condition':<26} | {'Success':<10} | {'Avg Steps':<10} | {'Jerk (Smooth)':<14} | {'Step Latency':<12}")
    print("-" * 85)

    for cond, runs in results_by_condition.items():
        total = len(runs)
        succ = sum(1 for r in runs if r["success"])
        sr = (succ / total) * 100.0 if total > 0 else 0.0
        avg_steps = float(np.mean([r["steps"] for r in runs]))
        avg_jerk = float(np.mean([r["jerk"] for r in runs]))
        avg_latency = float(np.mean([r["avg_step_latency"] for r in runs]))

        summary[cond] = {
            "success_count": succ,
            "total_episodes": total,
            "success_rate_pct": sr,
            "avg_steps": round(avg_steps, 1),
            "avg_jerk": round(avg_jerk, 6),
            "avg_latency_sec": round(avg_latency, 3),
        }
        print(f"{cond:<26} | {sr:>5.1f}% ({succ}/{total}) | {avg_steps:>9.1f} | {avg_jerk:>14.6f} | {avg_latency:>10.3f}s")
    print("=" * 85)

    # Save to json
    save_file = os.path.join(output_dir, f"libero_memory_ablation_tier{tier}_task{task_id}.json")
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tier": tier,
        "task_id": task_id,
        "task_name": task.name,
        "instruction": instruction,
        "episodes_per_condition": episodes_per_condition,
        "summary": summary,
        "raw_runs": {
            cond: [
                {
                    "success": r["success"],
                    "steps": r["steps"],
                    "duration": round(r["duration"], 2),
                    "jerk": round(r["jerk"], 6),
                    "gripper_variance": round(r["gripper_variance"], 6),
                    "avg_step_latency": round(r["avg_step_latency"], 3),
                }
                for r in runs
            ]
            for cond, runs in results_by_condition.items()
        },
    }
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Saved] Detailed ablation report written to: {save_file}\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirical Comparison on LIBERO Benchmark")
    parser.add_argument("--tier", type=int, default=1, help="LIBERO Tier (1: spatial, 2: object, 3: goal, 4: 10)")
    parser.add_argument("--task_id", type=int, default=0, help="Task ID within tier")
    parser.add_argument("--episodes", type=int, default=1, help="Number of rollouts per condition")
    parser.add_argument("--max_steps", type=int, default=80, help="Max steps per episode (to bound runtime)")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["raw", "visual_memory", "temporal_memory", "full_memory"],
        help="List of conditions to run",
    )
    args = parser.parse_args()

    evaluate_comparison(
        tier=args.tier,
        task_id=args.task_id,
        episodes_per_condition=args.episodes,
        max_steps=args.max_steps,
        conditions=args.conditions,
    )
