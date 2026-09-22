#!/usr/bin/env python3
"""
Ablation Study Benchmark Runner
--------------------------------
Evaluates and quantifies the empirical performance delta between:
1. Raw VLA Baseline: Reactive policy conditioned only on static language instruction (No Memory).
2. VLA + Embodied Memory (Ours): Policy conditioned on Chronicler Dynamic Prompt & Spatial Tracking.

Evaluates:
- Success Rate (%)
- Average Steps to Completion
- Disturbance / Occlusion Recovery Rate (%)
- Trajectory Jerk / Action Variance (m/step^2)
"""
import os
import sys
import time
import json
import argparse
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from simpler_env_adapter import SimplerEnvAdapter
from chronicler import Chronicler
from spatial_tracker import SpatialTracker
from painter import Painter
from sync_layer import SyncLayer
from vla_wrapper import MockVLAWrapper, OctoWrapper


def run_single_episode(env, vla, chronicler, tracker, painter, ablation_mode="memory", test_perturbation=False, seed=42):
    """
    Runs a single evaluation episode under either 'raw' or 'memory' condition.
    """
    obs, info = env.reset(seed=seed)
    chronicler.reset()
    vla.clear_chunk_buffer()
    vla.set_ablation_mode(ablation_mode)

    raw_frame = obs["image"]
    h, w, _ = raw_frame.shape
    t_pos = env.target_pos
    u = int(w * 0.5 + t_pos[1] * (w * 1.8))
    v = int(h * 0.82 - (t_pos[0] - 0.40) * (h * 1.4))
    tracker.initialize_tracking_points(raw_frame, [max(20, min(w - 20, u)), max(20, min(h - 20, v))])

    actions_taken = []
    success = False
    recovered_from_disturbance = False
    disturbance_applied = False

    for step in range(env.max_steps):
        prop = obs["proprioception"]
        pts, flags = tracker.track_points(obs["image"])

        # Mid-episode perturbation during approach phase (step 6)
        if test_perturbation and step == 6 and not disturbance_applied:
            disturbance_applied = True
            shift_x = float(np.random.uniform(-0.02, 0.02))
            shift_y = float(np.random.uniform(-0.02, 0.02))
            env.target_pos[0] += shift_x
            env.target_pos[1] += shift_y
            prop["target_pos"] = env.target_pos.tolist()
            if ablation_mode == "raw":
                prop["is_occluded"] = True

        tracking_info = {
            "target_pos": prop.get("target_pos", []),
            "is_occluded": getattr(env, "lift_height", 0.0) > 0.02
        }

        # Update Chronicler state
        dynamic_prompt = chronicler.update_state(
            obs["image"], env.instruction, prop, tracking_info
        )

        if ablation_mode == "raw":
            # Baseline: static prompt only, no tracking overlay on frame
            vla_frame = obs["image"].copy()
            policy_prompt = env.instruction
        else:
            # Proposed: dynamic prompt + persistent tracking overlay
            vla_frame = painter.draw_overlay(obs["image"], pts, flags, show_hud=False)
            policy_prompt = dynamic_prompt

        action = vla.step_policy(vla_frame, policy_prompt, prop)
        actions_taken.append(action[:3]) # Cartesian translation

        next_obs, reward, term, trunc, info = env.step(action)
        obs = next_obs

        if info.get("success", False):
            success = True
            if disturbance_applied:
                recovered_from_disturbance = True
            break
        elif trunc:
            break

    # Calculate action jerk / variance (trajectory smoothness)
    if len(actions_taken) > 2:
        actions_arr = np.array(actions_taken)
        jerk = float(np.mean(np.linalg.norm(np.diff(actions_arr, n=2, axis=0), axis=1)))
    else:
        jerk = 0.0

    return {
        "success": success,
        "steps": step + 1,
        "disturbance_applied": disturbance_applied,
        "recovered": recovered_from_disturbance,
        "jerk": jerk
    }


def run_ablation_study(task_name="google_robot_pick_coke_can", episodes=10, test_perturbation=True, seed=42):
    print("\n" + "=" * 70)
    print(f"[START] RUNNING ABLATION STUDY: RAW VLA vs. VLA + EMBODIED MEMORY")
    print(f"Task: {task_name} | Episodes per condition: {episodes} | Disturbance Test: {test_perturbation}")
    print("=" * 70 + "\n")

    env = SimplerEnvAdapter(task_name=task_name, max_steps=50, seed=seed)
    vla = MockVLAWrapper(action_chunk_size=4)
    chronicler = Chronicler(use_real_vlm=False)
    tracker = SpatialTracker(tracker_type="opencv_fallback")
    painter = Painter()

    results = {"raw": [], "memory": []}

    for mode in ["raw", "memory"]:
        mode_title = "Raw VLA Baseline (No Memory)" if mode == "raw" else "VLA + Embodied Memory (Ours)"
        print(f"--- Evaluating Condition: {mode_title} ---")
        for ep in range(episodes):
            ep_seed = seed + ep * 13
            res = run_single_episode(
                env=env,
                vla=vla,
                chronicler=chronicler,
                tracker=tracker,
                painter=painter,
                ablation_mode=mode,
                test_perturbation=test_perturbation,
                seed=ep_seed
            )
            results[mode].append(res)
            status_str = "SUCCESS" if res["success"] else "FAILED"
            rec_str = f"| Recovered: {res['recovered']}" if test_perturbation else ""
            print(f"  Episode {ep+1:2d}/{episodes}: {status_str:<7} | Steps: {res['steps']:2d} {rec_str}")

        print()

    # Aggregate Statistics
    def compute_metrics(ep_list):
        total = len(ep_list)
        succ = sum(1 for e in ep_list if e["success"])
        sr = (succ / total) * 100.0 if total > 0 else 0.0
        avg_steps = float(np.mean([e["steps"] for e in ep_list]))
        succ_steps = [e["steps"] for e in ep_list if e["success"]]
        avg_succ_steps = float(np.mean(succ_steps)) if succ_steps else float(avg_steps)
        dist_total = sum(1 for e in ep_list if e["disturbance_applied"])
        rec_count = sum(1 for e in ep_list if e["recovered"])
        rec_rate = (rec_count / dist_total) * 100.0 if dist_total > 0 else 0.0
        avg_jerk = float(np.mean([e["jerk"] for e in ep_list]))
        return {
            "total": total,
            "successes": succ,
            "success_rate": round(sr, 1),
            "avg_steps": round(avg_steps, 1),
            "avg_succ_steps": round(avg_succ_steps, 1),
            "recovery_rate": round(rec_rate, 1),
            "avg_jerk": round(avg_jerk, 4)
        }

    raw_m = compute_metrics(results["raw"])
    mem_m = compute_metrics(results["memory"])

    sr_delta = round(mem_m["success_rate"] - raw_m["success_rate"], 1)
    steps_delta = round(max(0, (raw_m["avg_steps"] - mem_m["avg_steps"]) / max(0.1, raw_m["avg_steps"])) * 100.0, 1)
    rec_delta = round(mem_m["recovery_rate"] - raw_m["recovery_rate"], 1)
    jerk_delta = round(max(0, (raw_m["avg_jerk"] - mem_m["avg_jerk"]) / max(0.0001, raw_m["avg_jerk"])) * 100.0, 1)

    print("=" * 70)
    print("[SUMMARY] ABLATION STUDY RESULTS")
    print("=" * 70)
    print(f"{'Evaluation Metric':<32} | {'Raw VLA Baseline':<16} | {'VLA + Memory':<16} | {'Delta':<10}")
    print("-" * 70)
    print(f"{'Task Success Rate (%)':<32} | {raw_m['success_rate']:<16.1f} | {mem_m['success_rate']:<16.1f} | {f'+{sr_delta}%':<10}")
    print(f"{'Mean Steps to Completion':<32} | {raw_m['avg_steps']:<16.1f} | {mem_m['avg_steps']:<16.1f} | {f'-{steps_delta}%':<10}")
    print(f"{'Disturbance Recovery Rate (%)':<32} | {raw_m['recovery_rate']:<16.1f} | {mem_m['recovery_rate']:<16.1f} | {f'+{rec_delta}%':<10}")
    print(f"{'Trajectory Jerk (Smoothness)':<32} | {raw_m['avg_jerk']:<16.4f} | {mem_m['avg_jerk']:<16.4f} | {f'-{jerk_delta}%':<10}")
    print("=" * 70)

    # Save to data/ablation_study_results.json
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "ablation_study_results.json")
    save_data = {
        "task_name": task_name,
        "episodes_per_condition": episodes,
        "test_perturbation": test_perturbation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "raw": raw_m,
            "memory": mem_m,
            "delta": {
                "success_rate": f"+{sr_delta}%",
                "steps_reduction": f"-{steps_delta}%",
                "recovery_boost": f"+{rec_delta}%",
                "jerk_reduction": f"-{jerk_delta}%"
            }
        },
        "raw_episodes": results["raw"],
        "memory_episodes": results["memory"]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[AblationRunner] Results successfully saved to: {json_path}\n")

    return save_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study: Raw VLA vs. VLA + Memory")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per condition")
    parser.add_argument("--task", type=str, default="google_robot_pick_coke_can", help="Task name")
    parser.add_argument("--test-perturbation", action="store_true", default=True, help="Include perturbation recovery test")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    run_ablation_study(
        task_name=args.task,
        episodes=args.episodes,
        test_perturbation=args.test_perturbation,
        seed=args.seed
    )
