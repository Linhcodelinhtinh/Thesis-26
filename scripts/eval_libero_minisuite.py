"""
LIBERO Mini-Suite Evaluator for MiniVLA
Evaluates policy across 4 official LIBERO Tiers:
- Tier 1 (Spatial): libero_spatial (e.g. pick bowl between plate/ramekin and place on plate)
- Tier 2 (Object): libero_object (e.g. pick alphabet soup / cream cheese / ketchup and place in basket)
- Tier 3 (Goal): libero_goal (e.g. open middle drawer, put bowl on plate)
- Tier 4 (Long-Horizon): libero_10 (e.g. put both soup and tomato sauce in basket, put bowl in drawer and close)
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any
from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("external/openvla-mini"))

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from src.minivla_policy import MiniVLAPolicy


TIER_TASKS = {
    1: {
        "suite": "libero_spatial",
        "tasks": [
            0,  # pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
            2,  # pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate
        ],
        "name": "Tier 1 — Spatial Manipulation",
    },
    2: {
        "suite": "libero_object",
        "tasks": [
            0,  # pick_up_the_alphabet_soup_and_place_it_in_the_basket
            1,  # pick_up_the_cream_cheese_and_place_it_in_the_basket
            4,  # pick_up_the_ketchup_and_place_it_in_the_basket
        ],
        "name": "Tier 2 — Diverse Object Manipulation",
    },
    3: {
        "suite": "libero_goal",
        "tasks": [
            0,  # open_the_middle_drawer_of_the_cabinet
            8,  # put_the_bowl_on_the_plate
        ],
        "name": "Tier 3 — Goal & Articulation",
    },
    4: {
        "suite": "libero_10",
        "tasks": [
            0,  # LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
            3,  # KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it
        ],
        "name": "Tier 4 — Long-Horizon Sequential Tasks",
    },
}


def run_episode(
    env: OffScreenRenderEnv,
    policy: MiniVLAPolicy,
    instruction: str,
    init_state=None,
    max_steps: int = 300,
    save_frames: bool = False,
):
    """Executes a single evaluation episode."""
    env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
    else:
        obs = env.reset()
    frames = []

    # Let objects settle in simulation (standard LIBERO 10 wait steps with gripper open)
    dummy_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    for _ in range(10):
        obs, _, _, _ = env.step(dummy_action)

    success = False
    step_count = 0
    t0 = time.time()

    for step in range(max_steps):
        step_count += 1
        if save_frames and "agentview_image" in obs:
            # MuJoCo OpenGL offscreen images are flipped vertically
            img = np.flipud(obs["agentview_image"])
            frames.append(img)

        # Query MiniVLA policy
        action = policy.predict_action(obs, instruction)

        # Step environment
        obs, reward, done, info = env.step(action.tolist())

        # LIBERO check success
        if env.check_success():
            success = True
            break

    duration = time.time() - t0
    return {
        "success": success,
        "steps": step_count,
        "duration": duration,
        "fps": step_count / max(duration, 0.001),
        "frames": frames if save_frames else None,
    }


def evaluate_tier(
    tier: int,
    policy: MiniVLAPolicy,
    episodes_per_task: int = 3,
    max_steps: int = 300,
    save_video: bool = False,
    task_id: Optional[int] = None,
    output_dir: str = "eval_results",
):
    """Evaluates all selected tasks for a given tier."""
    cfg = TIER_TASKS[tier]
    suite_name = cfg["suite"]
    tier_name = cfg["name"]
    task_indices = [task_id] if task_id is not None else cfg["tasks"]

    print(f"\n{'='*70}")
    print(f"EVALUATING {tier_name} ({suite_name})")
    print(f"{'='*70}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[suite_name]()

    tier_results = []
    os.makedirs(output_dir, exist_ok=True)

    for task_idx in task_indices:
        task = task_suite.get_task(task_idx)
        task_bddl = task_suite.get_task_bddl_file_path(task_idx)
        instruction = task.language

        # Get authentic LIBERO initial states for this benchmark task
        try:
            initial_states = task_suite.get_task_init_states(task_idx)
        except Exception as e:
            initial_states = None

        print(f"\n--- Task {task_idx}: {task.name} ---")
        print(f"Language Instruction: '{instruction}'")
        print(f"BDDL: {task_bddl}")
        if initial_states is not None:
            print(f"Loaded {len(initial_states)} official initial states.")

        env_args = {
            "bddl_file_name": task_bddl,
            "camera_heights": 256,
            "camera_widths": 256,
        }
        env = OffScreenRenderEnv(**env_args)
        try:
            env.seed(0)
        except Exception:
            pass

        task_successes = 0
        task_steps = []

        for ep in range(episodes_per_task):
            print(f"  [Episode {ep + 1}/{episodes_per_task}] Running...", end="", flush=True)
            init_st = initial_states[ep % len(initial_states)] if initial_states is not None else None
            res = run_episode(
                env,
                policy,
                instruction,
                init_state=init_st,
                max_steps=max_steps,
                save_frames=save_video,
            )
            status = "SUCCESS" if res["success"] else "FAIL"
            print(f" Result: {status} ({res['steps']} steps, {res['fps']:.1f} FPS)")

            if res["success"]:
                task_successes += 1
            task_steps.append(res["steps"])

            # Optionally save rollout video if requested
            if save_video and res["frames"]:
                try:
                    import cv2
                    video_path = os.path.join(
                        output_dir,
                        f"tier_{tier}_task_{task_idx}_ep_{ep}_{status.lower()}.mp4",
                    )
                    h, w, _ = res["frames"][0].shape
                    out = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        (w, h),
                    )
                    for frame in res["frames"]:
                        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    out.release()
                    print(f"    Saved rollout video to {video_path}")
                except Exception as e:
                    print(f"    Warning: Could not save video: {e}")

        env.close()

        sr = (task_successes / episodes_per_task) * 100.0
        avg_steps = float(np.mean(task_steps))
        print(f"  Task {task_idx} Summary: Success Rate = {sr:.1f}% ({task_successes}/{episodes_per_task}), Avg Steps = {avg_steps:.1f}")

        tier_results.append({
            "task_id": task_idx,
            "task_name": task.name,
            "instruction": instruction,
            "successes": task_successes,
            "total": episodes_per_task,
            "success_rate": sr,
            "avg_steps": avg_steps,
        })

    return tier_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniVLA on LIBERO Benchmark Mini-Suite")
    parser.add_argument("--tier", type=int, default=1, choices=[1, 2, 3, 4, 0], help="Tier to evaluate (1-4, or 0 for all)")
    parser.add_argument("--task_id", type=int, default=None, help="Specific task index to run within tier (optional)")
    parser.add_argument("--episodes", type=int, default=2, help="Number of rollouts per task")
    parser.add_argument("--max_steps", type=int, default=200, help="Max steps per episode")
    parser.add_argument("--save_video", action="store_true", help="Save rollout videos")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    print(f"Initializing MiniVLA Policy on {args.device}...")
    policy = MiniVLAPolicy(device=args.device)

    tiers_to_run = [1, 2, 3, 4] if args.tier == 0 else [args.tier]

    all_results = {}
    for tier in tiers_to_run:
        res = evaluate_tier(
            tier,
            policy,
            episodes_per_task=args.episodes,
            max_steps=args.max_steps,
            save_video=args.save_video,
            task_id=args.task_id,
        )
        all_results[tier] = res

    print("\n" + "="*75)
    print("                      LIBERO MINI-SUITE SUMMARY RESULTS")
    print("="*75)
    print(f"{'Tier':<8} | {'Task':<45} | {'Success Rate':<12}")
    print("-" * 75)
    for tier, tasks in all_results.items():
        for t in tasks:
            tname = t["task_name"][:43] + ".." if len(t["task_name"]) > 45 else t["task_name"]
            print(f"Tier {tier:<3} | {tname:<45} | {t['success_rate']:>5.1f}% ({t['successes']}/{t['total']})")
    print("="*75)


if __name__ == "__main__":
    main()
