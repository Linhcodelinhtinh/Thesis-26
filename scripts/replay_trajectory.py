#!/usr/bin/env python3
"""
Deterministic Trajectory Replay & Verification Engine
------------------------------------------------------
Replays previously recorded trajectories in SimplerEnv by:
1. Re-instantiating the task environment with the identical random seed & camera calibration
2. Re-injecting the exact recorded actions step-by-step
3. Verifying kinematic state determinism (TCP pos error <= threshold) and success reproducibility
"""
import sys
import os
import argparse
import time
import numpy as np
import cv2

# Add src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from simpler_env_adapter import SimplerEnvAdapter
from trajectory_recorder import TrajectoryRecorder


def replay_trajectory(file_path, visualize=False, tolerance=0.01):
    """
    Executes a deterministic replay of the specified trajectory.
    """
    print(f"\n=======================================================")
    print(f"[ReplayEngine] Loading trajectory: {file_path}")
    print(f"=======================================================")

    traj = TrajectoryRecorder.load_trajectory(file_path)
    metadata = traj["metadata"]
    task_name = metadata.get("task_name", "google_robot_pick_coke_can")
    seed = metadata.get("seed", 42)
    recorded_steps = traj.get("steps", [])

    print(f"Task Name: {task_name}")
    print(f"Environment Seed: {seed}")
    print(f"Recorded Steps: {len(recorded_steps)}")
    print(f"Original Status: {'SUCCESS' if metadata.get('final_success', False) else 'FAIL'}")

    # Initialize environment with exact recorded seed
    env = SimplerEnvAdapter(task_name=task_name, max_steps=len(recorded_steps) + 10, seed=seed)
    obs, reset_info = env.reset(seed=seed)

    max_deviation = 0.0
    matched_success = True
    replayed_frames = []

    if visualize:
        cv2.namedWindow(f"Replay: {task_name}", cv2.WINDOW_AUTOSIZE)

    print("\n--- Executing Replay Steps ---")
    for i, step_data in enumerate(recorded_steps):
        action = step_data["action"]
        orig_tcp = np.array(step_data["observation"]["tcp_pos"], dtype=np.float32)
        orig_success = step_data.get("success", False)

        # Step simulation with recorded action
        obs, reward, term, trunc, info = env.step(action)
        replayed_tcp = np.array(obs["proprioception"]["tcp_pos"], dtype=np.float32)
        replayed_success = info.get("success", False)

        # Calculate deviation
        dev = float(np.linalg.norm(replayed_tcp - orig_tcp))
        if dev > max_deviation:
            max_deviation = dev

        if orig_success != replayed_success:
            matched_success = False

        # Visual display if requested
        if visualize:
            frame = obs["image"].copy()
            status_text = "SUCCESS" if replayed_success else "IN PROGRESS"
            color = (0, 255, 0) if replayed_success else (0, 200, 255)
            cv2.putText(frame, f"REPLAY STEP {i+1}/{len(recorded_steps)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(frame, f"STATUS: {status_text}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            cv2.putText(frame, f"Dev: {dev*1000:.2f} mm", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
            cv2.imshow(f"Replay: {task_name}", frame)
            key = cv2.waitKey(40) & 0xFF
            if key == 27 or key == ord('q'):
                break

    if visualize:
        cv2.destroyAllWindows()

    env.close()

    # Evaluation Summary
    passed_determinism = max_deviation <= tolerance
    print("\n=======================================================")
    print("REPLAY VERIFICATION SUMMARY")
    print(f"=======================================================")
    print(f"Max Cartesian Deviation : {max_deviation * 1000.0:.3f} mm (Tolerance: {tolerance * 1000.0:.1f} mm)")
    print(f"Determinism Check       : {'PASSED (Zero-Drift)' if passed_determinism else 'FAILED (Drift Detected)'}")
    print(f"Success Flag Check      : {'MATCHED' if matched_success else 'MISMATCH'}")
    print(f"Overall Result          : {'REPLAY VERIFIED' if (passed_determinism and matched_success) else 'REPLAY FAILED'}")
    print(f"=======================================================\n")

    return {
        "passed": passed_determinism and matched_success,
        "max_deviation_mm": max_deviation * 1000.0,
        "matched_success": matched_success
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Trajectory Replay for SimplerEnv")
    parser.add_argument("--file", type=str, default=None, help="Path to trajectory JSON file")
    parser.add_argument("--visualize", action="store_true", help="Display visual replay in OpenCV window")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Allowed Cartesian drift tolerance in meters")
    args = parser.parse_args()

    target_file = args.file
    if not target_file:
        # If no file provided, generate an expert demo on the fly to verify replay
        print("[ReplayEngine] No input file provided. Generating a fresh expert trajectory for verification...")
        from teleop_controller import ScriptedOracleExpert

        test_env = SimplerEnvAdapter("google_robot_pick_coke_can", max_steps=40)
        recorder = TrajectoryRecorder(save_dir="data/trajectories", task_name="google_robot_pick_coke_can")
        expert = ScriptedOracleExpert("google_robot_pick_coke_can")

        obs, _ = test_env.reset(seed=999)
        recorder.start_trajectory(
            task_name="google_robot_pick_coke_can",
            seed=999,
            source="scripted_expert_for_replay"
        )

        for s in range(30):
            a = expert.get_action(obs)
            obs, r, term, trunc, inf = test_env.step(a)
            recorder.record_step(s, obs, a, r, term, trunc, inf)
            if inf["success"]:
                break

        res = recorder.save()
        target_file = res["json_path"]
        test_env.close()

    result = replay_trajectory(target_file, visualize=args.visualize, tolerance=args.tolerance)
    if not result["passed"]:
        sys.exit(1)
