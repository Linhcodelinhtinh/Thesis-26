"""
Lightweight verification that all 4 Tiers of the LIBERO Mini-Suite
can initialize with authentic official BDDL scenes, initial states,
and offscreen rendering.
"""

import os
import sys
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

TIER_SUITES = {
    1: ("libero_spatial", [0, 2]),
    2: ("libero_object", [0, 1, 4]),
    3: ("libero_goal", [0, 8]),
    4: ("libero_10", [0, 3]),
}

benchmark_dict = benchmark.get_benchmark_dict()

for tier, (suite_name, task_indices) in TIER_SUITES.items():
    print(f"\n{'='*60}\nChecking Tier {tier}: {suite_name}\n{'='*60}")
    task_suite = benchmark_dict[suite_name]()
    for task_idx in task_indices:
        task = task_suite.get_task(task_idx)
        bddl = task_suite.get_task_bddl_file_path(task_idx)
        init_states = task_suite.get_task_init_states(task_idx)
        print(f"Task {task_idx}: {task.name}")
        print(f"  Language: '{task.language}'")
        print(f"  BDDL exists: {os.path.exists(bddl)}")
        print(f"  Init states: {init_states.shape}")

        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128)
        env.reset()
        obs = env.set_init_state(init_states[0])
        assert "agentview_image" in obs, "Missing agentview_image in obs!"
        assert "robot0_eye_in_hand_image" in obs, "Missing eye_in_hand in obs!"
        
        # Settle step
        obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
        env.close()
        print(f"  [OK] Environment initialized and verified successfully.")

print(f"\n{'='*60}\nALL 4 TIERS VERIFIED WITH AUTHENTIC LIBERO SUITES!\n{'='*60}")
