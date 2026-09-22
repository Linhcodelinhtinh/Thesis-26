import os
import sys
import numpy as np

print("Testing LIBERO environment initialization...")
try:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    print(f"Loaded libero_spatial suite with {task_suite.get_num_tasks()} tasks.")
    task = task_suite.get_task(0)
    print(f"Task 0: {task.name}, language: {task.language}")

    task_bddl_file = task_suite.get_task_bddl_file_path(0)
    print(f"BDDL file: {task_bddl_file}")
    print(f"Exists: {os.path.exists(task_bddl_file)}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    print("Initializing environment (resetting)...")
    obs = env.reset()
    print(f"Env reset successful! Observation keys: {list(obs.keys())}")
    for k in obs:
        if isinstance(obs[k], np.ndarray):
            print(f"  {k}: shape={obs[k].shape}, dtype={obs[k].dtype}")

    # Test dummy step
    dummy_action = [0.0] * 7
    obs, reward, done, info = env.step(dummy_action)
    print("Step successful!")
    env.close()
    print("LIBERO test passed successfully!")

except Exception as e:
    import traceback
    print("Error during test:")
    traceback.print_exc()
