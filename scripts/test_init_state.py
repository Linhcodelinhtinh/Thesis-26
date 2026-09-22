from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = task_suite.get_task(0)
task_bddl = task_suite.get_task_bddl_file_path(0)
init_states = task_suite.get_task_init_states(0)
print(f"Init states shape: {init_states.shape}")

env = OffScreenRenderEnv(bddl_file_name=task_bddl, camera_heights=128, camera_widths=128)
env.reset()
print("Setting init state...")
obs = env.set_init_state(init_states[0])
print(f"Set init state success! Obs keys: {list(obs.keys())}")
env.close()
print("All good!")
