import os
import sys

# Ensure paths
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("src"))

from src.web_server import RoboticWebServer, LIBERO_TIER_TASKS

print("="*60)
print("Testing RoboticWebServer LIBERO Mini-Suite Integration...")
print("="*60)

# Instantiate server without starting HTTP daemon
server = RoboticWebServer(port=8899)

assert server.env_mode == "libero", f"Expected env_mode libero, got {server.env_mode}"
assert server.libero_env is not None, "server.libero_env is None!"
print(f"[OK] Initialized LIBERO Tier {server.libero_tier} Task {server.libero_task_id}: {server.libero_task_name}")
print(f"     Instruction: '{server.libero_instruction}'")

# Test run_libero_step()
payload = server.run_libero_step()
assert payload is not None, "run_libero_step returned None!"
assert "frame_top_b64" in payload and payload["frame_top_b64"].startswith("data:image/jpeg;base64,"), "Missing valid frame_top_b64"
assert "frame_wrist_b64" in payload and payload["frame_wrist_b64"].startswith("data:image/jpeg;base64,"), "Missing valid frame_wrist_b64"
assert "telemetry" in payload, "Missing telemetry in payload"

tel = payload["telemetry"]
assert "tcp_pos" in tel and len(tel["tcp_pos"]) == 3, f"Invalid tcp_pos: {tel.get('tcp_pos')}"
assert "action_chunk" in tel, "Missing action_chunk"
assert "progress_pct" in tel, "Missing progress_pct"
assert "gripper_state" in tel, "Missing gripper_state"
print(f"[OK] Telemetry: TCP={tel['tcp_pos']}, Gripper={tel['gripper_state']} ({tel['gripper_mm']}mm), Progress={tel['progress_pct']}%, Status={tel['status']}")

# Test execution state
server.execute_libero("pick up the black bowl")
assert server.libero_status == "RUNNING"
server.stop_libero()
assert server.libero_status == "IDLE"
server.reset_libero()
assert server.libero_step_idx == 0
print(f"[OK] Control flow execute/stop/reset verified.")

# Test switching tier
ok = server.init_libero_task(tier=2, task_id=0)
assert ok, "Failed to switch to Tier 2 Task 0"
assert server.libero_tier == 2
print(f"[OK] Switched to Tier 2 Task 0: {server.libero_task_name}")

# Test switching back
ok = server.init_libero_task(tier=1, task_id=0)
assert ok, "Failed to switch back to Tier 1 Task 0"

# Clean up
if server.libero_env is not None:
    server.libero_env.close()

print("\n" + "="*60)
print("ALL LIBERO MINI-SUITE WEB INTEGRATION TESTS PASSED!")
print("="*60)
