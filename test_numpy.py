import sys
import traceback

with open("numpy_output.txt", "w", encoding="utf-8") as f:
    try:
        import mujoco
        f.write(f"Success: {mujoco.__file__}\n")
    except Exception as e:
        f.write(f"Error:\n{traceback.format_exc()}\n")
