#!/usr/bin/env python3
"""
Native MuJoCo Interactive 3D Simulation & Evaluation Suite (CLI Shortcut)
========================================================================
Routes to src/mujoco_runner.py for native MuJoCo simulation.
Can also be launched directly via: python src/main.py --env mujoco
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.mujoco_runner import InteractiveMujocoSim, main

if __name__ == "__main__":
    main()
