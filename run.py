#!/usr/bin/env python3
"""
Root Unified Runner — Franka Emika Panda MuJoCo VLA Workspace
============================================================
Launches the Interactive Web Dashboard & Chat Console (Default)
or SimplerEnv Evaluation benchmark:

Examples:
  python run.py                     # Mở Web Dashboard tại http://localhost:8080
  python run.py --port 8080         # Khởi chạy trên cổng chỉ định
  python run.py --env simplerenv    # Đánh giá SimplerEnv benchmark
"""
import os
import sys

# Ensure src is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.main import main

if __name__ == "__main__":
    main()
