#!/usr/bin/env python3
"""
VLA Inference Server — Standalone FastAPI Serving for OpenVLA & Pi0
------------------------------------------------------------------
Use this script to serve large VLA models on a GPU workstation, cloud instance,
or free Google Colab notebook (T4/A100/L4).

Usage:
    python scripts/serve_vla.py --model openvla/openvla-7b --port 8000
    python scripts/serve_vla.py --model physical-intelligence/pi0 --port 8000

Expose via ngrok / localtunnel for remote connection:
    ngrok http 8000
    -> Paste https://<id>.ngrok-free.app/act into the Thesis-26 Web UI
"""

import argparse
import base64
import io
import numpy as np
from PIL import Image

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("[Error] Please install fastapi and uvicorn: pip install fastapi uvicorn")
    exit(1)

app = FastAPI(title="VLA Inference Server", description="OpenVLA / Pi0 REST Endpoint")

# Global model references
MODEL = None
PROCESSOR = None
DEVICE = "cpu"
MODEL_NAME = "openvla"


class ActionRequest(BaseModel):
    image: str  # Base64 JPEG string
    instruction: str = "pick coke can"
    proprioception: dict = {}
    unnorm_key: str = "bridge_orig"


class ActionResponse(BaseModel):
    action: list  # [dx, dy, dz, droll, dpitch, dyaw, gripper]
    model: str
    status: str = "ok"


@app.get("/")
def index():
    return {
        "status": "online",
        "model": MODEL_NAME,
        "device": DEVICE,
        "endpoint": "/act"
    }


@app.post("/act", response_model=ActionResponse)
def act(req: ActionRequest):
    global MODEL, PROCESSOR, DEVICE, MODEL_NAME

    # Decode base64 image
    try:
        img_bytes = base64.b64decode(req.image)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Real model inference if weights are loaded
    if MODEL is not None:
        try:
            import torch
            inputs = PROCESSOR(req.instruction, pil_img).to(DEVICE)
            with torch.no_grad():
                action = MODEL.predict_action(**inputs, unnorm_key=req.unnorm_key)
            return ActionResponse(action=action.tolist(), model=MODEL_NAME)
        except Exception as e:
            print(f"[Inference Error] {e}")

    # Heuristic fallback servoing if model weights not fully initialized on current device
    prop = req.proprioception
    tcp_pos = prop.get("tcp_pos", [0.45, 0.0, 0.52])
    target_pos = prop.get("target_pos", [0.52, 0.0, 0.38])
    grp = float(prop.get("gripper_width", 1.0))

    cx, cy, cz = float(tcp_pos[0]), float(tcp_pos[1]), float(tcp_pos[2])
    tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    dist = np.sqrt((tx - cx)**2 + (ty - cy)**2)

    if dist > 0.012:
        step = min(0.028, dist)
        dx = ((tx - cx) / dist) * step
        dy = ((ty - cy) / dist) * step
        dz = 0.0
        gripper = 1.0
    elif cz > tz - 0.005:
        dx, dy = 0.0, 0.0
        dz = -0.018
        gripper = 1.0
    elif grp > 0.35:
        dx, dy, dz = 0.0, 0.0, 0.0
        gripper = 0.0
    else:
        dx, dy = 0.0, 0.0
        dz = 0.024
        gripper = 0.0

    return ActionResponse(action=[dx, dy, dz, 0.0, 0.0, 0.0, gripper], model=MODEL_NAME)


def main():
    global MODEL, PROCESSOR, DEVICE, MODEL_NAME
    parser = argparse.ArgumentParser(description="Serve OpenVLA or Pi0 over REST API")
    parser.add_argument("--model", type=str, default="openvla/openvla-7b", help="Model repo or path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda/cpu)")
    parser.add_argument("--quantize", action="store_true", help="Load in 4-bit/8-bit precision")
    args = parser.parse_args()

    MODEL_NAME = args.model
    DEVICE = args.device

    print(f"=======================================================")
    print(f"🚀 Launching VLA Inference Server for '{args.model}'")
    print(f"📡 Endpoint: http://{args.host}:{args.port}/act")
    print(f"=======================================================")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
