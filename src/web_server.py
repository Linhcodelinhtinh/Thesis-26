#!/usr/bin/env python3
"""
Web Visualization & Digital Twin Server
---------------------------------------
Serves the HTML5/Three.js Web Dashboard and streams real-time MuJoCo off-screen
camera views (encoded as Base64 JPEG with Painter overlays) and 3D telemetry
data via Server-Sent Events (SSE).
"""
import os
import sys
import time
import json
import base64
import threading
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mujoco_env import MujocoRoboticEnv
from chronicler import Chronicler
from spatial_tracker import SpatialTracker
from painter import Painter
from sync_layer import SyncLayer
from vla_wrapper import MockVLAWrapper, OctoWrapper, OpenVLAWrapper


class RoboticWebServer:
    def __init__(self, host="0.0.0.0", port=8080, config_path="configs/config.yaml"):
        self.host = host
        self.port = port
        self.config_path = config_path
        self.web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
        self.sim_lock = threading.Lock()

        # Initialize Physics & Memory components
        print("[WebServer] Initializing MuJoCo Physics Environment...", flush=True)
        self.env = MujocoRoboticEnv()
        self.sync_layer = SyncLayer()
        self.painter = Painter()

        # Selection of VLA Wrapper
        self.vla_name = "MockVLA (Fast Controller)"
        self.vla = MockVLAWrapper()
        try:
            if os.path.exists(config_path):
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if cfg.get("vla_model") == "octo":
                        try:
                            self.vla = OctoWrapper()
                            self.vla_name = "Octo-Small (VLA)"
                        except Exception as e:
                            print(f"[WebServer] Octo model not loaded ({e}), using MockVLA.")
        except Exception as e:
            print(f"[WebServer] Config load exception ({e}), using default MockVLA.")


        self.tracker = SpatialTracker(tracker_type="opencv_fallback")
        self.chronicler = Chronicler(use_real_vlm=False)

        # Global Control State
        self.running = False
        self.active_camera = "overhead_cam"
        self.human_command = "Grasp the orange object"
        self.last_chronicler_log = ""
        self.clients_lock = threading.Lock()
        
        # Initial Target Placement using exact 3D to 2D Camera Projection
        raw_frame = self.env.render_camera(self.active_camera)
        proj = self.env.project_point_to_camera(self.env.get_target_pos(), self.active_camera)
        target_pos_2d = list(proj) if proj else [320.0, 240.0]
        self.tracker.initialize_tracking_points(raw_frame, target_pos_2d)

    def run_simulation_step(self):
        """
        Executes one physics & control iteration, updates Painter overlays and Chronicler state.
        """
        with self.sim_lock:
            # 1. Fetch MuJoCo Camera Frame
            raw_frame = self.env.render_camera(self.active_camera)

            # 2. Get Proprioception & Track Points
            prop = self.env.get_proprioception()
            is_occluded = self.env.check_occlusion()
            pts, flags = self.tracker.track(raw_frame, gripper_pos=prop.get("gripper_pos_2d", prop["gripper_pos"]))
            if is_occluded:
                flags = [True] * len(flags)

            # 3. Update Chronicler State
            tracking_info = {
                "target_pos": prop["target_pos"],
                "is_occluded": is_occluded
            }
            dynamic_prompt = self.chronicler.update_state(raw_frame, self.human_command, prop, tracking_info)
            self.sync_layer.update_prompt(dynamic_prompt)
            self.last_chronicler_log = dynamic_prompt

            # 4. Apply Painter Visual Overlay
            _, latest_prompt = self.sync_layer.get_latest_state()
            painted_frame = self.painter.draw_overlay(raw_frame, pts, flags, latest_prompt or "")
            self.sync_layer.update_image(painted_frame)

            # 5. Query VLA Model & Step Physics Simulation
            action = self.vla.predict_action(painted_frame, latest_prompt or "", prop)
            self.env.step(action)

            # Return Web Streaming Package
            frame_b64 = self.env.render_camera_b64(overlay_image=painted_frame, quality=75)
            telemetry = self.env.get_web_telemetry()

            return {
                "vla_model": self.vla_name,
                "frame_b64": frame_b64,
                "dynamic_prompt": latest_prompt,
                "telemetry": telemetry,
                "chronicler_log": self.last_chronicler_log
            }

    def start(self):
        self.running = True

        # Custom Request Handler class
        server_instance = self

        class DashboardRequestHandler(BaseHTTPRequestHandler):
            def handle(self):
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                    pass

            def log_message(self, format, *args):
                return  # Suppress default noisy HTTP logging

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/stream":
                    # Server-Sent Events (SSE) Streaming Endpoint
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "keep-alive")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

                    try:
                        while server_instance.running:
                            payload = server_instance.run_simulation_step()
                            data_str = f"data: {json.dumps(payload)}\n\n"
                            self.wfile.write(data_str.encode("utf-8"))
                            self.wfile.flush()
                            time.sleep(0.066)  # ~15-20 FPS stream rate
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                        pass

                else:
                    # Serve Static Frontend Files (web/index.html, style.css, app.js)
                    if path == "/" or path == "/index.html":
                        filepath = os.path.join(server_instance.web_dir, "index.html")
                        content_type = "text/html"
                    else:
                        filepath = os.path.join(server_instance.web_dir, path.lstrip("/"))
                        if filepath.endswith(".css"):
                            content_type = "text/css"
                        elif filepath.endswith(".js"):
                            content_type = "application/javascript"
                        elif filepath.endswith(".png"):
                            content_type = "image/png"
                        elif filepath.endswith(".jpg") or filepath.endswith(".jpeg"):
                            content_type = "image/jpeg"
                        else:
                            content_type = "text/plain"

                    if os.path.exists(filepath) and os.path.isfile(filepath):
                        self.send_response(200)
                        self.send_header("Content-Type", content_type)
                        self.end_headers()
                        with open(filepath, "rb") as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_error(404, "File Not Found")

            def do_POST(self):
                content_length = int(self.headers.get("Content-Length", 0))
                body_bytes = self.rfile.read(content_length) if content_length > 0 else b""
                
                try:
                    body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                except Exception:
                    body = {}

                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/api/command":
                    cmd = body.get("command", "Grasp the orange object")
                    server_instance.human_command = cmd
                    server_instance.chronicler.reset()
                    res = {"status": "ok", "command": cmd}

                elif path == "/api/reset":
                    with server_instance.sim_lock:
                        server_instance.env.reset()
                        server_instance.chronicler.reset()
                        raw_frame = server_instance.env.render_camera(server_instance.active_camera)
                        proj = server_instance.env.project_point_to_camera(server_instance.env.get_target_pos(), server_instance.active_camera)
                        server_instance.tracker.initialize_tracking_points(raw_frame, list(proj) if proj else [320.0, 240.0])
                    res = {"status": "ok", "message": "Simulation reset to home keyframe."}

                elif path == "/api/move_target":
                    # Randomize target object position on the table surface
                    import numpy as np
                    new_x = float(np.random.uniform(0.40, 0.60))
                    new_y = float(np.random.uniform(-0.18, 0.18))
                    new_target = [new_x, new_y, 0.40]
                    with server_instance.sim_lock:
                        server_instance.env.reset(target_pos=new_target)
                        server_instance.chronicler.reset()
                        raw_frame = server_instance.env.render_camera(server_instance.active_camera)
                        proj = server_instance.env.project_point_to_camera(new_target, server_instance.active_camera)
                        server_instance.tracker.initialize_tracking_points(raw_frame, list(proj) if proj else [320.0, 240.0])
                    res = {"status": "ok", "target_pos": new_target}

                elif path == "/api/set_camera":
                    query = urllib.parse.parse_qs(parsed.query)
                    cam_name = query.get("name", ["overhead_cam"])[0]
                    with server_instance.sim_lock:
                        server_instance.active_camera = cam_name
                        raw_frame = server_instance.env.render_camera(cam_name)
                        proj = server_instance.env.project_point_to_camera(server_instance.env.get_target_pos(), cam_name)
                        server_instance.tracker.initialize_tracking_points(raw_frame, list(proj) if proj else [320.0, 240.0])
                    res = {"status": "ok", "active_camera": cam_name}

                else:
                    res = {"error": "Invalid API endpoint"}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))

        self.httpd = ThreadingHTTPServer((self.host, self.port), DashboardRequestHandler)
        print(f"\n=======================================================", flush=True)
        print(f"  VLA Web Dashboard & 3D Digital Twin Active!", flush=True)
        print(f"  Local Access: http://localhost:{self.port}", flush=True)
        print(f"  Network Access: http://{self.host}:{self.port}", flush=True)
        print(f"=======================================================\n", flush=True)
        
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[WebServer] Stopping server...", flush=True)
            self.running = False
            self.httpd.server_close()


if __name__ == "__main__":
    server = RoboticWebServer(host="0.0.0.0", port=8080)
    server.start()
