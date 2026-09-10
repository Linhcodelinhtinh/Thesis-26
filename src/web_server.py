#!/usr/bin/env python3
"""
Web Visualization & Digital Twin Server for SimplerEnv
-------------------------------------------------------
Serves HTML5/Three.js Web Dashboard and streams real-time SimplerEnv camera views
(encoded as Base64 JPEG with Painter visual overlays) and robot telemetry
via Server-Sent Events (SSE).
"""
import os
import sys
import time
import json
import base64
import threading
import urllib.parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simpler_env_adapter import SimplerEnvAdapter
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

        # Load config
        task_name = "google_robot_pick_coke_can"
        vla_choice = "octo"
        try:
            if os.path.exists(config_path):
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    task_name = cfg.get("simpler_env", {}).get("task_name", task_name)
                    vla_choice = cfg.get("vla_model", "octo")
        except Exception:
            pass

        # Initialize SimplerEnv Physics & Memory components
        print(f"[WebServer] Initializing SimplerEnv for task '{task_name}'...", flush=True)
        self.task_name = task_name
        self.env = SimplerEnvAdapter(task_name=task_name, max_steps=80)
        self.sync_layer = SyncLayer()
        self.painter = Painter()

        if vla_choice == "octo":
            self.vla = OctoWrapper(action_chunk_size=4)
            self.vla_name = "Octo-Small (VLA + Chunking)"
        elif vla_choice == "openvla":
            self.vla = OpenVLAWrapper(action_chunk_size=1)
            self.vla_name = "OpenVLA-7B"
        else:
            self.vla = MockVLAWrapper(action_chunk_size=4)
            self.vla_name = "MockVLA (Fast Controller)"

        self.tracker = SpatialTracker(tracker_type="opencv_fallback")
        self.chronicler = Chronicler(use_real_vlm=False)

        # Global Control State
        self.running = False
        self.step_idx = 0
        self.task_status = "RUNNING"  # "RUNNING", "SUCCESS", "FAILED"
        self.last_chronicler_log = ""
        self.current_obs = None

        self._reset_sim()

    def _reset_sim(self):
        self.current_obs, _ = self.env.reset()
        self.step_idx = 0
        self.task_status = "RUNNING"
        self.chronicler.reset()
        self.vla.clear_chunk_buffer()

        raw_frame = self.current_obs["image"]
        h, w, _ = raw_frame.shape
        self.tracker.initialize_tracking_points(raw_frame, [w * 0.5, h * 0.6])

    def run_simulation_step(self):
        """
        Executes one physics & control iteration, updates Painter overlays and Chronicler state.
        """
        with self.sim_lock:
            self.step_idx += 1
            raw_frame = self.current_obs["image"]
            prop = self.current_obs["proprioception"]

            # 1. Tracker step
            pts, flags = self.tracker.track(raw_frame)

            # 2. Chronicler update
            tracking_info = {
                "target_pos": prop.get("target_pos", []),
                "is_occluded": self.task_status == "SUCCESS"
            }
            dynamic_prompt = self.chronicler.update_state(
                raw_frame, self.env.instruction, prop, tracking_info
            )
            self.sync_layer.update_prompt(dynamic_prompt)
            self.last_chronicler_log = dynamic_prompt

            # 3. Apply Painter visual overlay
            _, latest_prompt = self.sync_layer.get_latest_state()
            painted_frame = self.painter.draw_overlay(raw_frame, pts, flags, latest_prompt or "")
            self.sync_layer.update_image(painted_frame)

            # 4. Query VLA Model & Step SimplerEnv
            if self.task_status == "RUNNING":
                action = self.vla.step_policy(painted_frame, latest_prompt or "", prop)
                next_obs, reward, term, trunc, info = self.env.step(action)
                self.current_obs = next_obs

                if info.get("success", False):
                    self.task_status = "SUCCESS"
                elif trunc:
                    self.task_status = "FAILED"
            else:
                # Idle step if finished
                info = {"success": self.task_status == "SUCCESS"}

            # 5. Encode camera frame as JPEG Base64
            bgr_img = cv2.cvtColor(painted_frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            jpg_b64 = base64.b64encode(buffer).decode('utf-8')
            frame_b64 = f"data:image/jpeg;base64,{jpg_b64}"

            telemetry = {
                "qpos": prop.get("joint_positions", []),
                "tcp_pos": prop.get("tcp_pos", []),
                "gripper_width": prop.get("gripper_width", 1.0),
                "target_pos": prop.get("target_pos", []),
                "step": self.step_idx,
                "task_name": self.task_name,
                "instruction": self.env.instruction,
                "status": self.task_status,
                "is_success": self.task_status == "SUCCESS"
            }

            return {
                "vla_model": self.vla_name,
                "task_name": self.task_name,
                "instruction": self.env.instruction,
                "status": self.task_status,
                "step": self.step_idx,
                "frame_b64": frame_b64,
                "dynamic_prompt": latest_prompt,
                "telemetry": telemetry,
                "chronicler_log": self.last_chronicler_log
            }

    def start(self):
        self.running = True
        server_instance = self

        class DashboardRequestHandler(BaseHTTPRequestHandler):
            def handle(self):
                try:
                    super().handle()
                except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                    pass

            def log_message(self, format, *args):
                return

            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                path = parsed.path

                if path == "/stream":
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
                            time.sleep(0.08)  # ~12 FPS
                    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, OSError):
                        pass

                else:
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

                if path == "/api/reset":
                    with server_instance.sim_lock:
                        server_instance._reset_sim()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "message": "Simulation reset"}).encode("utf-8"))

                elif path == "/api/command":
                    cmd = body.get("command", "")
                    if cmd:
                        server_instance.env.instruction = cmd
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "ok", "instruction": server_instance.env.instruction}).encode("utf-8"))

                else:
                    self.send_error(404, "Endpoint Not Found")

        print(f"\n=======================================================")
        print(f"🚀 SimplerEnv Web Visualization Dashboard is LIVE at:")
        print(f"👉 http://localhost:{self.port}")
        print(f"=======================================================\n", flush=True)

        httpd = ThreadingHTTPServer((self.host, self.port), DashboardRequestHandler)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[WebServer] Stopping server...")
        finally:
            self.running = False
            httpd.server_close()


if __name__ == "__main__":
    port = 8080
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    server = RoboticWebServer(host="0.0.0.0", port=port)
    server.start()
