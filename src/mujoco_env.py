#!/usr/bin/env python3
"""
MuJoCo Robotic Environment for VLA Grasp Evaluation
---------------------------------------------------
Simulates a 7-DoF Franka Emika Panda robot with parallel gripper,
offscreen rendering, Jacobian-based IK controller, and physical metrics logging.
"""
import os
import time
import threading
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None




OBJECT_CONFIGS = {
    "can": {
        "name": "can",
        "label": "Campbell Tomato Soup Can",
        "body_name": "target_object",
        "joint_name": "target_joint",
        "geom_name": "target_geom",
        "color": "#e11d48",
        "shape": "can"
    },
    "mustard": {
        "name": "mustard",
        "label": "French's Mustard Bottle",
        "body_name": "mustard_object",
        "joint_name": "mustard_joint",
        "geom_name": "mustard_geom",
        "color": "#eab308",
        "shape": "bottle"
    },
    "blue_box": {
        "name": "blue_box",
        "label": "Cobalt Blue Box",
        "body_name": "blue_box_object",
        "joint_name": "blue_box_joint",
        "geom_name": "blue_box_geom",
        "color": "#2563eb",
        "shape": "box"
    },
    "red_cylinder": {
        "name": "red_cylinder",
        "label": "Ruby Red Cylinder",
        "body_name": "red_cylinder_object",
        "joint_name": "red_cylinder_joint",
        "geom_name": "red_cylinder_geom",
        "color": "#dc2626",
        "shape": "cylinder"
    },
    "green_cube": {
        "name": "green_cube",
        "label": "Emerald Green Cube",
        "body_name": "green_cube_object",
        "joint_name": "green_cube_joint",
        "geom_name": "green_cube_geom",
        "color": "#16a34a",
        "shape": "cube"
    }
}


class MujocoRoboticEnv:
    """
    Standard MuJoCo Physics Environment for Franka Emika Panda Pick-and-Place tasks
    with full 7-DoF CAD meshes, parallel gripper, and diverse physical objects.
    """
    def __init__(
        self,
        scene_path="assets/franka_table_scene.xml",
        render_width=640,
        render_height=480,
        substeps_per_action=20
    ):
        if mujoco is None:
            raise ImportError(
                "MuJoCo is not installed in the current Python environment. "
                "Please install via: pip install mujoco"
            )

        self.scene_path = os.path.abspath(scene_path)
        if not os.path.exists(self.scene_path):
            raise FileNotFoundError(f"Scene XML file not found at: {self.scene_path}")

        # Read XML content in Python with UTF-8 to prevent path encoding issues in C++
        with open(self.scene_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

        # Load MuJoCo Model & Data
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)

        self.render_width = render_width
        self.render_height = render_height
        self.substeps_per_action = substeps_per_action

        # Thread-local storage for offscreen renderer to isolate OpenGL/GLFW/WGL context per thread
        self._local_tls = threading.local()

        # Body, Geom, and Site IDs
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "panda_tcp")
        self.basket_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "basket")

        # Diverse objects map
        self.object_configs = OBJECT_CONFIGS
        self.object_ids = {}
        for k, cfg in self.object_configs.items():
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg["body_name"])
            j_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, cfg["joint_name"])
            g_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, cfg["geom_name"])
            self.object_ids[k] = {
                "body_id": b_id,
                "joint_id": j_id,
                "geom_id": g_id
            }

        # Active target for Pick-and-Place
        self.active_target_key = "can"
        self.target_body_id = self.object_ids["can"]["body_id"]
        self.target_geom_id = self.object_ids["can"]["geom_id"]

        # Initial table surface level
        self.table_surface_z = 0.37
        self.reset()

    @property
    def renderer(self):
        """Thread-isolated MuJoCo classic renderer guaranteeing WGL context safety across threads."""
        if not hasattr(self._local_tls, "renderer") or self._local_tls.renderer is None:
            self._local_tls.renderer = mujoco.Renderer(
                self.model, height=self.render_height, width=self.render_width
            )
        return self._local_tls.renderer

    def set_active_target(self, obj_key):
        """Sets the active object to be manipulated cleanly without biased alias heuristics."""
        k = str(obj_key).lower().strip()
        canonical_map = {
            "can": "can", "soup": "can", "soup_can": "can",
            "mustard": "mustard", "mustard_bottle": "mustard",
            "blue_box": "blue_box", "box": "blue_box",
            "red_cylinder": "red_cylinder", "cylinder": "red_cylinder",
            "green_cube": "green_cube", "cube": "green_cube"
        }
        mapped_key = canonical_map.get(k, k)
        if mapped_key in self.object_ids:
            self.active_target_key = mapped_key
            self.target_body_id = self.object_ids[mapped_key]["body_id"]
            self.target_geom_id = self.object_ids[mapped_key]["geom_id"]
            return True, mapped_key
        return False, self.active_target_key

    def reset(self, target_pos=None):
        """
        Resets simulation state to home keyframe and settles physics.
        """
        mujoco.mj_resetData(self.model, self.data)

        # Set to home keyframe if available
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            # Default home joint positions for official Menagerie Franka
            self.data.qpos[:7] = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
            self.data.qpos[7:9] = [0.04, 0.04]  # Gripper open

        # Set specific active target object position if requested
        if target_pos is not None:
            active_cfg = self.object_configs.get(self.active_target_key, self.object_configs["can"])
            jnt_name = active_cfg["joint_name"]
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[adr:adr+3] = target_pos
                self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]

        # Settle physics for 80 steps with steady home posture control
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
        self.data.ctrl[7:9] = self.data.qpos[7:9]
        for _ in range(80):
            self.data.ctrl[:7] = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
            mujoco.mj_step(self.model, self.data)
        self.data.qvel[:7] = 0.0
        self.data.ctrl[:7] = np.copy(self.data.qpos[:7])

        return self.get_proprioception()

    def get_tcp_pos(self):
        """Returns 3D Cartesian position of the Tool Center Point."""
        return np.copy(self.data.site_xpos[self.tcp_site_id])

    def get_target_pos(self, target_key=None):
        """Returns 3D Cartesian position of the specified or active target object."""
        k = target_key or self.active_target_key
        if k in self.object_ids and self.object_ids[k]["body_id"] >= 0:
            return np.copy(self.data.xpos[self.object_ids[k]["body_id"]])
        return np.copy(self.data.xpos[self.target_body_id])

    def get_basket_pos(self):
        """Returns 3D Cartesian position of the receptacle basket."""
        if hasattr(self, "basket_body_id") and self.basket_body_id >= 0:
            return np.copy(self.data.xpos[self.basket_body_id])
        return np.array([0.45, -0.22, 0.37], dtype=np.float32)

    def get_all_objects_info(self):
        """Returns dictionary of all objects with positions, labels, and colors."""
        res = {}
        for k, cfg in self.object_configs.items():
            bid = self.object_ids[k]["body_id"]
            pos = np.copy(self.data.xpos[bid]).tolist() if bid >= 0 else [0.0, 0.0, 0.0]
            res[k] = {
                "name": cfg["name"],
                "label": cfg["label"],
                "color": cfg["color"],
                "shape": cfg["shape"],
                "pos": [round(p, 4) for p in pos],
                "is_active": (k == self.active_target_key)
            }
        # Include basket
        res["basket"] = {
            "name": "basket",
            "label": "Receptacle Basket",
            "color": "#3b82f6",
            "shape": "tray",
            "pos": [round(p, 4) for p in self.get_basket_pos().tolist()],
            "is_active": False
        }
        return res

    def check_basket_containment(self, target_key=None):
        """Checks whether the specified or active target object has been deposited into the basket."""
        target_pos = self.get_target_pos(target_key)
        basket_pos = self.get_basket_pos()
        dist_xy = float(np.linalg.norm(target_pos[:2] - basket_pos[:2]))
        is_low_enough = bool(target_pos[2] <= basket_pos[2] + 0.075)
        return bool(dist_xy < 0.09 and is_low_enough)

    def reposition_target(self, x=None, y=None, z=None):
        """Randomizes or sets the active target object position on the table surface."""
        rng = np.random.RandomState()
        rx = x if x is not None else float(rng.uniform(0.42, 0.60))
        ry = y if y is not None else float(rng.uniform(-0.16, 0.22))
        rz = z if z is not None else 0.40

        active_cfg = self.object_configs.get(self.active_target_key, self.object_configs["can"])
        jnt_id = self.object_ids.get(self.active_target_key, {}).get("joint_id", -1)
        if jnt_id >= 0:
            adr = self.model.jnt_qposadr[jnt_id]
            self.data.qpos[adr:adr+3] = [rx, ry, rz]
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            v_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[v_adr:v_adr+6] = 0.0

            if x is not None and y is not None:
                mujoco.mj_forward(self.model, self.data)
            else:
                arm_qpos = np.copy(self.data.qpos[:7])
                gripper_ctrl = np.copy(self.data.ctrl[7:9])
                for _ in range(30):
                    self.data.qpos[:7] = arm_qpos
                    self.data.qvel[:7] = 0.0
                    self.data.ctrl[:7] = arm_qpos
                    self.data.ctrl[7:9] = gripper_ctrl
                    self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
                    mujoco.mj_step(self.model, self.data)

        return self.get_target_pos().tolist()

    def reposition_all_objects(self, seed=None):
        """
        Randomizes the placement of all diverse objects across the widened table workspace
        with wide spacing without mutual collisions or falling off the table surface.
        Keeps robot arm configuration rigidly locked and stationary.
        """
        rng = np.random.RandomState(seed)
        placed_positions = []
        min_distance = 0.13  # Minimum 13cm spacing between objects

        # Freeze current robot arm joint state and gripper posture
        arm_qpos = np.copy(self.data.qpos[:7])
        gripper_ctrl = np.copy(self.data.ctrl[7:9])

        for k in self.object_configs.keys():
            jnt_id = self.object_ids[k]["joint_id"]
            if jnt_id < 0:
                continue

            # Find collision-free spot on table
            cand_x, cand_y = 0.50, 0.0
            for _ in range(60):
                cx = float(rng.uniform(0.40, 0.62))
                cy = float(rng.uniform(-0.16, 0.24))
                # Check distance against basket
                basket_pos = self.get_basket_pos()
                if np.linalg.norm([cx - basket_pos[0], cy - basket_pos[1]]) < 0.15:
                    continue
                # Check distance against already placed objects
                too_close = False
                for px, py in placed_positions:
                    if np.linalg.norm([cx - px, cy - py]) < min_distance:
                        too_close = True
                        break
                if not too_close:
                    cand_x, cand_y = cx, cy
                    break

            placed_positions.append((cand_x, cand_y))
            adr = self.model.jnt_qposadr[jnt_id]
            # Table surface is 0.37m, resting height adjusted per object mesh minimum z
            z_map = {
                "can": 0.420,
                "mustard": 0.449,
                "blue_box": 0.400,
                "red_cylinder": 0.408,
                "green_cube": 0.390
            }
            cz = z_map.get(k, 0.400)
            self.data.qpos[adr:adr+3] = [cand_x, cand_y, cz]
            self.data.qpos[adr+3:adr+7] = [1, 0, 0, 0]
            v_adr = self.model.jnt_dofadr[jnt_id]
            self.data.qvel[v_adr:v_adr+6] = 0.0

        # Settle physics for 60 steps with arm rigidly clamped to prevent gravity sagging
        for _ in range(60):
            self.data.qpos[:7] = arm_qpos
            self.data.qvel[:7] = 0.0
            self.data.ctrl[:7] = arm_qpos
            self.data.ctrl[7:9] = gripper_ctrl
            self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
            mujoco.mj_step(self.model, self.data)

        return self.get_all_objects_info()

    def render_dual_cameras_b64(self, quality=80):
        """
        Renders both Top/Overhead camera and Wrist camera offscreen directly from MuJoCo,
        encoding them as Base64 JPEG data URLs for ultra-low latency real-time web streaming.
        """
        import cv2
        import base64

        # 1. Top Camera (Overhead view)
        self.renderer.update_scene(self.data, camera="overhead_cam")
        top_rgb = self.renderer.render()
        top_bgr = cv2.cvtColor(top_rgb, cv2.COLOR_RGB2BGR)
        _, top_buf = cv2.imencode('.jpg', top_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        top_b64 = base64.b64encode(top_buf).decode('utf-8')

        # 2. Wrist Camera (End-effector view)
        self.renderer.update_scene(self.data, camera="wrist_cam")
        wrist_rgb = self.renderer.render()
        wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
        _, wrist_buf = cv2.imencode('.jpg', wrist_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        wrist_b64 = base64.b64encode(wrist_buf).decode('utf-8')

        return {
            "top": f"data:image/jpeg;base64,{top_b64}",
            "wrist": f"data:image/jpeg;base64,{wrist_b64}"
        }

    def get_proprioception(self):
        """
        Returns full proprioceptive state dictionary.
        """
        tcp_pos = self.get_tcp_pos()
        gripper_w = self.data.qpos[7] + self.data.qpos[8] # Total finger spacing
        joint_positions = np.copy(self.data.qpos[:7])
        target_pos = self.get_target_pos()
        basket_pos = self.get_basket_pos()
        all_objects = self.get_all_objects_info()

        return {
            "tcp_pos": tcp_pos.tolist(),
            "target_pos": target_pos.tolist(),
            "basket_pos": basket_pos.tolist(),
            "active_target": self.active_target_key,
            "objects": all_objects,
            "gripper_pos": tcp_pos.tolist(), # 3D Cartesian coordinates [x, y, z]
            "gripper_pos_2d": [float(tcp_pos[0]), float(tcp_pos[1])], # 2D projected coordinates
            "gripper_width": float(np.clip(gripper_w / 0.08, 0.0, 1.0)), # Normalized 0..1
            "gripper_width_mm": float(round(gripper_w * 1000.0, 1)), # in millimeters (0..80mm)
            "joint_positions": joint_positions.tolist()
        }

    def compute_ik_step(self, target_dx, target_dy, target_dz, target_droll=0.0, target_dpitch=0.0, target_dyaw=0.0, damping=0.05):
        """
        Computes 7-DoF joint delta to achieve Cartesian end-effector delta using Damped Jacobian Pseudo-Inverse
        for translation, combined with direct smooth wrist yaw rotation (Joint 7) and Nullspace Posture Regularization.
        """
        # Calculate End-Effector Jacobian (3x7 translational, well-conditioned)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.tcp_site_id)
        J = jacp[:, :7]  # 3x7 arm joints

        delta_x = np.array([target_dx, target_dy, target_dz], dtype=np.float64)

        # Damped Least Squares (DLS): J_dls = J^T * (J * J^T + lambda^2 * I)^-1
        lambda_sq = damping ** 2
        JJT = J @ J.T + lambda_sq * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)
        dq_task = J_pinv @ delta_x

        # Nullspace posture regularization: keep elbow and wrist pointing downward in home posture
        q_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        q_current = np.copy(self.data.qpos[:7])
        q_err = q_home - q_current
        q_err[6] = 0.0  # Allow wrist rotation free compliance without nullspace resistance

        nullspace_proj = np.eye(7) - J_pinv @ J
        dq_null = nullspace_proj @ (0.05 * q_err)

        dq = dq_task + dq_null

        # Apply commanded wrist yaw rotation directly to Joint 7 (wrist rotation axis)
        dq[6] += float(target_dyaw)

        # Smooth velocity limiter to prevent any sudden joint jerks
        max_joint_step = 0.05
        scale = float(np.max(np.abs(dq)) / max_joint_step)
        if scale > 1.0:
            dq = dq / scale

        return dq

    def step(self, action_7dof):
        """
        Executes a 7-DoF Cartesian action step with smooth closed-loop joint tracking.
        
        Args:
            action_7dof: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
                         dx, dy, dz in meters.
                         droll, dpitch, dyaw in radians.
                         gripper_cmd: 0.0 (closed) to 1.0 (open).
        """
        if len(action_7dof) < 7:
            padded = list(action_7dof) + [0.0] * (7 - len(action_7dof))
            action_7dof = padded

        dx = float(action_7dof[0])
        dy = float(action_7dof[1])
        dz = float(action_7dof[2])
        droll = float(action_7dof[3])
        dpitch = float(action_7dof[4])
        dyaw = float(action_7dof[5])
        gripper_cmd = float(np.clip(action_7dof[6], 0.0, 1.0))

        # 1. Compute 7-DoF joint position delta
        dq = self.compute_ik_step(dx, dy, dz, droll, dpitch, dyaw)

        # 2. Update target joint positions from current joint state
        target_q = self.data.qpos[:7] + dq

        # Clip within joint limits
        for i in range(7):
            limits = self.model.jnt_range[i]
            target_q[i] = np.clip(target_q[i], limits[0], limits[1])

        # 3. Apply position control commands
        self.data.ctrl[:7] = target_q
        # Finger position: 0.04m is fully open per finger, 0.00m is closed
        finger_target = gripper_cmd * 0.04
        self.data.ctrl[7] = finger_target
        self.data.ctrl[8] = finger_target

        # 4. Step physics simulation forward with active gravity compensation
        for _ in range(self.substeps_per_action):
            self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
            mujoco.mj_step(self.model, self.data)
            # Physical stroke limit clamp [0.0m, 0.04m] to prevent penetration
            self.data.qpos[7:9] = np.clip(self.data.qpos[7:9], 0.0, 0.04)

        # Return observation and metrics
        obs = {
            "proprioception": self.get_proprioception(),
            "metrics": self.get_metrics()
        }
        return obs

    def return_to_home_step(self, alpha=0.15):
        """
        Smoothly interpolates the robot arm joints back towards the initial home keyframe
        and opens the gripper fully.
        Returns:
            True when the arm is within 0.03 rad of home posture and gripper is open.
        """
        q_home = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
        current_q = np.copy(self.data.qpos[:7])
        diff = q_home - current_q
        step_dq = np.clip(diff * alpha, -0.04, 0.04)
        target_q = current_q + step_dq

        self.data.ctrl[:7] = target_q
        self.data.ctrl[7:9] = [0.04, 0.04]  # Full open stroke

        for _ in range(self.substeps_per_action):
            self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
            mujoco.mj_step(self.model, self.data)
            self.data.qpos[7:9] = np.clip(self.data.qpos[7:9], 0.0, 0.04)

        max_err = float(np.max(np.abs(q_home - self.data.qpos[:7])))
        gripper_open = bool(self.data.qpos[7] >= 0.032)
        return bool(max_err < 0.03 and gripper_open)

    def render_camera(self, camera_name="overhead_cam"):
        """
        Renders an RGB image from the specified camera.
        Returns: numpy array of shape (H, W, 3), dtype=np.uint8.
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            # Fallback to free camera
            self.renderer.update_scene(self.data)
        else:
            self.renderer.update_scene(self.data, camera=camera_name)

        rgb = self.renderer.render()
        return rgb

    def render_camera_b64(self, camera_name="overhead_cam", quality=80, overlay_image=None):
        """
        Renders specified camera or uses overlay_image, encodes as JPEG Base64 data URL string.
        """
        import cv2
        import base64
        
        if overlay_image is not None:
            rgb_img = overlay_image
        else:
            rgb_img = self.render_camera(camera_name=camera_name)
            
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        jpg_b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{jpg_b64}"

    def get_web_telemetry(self):
        """
        Extracts telemetry payload formatted specifically for Web 3D Digital Twin (Three.js).
        """
        joint_positions = np.copy(self.data.qpos[:7]).tolist()
        finger_pos = float(self.data.qpos[7]) # finger slide position
        tcp_pos = self.get_tcp_pos().tolist()
        target_pos = self.get_target_pos().tolist()
        metrics = self.get_metrics()
        
        return {
            "qpos": joint_positions,
            "finger": finger_pos,
            "tcp_pos": tcp_pos,
            "target_pos": target_pos,
            "metrics": metrics
        }
        
    def project_point_to_camera(self, point_3d, camera_name="overhead_cam"):
        """
        Projects a 3D Cartesian world point into 2D camera pixel coordinates (u, v).
        Returns: (u, v) float tuple or None if behind camera.
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            return 320.0, 240.0
            
        cam_pos = self.data.cam_xpos[cam_id]
        cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3)
        
        # Transform to camera coordinate frame (camera looks along -Z)
        p_cam = cam_mat.T @ (np.array(point_3d) - cam_pos)
        depth = -p_cam[2]
        if depth <= 0.01:
            return None
            
        fovy_rad = np.radians(self.model.cam_fovy[cam_id])
        f_y = (self.render_height / 2.0) / np.tan(fovy_rad / 2.0)
        f_x = f_y
        
        c_x = self.render_width / 2.0
        c_y = self.render_height / 2.0
        
        u = c_x + (p_cam[0] / depth) * f_x
        v = c_y - (p_cam[1] / depth) * f_y
        
        return float(np.clip(u, 10.0, self.render_width - 10.0)), float(np.clip(v, 10.0, self.render_height - 10.0))

    def pixel_to_world_on_plane(self, u, v, plane_z=0.40, camera_name="overhead_cam"):
        """
        Inverses a 2D camera pixel coordinate (u, v) by casting a 3D ray from the camera optical center
        and finding its intersection with the horizontal workspace plane at z = plane_z.
        Enables 100% perception-driven targeting without reading ground-truth simulation coordinates.
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            return np.array([0.48, 0.0, plane_z], dtype=np.float32)

        cam_pos = np.copy(self.data.cam_xpos[cam_id])
        cam_mat = np.copy(self.data.cam_xmat[cam_id].reshape(3, 3))

        fovy_rad = np.radians(self.model.cam_fovy[cam_id])
        f_y = (self.render_height / 2.0) / np.tan(fovy_rad / 2.0)
        f_x = f_y

        c_x = self.render_width / 2.0
        c_y = self.render_height / 2.0

        x_c = (float(u) - c_x) / f_x
        y_c = -(float(v) - c_y) / f_y
        ray_cam = np.array([x_c, y_c, -1.0])
        ray_world = cam_mat @ ray_cam
        norm = np.linalg.norm(ray_world)
        if norm > 1e-6:
            ray_world = ray_world / norm

        if abs(ray_world[2]) < 1e-5:
            return np.array([0.48, 0.0, plane_z], dtype=np.float32)

        lambd = (plane_z - cam_pos[2]) / ray_world[2]
        world_pt = cam_pos + lambd * ray_world
        return np.array([float(world_pt[0]), float(world_pt[1]), float(plane_z)], dtype=np.float32)



    def check_occlusion(self):
        """
        Checks whether the target object is occluded by the gripper end-effector hovering directly above it.
        """
        target_pos = self.get_target_pos()
        tcp_pos = self.get_tcp_pos()
        dist_xy = float(np.linalg.norm(target_pos[:2] - tcp_pos[:2]))
        is_under_gripper = (dist_xy < 0.045) and (tcp_pos[2] > target_pos[2])
        return bool(is_under_gripper)

    def get_metrics(self):
        """
        Extracts physical evaluation metrics:
        - lift_height_meters: Height above table
        - max_contact_force_n: Normal contact forces on target
        - distance_to_target: Distance from TCP to target
        - is_grasped: Successful grasp & lift flag
        """
        target_pos = self.get_target_pos()
        tcp_pos = self.get_tcp_pos()

        lift_height = float(max(0.0, target_pos[2] - self.table_surface_z - 0.02))
        dist_to_target = float(np.linalg.norm(tcp_pos - target_pos))

        # Calculate contact forces on target object
        total_contact_force = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Check if contact involves target_geom
            if contact.geom1 == self.target_geom_id or contact.geom2 == self.target_geom_id:
                # Approximate normal force
                c_array = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, c_array)
                total_contact_force += float(np.linalg.norm(c_array[:3]))

        is_grasped = bool(lift_height > 0.03 and self.data.qpos[7] < 0.025)

        return {
            "lift_height_meters": round(lift_height, 4),
            "max_contact_force_n": round(total_contact_force, 2),
            "distance_to_target_meters": round(dist_to_target, 4),
            "is_grasped": is_grasped,
            "is_occluded": self.check_occlusion()
        }


if __name__ == "__main__":
    print("[MujocoRoboticEnv] Testing standalone environment initialization...")
    env = MujocoRoboticEnv()
    print("TCP Position:", env.get_tcp_pos())
    print("Target Position:", env.get_target_pos())
    frame = env.render_camera("overhead_cam")
    print("Rendered RGB frame shape:", frame.shape)
    metrics = env.get_metrics()
    print("Initial Metrics:", metrics)
    print("[MujocoRoboticEnv] Environment test passed successfully!")
