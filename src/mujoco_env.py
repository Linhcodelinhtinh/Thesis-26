#!/usr/bin/env python3
"""
MuJoCo Robotic Environment for VLA Grasp Evaluation
---------------------------------------------------
Simulates a 7-DoF Franka Emika Panda robot with parallel gripper,
offscreen rendering, Jacobian-based IK controller, and physical metrics logging.
"""
import os
import time
import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None




class MujocoRoboticEnv:
    """
    Standard MuJoCo Physics Environment for Franka Emika Panda Pick-and-Place tasks.
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

        # Offscreen renderer
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)

        # Body, Geom, and Site IDs
        self.tcp_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "panda_tcp")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
        self.target_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        self.wall_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "occlusion_wall")

        # Initial table surface level
        self.table_surface_z = 0.37
        self.reset()

    def reset(self, target_pos=None):
        """
        Resets simulation state to home keyframe and randomizes/sets target position.
        """
        mujoco.mj_resetData(self.model, self.data)

        # Set to home keyframe if available
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            # Default home joint positions
            self.data.qpos[:7] = [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.785]
            self.data.qpos[7:9] = [0.04, 0.04] # Gripper open

        # Set target object position
        if target_pos is not None:
            # target joint is freejoint: 3 pos + 4 quat
            target_qpos_adr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")]
            self.data.qpos[target_qpos_adr:target_qpos_adr+3] = target_pos
            self.data.qpos[target_qpos_adr+3:target_qpos_adr+7] = [1, 0, 0, 0]

        # Settle physics for 50 steps
        for _ in range(50):
            # Hold joint position targets
            self.data.ctrl[:7] = self.data.qpos[:7]
            self.data.ctrl[7:9] = self.data.qpos[7:9]
            mujoco.mj_step(self.model, self.data)

        return self.get_proprioception()

    def get_tcp_pos(self):
        """Returns 3D Cartesian position of the Tool Center Point."""
        return np.copy(self.data.site_xpos[self.tcp_site_id])

    def get_target_pos(self):
        """Returns 3D Cartesian position of the target object."""
        return np.copy(self.data.xpos[self.target_body_id])

    def get_proprioception(self):
        """
        Returns full proprioceptive state dictionary.
        """
        tcp_pos = self.get_tcp_pos()
        gripper_w = self.data.qpos[7] + self.data.qpos[8] # Total finger spacing
        joint_positions = np.copy(self.data.qpos[:7])

        return {
            "tcp_pos": tcp_pos.tolist(),
            "target_pos": self.get_target_pos().tolist(),
            "gripper_pos": tcp_pos.tolist(), # 3D Cartesian coordinates [x, y, z]
            "gripper_pos_2d": [float(tcp_pos[0]), float(tcp_pos[1])], # 2D projected coordinates
            "gripper_width": float(np.clip(gripper_w / 0.08, 0.0, 1.0)), # Normalized 0..1
            "joint_positions": joint_positions.tolist()
        }

    def compute_ik_step(self, target_dx, target_dy, target_dz, damping=0.05):
        """
        Computes 7-DoF joint delta to achieve Cartesian end-effector delta using Damped Jacobian Pseudo-Inverse.
        """
        # Calculate End-Effector Jacobian (3x7 translational)
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self.tcp_site_id)
        J = jacp[:, :7] # Only 7 arm joints

        delta_x = np.array([target_dx, target_dy, target_dz])

        # Damped Least Squares (DLS): J_dls = J^T * (J * J^T + lambda^2 * I)^-1
        lambda_sq = damping ** 2
        JJT = J @ J.T + lambda_sq * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, delta_x)

        # Scale down step size for stability
        max_joint_step = 0.08
        scale = np.max(np.abs(dq)) / max_joint_step
        if scale > 1.0:
            dq = dq / scale

        return dq

    def step(self, action_7dof):
        """
        Executes a 7-DoF Cartesian action step.
        
        Args:
            action_7dof: [dx, dy, dz, droll, dpitch, dyaw, gripper_cmd]
                         dx, dy, dz in meters.
                         gripper_cmd: 0.0 (closed) to 1.0 (open).
        """
        dx, dy, dz = action_7dof[0], action_7dof[1], action_7dof[2]
        gripper_cmd = float(np.clip(action_7dof[6], 0.0, 1.0))

        # 1. Compute joint position delta from Cartesian delta
        dq = self.compute_ik_step(dx, dy, dz)

        # 2. Update target joint positions
        target_q = self.data.qpos[:7] + dq

        # Clip within joint limits
        for i in range(7):
            joint_id = self.model.jnt_qposadr[i]
            limits = self.model.jnt_range[i]
            target_q[i] = np.clip(target_q[i], limits[0], limits[1])

        # 3. Apply position control commands
        self.data.ctrl[:7] = target_q
        # Finger position: 0.04m is fully open per finger, 0.00m is closed
        finger_target = gripper_cmd * 0.04
        self.data.ctrl[7] = finger_target
        self.data.ctrl[8] = finger_target

        # 4. Step physics simulation forward
        for _ in range(self.substeps_per_action):
            mujoco.mj_step(self.model, self.data)

        # Return observation and metrics
        obs = {
            "proprioception": self.get_proprioception(),
            "metrics": self.get_metrics()
        }
        return obs

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



    def check_occlusion(self):
        """
        Checks whether the target object is geometrically occluded by the obstacle wall or gripper.
        """
        target_pos = self.get_target_pos()
        tcp_pos = self.get_tcp_pos()
        wall_pos = self.data.xpos[self.wall_body_id]

        # 1. Check if behind occlusion wall
        # Wall is around x=0.42, y=[-0.12, 0.12]
        is_behind_wall = (
            abs(target_pos[0] - wall_pos[0]) < 0.05 and
            abs(target_pos[1] - wall_pos[1]) < 0.12
        )

        # 2. Check if gripper is directly hovering over the object
        dist_xy = np.linalg.norm(target_pos[:2] - tcp_pos[:2])
        is_under_gripper = (dist_xy < 0.04) and (tcp_pos[2] > target_pos[2])

        return bool(is_behind_wall or is_under_gripper)

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
