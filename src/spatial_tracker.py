#!/usr/bin/env python3
"""
The Spatial Tracker — Persistent Visual Tracking
-----------------------------------------------
Tracks physical object points under high occlusion.
Provides persistent spatial awareness (Green = Visible, Red = Occluded).
Runs at 10-20 Hz.
"""
import numpy as np
import cv2

class SpatialTracker:
    def __init__(self, tracker_type="opencv_fallback", config=None):
        self.tracker_type = tracker_type
        self.config = config or {}
        
        # Grid points being tracked: list of [x, y]
        self.points = []
        self.occluded_flags = []
        
        # Keep track of previous frame for optical flow
        self.prev_gray = None
        
        # Persistent memory coordinates
        self.last_known_points = None
        
        # Simulation parameters for geometric occlusion
        self.occlusion_radius = self.config.get("simulation", {}).get("occlusion_radius", 40)

    def initialize_tracking_points(self, initial_frame, target_centroid, num_points=9):
        """
        Spawns a persistent grid of points centered around the target centroid.
        
        Args:
            initial_frame: RGB/BGR image array.
            target_centroid: [x, y] coordinates of the target object.
            num_points: number of tracking points to spawn (default 9 for 3x3 grid).
        """
        self.points = []
        self.occluded_flags = []
        
        cx, cy = float(target_centroid[0]), float(target_centroid[1])
        
        # Create a grid of points around the centroid
        grid_size = max(1, int(np.sqrt(num_points)))
        spacing = 8  # pixels spacing between grid points
        
        start_x = cx - (grid_size - 1) * spacing / 2.0
        start_y = cy - (grid_size - 1) * spacing / 2.0
        
        for i in range(grid_size):
            for j in range(grid_size):
                px = start_x + i * spacing
                py = start_y + j * spacing
                self.points.append([px, py])
                self.occluded_flags.append(False)
                
        # Convert to numpy array for OpenCV LK Tracker
        self.points = np.array(self.points, dtype=np.float32).reshape(-1, 1, 2)
        self.last_known_points = self.points.copy()
        
        # Cache grayscale frame
        if initial_frame is not None and isinstance(initial_frame, np.ndarray):
            self.prev_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY) if initial_frame.ndim == 3 else initial_frame
            
        print(f"[Tracker] Initialized {len(self.points)} persistent tracking points around {target_centroid}")

    def track(self, current_frame, gripper_pos=None):
        """
        Updates tracking points across consecutive frames.
        
        Args:
            current_frame: BGR/RGB image array.
            gripper_pos: [x, y] coordinate of the gripper (used to compute geometric occlusion).
            
        Returns:
            flat_points: Coordinates list of tracking points.
            occluded_flags: List of booleans (True = Occluded).
        """
        if len(self.points) == 0:
            return [], []

        if current_frame is None or not isinstance(current_frame, np.ndarray):
            flat_points = self.points.reshape(-1, 2)
            return flat_points.tolist(), self.occluded_flags

        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if current_frame.ndim == 3 else current_frame
        
        if self.tracker_type == "opencv_fallback" and self.prev_gray is not None:
            # Lucas-Kanade optical flow tracking
            lk_params = dict(
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
            )
            try:
                new_points, st, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.points, None, **lk_params
                )
                
                # Update points that were successfully tracked, maintain memory for lost points
                for idx in range(len(new_points)):
                    if st is not None and st[idx] == 1:
                        self.points[idx] = new_points[idx]
                        self.last_known_points[idx] = new_points[idx]
                    else:
                        # Fallback to persistent spatial memory
                        self.points[idx] = self.last_known_points[idx]
            except Exception as e:
                # Keep points in place
                pass
            
            self.prev_gray = gray
            
        elif self.tracker_type == "tapir":
            # Modular plug-in for deep TAPIR / CoTracker point tracking
            pass

        # Occlusion estimation based on distance to the gripper
        self.occluded_flags = []
        flat_points = self.points.reshape(-1, 2)
        
        for pt in flat_points:
            is_occ = False
            if gripper_pos is not None:
                g_p = np.array(gripper_pos[:2], dtype=np.float32)
                # Only compare directly if in pixel space (> 10 pixels)
                if np.max(g_p) > 10.0:
                    dist = np.linalg.norm(np.array(pt) - g_p)
                    if dist < self.occlusion_radius:
                        is_occ = True
            self.occluded_flags.append(is_occ)
            
        return flat_points, self.occluded_flags


if __name__ == "__main__":
    tracker = SpatialTracker()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    tracker.initialize_tracking_points(dummy_frame, [320, 240], num_points=9)
    pts, flags = tracker.track(dummy_frame, [330, 245])
    print(f"Tracked Points count: {len(pts)}")
    print(f"Occlusion status: {flags}")
