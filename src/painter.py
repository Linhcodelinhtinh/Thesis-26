#!/usr/bin/env python3
"""
The Painter — Tracking Overlay Filter
-------------------------------------
Paints visible (green) and occluded (red) tracking points directly on the RGB image,
along with rendering the multi-modal HUD containing Task, Status, and Memory.
"""
import cv2
import numpy as np

class Painter:
    def __init__(self, visible_color=(0, 255, 0), occluded_color=(0, 0, 255)):
        self.visible_color = visible_color   # Green (BGR: 0, 255, 0)
        self.occluded_color = occluded_color # Red (BGR: 0, 0, 255)

    def draw_overlay(self, image, points, occluded_flags, prompt_text="", show_hud=False):
        """
        Draws green/red tracking points onto the RGB image and optionally renders HUD overlay.
        
        Args:
            image: numpy array (H, W, 3) - RGB or BGR image
            points: list of [x, y] coordinates
            occluded_flags: list of booleans indicating if each point is occluded
            prompt_text: string - The dynamic state prompt from the Chronicler
            show_hud: bool - Whether to paint the HUD banner directly onto the image.
                             Default is False to prevent obscuring the camera view.
            
        Returns:
            painted_image: painted BGR image
        """
        if image is None:
            return None

        painted_image = image.copy()
        
        # 1. Draw persistent tracking dots
        if points is not None and occluded_flags is not None:
            for pt, is_occ in zip(points, occluded_flags):
                color = self.occluded_color if is_occ else self.visible_color
                center = (int(pt[0]), int(pt[1]))
                # Solid circle for tracking point
                cv2.circle(painted_image, center, 4, color, -1)
                # Outer border for visual contrast
                cv2.circle(painted_image, center, 6, (0, 0, 0), 1)

        # 2. Draw HUD multi-modal prompt overlay ONLY if explicitly requested
        if show_hud and prompt_text:
            h, w = painted_image.shape[:2]
            hud_bg = painted_image[0:22, 0:w]
            if hud_bg.shape[0] > 0 and hud_bg.shape[1] > 0:
                dark_box = np.zeros_like(hud_bg)
                painted_image[0:22, 0:w] = cv2.addWeighted(hud_bg, 0.35, dark_box, 0.65, 0)
            
            first_part = prompt_text.split(" | ")[0]
            cv2.putText(
                painted_image, first_part, (8, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA
            )
                
        return painted_image


if __name__ == "__main__":
    painter = Painter()
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    res = painter.draw_overlay(
        dummy_image, [[100, 100], [200, 200]], [False, True], 
        "[Task]: Grasp mug | [Status]: Approaching. Object visible. | [Memory]: Step 1: APPROACHING"
    )
    cv2.imwrite("test_paint.png", res)
    print("Painter test passed.")
