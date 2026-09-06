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

    def draw_overlay(self, image, points, occluded_flags, prompt_text=""):
        """
        Draws green/red tracking points onto the RGB image and renders HUD overlay.
        
        Args:
            image: numpy array (H, W, 3) - RGB or BGR image
            points: list of [x, y] coordinates
            occluded_flags: list of booleans indicating if each point is occluded
            prompt_text: string - The dynamic state prompt from the Chronicler
            
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

        # 2. Draw HUD multi-modal prompt overlay
        if prompt_text:
            parts = prompt_text.split(" | ")
            y_offset = 25
            
            # Semi-transparent HUD background banner
            hud_height = min(120, len(parts) * 24 + 15)
            h, w = painted_image.shape[:2]
            hud_bg = painted_image[5:5+hud_height, 10:w-10]
            if hud_bg.shape[0] > 0 and hud_bg.shape[1] > 0:
                dark_box = np.zeros_like(hud_bg)
                cv2.rectangle(dark_box, (0, 0), (dark_box.shape[1], dark_box.shape[0]), (20, 20, 20), -1)
                painted_image[5:5+hud_height, 10:w-10] = cv2.addWeighted(hud_bg, 0.35, dark_box, 0.65, 0)
                cv2.rectangle(painted_image, (10, 5), (w - 10, 5 + hud_height), (80, 80, 80), 1)

            for part in parts:
                # Text color based on section type
                text_color = (255, 255, 255)
                if part.startswith("[Task]"):
                    text_color = (0, 220, 255) # Yellowish
                elif part.startswith("[Status]"):
                    text_color = (100, 255, 100) # Light Green
                elif part.startswith("[Memory]"):
                    text_color = (220, 180, 255) # Light Violet

                # Drop shadow
                cv2.putText(
                    painted_image, part, (18, y_offset + 1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA
                )
                # Main text
                cv2.putText(
                    painted_image, part, (18, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1, cv2.LINE_AA
                )
                y_offset += 22
                
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
