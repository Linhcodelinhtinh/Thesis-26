"""
MiniVLA Policy Wrapper for LIBERO Benchmark
Uses Stanford-ILIAD/minivla-vq-libero90 with DinoSigLIP vision backbone,
Qwen2.5-0.5B LLM, and VQ-VAE action chunk tokenizer.
"""

import os
import math
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import torch
from PIL import Image

from prismatic.models import load_vla


def apply_center_crop(image: np.ndarray, t_h: int, t_w: int) -> np.ndarray:
    """Center crop image according to LIBERO evaluation standard."""
    h, w = image.shape[:2]
    top = (h - t_h) // 2
    left = (w - t_w) // 2
    return image[top : top + t_h, left : left + t_w]


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
    Formula: y = 2 * (x - 0) / (1 - 0) - 1
    """
    action = np.copy(action)
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flips the sign of gripper action (-1 = open, +1 = close) for Robosuite/MuJoCo execution.
    """
    action = np.copy(action)
    action[..., -1] = action[..., -1] * -1.0
    return action


class MiniVLAPolicy:
    """
    High-performance policy wrapper for MiniVLA Libero90 checkpoint.
    """
    _instance = None
    _loaded_ckpt = None

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        center_crop: bool = True,
        unnorm_key: str = "libero_90",
    ):
        if checkpoint_path is None:
            checkpoint_path = os.path.abspath(
                r"checkpoints/minivla_vq_libero90/checkpoints/step-150000-epoch-67-loss=0.0934.pt"
            )
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.center_crop = center_crop
        self.unnorm_key = unnorm_key
        self.vla = None

        self._load_model()

    def _load_model(self):
        # Cache single instance in memory
        if MiniVLAPolicy._instance is not None and MiniVLAPolicy._loaded_ckpt == self.checkpoint_path:
            self.vla = MiniVLAPolicy._instance
            print("Reusing cached MiniVLA instance.")
            return

        print(f"[MiniVLA] Loading model from: {self.checkpoint_path} on {self.device}...")
        self.vla = load_vla(self.checkpoint_path)
        self.vla.eval()
        MiniVLAPolicy._instance = self.vla
        MiniVLAPolicy._loaded_ckpt = self.checkpoint_path
        print("[MiniVLA] Model loaded and ready.")

    def preprocess_image(self, img_array: np.ndarray, flip_ud: bool = True) -> Image.Image:
        """
        Preprocesses image from LIBERO observation:
        1. Apply np.flipud to correct MuJoCo's OpenGL vertical flip
        2. Convert uint8 RGB numpy array to PIL Image
        3. Resize to 224x224 via LANCZOS (model input resolution)
        4. Apply center crop with sqrt(0.9) factor and resize back via BILINEAR
        """
        if flip_ud:
            img_array = np.flipud(img_array)
        image = Image.fromarray(img_array).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        if self.center_crop:
            temp = np.array(image)
            crop_scale = 0.9
            sqrt_crop = math.sqrt(crop_scale)
            t_h = int(sqrt_crop * temp.shape[0])
            t_w = int(sqrt_crop * temp.shape[1])
            cropped = apply_center_crop(temp, t_h, t_w)
            image = Image.fromarray(cropped).resize((224, 224), Image.Resampling.BILINEAR)

        return image

    def predict_action(
        self,
        obs: Dict[str, Any],
        instruction: str,
    ) -> np.ndarray:
        """
        Generates 7D continuous action [dx, dy, dz, droll, dpitch, dyaw, gripper]
        for a given observation and natural language instruction.
        """
        if "agentview_image" in obs:
            raw_img = obs["agentview_image"]
        elif "image" in obs:
            raw_img = obs["image"]
        else:
            raise KeyError(f"No agentview_image found in obs keys: {list(obs.keys())}")

        processed_img = self.preprocess_image(raw_img)

        # Predict continuous action from VLA
        raw_action = self.vla.predict_action(
            processed_img,
            instruction,
            unnorm_key=self.unnorm_key,
        )

        # Ensure 1D array of 7 elements
        action = np.array(raw_action, dtype=np.float64)
        if action.ndim > 1:
            action = action.squeeze()
        if action.shape != (7,):
            action = action[:7]

        # Normalize gripper [0, 1] -> [-1, 1]
        action = normalize_gripper_action(action, binarize=True)

        # Invert gripper for Robosuite convention (-1 = open, +1 = close)
        action = invert_gripper_action(action)

        return action
