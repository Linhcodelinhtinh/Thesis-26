import os
import sys
import torch
from prismatic.models import load_vla

ckpt_path = os.path.abspath(r"checkpoints/minivla_vq_libero90/checkpoints/step-150000-epoch-67-loss=0.0934.pt")
print(f"Loading VLA from {ckpt_path}...")
vla = load_vla(ckpt_path)
print("VLA loaded successfully!")
print(f"VLA type: {type(vla)}")
print(f"Action tokenizer: {vla.action_tokenizer}")
print(f"LLM backbone: {type(vla.llm_backbone)}")
print(f"Vision backbone: {type(vla.vision_backbone)}")

from PIL import Image
import numpy as np

print("Testing action prediction with dummy image...")
dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
instruction = "pick up the black bowl and place it on the plate"
action = vla.predict_action(dummy_img, instruction, unnorm_key="libero_90")
print("Action prediction successful!")
print(f"Action shape: {action.shape if hasattr(action, 'shape') else len(action)}")
print(f"Action values:\n{action}")
