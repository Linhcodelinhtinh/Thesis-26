#!/usr/bin/env python3
"""
The Chronicler — Memory-as-a-Prompt Mechanism
---------------------------------------------
Maintains temporal awareness, state transitions, and logic memory (Dynamic Prompt).
Supports both Real VLM Inference (Hugging Face) and Fast Temporal State Machine.
Runs at 5-10 Hz.
"""
import time
from collections import deque
import numpy as np
from PIL import Image

# Dynamic import for real VLM packages
try:
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, pipeline
except ImportError:
    torch = None


class Chronicler:
    """
    The Chronicler generates and maintains the Dynamic Prompt state.
    It tracks the temporal progression of tasks and outputs structured prompts
    for downstream VLA policy conditioning.
    """
    def __init__(
        self,
        use_real_vlm=False,
        model_id="HuggingFaceTB/SmolVLM-Instruct",
        device="auto",
        max_history_length=5
    ):
        self.use_real_vlm = use_real_vlm
        self.model_id = model_id
        self.max_history_length = max_history_length
        self.device = self._resolve_device(device)
        
        # Real VLM components
        self.model = None
        self.processor = None
        self.vlm_pipeline = None
        
        # Temporal Memory Buffer
        self.history = deque(maxlen=max_history_length)
        self.current_phase = "INIT"
        self.step_count = 0
        self.was_occluded = False
        self.grasped_step = None
        
        if self.use_real_vlm:
            self._load_vlm()

    def _resolve_device(self, device_str):
        if device_str == "auto":
            return "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
        return device_str

    def _load_vlm(self):
        """
        Attempts to load real Vision-Language Model weights.
        Falls back to fast temporal state machine if dependencies or weights are unavailable.
        """
        if torch is None:
            print("[Chronicler] Warning: PyTorch/Transformers not installed. Falling back to Fast Temporal State Machine.")
            self.use_real_vlm = False
            return

        print(f"[Chronicler] Initializing Real VLM '{self.model_id}' on device: {self.device}...")
        try:
            # Try loading pipeline or generic AutoModel
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                try:
                    self.model = AutoModelForImageTextToText.from_pretrained(
                        self.model_id,
                        dtype=dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).to(self.device).eval()
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        dtype=dtype,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    ).to(self.device).eval()
                print(f"[Chronicler] Real VLM '{self.model_id}' loaded successfully.")
            except Exception:
                # Fallback to general vision-language / image-captioning pipeline
                self.vlm_pipeline = pipeline(
                    "image-text-to-text",
                    model=self.model_id,
                    device=0 if self.device == "cuda" else -1,
                    dtype=dtype
                )
                print(f"[Chronicler] Real VLM pipeline initialized for '{self.model_id}'.")
                
        except Exception as e:
            print(f"[Chronicler] Notice: Could not load local/online weights for '{self.model_id}' ({e}).")
            print("[Chronicler] Operating in Real-Time Fast Temporal State Machine mode.")
            self.use_real_vlm = False

    def run_vlm_inference(self, camera_frame, human_command):
        """
        Runs real VLM inference on the visual frame to describe scene logic.
        """
        if camera_frame is None:
            return "No visual feed"

        try:
            # Ensure PIL Image format
            if isinstance(camera_frame, np.ndarray):
                if camera_frame.ndim == 3 and camera_frame.shape[2] == 3:
                    # Assume RGB/BGR
                    pil_img = Image.fromarray(camera_frame)
                else:
                    pil_img = Image.fromarray((camera_frame * 255).astype(np.uint8))
            elif isinstance(camera_frame, Image.Image):
                pil_img = camera_frame
            else:
                return "Invalid image format"

            if self.vlm_pipeline is not None:
                res = self.vlm_pipeline(pil_img, max_new_tokens=30)
                if isinstance(res, list) and len(res) > 0 and "generated_text" in res[0]:
                    return res[0]["generated_text"].strip()

            if self.model is not None and self.processor is not None:
                prompt_text = f"User: Describe the robot gripper and object state for task: '{human_command}'. Assistant:"
                inputs = self.processor(images=pil_img, text=prompt_text, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, max_new_tokens=35)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return generated_text.replace(prompt_text, "").strip()

        except Exception as e:
            return f"VLM parsing error: {e}"

        return "Scene analyzed"

    def update_state(self, camera_frame, human_command, proprioception, tracking_info=None):
        """
        Generates and updates the structured Dynamic State Prompt.
        
        Output format:
        [Task]: <Command> | [Status]: <Action Phase>. <Object State>. <Gripper State>. | [Memory]: <Past Context>
        
        Args:
            camera_frame: RGB/BGR image array or PIL Image.
            human_command: string command (e.g., "Grasp the orange object").
            proprioception: dict with gripper_pos/tcp_pos, gripper_width.
            tracking_info: dict with target_pos, is_occluded.
        """
        self.step_count += 1
        
        # 1. Parse robot physical state
        tcp_pos = proprioception.get("tcp_pos", None)
        gripper_pos = proprioception.get("gripper_pos", tcp_pos if tcp_pos is not None else [0.0, 0.0, 0.0])
        gripper_width = float(proprioception.get("gripper_width", 1.0)) # 1.0 = open, 0.0 = closed
        
        # 2. Parse tracking & spatial memory
        target_pos = None
        is_occluded = False
        
        if tracking_info:
            target_pos = tracking_info.get("target_pos", None)
            is_occluded = bool(tracking_info.get("is_occluded", False))
            
        # Target distance calculation with robust dimensional alignment
        dist = None
        if target_pos is not None:
            # Prefer full 3D tcp_pos if available and target_pos is 3D
            ref_pos = tcp_pos if (tcp_pos is not None and len(target_pos) == 3) else gripper_pos
            if ref_pos is not None:
                g_arr = np.array(ref_pos, dtype=np.float32)
                t_arr = np.array(target_pos, dtype=np.float32)
                min_dim = min(len(g_arr), len(t_arr))
                if min_dim > 0:
                    dist = float(np.linalg.norm(g_arr[:min_dim] - t_arr[:min_dim]))

        # 3. Object Visibility State
        if is_occluded:
            object_state = "Object is OCCLUDED (relying on persistent spatial tracker)"
            self.was_occluded = True
        else:
            object_state = "Object is VISIBLE"

        # 4. Gripper Mechanical State
        if gripper_width < 0.2:
            gripper_state = "Gripper is FULLY CLOSED"
        elif gripper_width < 0.65:
            gripper_state = "Gripper is CLOSED ON OBJECT"
        elif gripper_width < 0.85:
            gripper_state = "Gripper is CLOSING"
        else:
            gripper_state = "Gripper is OPEN"

        # 5. Temporal Phase & State Machine Transitions
        if self.use_real_vlm and (self.model is not None or self.vlm_pipeline is not None):
            vlm_description = self.run_vlm_inference(camera_frame, human_command)
            action_state = f"VLM Analysis: {vlm_description}"
        else:
            # Fast Temporal State Machine
            if dist is not None:
                # MuJoCo metric units (< 0.04m) vs 2D pixel units (< 20px)
                is_near = (dist < 0.04) if dist < 1.0 else (dist < 20.0)
                is_descended = (dist < 0.015) if dist < 1.0 else (dist < 10.0)
                
                # Check if gripper holds object (for rigid objects, width is ~0.4-0.6)
                if (gripper_width < 0.65 and is_near) or self.current_phase == "LIFTING":
                    if self.grasped_step is None:
                        self.grasped_step = self.step_count
                    action_state = "GRASP_LOCKED: Object secured in fingers, executing lift"
                    self.current_phase = "LIFTING"
                elif is_descended:
                    action_state = "ALIGNMENT_REACHED: Ready to grasp target object"
                    self.current_phase = "GRASPING"
                elif is_near:
                    action_state = "TARGET_PROXIMITY: Descending end-effector onto target"
                    self.current_phase = "DESCENDING"
                elif is_occluded:
                    action_state = "OCCLUSION_RECOVERY: Executing blind visual servoing trajectory"
                    self.current_phase = "OCCLUDED_APPROACH"
                else:
                    action_state = "APPROACHING: Navigating towards object coordinates"
                    self.current_phase = "APPROACHING"
            else:
                action_state = "APPROACHING: Navigating towards target"
                self.current_phase = "APPROACHING"

        # 6. Update Temporal History Queue
        history_entry = f"Step {self.step_count}: {self.current_phase}"
        if not self.history or self.history[-1].split(": ")[1] != self.current_phase:
            self.history.append(history_entry)

        memory_str = " -> ".join(list(self.history)[-3:]) if self.history else "Initial state"

        # 7. Construct Full Dynamic Prompt Output
        dynamic_prompt = (
            f"[Task]: {human_command} | "
            f"[Status]: {action_state}. {object_state}. {gripper_state}. | "
            f"[Memory]: {memory_str}"
        )

        return dynamic_prompt

    def get_state_dict(self):
        """
        Returns machine-readable state summary.
        """
        return {
            "current_phase": self.current_phase,
            "step_count": self.step_count,
            "was_occluded": self.was_occluded,
            "history": list(self.history)
        }

    def reset(self):
        """
        Resets temporal history on episode restart.
        """
        self.history.clear()
        self.current_phase = "INIT"
        self.step_count = 0
        self.was_occluded = False
        self.grasped_step = None


if __name__ == "__main__":
    chronicler = Chronicler(use_real_vlm=False)
    
    demo_prop = {"gripper_pos": [100.0, 100.0, 0.5], "gripper_width": 1.0}
    demo_track = {"target_pos": [105.0, 105.0, 0.4], "is_occluded": False}
    
    prompt = chronicler.update_state(None, "Grasp the red cube", demo_prop, demo_track)
    print("=" * 60)
    print("[Chronicler Test Output]")
    print(prompt)
    print("=" * 60)
