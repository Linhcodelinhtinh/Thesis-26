import os

target_path = r"C:\Users\Admin\AppData\Local\Programs\Python\Python3\Lib\site-packages\robosuite\environments\manipulation\single_arm_env.py"

content = """from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
import copy

class SingleArmEnv(ManipulationEnv):
    def __init__(self, *args, **kwargs):
        if 'mount_types' in kwargs:
            mount = kwargs.pop('mount_types')
            if 'base_types' not in kwargs:
                kwargs['base_types'] = mount

        if 'controller_configs' in kwargs and kwargs['controller_configs'] is not None:
            cfg = kwargs['controller_configs']
            if isinstance(cfg, dict) and cfg.get('type') not in ['BASIC', 'HYBRID_MOBILE_BASE', 'WHOLE_BODY_COMPOSITE', 'WHOLE_BODY_IK']:
                arm_cfg = copy.deepcopy(cfg)
                if "gripper" not in arm_cfg:
                    arm_cfg["gripper"] = {"type": "GRIP"}
                kwargs['controller_configs'] = {
                    "type": "BASIC",
                    "body_parts": {
                        "right": arm_cfg
                    }
                }

        super().__init__(*args, **kwargs)
"""

with open(target_path, "w", encoding="utf-8") as f:
    f.write(content)

# Patch MountedPanda and OnTheGroundPanda in libero
for name in ["mounted_panda.py", "on_the_ground_panda.py"]:
    p = os.path.join(r"C:\Users\Admin\AppData\Local\Programs\Python\Python3\Lib\site-packages\libero\libero\envs\robots", name)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            code = f.read()

        # Fix arms
        if "arms = ['right']" not in code and 'arms = ["right"]' not in code:
            code = code.replace("class MountedPanda(ManipulatorModel):", "class MountedPanda(ManipulatorModel):\n    arms = ['right']")
            code = code.replace("class OnTheGroundPanda(ManipulatorModel):", "class OnTheGroundPanda(ManipulatorModel):\n    arms = ['right']")

        # Fix default_base
        if "default_base" not in code:
            code = code.replace(
                "def default_mount(self):",
                "def default_base(self):\n        return self.default_mount\n\n    @property\n    def default_mount(self):"
            )

        # Fix default_gripper
        code = code.replace(
            'return "PandaGripper"',
            'return {"right": "PandaGripper"}'
        )

        # Fix default_controller_config
        code = code.replace(
            'return "default_panda"',
            'return {"right": "default_panda"}'
        )

        with open(p, "w", encoding="utf-8") as f:
            f.write(code)

# Patch controller.py for modern MuJoCo 3.x mj_fullM signature
ctrl_p = r"C:\Users\Admin\AppData\Local\Programs\Python\Python3\Lib\site-packages\robosuite\controllers\parts\controller.py"
if os.path.exists(ctrl_p):
    with open(ctrl_p, "r", encoding="utf-8") as f:
        code = f.read()
    old_call = "mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)"
    new_call = "mujoco.mj_fullM(self.sim.model._model, self.sim.data._data, mass_matrix) if not hasattr(self.sim.data, 'qM') else mujoco.mj_fullM(self.sim.model._model, mass_matrix, self.sim.data.qM)"
    if old_call in code:
        code = code.replace(old_call, new_call)
        with open(ctrl_p, "w", encoding="utf-8") as f:
            f.write(code)
        print("Patched controller.py for MuJoCo 3 mj_fullM!")

# Patch robot_base_factory.py for missing or None base handling
base_factory_p = r"C:\Users\Admin\AppData\Local\Programs\Python\Python3\Lib\site-packages\robosuite\models\bases\robot_base_factory.py"
if os.path.exists(base_factory_p):
    factory_code = """from typing import Optional
from robosuite.models.bases.robot_base_model import RobotBaseModel

def robot_base_factory(name: Optional[str], idn=0) -> RobotBaseModel:
    from robosuite.models.bases import BASE_MAPPING
    if name is None or name not in BASE_MAPPING:
        name = "NullMount"
    return BASE_MAPPING[name](idn=idn)
"""
    with open(base_factory_p, "w", encoding="utf-8") as f:
        f.write(factory_code)
    print("Patched robot_base_factory.py!")

# Ensure on_the_ground_panda.py defaults to NullMount
on_ground_p = r"C:\Users\Admin\AppData\Local\Programs\Python\Python3\Lib\site-packages\libero\libero\envs\robots\on_the_ground_panda.py"
if os.path.exists(on_ground_p):
    with open(on_ground_p, "r", encoding="utf-8") as f:
        pcode = f.read()
    pcode = pcode.replace("return None", 'return "NullMount"')
    with open(on_ground_p, "w", encoding="utf-8") as f:
        f.write(pcode)
    print("Patched on_the_ground_panda.py!")

print("single_arm_env.py, panda classes, controller.py, and robot_base_factory.py updated successfully!")
