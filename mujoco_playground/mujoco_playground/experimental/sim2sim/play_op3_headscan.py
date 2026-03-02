import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
from pynput import keyboard

from mujoco_playground._src.locomotion.op3 import op3_constants as consts
from mujoco_playground._src.locomotion.op3.base import get_assets

# --- Global Target State ---
class HeadState:
    def __init__(self):
        self.target_yaw = 0.0  # The target the head should look at

state = HeadState()

# --- Keyboard Handling ---
def on_press(key):
    try:
        # Control the target yaw manually
        if key.char == 'a': state.target_yaw = np.clip(state.target_yaw + 0.1, -1.5, 1.5)
        elif key.char == 'd': state.target_yaw = np.clip(state.target_yaw - 0.1, -1.5, 1.5)
        elif key.char == 'r': state.target_yaw = 0.0 # Reset
        print(f"Target Yaw: {state.target_yaw:.2f}")
    except AttributeError:
        pass

# Start Listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# --- Controller Logic ---
class HeadScanController:
    def __init__(self, policy_path, action_scale, head_pan_idx):
        # Your model had 1 output
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        
        self._action_scale = action_scale
        self._head_pan_qpos_idx = head_pan_idx # Usually 7 for OP3 head pan
        self._counter = 0

    def get_obs(self, data) -> np.ndarray:
        # Match your training code: [current_head_pos, target_yaw]
        current_yaw = data.qpos[self._head_pan_qpos_idx]
        obs = np.array([current_yaw, state.target_yaw], dtype=np.float32)
        print(obs)
        return obs

    def get_control(self, model, data):
        # Step physics at a fixed rate (matching training ctrl_dt)
        obs = self.get_obs(data)
        
        # Inference
        onnx_input = {"obs": obs.reshape(1, -1)}
        action = self._policy.run(self._output_names, onnx_input)[0][0]
        
        # The model outputs a delta for the head. 
        # In your training: motor_targets = default_pose.at[head_idx].add(action * scale)
        # Here we just set the specific actuator.
        head_motor_target = action[0] * self._action_scale
        
        # Apply only to the head actuator (index 0 based on your training script)
        data.ctrl[0] = head_motor_target

# --- Simulation Entry ---
def load_callback(model=None, data=None):
    # Load model
    model = mujoco.MjModel.from_xml_path(consts.FEET_ONLY_XML.as_posix(), assets=get_assets())
    data = mujoco.MjData(model)

    # Reset to standing pose
    k_id = model.keyframe("stand_bent_knees").id
    mujoco.mj_resetDataKeyframe(model, data, k_id)

    # Match training physics
    ctrl_dt, sim_dt = 0.02, 0.004
    model.opt.timestep = sim_dt
    
    # Initialize Controller
    # Path should point to the ONNX file you just exported
    policy = HeadScanController(
        policy_path="onnx/Op3_head_scanpolicy.onnx", 
        action_scale=0.5, # Matches your training config.action_scale
        head_pan_idx=7    # Matches your training self._head_pan_qpos_idx
    )

    mujoco.set_mjcb_control(policy.get_control)
    return model, data

if __name__ == "__main__":
    print("-" * 30)
    print("OP3 HEAD SCAN INTERACTIVE")
    print("-" * 30)
    print("ADJUST TARGET: A / D")
    print("RESET: R")
    print("-" * 30)
    
    viewer.launch(loader=load_callback)