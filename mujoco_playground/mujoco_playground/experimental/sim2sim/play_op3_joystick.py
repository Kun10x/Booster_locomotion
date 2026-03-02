import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
from collections import deque
from pynput import keyboard

from mujoco_playground._src.locomotion.op3 import op3_constants as consts
from mujoco_playground._src.locomotion.op3.base import get_assets

# --- Global Command State ---
class CommandState:
    def __init__(self):
        self.x = 0.0      # Forward/Back
        self.y = 0.0      # Lateral (Side-step)
        self.yaw = 0.0    # Rotation
        self.push = np.zeros(3) # External Force [x, y, z]

cmd = CommandState()

# --- Keyboard Handling ---
def on_press(key):
    try:
        # Walking Controls
        if key.char == 'w': cmd.x = min(cmd.x + 0.1, 1.0)
        elif key.char == 's': cmd.x = max(cmd.x - 0.1, -0.5)
        elif key.char == 'a': cmd.yaw = min(cmd.yaw + 0.2, 0.7)
        elif key.char == 'd': cmd.yaw = max(cmd.yaw - 0.2, -0.7)
        elif key.char == 'q': cmd.y = min(cmd.y + 0.1, 0.5)
        elif key.char == 'e': cmd.y = max(cmd.y - 0.1, -0.5)
    except AttributeError:
        # Push Testing (Arrow Keys)
        force = 8.0 # Newtons. Adjust based on OP3 weight.
        if key == keyboard.Key.up:    cmd.push[0] = force
        elif key == keyboard.Key.down:  cmd.push[0] = -force
        elif key == keyboard.Key.left:  cmd.push[1] = force
        elif key == keyboard.Key.right: cmd.push[1] = -force
        # Emergency Stop
        elif key == keyboard.Key.space:
            cmd.x, cmd.y, cmd.yaw = 0.0, 0.0, 0.0
            print(">> Reset Velocity Commands")
    print(f"Command: x={cmd.x:.2f}, y={cmd.y:.2f}, yaw={cmd.yaw:.2f}")

def on_release(key):
    # Release the push force immediately when key is up
    if key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
        cmd.push.fill(0)

# Start Listenerconsts
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# --- Controller and Observation Logic ---
class OnnxController:
    def __init__(self, policy_path, default_angles, default_angles_mujoco, ctrl_dt, n_substeps):
        self._output_names = ["continuous_actions"]
        self._policy = rt.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
        
        self._action_scale = 0.3
        self._default_angles = default_angles
        self._last_action = np.zeros(20, dtype=np.float32)
        
        self._counter = 0
        self._n_substeps = n_substeps
        
        # History Buffer: Model expects 147 (3 frames of 49)
        self._history_maxlen = 3
        self._single_obs_size = 49
        self._obs_buffer = deque(maxlen=self._history_maxlen)
        self.default_angles_mujoco = default_angles_mujoco
        for _ in range(self._history_maxlen):
            self._obs_buffer.append(np.zeros(self._single_obs_size, dtype=np.float32))

    def get_obs(self, data) -> np.ndarray:
        gyro = data.sensor(consts.GYRO_SENSOR).data  
        # print("GYRO DATA: ", gyro)        
        gravity = data.sensor(consts.GRAVITY_SENSOR).data
        gx = 2 * (data.qpos[4] * data.qpos[6] - data.qpos[3] * data.qpos[5])
        gy = 2 * (data.qpos[5] * data.qpos[6] + data.qpos[3] * data.qpos[4])
        gz = data.qpos[3] * data.qpos[3] - data.qpos[4] * data.qpos[4] - data.qpos[5] * data.qpos[5] + data.qpos[6] * data.qpos[6]
        # print("GRAVITY CALCULATED: ", gx,gy,gz)
        # print("GRAVITY DATA: ", gravity)    
        command = np.array([cmd.x, cmd.y, cmd.yaw])
        # print("COMMAND DATA: ", command)
        joint_angles = data.qpos 
        # print("JOINTS ANGLES BEFORE FIXED: ", joint_angles)
        joint_angles = data.qpos[7:] - self._default_angles  
        last_action = self._last_action                      
        # print("JOINTS: ", joint_angles)
        # print("JOINT FROM 7: ", joint_angles[7:])
        # Single frame (49 dims)
        current_obs = np.concatenate([
            gyro, gravity, command, joint_angles, last_action
        ]).astype(np.float32)
        # print("Number of joints: ",len(joint_angles))
        # print("Number of last action :",len(last_action))
        # self.print_detailed_obs(current_obs)
        self._obs_buffer.appendleft(current_obs)
        return np.concatenate(list(self._obs_buffer)).astype(np.float32)
    def print_detailed_obs(self, obs):
        # obs is the 49-dim array
        print("\n--- MUJOCO RAW OBSERVATION (49 DIMS) ---")
        print(f"GYRO (0-2):      {obs[0:3]}")
        print(f"GRAVITY (3-5):   {obs[3:6]}")
        print(f"COMMAND (6-8):   {obs[6:9]}")
        print("\nJOINT RELATIVE ANGLES (9-28):")
        # Prints in 2 rows of 10
        print(f"L_Leg/Head:      {obs[9:19]}")
        print(f"R_Leg/Arms:      {obs[19:29]}")
        print("\nLAST ACTION (29-48):")
        print(f"Actions 0-9:     {obs[29:39]}")
        print(f"Actions 10-19:   {obs[39:49]}")
        print("-" * 40)
    def get_control(self, model, data):
        # Apply external pushes to the torso body
        # Usually body ID 1 is the torso after the worldbody
        data.xfrc_applied[1, :3] = cmd.push
        
        self._counter += 1
        if self._counter % self._n_substeps == 0:
            obs = self.get_obs(data)
            onnx_input = {"obs": obs.reshape(1, -1)}
            onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
            
            self._last_action = onnx_pred.copy()
            motor_targets = self._default_angles + onnx_pred * self._action_scale
            
            # Clip and apply motor targets
            ctrl_range = model.actuator_ctrlrange
            data.ctrl[:] = np.clip(motor_targets, ctrl_range[:, 0], ctrl_range[:, 1])

# --- Simulation Entry ---
def load_callback(model=None, data=None):
    mujoco.set_mjcb_control(None)
    
    # Load model with correct assets
    model = mujoco.MjModel.from_xml_path(consts.FEET_ONLY_XML.as_posix(), assets=get_assets())
    data = mujoco.MjData(model)

    # Reset to trained keyframe
    try:
        k_id = model.keyframe("stand_bent_knees").id
        mujoco.mj_resetDataKeyframe(model, data, k_id)
    except:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    # Physics Params from OP3 Config
    model.dof_damping[6:] = 1.084
    model.actuator_gainprm[:, 0] = 21.1
    model.actuator_biasprm[:, 1] = -21.1
    model.opt.ccd_iterations = 10 
    
    # Timesteps from config
    ctrl_dt, sim_dt = 0.02, 0.004
    model.opt.timestep = sim_dt
    # PRINT THE PID/GAIN VALUES
    print_actuator_params(model)
    
    # Print additional physics parameters
    print("\n" + "="*60)
    print("PHYSICS PARAMETERS")
    print("="*60)
    print(f"Simulation timestep: {model.opt.timestep:.6f} s")
    print(f"Control timestep:    {ctrl_dt:.3f} s")
    print(f"Control substeps:    {int(round(ctrl_dt / sim_dt))}")
    print(f"CCD iterations:      {model.opt.ccd_iterations}")
    print(f"Solver iterations:   {model.opt.iterations}")
    print(f"Solver tolerance:    {model.opt.tolerance:.6f}")
    
    # Print joint names in order (to help with mapping to Webots)
    print("\n" + "="*60)
    print("JOINT ORDER (from qpos[7:])")
    print("="*60)
    for i in range(0, model.nq):  # Start from index 7 (skip 6DOF root)
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i-1)  # -1 because root has no joint
        if joint_name:
            print(f"Index {i:2d}: {joint_name}")
        else:
            print(f"Index {i:2d}: unnamed joint")

    # Initialize Policy
    policy = OnnxController(
        policy_path="onnx/Op3_policy.onnx", 
        default_angles=data.qpos[7:].copy(),
        default_angles_mujoco= [0.0,-0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0,-0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ctrl_dt=ctrl_dt,
        n_substeps=int(round(ctrl_dt / sim_dt))
    )

    # Register Control Callback
    mujoco.set_mjcb_control(policy.get_control)
    return model, data
def print_actuator_params(model):
    """Print actuator gain and bias parameters (MuJoCo's equivalent of PID)"""
    print("\n" + "="*60)
    print("ACTUATOR PARAMETERS (MuJoCo PID equivalents)")
    print("="*60)
    
    # Get actuator names
    actuator_names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        actuator_names.append(name if name else f"actuator_{i}")
    
    # Print header
    print(f"{'Actuator':<20} {'Gainprm[0]':<12} {'Biasprm[0]':<12} {'Biasprm[1]':<12} {'Ctrl Range':<20}")
    print("-" * 80)
    
    # Print each actuator
    for i in range(model.nu):
        name = actuator_names[i]
        gain = model.actuator_gainprm[i, 0]
        bias0 = model.actuator_biasprm[i, 0]
        bias1 = model.actuator_biasprm[i, 1]
        ctrl_min = model.actuator_ctrlrange[i, 0] if model.actuator_ctrlrange.any() else -np.inf
        ctrl_max = model.actuator_ctrlrange[i, 1] if model.actuator_ctrlrange.any() else np.inf
        
        print(f"{name:<20} {gain:<12.3f} {bias0:<12.3f} {bias1:<12.3f} [{ctrl_min:>6.3f}, {ctrl_max:<6.3f}]")
    
    # Also print dof damping (joint-level PD)
    print("\n" + "="*60)
    print("JOINT DAMPING (velocity-dependent friction)")
    print("="*60)
    
    for i in range(model.nv):
        if i < 6:  # Skip 6DOF root joint
            continue
        dof_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_DOF, i)
        damping = model.dof_damping[i]
        print(f"{dof_name or f'dof_{i}':<20} {damping:.3f}")
if __name__ == "__main__":
    print("-" * 30)
    print("OP3 INTERACTIVE DEPLOYMENT")
    print("-" * 30)
    print("MOVEMENT: W/A/S/D (Q/E for strafe)")
    print("PUSH TEST: ARROW KEYS (Up/Down/Left/Right)")
    print("STOP: SPACEBAR")
    print("-" * 30)
    
    viewer.launch(loader=load_callback)