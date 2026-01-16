import argparse
from isaaclab.app import AppLauncher

# 1. Setup Argument Parser
parser = argparse.ArgumentParser(description="Booster T1 Multi-Environment Scene.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# 3. Import your Booster Config
from booster_isaac_lab.assets.booster import BOOSTER_T1_CFG

@configclass
class BoosterSceneCfg(InteractiveSceneCfg):
    """Configuration for a Booster T1 scene."""
    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Replace cartpole with Booster
    robot: ArticulationCfg = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    robot = scene["robot"]  # Using the name we defined in BoosterSceneCfg
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset every 500 steps
        if count % 500 == 0:
            count = 0
            # Reset Root (Position/Velocity)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            
            # Reset Joints (Apply default posture from your booster.py)
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # Add a tiny bit of noise to joints
            joint_pos += torch.randn_like(joint_pos) * 0.05
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            
            scene.reset()
            print("[INFO]: Resetting Booster robots...")

        # Apply random joint effort targets
        # Booster has 23+ DOFs, so we generate a vector of size (num_envs, num_dof)
        efforts = torch.randn_like(robot.data.joint_pos) * 2.0
        robot.set_joint_effort_target(efforts)

        # Write and Step
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Adjusted Camera for a 1.7m tall Humanoid
    sim.set_camera_view([4.0, 4.0, 3.0], [0.0, 0.0, 1.0])
    
    # Initialize Scene
    scene_cfg = BoosterSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: Booster Scene Setup complete...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()