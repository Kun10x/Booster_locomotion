import argparse
from isaaclab.app import AppLauncher

# 1. Setup Argument Parser
parser = argparse.ArgumentParser(description="Booster T1 Manager-Based Environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Imports follow launching the app."""
import torch
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# Import your Booster Config
from booster_isaac_lab.assets.booster import BOOSTER_T1_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

@configclass
class BoosterSceneCfg(InteractiveSceneCfg):
    """Configuration for the Booster scene."""
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", 
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # The robot
    robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

@configclass
class ActionsCfg:
    """Action specifications: Mapping RL output to joint commands."""
    # We use JointPositionAction to control the humanoid's posture
    joint_positions = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], # All joints
        scale=0.5 
    )

@configclass
class ObservationsCfg:
    """Observation specifications: What the 'brain' sees."""
    @configclass
    class PolicyCfg(ObsGroup):
        # Humanoids need more than just joint states
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class EventCfg:
    """Physics randomization events."""
    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.01, 0.01),
        },
    )

@configclass
class BoosterEnvCfg(ManagerBasedEnvCfg):
    """Full environment configuration."""
    scene = BoosterSceneCfg(num_envs=16, env_spacing=4.0)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        self.viewer.eye = [6.0, 6.0, 4.0]
        self.viewer.lookat = [0.0, 0.0, 1.0]
        self.decimation = 4  # 200Hz sim -> 50Hz control
        self.sim.dt = 0.005

def main():
    env_cfg = BoosterEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    
    env = ManagerBasedEnv(cfg=env_cfg)

    while simulation_app.is_running():
        with torch.inference_mode():
            # Standard RL Loop: Action -> Step -> Obs
            # Sample random joint targets
            random_actions = torch.randn_like(env.action_manager.action)
            obs, _ = env.step(random_actions)
            
            # Print base orientation (Projected gravity Z should be ~ -1.0 if standing)
            print("[Env 0] Projected Gravity (Z): ", obs["policy"][0][8].item())

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()