# booster_rough_env_cfg.py
# Full working Booster T1 rough locomotion config (G1-aligned) with regex-safe terms.
import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from isaaclab.managers import SceneEntityCfg
from booster_isaac_lab.assets.booster import BOOSTER_T1_CFG


import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def reward_base_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_terrain_relative: bool = True,
    mode: str = "square_error"  # "square_error" or "gaussian"
) -> torch.Tensor:
    """
    Reward/penalty for maintaining base height relative to terrain.
    
    Args:
        target_height: Desired height above terrain (meters)
        asset_cfg: Which robot/asset to check
        use_terrain_relative: If True, subtract terrain height. If False, use absolute world height.
        mode: "square_error" for squared penalty, "gaussian" for exponential reward
    """
    asset = env.scene[asset_cfg.name]
    
    # Get base position (z coordinate)
    base_z = asset.data.root_pos_w[:, 2]  # [num_envs]
    
    if use_terrain_relative:
        # Get terrain heights at base XY position
        base_xy = asset.data.root_pos_w[:, :2]  # [num_envs, 2]
        
        # Access terrain heights (requires terrain sensor)
        if "height_scanner" in env.scene.sensors:
            # Get height at base position (approximate)
            # This is simplified - actual implementation depends on your terrain setup
            terrain_z = env.scene.sensors["height_scanner"].data.measured_heights
            
            # For simplicity, let's use the height at base position
            # You might need to interpolate from height field
            height_above_terrain = base_z - terrain_z
        else:
            # Fallback: assume flat terrain at z=0
            height_above_terrain = base_z
    else:
        # Use absolute world height
        height_above_terrain = base_z
    
    # Calculate error from target
    height_error = height_above_terrain - target_height
    
    if mode == "square_error":
        # Your original: squared error (negative reward/penalty)
        return torch.square(height_error)
    
    elif mode == "gaussian":
        # Alternative: exponential reward (positive reward)
        # Similar to velocity tracking
        sigma = 0.1  # How strict: smaller = more strict
        return torch.exp(-torch.square(height_error) / (sigma**2))
    
    elif mode == "tolerance_band":
        # Binary reward within tolerance band
        tolerance = 0.05  # ±5cm tolerance
        return (torch.abs(height_error) < tolerance).float()
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'square_error', 'gaussian', or 'tolerance_band'")

def penalty_base_height_squared(
    env: ManagerBasedRLEnv,
    target_height: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Direct implementation matching your original _reward_base_height.
    
    Note: This is actually a PENALTY (positive values = bad).
    Use with negative weight.
    """
    asset = env.scene[asset_cfg.name]
    
    # Get base height (z coordinate)
    base_z = asset.data.root_pos_w[:, 2]
    
    # Try to get terrain height
    terrain_height = torch.zeros_like(base_z)
    
    # Check if we have a height scanner or terrain
    if "height_scanner" in env.scene.sensors:
        # Get height at base position
        # This part depends on your specific terrain implementation
        scanner = env.scene.sensors["height_scanner"]
        
        # Get base XY position
        base_xy = asset.data.root_pos_w[:, :2]
        
        # Convert to grid coordinates (depends on your terrain setup)
        # This is approximate - adjust based on your actual terrain
        grid_size = scanner.cfg.resolution
        terrain_origin = scanner.cfg.prim_path  # Or however origin is stored
        
        # For now, use simple approach if available
        if hasattr(scanner.data, 'terrain_heights_at_pos'):
            terrain_height = scanner.data.terrain_heights_at_pos(base_xy)
        else:
            # Fallback: use the center measurement
            terrain_height = scanner.data.measured_heights.mean(dim=1)
    
    # Height above terrain
    height_above_terrain = base_z - terrain_height
    
    # Squared error from target (your original formula)
    return torch.square(height_above_terrain - target_height)

# def reward_base_height_simple(
#     env: ManagerBasedRLEnv,
#     target_height: float = 0.5,
#     tolerance: float = 0.1,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """
#     Simplified version: Gaussian reward around target height.
#     Returns POSITIVE reward (use with positive weight).
    
#     reward = exp(-(height - target)² / (2*tolerance²))
#     """
#     asset = env.scene[asset_cfg.name]
    
#     # Get base height
#     base_z = asset.data.root_pos_w[:, 2]
    
#     # For now, assume flat terrain at z=0
#     # You can modify to include terrain if needed
#     height = base_z
    
#     # Gaussian reward
#     error = height - target_height
#     return torch.exp(-torch.square(error) / (2 * tolerance**2))



def reward_survival(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for simply surviving (being alive)."""
    # Return 1.0 for every environment that's still running
    return torch.ones(env.num_envs, device=env.device, dtype=torch.float32)

def reward_survival_with_termination_check(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Survival reward that gives 0.0 if terminated."""
    # Check which environments are still active
    terminated = env.termination_manager.is_terminated()
    # Give 1.0 if alive, 0.0 if terminated
    return (~terminated).float()
# -----------------------------------------------------------------------------
# Rewards (start G1-like; keep only terms that don't require risky body regexes)
# -----------------------------------------------------------------------------
@configclass
class BoosterRewards(RewardsCfg):
    """G1-equivalent locomotion rewards for Booster T1 (regex-safe)."""
    # ---------- NEW CUSTOM REWARDS ----------

    # survival: 0.25
    survival_bonus = RewTerm(
        func=reward_survival,  # Your custom function
        weight=0.25,  # Start small
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # Track_lin_vel_xy: 1.0
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Track_ang_vel: 0.5
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    #Base height: 0.68
    base_height_penalty = RewTerm(
        func=mdp.base_height_l2,  # Use the existing function!
        weight=-20.0,  # Negative weight = penalty
        params={
            "target_height": 0.68,  # Target 0.68m above ground
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": None,  # No terrain adjustment
        },
    )

    #lin_vel_z : -2.0
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)

    #ang_vel_xy: -0.2
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)

    #orientation: -5.0
    orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight= -5.0)

    #torques: -2.e-4
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.e-4)


    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_link"]),
            "threshold": 0.2,
        },
    )
    
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_foot_link"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_foot_link"]),
        },
    )


    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )

    # Penalize deviation from default joints
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Pitch", ".*_Hip_Yaw"])},
    )
    joint_deviation_knee = RewTerm(
        func= mdp.joint_deviation_l1,
        weight = -0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Knee_Pitch"])},
    )
    joint_deviation_ankle = RewTerm(
        func= mdp.joint_deviation_l1,
        weight = -0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Shoulder_Pitch",
                    ".*_Shoulder_Roll",
                    ".*_Elbow_Pitch",
                    ".*_Elbow_Yaw",
                ],
            )
        },
    )
    
    joint_deviation_head_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "AAHead_yaw",
                    "Head_pitch",
                    "Waist",
                ],
            )
        },
    )
    
    # In BoosterRewards class - FIXED:
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,  # Start mild
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", 
                body_names=["Trunk",
                    "Hip_Pitch_Left", "Hip_Roll_Left",
                    "Hip_Pitch_Right", "Hip_Roll_Right",
                ]
            ),
            "threshold": 1.0,  # ← ADD THIS! Minimum force to count as contact (Newtons)
        }
    )
    

@configclass
class BoosterRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: BoosterRewards = BoosterRewards()

    def __post_init__(self):
        # Parent setup 
        super().__post_init__()

        # ---------------------------------------------------------------------
        # Robot asset
        # ---------------------------------------------------------------------
        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Height scanner base link (Booster uses Trunk)
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"
        # Randomization

        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["Trunk"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None
        self.rewards.dof_pos_limits.weight = -1.0

        # STRICTER TERMINATIONS
        self.terminations.base_contact.params["sensor_cfg"].body_names = ["Trunk",
                    "Waist", "AL1", "AL2", "AL3", "AR1", "AR2", "AR3","H1","H2",
                    "Hip_Pitch_Left", "Hip_Roll_Left",
                    "Hip_Pitch_Right", "Hip_Roll_Right",
                    "Shank_Left", "Shank_Right", "Ankle_Cross_Left", "Ankle_Cross_Right"
                ]



@configclass
class BoosterRoughEnvCfg_PLAY(BoosterRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0

        # No corruption / no pushes for evaluation
        if hasattr(self.observations, "policy"):
            self.observations.policy.enable_corruption = False

        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None