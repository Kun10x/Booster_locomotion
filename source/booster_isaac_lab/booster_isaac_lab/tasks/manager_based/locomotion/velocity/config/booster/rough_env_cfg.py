# booster_rough_env_cfg.py
# Full working Booster T1 rough locomotion config (G1-aligned) with regex-safe terms.

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from isaaclab.managers import SceneEntityCfg
from booster_isaac_lab.assets.booster import BOOSTER_T1_CFG


# -----------------------------------------------------------------------------
# Rewards (start G1-like; keep only terms that don't require risky body regexes)
# -----------------------------------------------------------------------------
@configclass
@configclass
class BoosterRewards(RewardsCfg):
    """G1-equivalent locomotion rewards for Booster T1 (regex-safe)."""
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    
    # Positive rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            # Use the class directly since it's imported
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Ankle_Cross_(Left|Right)"), # Use existing link names
            "asset_cfg": SceneEntityCfg("robot", body_names="Ankle_Cross_(Left|Right)"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Ankle_Pitch", ".*_Ankle_Roll"])},
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk", "Hip_Pitch_(Left|Right)", "Shank_(Left|Right)"]),
            "threshold": 1.0,
        },
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
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch"]
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Knee_Pitch", ".*_Ankle_.*"]
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "Trunk"




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
