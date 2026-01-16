from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from booster_isaac_lab.assets.booster import BOOSTER_T1_CFG

@configclass
class BoosterRewards(RewardsCfg):
    """Refined reward terms for the Booster T1."""
    # Positive Rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0, 
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, 
        weight=0.25, 
        params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot_link"),
            "threshold": 0.2, 
        },
    )

    # Negative Rewards (Standardized to newest Isaac Lab API)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk", "Waist", ".*Hip_Pitch.*", ".*Shank.*"]),
            "threshold": 1.0,
        },
    )

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

@configclass
class BoosterRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: BoosterRewards = BoosterRewards()

    def __post_init__(self):
            super().__post_init__()

            # 1. Asset Setup
            self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            
            # 2. Fix Sensors and Height Scanner
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Trunk"

            # 3. Fix Events and Terminations naming
            if hasattr(self.events, "add_base_mass"):
                self.events.add_base_mass.params["asset_cfg"].body_names = "Trunk"
                
            self.terminations.base_contact.params["sensor_cfg"].body_names = ["Trunk", "Waist"]

            # 4. FIX ALL OBSERVATIONS (Safe version)
            # We only iterate over 'policy' since 'critic' doesn't exist in your config
            if hasattr(self.observations, "policy"):
                for term_name in self.observations.policy.__dict__:
                    term_cfg = getattr(self.observations.policy, term_name)
                    # Check if the term has a 'params' dict with an 'asset_cfg' or 'sensor_cfg'
                    if hasattr(term_cfg, "params"):
                        for param_key in ["asset_cfg", "sensor_cfg"]:
                            if param_key in term_cfg.params:
                                cfg = term_cfg.params[param_key]
                                # If it's looking for "base", point it to "Trunk"
                                if hasattr(cfg, "body_names") and cfg.body_names == "base":
                                    cfg.body_names = "Trunk"

            # 5. Command Ranges
            self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
            self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)

@configclass
class BoosterRoughEnvCfg_PLAY(BoosterRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False