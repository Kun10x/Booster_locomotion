import gymnasium as gym

gym.register(
    id="Isaac-Velocity-Rough-Booster-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "booster_isaac_lab.tasks.manager_based.locomotion.velocity.config.booster.rough_env_cfg:BoosterRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "booster_isaac_lab.tasks.manager_based.locomotion.velocity.config.booster.agents.rsl_rl_ppo_cfg:BoosterRoughPPORunnerCfg",
    },
)