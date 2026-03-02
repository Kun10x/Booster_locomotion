# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cartpole environment."""

from typing import Any, Dict, Optional, Union
import warnings

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common
from mujoco_playground._src.locomotion.op3 import op3_constants as consts
from mujoco_playground._src.locomotion.op3 import base as op3_base



def default_config() -> config_dict.ConfigDict:

    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.004,
        episode_length=1000,  # Shorter for a simple task
        impl="jax",
        # Motor gains from the original OP3 config
        Kp=21.1,
        Kd=1.084,
        
        action_repeat=1,
        action_scale=0.3,    # Limits how "far" the RL action can push the motor
        
        # Simple Reward Weights
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_yaw=1.0,     # Primary goal
                action_rate=-0.01,    # Smoothness
                torques=-0.0001,      # Energy efficiency
            ),
        ),
        
        # MJX specific physics limits
        nconmax=160, 
        njmax=400,
    )

class OP3HeadScan(op3_base.Op3Env):
    """A simple head scanning task for OP3."""

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        # Op3Env handle the model loading, but needs the xml_path string
        super().__init__(
            xml_path=str(consts.FEET_ONLY_XML),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    @property
    def xml_path(self) -> str:
        return str(consts.FEET_ONLY_XML)

    @property
    def action_size(self) -> int:
        # Only controlling the head_pan actuator
        return 1

    def _post_init(self) -> None:
        # Keyframe and joint indices
        self._init_q = jp.array(self._mj_model.keyframe("stand_bent_knees").qpos)
        self._default_pose = jp.array(self._mj_model.keyframe("stand_bent_knees").qpos[7:])
        
        # We find the specific index for the head actuator
        # In the robotis_op3, this is usually the last actuator or index 19
        self._head_pan_qpos_idx = 7 
        self._head_pan_ctrl_idx = 0 

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, goal_rng = jax.random.split(rng)

        data = mjx_env.make_data(self.mj_model, qpos=self._init_q, qvel=jp.zeros(self.mjx_model.nv))
        data = mjx.forward(self.mjx_model, data)

        target_yaw = jax.random.uniform(goal_rng, (1,), minval=-5, maxval=5)

        info = {
            "rng": rng,
            "target_yaw": target_yaw,
            "last_act": jp.zeros(1),
            "step": 0,
        }

        metrics = {f"reward/{k}": jp.zeros(()) for k in self._config.reward_config.scales.keys()}
        obs = self._get_obs(data, info)
        
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, _ = jax.random.split(state.info["rng"])

        # Map the 1D action to the head pan while keeping rest of body at default pose
        motor_targets = self._default_pose
        motor_targets = motor_targets.at[self._head_pan_ctrl_idx].add(action[0] * self._config.action_scale)
        
        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

        # 3. Calculate Reward
        current_yaw = data.qpos[self._head_pan_qpos_idx]
        # Ensure error is shape () per env, not (1,)
        target_yaw = state.info["target_yaw"].squeeze()
        error = current_yaw - target_yaw
        
        rewards = {
            # Use .squeeze() to ensure these are flat [16] arrays
            "tracking_yaw": jp.exp(-15.0 * jp.square(error)).squeeze(),
            "action_rate": (-jp.square(action[0] - state.info["last_act"][0])).squeeze(),
            "torques": (-jp.sqrt(jp.sum(jp.square(data.actuator_force)))).squeeze(),
        }
        
        weighted_rewards = {k: v * self._config.reward_config.scales[k] for k, v in rewards.items()}
        
        # result in shape [16] instead of [16, 1]
        reward = sum(weighted_rewards.values())

        # State Update
        state.info.update({"last_act": action, "step": state.info["step"] + 1, "rng": rng})
        obs = self._get_obs(data, state.info)
        
        for k, v in weighted_rewards.items():
            state.metrics[f"reward/{k}"] = v

        return state.replace(data=data, obs=obs, reward=reward, done=jp.zeros(()))

    def _get_obs(self, data: mjx.Data, info: dict) -> jax.Array:
        return jp.concatenate([
            data.qpos[self._head_pan_qpos_idx:self._head_pan_qpos_idx+1], 
            info["target_yaw"]
        ])