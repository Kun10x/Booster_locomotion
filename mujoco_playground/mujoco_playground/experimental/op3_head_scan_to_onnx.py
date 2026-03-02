import os
import functools
import numpy as np
if not hasattr(np, 'cast'):
    np.cast = {dest_t: lambda x: np.asarray(x, dtype=dest_t) for dest_t in [np.float32, np.float64, np.int32, np.int64]}
import jax
import jax.numpy as jp
import tensorflow as tf
import tf2onnx
import onnxruntime as rt
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.checkpoint import load
from brax.training.acme import running_statistics
from mujoco_playground.config import locomotion_params
from mujoco_playground import locomotion

# 1. Environment Setup
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

env_name = "Op3HeadScan"
ppo_params = locomotion_params.brax_ppo_config(env_name)
env_cfg = locomotion.get_default_config(env_name)
env = locomotion.load(env_name, config=env_cfg)

obs_size = env.observation_size
act_size = env.action_size

# Handle whether obs_size is a dict or a plain int
if isinstance(obs_size, dict):
    # It's a dict, get the 'state' key
    state_shape = obs_size["state"]
    input_dim = state_shape[0] if isinstance(state_shape, (tuple, list)) else state_shape
else:
    # It's just an int (e.g., 147)
    input_dim = obs_size

print(f"Obs Size: {obs_size}, Act Size: {act_size}, Input Dim: {input_dim}")

# 2. Load JAX Checkpoint
ckpt_path = "/home/quangnam/mujoco_playground/logs/Op3HeadScan-20260211-230831/checkpoints/000014745600"
params = load(ckpt_path)
output_path = "Op3_head_scanpolicy_cmd.onnx"

# Prepare JAX Inference Function
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **ppo_params.network_factory,
    preprocess_observations_fn=running_statistics.normalize,
)
ppo_network = network_factory(obs_size, act_size)
# metadata confirms params[0] is stats, params[1] is policy
inference_fn = ppo_networks.make_inference_fn(ppo_network)((params[0], params[1]), deterministic=True)

# 3. Define Keras Equivalent Model
class MLP(tf.keras.Model):
    def __init__(self, layer_sizes, activation, mean_std):
        super().__init__()
        self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
        self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
        
        self.mlp_block = tf.keras.Sequential(name="MLP_0")
        for i, size in enumerate(layer_sizes):
            self.mlp_block.add(layers.Dense(
                size, 
                activation=activation if i < len(layer_sizes)-1 else None,
                name=f"hidden_{i}",
                kernel_initializer="lecun_uniform"
            ))

    def call(self, inputs):
        # Apply Normalization
        x = (inputs - self.mean) / self.std
        logits = self.mlp_block(x)
        # PPO standard: Split output in half, return tanh of the first half (mean)
        loc, _ = tf.split(logits, 2, axis=-1)
        return tf.tanh(loc)

# 4. Initialize Keras Model and Transfer Weights
mean_std = (tf.convert_to_tensor(params[0].mean, dtype=tf.float32), 
            tf.convert_to_tensor(params[0].std, dtype=tf.float32))

# metadata confirmed 4 hidden layers of 128 + 1 final layer of 40
tf_policy_network = MLP(
    layer_sizes=[128, 128, 128, 128, 40], 
    activation=tf.nn.swish, 
    mean_std=mean_std
)

# Build model
_ = tf_policy_network(tf.zeros((1, input_dim)))

def transfer_weights(jax_params, tf_model):
    for layer_name, layer_params in jax_params.items():
        try:
            tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params['kernel'])
                bias = np.array(layer_params['bias'])
                tf_layer.set_weights([kernel, bias])
                print(f"Transferred: {layer_name}")
        except ValueError:
            print(f"Skipped: {layer_name}")

transfer_weights(params[1]['params'], tf_policy_network)

# 5. Convert to ONNX
spec = [tf.TensorSpec(shape=(1, input_dim), dtype=tf.float32, name="obs")]
tf_policy_network.output_names = ['continuous_actions']
model_proto, _ = tf2onnx.convert.from_keras(
    tf_policy_network, input_signature=spec, opset=11, output_path=output_path
)

# 6. Verification
test_input_np = np.ones((1, input_dim), dtype=np.float32)

# TF Prediction
tf_pred = tf_policy_network(test_input_np).numpy()[0]

# ONNX Prediction
m = rt.InferenceSession(output_path, providers=['CPUExecutionProvider'])
onnx_pred = m.run(['continuous_actions'], {'obs': test_input_np})[0][0]

# JAX Prediction - FIXED LOGIC
if isinstance(obs_size, dict):
    jax_obs = {'state': jp.ones((input_dim,))}
    if "privileged_state" in obs_size:
        jax_obs['privileged_state'] = jp.zeros(obs_size["privileged_state"])
else:
    # This matches your Op3 setup where obs_size is just an int
    jax_obs = jp.ones((input_dim,))

jax_pred, _ = inference_fn(jax_obs, jax.random.PRNGKey(0))

print("\n--- Numerical Verification (First 3 actions) ---")
print(f"JAX:  {np.array(jax_pred[:3])}")
print(f"TF:   {tf_pred[:3]}")
print(f"ONNX: {onnx_pred[:3]}")

plt.plot(onnx_pred, label='ONNX', linestyle='--')
plt.plot(tf_pred, label='TensorFlow', alpha=0.5)
plt.plot(jax_pred, label='JAX', alpha=0.5)
plt.legend()
plt.title(f"OP3 Action Comparison ({env_name})")
plt.show()