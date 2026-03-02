from mujoco_playground import registry

# List all registered environment names
print("Available Environments:")
for env_id in registry.ALL_ENVS:
    print(f" - {env_id}")
