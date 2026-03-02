import pickle
import numpy as np
import gym
import mujoco_py  # or import mujoco if using MuJoCo 2.1+

# Load the policy (adjust based on how it was saved)
def test_policy(checkpoint_path, env_name="Op3Joystick", num_episodes=10):
    # Load the saved policy
    with open(checkpoint_path, 'rb') as f:
        policy = pickle.load(f)  # Might be .pt, .pth, .pkl, etc.
    
    # Create environment
    env = gym.make(env_name)
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action from policy (adjust based on your policy interface)
            action = policy.predict(obs, deterministic=True)[0]
            # Or if it's a neural network directly:
            # action = policy(obs)
            
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            env.render()  # Visualize
            
        print(f"Episode {episode}: Reward = {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    checkpoint = "/home/quangnam/mujoco_playground/logs/Op3Joystick-20260127-194712/checkpoints/000103219200"
    test_policy(checkpoint)