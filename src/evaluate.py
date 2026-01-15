import torch
import numpy as np
import gymnasium as gym
import os

from agent import Agent
from utils2 import make_env, get_device

def evaluate(model_path, episodes=5):
    """
    Loads a trained model and evaluates its performance.

    Args:
        model_path (str): Path to the .pth model file.
        episodes (int): Number of test episodes to run.
    """
    device = get_device()
    print(f"Evaluating model: {model_path} on {device}")

    # Environment Setup
    # We use capture_video=True to save the run in the 'videos_T1/' folder
    run_name = "eval_" + os.path.basename(model_path).replace(".pth", "")
    env = make_env("CarRacing-v2", seed=100, idx=0, capture_video=True, run_name=run_name)()

    # Initialize Agent and Load Weights
    # Note: We must initialize the agent with a dummy environment to get the correct shapes
    dummy_envs = gym.vector.SyncVectorEnv([lambda: env])
    agent = Agent(dummy_envs).to(device)

    # Load the state dictionary safely
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval() # Switch to evaluation mode

    total_rewards = []

    for episode in range(episodes):
        obs, _ = env.reset()
        obs = torch.as_tensor(np.array(obs)).float().unsqueeze(0).to(device) # Add batch dimension (1, 4, 96, 96)

        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                # In evaluation, we often behave deterministically (mean) or sample.
                # Standard PPO evaluation usually keeps sampling but without exploration noise updates.
                action, _, _, _ = agent.get_action_and_value(obs)

            # Execute action
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            episode_reward += reward

            obs = torch.as_tensor(np.array(next_obs)).float().unsqueeze(0).to(device)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    env.close()

    avg_score = np.mean(total_rewards)
    print(f"Average Score over {episodes} episodes: {avg_score:.2f}")
    if avg_score > 900:
        print("SUCCESS: The agent consistently scores > 900 points.")
    else:
        print("Keep Training: The agent has not yet reached the target score.")


if __name__ == "__main__":

    modelo_a_probar = r"C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_B\model_0500k.pth"
    evaluate(modelo_a_probar)