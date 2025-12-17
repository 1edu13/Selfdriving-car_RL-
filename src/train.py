import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import time
import os

from agent import Agent
from utils import make_env, get_device

def train():
    # --- Hyperparameters (Tune these based on Section 5.1 of your report) ---
    run_name = "ppo_carracing_v1"
    env_id = "CarRacing-v3"
    seed = 42
    total_timesteps = 1000000  # As mentioned in Exp 5.1
    learning_rate = 3e-4
    num_envs = 8               # Parallel environments for faster data collection
    num_steps = 1024           # Steps per environment per update (The 'Buffer' size)
    anneal_lr = True           # Decay learning rate
    gamma = 0.99               # Discount factor
    gae_lambda = 0.95          # GAE parameter
    num_minibatches = 32
    update_epochs = 10
    norm_adv = True            # Normalize advantages
    clip_coef = 0.2            # The epsilon for clipping (Section 2.2)
    ent_coef = 0.01            # Entropy coefficient to encourage exploration
    vf_coef = 0.5              # Value function coefficient
    max_grad_norm = 0.5        # Gradient clipping
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    # --- Setup ---
    device = get_device()
    print(f"Training on device: {device}")

    # Create directories for saving models
    os.makedirs("models", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    # Vectorized Environment (Parallel data collection)
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed + i, i, capture_video=False, run_name=run_name) for i in range(num_envs)]
    )

    # Initialize Agent
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # --- Storage Buffers (The "Buffer" from your diagram) ---
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Start the game
    global_step = 0
    start_time = time.time()

    # Initial observation
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    print(f"Starting training loop... Total Updates: {num_updates}")

    for update in range(1, num_updates + 1):
        # 1. Rollout Phase (Data Collection)
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Action logic (no grad needed for collection)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Execute step in environment
            # Note: We move action to cpu for numpy compatibility
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(terminations | truncations).to(device)

            # Print episodic return if available (for logging)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"Global Step: {global_step} | Episode Reward: {info['episode']['r']}")

        # 2. Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

            returns = advantages + values

        # 3. Optimization Phase (Backpropagation)
        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Calculate Advantages
                with torch.no_grad():
                    # Normalize advantages (optional but recommended)
                    mb_advantages = b_advantages[mb_inds]
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss (The Clipped Objective from Section 2.2)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # Entropy Loss (to encourage exploration)
                entropy_loss = entropy.mean()

                # Total Loss
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # Save Model periodically
        if update % 10 == 0:
            torch.save(agent.state_dict(), f"models/ppo_car_racing_step_{global_step}.pth")
            print(f"Model saved at step {global_step}")

    # Save final model
    torch.save(agent.state_dict(), "models/ppo_car_racing_final.pth")
    envs.close()
    print("Training Completed.")

if __name__ == "__main__":
    train()