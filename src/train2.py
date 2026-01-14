import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os

from agent import Agent
from utils2 import make_env, get_device


def train():
    # --- Hyperparameters ---
    run_name = "ppo_carracing_T5_grass_penalty"  # Nombre para identificar esta runn
    env_id = "CarRacing-v2"
    seed = 42
    total_timesteps = 3000000
    learning_rate = 3e-4
    num_envs = 8
    num_steps = 1024
    anneal_lr = True
    gamma = 0.99
    gae_lambda = 0.95
    num_minibatches = 32
    update_epochs = 10
    norm_adv = True
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    batch_size = int(num_envs * num_steps)
    minibatch_size = int(batch_size // num_minibatches)

    # --- Setup ---
    device = get_device()

    # Crear directorios para guardar los nuevos modelos T4
    os.makedirs("../Models/models_T6_grass", exist_ok=True)

    # Nota: La carpeta de videos se crea automáticamente por el wrapper,
    # pero nos aseguramos de que exista la base.
    os.makedirs("videos_T6", exist_ok=True)

    # Vectorized Environment
    # AQUI ESTA LA CLAVE: apply_grass_penalty=True
    envs = gym.vector.AsyncVectorEnv(
        [make_env(env_id, seed + i, i, capture_video=False, run_name=run_name, apply_grass_penalty=True)
         for i in range(num_envs)]
    )

    # Initialize Agent
    agent = Agent(envs).to(device)

    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # --- Storage Buffers ---
    obs = torch.zeros((num_steps, num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((num_steps, num_envs)).to(device)
    rewards = torch.zeros((num_steps, num_envs)).to(device)
    dones = torch.zeros((num_steps, num_envs)).to(device)
    values = torch.zeros((num_steps, num_envs)).to(device)

    # Start the game
    global_step = 0

    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(num_envs).to(device)
    num_updates = total_timesteps // batch_size

    print(f"Starting training loop (T4 - Grass Penalty)... Total Updates: {num_updates}")

    for update in range(1, num_updates + 1):
        # 1. Rollout Phase
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            # Execute step
            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(terminations | truncations).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # Log extra para ver si el timeout de hierba está funcionando
                        timeout_str = " (Grass Timeout)" if info.get('off_track_timeout') else ""
                        print(f"Global Step: {global_step} | Episode Reward: {info['episode']['r']:.2f}{timeout_str}")

        # 2. GAE
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

        # 3. Optimization
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    mb_advantages = b_advantages[mb_inds]
                    if norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # Save Model periodically
        if update % 25 == 0:
            save_path = f"../Models/models_T5_grass/ppo_car_racing_step_{global_step}.pth"
            torch.save(agent.state_dict(), save_path)
            print(f"Model saved at step {global_step} -> {save_path}")

    # Save final model
    torch.save(agent.state_dict(), "../Models/models_T5_grass/model_3000k.pth")
    envs.close()
    print("Training Completed.")


if __name__ == "__main__":
    train()