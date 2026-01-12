import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, FrameStack, RecordVideo

# Import internal modules
from agent import Agent
from utils import get_device

# Suppress harmless warnings from Gymnasium/Deprecation
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# Constants
CHECKPOINT_FREQ = 200000
TOTAL_TIMESTEPS = 3000000
LEARNING_RATE = 3e-4
NUM_ENVS = 8
NUM_STEPS = 1024
GAMMA = 0.99
GAE_LAMBDA = 0.95
UPDATE_EPOCHS = 10
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5


class FixedSeedWrapper(gym.Wrapper):
    """
    Wrapper to force the environment to use a specific fixed seed on every reset.
    """

    def __init__(self, env, fixed_seed):
        super().__init__(env)
        self.fixed_seed = fixed_seed

    def reset(self, **kwargs):
        return self.env.reset(seed=self.fixed_seed, options=kwargs.get('options'))


def make_specialist_env(env_id, fixed_seed, idx, capture_video, run_name):
    """
    Factory function to create the environment with the FixedSeedWrapper.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env = FixedSeedWrapper(env, fixed_seed=fixed_seed)
        if capture_video and idx == 0:
            env = RecordVideo(env, f"videos_specialist/{run_name}")
        env = GrayScaleObservation(env, keep_dim=False)
        env = FrameStack(env, 4)
        return env

    return thunk


def compute_gae(rewards, values, next_value, dones, next_done, num_steps, gamma, gae_lambda, device):
    """
    Calculates Generalized Advantage Estimation (GAE) to reduce duplication in main loop.
    """
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
    return advantages, returns


def ppo_update(agent, optimizer, b_obs, b_logprobs, b_actions, b_advantages, b_returns, batch_size, minibatch_size):
    """
    Performs the PPO update epochs.
    """
    b_inds = np.arange(batch_size)
    for _ in range(UPDATE_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + VF_COEF * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()


def train_specialist():
    run_name = "ppo_specialist_3M"
    env_id = "CarRacing-v2"
    fixed_track_seed = 12345

    # Configuration
    batch_size = int(NUM_ENVS * NUM_STEPS)
    minibatch_size = int(batch_size // 32)
    device = get_device()

    os.makedirs("../Models/specialist", exist_ok=True)
    os.makedirs(f"videos_specialist/{run_name}", exist_ok=True)

    # Initialize Envs
    envs = gym.vector.AsyncVectorEnv(
        [make_specialist_env(env_id, fixed_track_seed, i, False, run_name) for i in range(NUM_ENVS)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # Buffers
    obs = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((NUM_STEPS, NUM_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    rewards = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    dones = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)
    values = torch.zeros((NUM_STEPS, NUM_ENVS)).to(device)

    # Start
    global_step = 0
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(NUM_ENVS).to(device)
    num_updates = TOTAL_TIMESTEPS // batch_size
    next_checkpoint = CHECKPOINT_FREQ

    print(f"🏁 STARTING SPECIALIST TRAINING")
    print(f"📍 Fixed Track Seed: {fixed_track_seed}")

    for update in range(1, num_updates + 1):
        # Annealing
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow

        # Rollout
        for step in range(0, NUM_STEPS):
            global_step += 1 * NUM_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(terminations | truncations).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"Global Step: {global_step} | Fixed Track Reward: {info['episode']['r']:.2f}")

        # Compute GAE (Refactored)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages, returns = compute_gae(
                rewards, values, next_value, dones, next_done,
                NUM_STEPS, GAMMA, GAE_LAMBDA, device
            )

        # Flatten
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Optimization (Refactored)
        ppo_update(
            agent, optimizer, b_obs, b_logprobs, b_actions,
            b_advantages, b_returns, batch_size, minibatch_size
        )

        # Checkpoint
        if global_step >= next_checkpoint:
            checkpoint_path = f"../Models/specialist/ppo_specialist_step_{global_step}.pth"
            torch.save(agent.state_dict(), checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            next_checkpoint += CHECKPOINT_FREQ

    # Final Save
    final_save_path = "../Models/specialist/ppo_specialist_final_3M.pth"
    torch.save(agent.state_dict(), final_save_path)
    print(f"✅ Finished. Saved: {final_save_path}")
    envs.close()


if __name__ == "__main__":
    train_specialist()