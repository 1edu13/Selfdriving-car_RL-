import gymnasium as gym
import torch
from gymnasium.wrappers import GrayScaleObservation, FrameStack

def make_env(env_id, seed, idx, capture_video, run_name):
    """
    Utility function to create and configure the environment.

    Args:
        env_id (str): The environment ID (e.g., "CarRacing-v3").
        seed (int): Global seed for reproducibility.
        idx (int): Index of the environment (for vectorized environments).
        capture_video (bool): Whether to save videos of the agent driving.
        run_name (str): Name of the experiment for video saving.
    """
    def thunk():
        # Initialize the environment
        # render_mode="rgb_array" is required for capturing video or processing pixels
        env = gym.make(env_id, render_mode="rgb_array")

        # Wrapper to record videos (optional, usually for evaluation or the first env)
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # 1. Grayscale Conversion
        # Transforms (96, 96, 3) -> (96, 96)
        # We keep_dim=False to remove the channel dimension before stacking
        env = GrayScaleObservation(env, keep_dim=False)

        # 2. Frame Stacking
        # Stacks 4 frames. New shape will be (4, 96, 96).
        # This provides temporal context (velocity/acceleration).
        env = FrameStack(env, 4)

        # Seed the environment for reproducibility. Adding idx ensures different seeds for different envs.
        env.action_space.seed(seed + idx)

        return env

    return thunk

def get_device():
    """Returns the device (CPU or CUDA) available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")