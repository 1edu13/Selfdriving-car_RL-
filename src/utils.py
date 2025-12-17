import gymnasium as gym
import torch
import numpy as np
# CORRECCIÓN: Usamos GrayscaleObservation y FrameStackObservation
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, TransformObservation


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # 1. Grayscale Conversion
        # CORRECCIÓN: GrayscaleObservation (con 's' minúscula)
        env = GrayscaleObservation(env, keep_dim=False)

        # 2. Frame Stacking
        # CORRECCIÓN: FrameStackObservation
        env = FrameStackObservation(env, 4)

        env.action_space.seed(seed + idx)
        return env

    return thunk


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")