import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, FrameStack


class GrassPenaltyWrapper(gym.Wrapper):
    """
    Wrapper that penalizes if the agent drives on grass.
    Version 3.0:
    - ROI Logic (Region of Interest): Only looks at the car zone.
    - Avoids false positives from background grass.
    """

    def __init__(self, env, grass_penalty=0.8, max_off_track=50):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.max_off_track = max_off_track
        self.off_track_frames = 0
        self.episode_steps = 0
        self.debug_printed = False

    def reset(self, **kwargs):
        """Resets counters when starting a new episode."""
        self.off_track_frames = 0
        self.episode_steps = 0
        self.debug_printed = False
        return self.env.reset(**kwargs)

    def step(self, action):
        self.episode_steps += 1

        # 1. Standard environment step
        obs, rew, terminated, truncated, info = self.env.step(action)

        # --- GRACE PERIOD (INITIAL ZOOM) ---
        if self.episode_steps < 60:
            return obs, rew, terminated, truncated, info

        # 2. DEFINE REGION OF INTEREST (ROI)
        # The car is always centered at the bottom.
        # We crop only a rectangle around the car to ignore the landscape.
        # Vertical: 60 to 80 (Right in front and over the car)
        # Horizontal: 38 to 58 (The width of the central road)
        roi = obs[60:80, 38:58, :]

        # 3. DETECT GRASS IN THE ROI
        # We use dominant color logic on the crop (roi), not on the whole obs
        is_green = (
                (roi[:, :, 1] > roi[:, :, 0] + 10) &  # Green > Red
                (roi[:, :, 1] > roi[:, :, 2] + 10) &  # Green > Blue
                (roi[:, :, 1] > 100)  # Minimum brightness
        )

        # Grass ratio IN THE CAR ZONE
        green_ratio = np.mean(is_green)

        # 4. PENALTY
        # Now we can be stricter with the threshold (0.4) because we only look at the road.
        # If 40% of the car zone is green, it means you went off-track.
        if green_ratio > 0.40:
            rew -= self.grass_penalty
            self.off_track_frames += 1

            # Visual debug in console (only the first time it goes off-track in the episode)
            if not self.debug_printed and self.off_track_frames > 5:
                print(f"⚠️  OFF-TRACK DETECTED (Step {self.episode_steps})")
                self.debug_printed = True

            # Sudden death
            if self.off_track_frames > self.max_off_track:
                # print(f"💀 Sudden death (Step {self.episode_steps})")
                terminated = True
                info['off_track_timeout'] = True
        else:
            # If it returns to the track, we reset the counter
            if self.off_track_frames > 0:
                self.off_track_frames = 0
                self.debug_printed = False  # Allow printing again

        info['grass_ratio'] = green_ratio

        return obs, rew, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video, run_name, apply_grass_penalty=False):
    """
    Function to create and configure the environment.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")

        # --- APPLY WRAPPER ---
        if apply_grass_penalty:
            env = GrassPenaltyWrapper(env)
        # -----------------------

        env = GrayScaleObservation(env, keep_dim=False)
        env = FrameStack(env, 4)

        env.action_space.seed(seed + idx)
        return env

    return thunk


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")