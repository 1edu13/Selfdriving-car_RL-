import os
import json
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from utils import make_env, get_device


class RobustEvaluator:
    """
    Professional evaluator for PPO models on CarRacing-v2.
    Captures detailed metrics, videos, and generates reports.
    """

    def __init__(self, model_path: str, num_episodes: int = 30, seed: int = 100):
        """
        Args:
            model_path (str): Path to the model .pth file.
            num_episodes (int): Number of episodes to evaluate.
            seed (int): Seed for reproducibility.
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.seed = seed
        self.device = get_device()
        self.model_name = os.path.basename(model_path).replace(".pth", "")

        # Output directories
        self.output_dir = Path("evaluation_results") / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir = self.output_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print("RL Evaluator - Self-Driving Car")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Seed: {self.seed}")
        print(f"Output Directory: {self.output_dir}")
        print(f"{'='*60}\n")

    # --------------------------------------------------------------------- #
    #  Load Agent
    # --------------------------------------------------------------------- #
    def load_agent(self) -> Agent:
        """Loads the trained model."""
        # Create dummy env to get correct shapes
        dummy_env = make_env(
            "CarRacing-v2", seed=self.seed, idx=0,
            capture_video=False, run_name="dummy"
        )()
        dummy_envs = gym.vector.SyncVectorEnv([lambda: dummy_env])

        # Initialize and load agent
        agent = Agent(dummy_envs).to(self.device)
        agent.load_state_dict(torch.load(self.model_path, map_location=self.device))
        agent.eval()

        dummy_env.close()
        return agent

    # --------------------------------------------------------------------- #
    #  Episode Evaluation
    # --------------------------------------------------------------------- #
    def evaluate_episode(self, agent: Agent, episode_num: int,
                         capture_video: bool = True) -> dict:
        """
        Runs a single episode and captures detailed metrics.

        Returns:
            dict: Episode metrics.
        """
        run_name = f"eval_{self.model_name}_ep{episode_num:03d}"

        if capture_video:
            env = make_env(
                "CarRacing-v2",
                seed=self.seed + episode_num,
                idx=0,
                capture_video=True,
                run_name=run_name,
            )()
        else:
            env = make_env(
                "CarRacing-v2",
                seed=self.seed + episode_num,
                idx=0,
                capture_video=False,
                run_name=run_name,
            )()

        obs, _ = env.reset(seed=self.seed + episode_num)
        obs = torch.as_tensor(np.array(obs)).float().unsqueeze(0).to(self.device)

        episode_reward = 0.0
        episode_length = 0
        done = False
        actions_taken = []
        rewards_per_step = []

        with torch.no_grad():
            while not done:
                # Deterministic action: policy mean
                hidden = agent.network(obs / 255.0)
                action_mean = agent.actor_mean(hidden)

                next_obs, reward, terminated, truncated, _ = env.step(
                    action_mean.cpu().numpy()[0]
                )

                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                actions_taken.append(action_mean.cpu().numpy()[0])
                rewards_per_step.append(reward)

                obs = torch.as_tensor(np.array(next_obs)).float().unsqueeze(0).to(self.device)

        env.close()

        # Compile metrics
        metrics = {
            "episode_num": episode_num,
            "total_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "avg_reward_per_step": float(
                episode_reward / max(episode_length, 1)
            ),
            "max_reward_step": float(max(rewards_per_step)) if rewards_per_step else 0.0,
            "min_reward_step": float(min(rewards_per_step)) if rewards_per_step else 0.0,
        }

        if actions_taken:
            actions_arr = np.array(actions_taken)
            metrics.update(
                {
                    "steering_mean": float(actions_arr[:, 0].mean()),
                    "steering_std": float(actions_arr[:, 0].std()),
                    "throttle_mean": float(actions_arr[:, 1].mean()),
                    "brake_mean": float(actions_arr[:, 2].mean()),
                }
            )
        else:
            metrics.update(
                {
                    "steering_mean": 0.0,
                    "steering_std": 0.0,
                    "throttle_mean": 0.0,
                    "brake_mean": 0.0,
                }
            )

        return metrics

    # --------------------------------------------------------------------- #
    #  Full Evaluation Loop
    # --------------------------------------------------------------------- #
    def run_full_evaluation(self):
        """Runs evaluation for all episodes."""
        print("Loading model...")
        agent = self.load_agent()

        all_metrics = []
        print(f"\nRunning {self.num_episodes} episodes...\n")

        for ep in range(self.num_episodes):
            capture_video = (ep % 5 == 0)  # Capture video every 5 episodes
            metrics = self.evaluate_episode(agent, ep, capture_video=capture_video)
            all_metrics.append(metrics)

            reward_str = f"{metrics['total_reward']:7.1f}"
            length_str = f"{metrics['episode_length']:4d}"
            status = "📹" if capture_video else "  "

            print(
                f"[{ep+1:2d}/{self.num_episodes}] "
                f"Reward: {reward_str} | Steps: {length_str} | {status}"
            )

        return all_metrics

    # --------------------------------------------------------------------- #
    #  Aggregated Statistics
    # --------------------------------------------------------------------- #
    @staticmethod
    def compute_statistics(all_metrics):
        """Computes aggregated statistics."""
        rewards = [m["total_reward"] for m in all_metrics]
        lengths = [m["episode_length"] for m in all_metrics]

        stats = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "median_reward": float(np.median(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "win_rate": float(
                np.sum([1 for r in rewards if r > 900]) / len(rewards) * 100
            ),
            "success_rate": float(
                np.sum([1 for r in rewards if r > 0]) / len(rewards) * 100
            ),
            "steering_mean": float(np.mean([m["steering_mean"] for m in all_metrics])),
            "throttle_mean": float(np.mean([m["throttle_mean"] for m in all_metrics])),
            "brake_mean": float(np.mean([m["brake_mean"] for m in all_metrics])),
        }
        return stats

    # --------------------------------------------------------------------- #
    #  Save Results
    # --------------------------------------------------------------------- #
    def save_results(self, all_metrics, stats):
        """Saves results to JSON."""
        results = {
            "model": self.model_name,
            "evaluation_date": datetime.now().isoformat(),
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "device": str(self.device),
            "statistics": stats,
            "episodes": all_metrics,
        }

        results_file = self.output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Results saved to: {results_file}")
        return results_file

    # --------------------------------------------------------------------- #
    #  Plots
    # --------------------------------------------------------------------- #
    def generate_plots(self, all_metrics, stats):
        """Generates visualization plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Evaluation: {self.model_name}", fontsize=16, fontweight="bold")

        rewards = [m["total_reward"] for m in all_metrics]
        lengths = [m["episode_length"] for m in all_metrics]
        episodes = list(range(1, len(rewards) + 1))

        # Plot 1: Rewards per Episode
        axes[0, 0].plot(episodes, rewards, "b-o", alpha=0.7)
        axes[0, 0].axhline(
            y=stats["mean_reward"],
            color="r",
            linestyle="--",
            label=f"Mean: {stats['mean_reward']:.1f}",
        )
        axes[0, 0].axhline(y=900, color="g", linestyle="--", label="Target (900)")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].set_title("Reward per Episode")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: Reward Histogram
        axes[0, 1].hist(
            rewards, bins=15, color="steelblue", edgecolor="black", alpha=0.7
        )
        axes[0, 1].axvline(
            x=stats["mean_reward"],
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {stats['mean_reward']:.1f}",
        )
        axes[0, 1].set_xlabel("Total Reward")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].set_title("Reward Distribution")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Episode Length
        axes[1, 0].plot(episodes, lengths, "g-o", alpha=0.7)
        axes[1, 0].axhline(
            y=stats["mean_length"],
            color="r",
            linestyle="--",
            label=f"Mean: {stats['mean_length']:.0f}",
        )
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Steps")
        axes[1, 0].set_title("Episode Length")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: Control Statistics
        control_actions = ["Steering", "Throttle", "Brake"]
        control_values = [
            stats["steering_mean"],
            stats["throttle_mean"],
            stats["brake_mean"],
        ]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        axes[1, 1].bar(
            control_actions,
            control_values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        axes[1, 1].set_ylabel("Average Value")
        axes[1, 1].set_title("Average Control Actions")
        axes[1, 1].grid(alpha=0.3, axis="y")
        axes[1, 1].set_ylim([-0.5, 1.0])

        plt.tight_layout()
        plot_file = self.output_dir / "evaluation_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"📊 Plots saved to: {plot_file}")
        plt.close()

    # --------------------------------------------------------------------- #
    #  Text Report
    # --------------------------------------------------------------------- #
    def generate_report(self, stats):
        """Generates a formatted text report."""
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║                  EVALUATION REPORT - RL MODEL                      ║
╚════════════════════════════════════════════════════════════════════╝

📋 MODEL INFORMATION
{'─' * 64}
Model Name:            {self.model_name}
File Path:             {self.model_path}
Evaluation Date:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Device:                {self.device}

📊 EVALUATION CONFIGURATION
{'─' * 64}
Evaluated Episodes:    {self.num_episodes}
Seed:                  {self.seed}
Environment:           CarRacing-v2
Video Captured:        Yes (every 5 episodes)

📈 PERFORMANCE STATISTICS
{'─' * 64}
Mean Reward:           {stats['mean_reward']:>10.2f} ± {stats['std_reward']:.2f}
Min Reward:            {stats['min_reward']:>10.2f}
Max Reward:            {stats['max_reward']:>10.2f}
Median Reward:         {stats['median_reward']:>10.2f}

Win Rate (>900):       {stats['win_rate']:>6.1f}%
Success Rate (>0):     {stats['success_rate']:>6.1f}%

Average Steps:         {stats['mean_length']:>10.1f} ± {stats['std_length']:.1f}

🎮 AVERAGE CONTROL ACTIONS
{'─' * 64}
Steering:              {stats['steering_mean']:>10.4f}
Throttle:              {stats['throttle_mean']:>10.4f}
Brake:                 {stats['brake_mean']:>10.4f}

💾 GENERATED FILES
{'─' * 64}
✓ results.json              - Detailed data (JSON)
✓ evaluation_plots.png      - Performance plots
✓ videos/                   - Episode videos
✓ report.txt                - This report

╔════════════════════════════════════════════════════════════════════╗
║                        END OF REPORT                               ║
╚════════════════════════════════════════════════════════════════════╝
"""
        report_file = self.output_dir / "report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(report)
        print(f"\n📄 Report saved to: {report_file}")

    # --------------------------------------------------------------------- #
    #  Full Pipeline
    # --------------------------------------------------------------------- #
    def run(self):
        """Runs the full evaluation pipeline."""
        all_metrics = self.run_full_evaluation()
        stats = self.compute_statistics(all_metrics)
        self.save_results(all_metrics, stats)
        self.generate_plots(all_metrics, stats)
        self.generate_report(stats)

        print(f"\n{'='*60}")
        print("✅ EVALUATION COMPLETED")
        print(f"{'='*60}\n")

        return all_metrics, stats


if __name__ == "__main__":
    # ========== MANUAL CONFIGURATION ==========
    # Example paths (replace with actual paths if running standalone)
    model_500k = r"path/to/ppo_car_racing_step_512000.pth"
    model_1m = r"path/to/ppo_car_racing_step_1024000.pth"
    model_2m = r"path/to/ppo_car_racing_step_2048000.pth"
    model_3m = r"../Models/models_T4/ppo_car_racing_final_3M.pth"

    # Uncomment the model you want to evaluate:
    # evaluator = RobustEvaluator(modelo_500k, num_episodes=30)
    # evaluator = RobustEvaluator(modelo_1m, num_episodes=30)
    evaluator = RobustEvaluator(model_2m, num_episodes=30)

    evaluator.run()