import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ComparativeAnalysis:
    """
    Analyzes and compares evaluation results from multiple models.
    Generates comparative reports and visualizations.
    """

    def __init__(self, evaluation_results_dir: str = "evaluation_results"):
        """
        Args:
            evaluation_results_dir (str): Directory containing the results.
        """
        self.eval_dir = Path(evaluation_results_dir)
        self.comparison_dir = Path("comparison_analysis")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = {}

    # ------------------------------------------------------------------ #
    #  Load Results
    # ------------------------------------------------------------------ #
    def load_model_results(self, model_name: str):
        """Loads results for a specific model."""
        results_file = self.eval_dir / model_name / "results.json"

        if not results_file.exists():
            print(f"⚠️  Not found: {results_file}")
            return None

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        return results

    def load_all_models(self, model_names):
        """Loads results for all specified models."""
        print("\n📂 Loading model results...")

        for model_name in model_names:
            print(f"  Loading: {model_name}...", end=" ")
            results = self.load_model_results(model_name)
            if results is not None:
                self.all_results[model_name] = results
                print("✅")
            else:
                print("❌")

        if not self.all_results:
            raise ValueError("No valid results were loaded.")

        print(f"\n✓ {len(self.all_results)} models loaded successfully\n")

    # ------------------------------------------------------------------ #
    #  Comparison DataFrame
    # ------------------------------------------------------------------ #
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Creates a DataFrame with statistics from all models."""
        comparison_data = []

        for model_name, results in self.all_results.items():
            stats = results["statistics"]
            comparison_data.append(
                {
                    "Model": model_name,
                    "Mean Reward": stats["mean_reward"],
                    "Std Dev": stats["std_reward"],
                    "Min": stats["min_reward"],
                    "Max": stats["max_reward"],
                    "Win Rate (%)": stats["win_rate"],
                    "Success Rate (%)": stats["success_rate"],
                    "Avg Steps": stats["mean_length"],
                    "Episodes": results["num_episodes"],
                }
            )

        return pd.DataFrame(comparison_data)

    # ------------------------------------------------------------------ #
    #  Text Report
    # ------------------------------------------------------------------ #
    def _format_table(self, df: pd.DataFrame) -> str:
        """Formats a DataFrame as an ASCII table."""
        df_display = df.copy()
        for col in df_display.columns:
            if col not in ("Model", "Episodes"):
                df_display[col] = df_display[col].round(2)
        return df_display.to_string(index=False)

    def generate_comparison_report(self) -> str:
        """Generates a textual comparison report."""
        df = self.create_comparison_dataframe()

        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                COMPARATIVE ANALYSIS - SELF-DRIVING CAR RL                  ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 EXECUTIVE SUMMARY
{'─' * 80}
Evaluated Models:  {len(self.all_results)}
Analysis Date:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

┌────────────────────────────────────────────────────────────────────────────┐
│ PERFORMANCE COMPARISON TABLE                                               │
└────────────────────────────────────────────────────────────────────────────┘

{self._format_table(df)}

╔════════════════════════════════════════════════════════════════════════════╗
║                          DETAILED ANALYSIS                                ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

        # Analysis by model
        for model_name, results in self.all_results.items():
            stats = results["statistics"]
            report += f"""
📌 {model_name.upper()}
{'─' * 80}
Evaluated Episodes:     {results['num_episodes']}
Device:                 {results['device']}
Seed:                   {results['seed']}

PERFORMANCE:
  • Mean Reward:        {stats['mean_reward']:>8.2f} ± {stats['std_reward']:>6.2f}
  • Range:              [{stats['min_reward']:>7.1f}, {stats['max_reward']:>7.1f}]
  • Median:             {stats['median_reward']:>8.2f}
  • Win Rate:           {stats['win_rate']:>8.1f}% (reward > 900)
  • Success Rate:       {stats['success_rate']:>8.1f}% (reward > 0)

DURATION:
  • Avg Steps:          {stats['mean_length']:>8.1f} ± {stats['std_length']:>6.1f}

CONTROL:
  • Steering (mean):    {stats['steering_mean']:>8.4f}
  • Throttle (mean):    {stats['throttle_mean']:>8.4f}
  • Brake (mean):       {stats['brake_mean']:>8.4f}
"""

        # Improvement comparison
        model_names = list(self.all_results.keys())
        if len(model_names) > 1:
            first = self.all_results[model_names[0]]["statistics"]
            last = self.all_results[model_names[-1]]["statistics"]
            improvement = (
                (last["mean_reward"] - first["mean_reward"])
                / (abs(first["mean_reward"]) + 1e-6)
                * 100
            )

            report += f"""
{'═' * 80}
🚀 IMPROVEMENT ANALYSIS (First vs Last)
{'─' * 80}

Base Model:            {model_names[0]}
Final Model:           {model_names[-1]}

Base Reward:           {first['mean_reward']:>10.2f}
Final Reward:          {last['mean_reward']:>10.2f}
Absolute Improvement:  {last['mean_reward'] - first['mean_reward']:>10.2f}
Relative Improvement:  {improvement:>10.1f}%

Base Win Rate:         {first['win_rate']:>10.1f}%
Final Win Rate:        {last['win_rate']:>10.1f}%
Win Rate Improvement:  {last['win_rate'] - first['win_rate']:>10.1f}%

{'═' * 80}
"""

        report += """
💡 CONCLUSIONS AND RECOMMENDATIONS
{'─' * 80}

1. OVERALL PERFORMANCE:
   - Identify the model with the highest stable mean reward.
   - Models trained longer tend to be more stable.

2. STABILITY (Standard Deviation):
   - Lower deviation = more predictable behavior.

3. WIN RATE:
   - Target: reward > 900.

4. EPISODE DURATION:
   - Longer duration (without reaching 1000 limit aimlessly) often indicates
     the agent is staying on track.

5. CONTROL PATTERNS:
   - Extreme values for steering/brake might indicate jittery driving.

╔════════════════════════════════════════════════════════════════════════════╗
║                          END OF ANALYSIS                                  ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
        return report

    # ------------------------------------------------------------------ #
    #  Plots
    # ------------------------------------------------------------------ #
    def plot_comparison(self):
        """Generates comparison plots between models."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Comparative Analysis of RL Models", fontsize=18, fontweight="bold")

        model_names = list(self.all_results.keys())
        rewards_means = [self.all_results[m]["statistics"]["mean_reward"] for m in model_names]
        rewards_stds = [self.all_results[m]["statistics"]["std_reward"] for m in model_names]
        win_rates = [self.all_results[m]["statistics"]["win_rate"] for m in model_names]
        success_rates = [self.all_results[m]["statistics"]["success_rate"] for m in model_names]

        x_pos = np.arange(len(model_names))
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"][: len(model_names)]

        # Plot 1: Mean Reward
        axes[0, 0].bar(
            x_pos,
            rewards_means,
            yerr=rewards_stds,
            capsize=10,
            color=colors,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )
        axes[0, 0].axhline(y=900, color="g", linestyle="--", linewidth=2, label="Target (900)")
        axes[0, 0].axhline(
            y=np.mean(rewards_means),
            color="r",
            linestyle=":",
            linewidth=2,
            label="Average",
        )
        axes[0, 0].set_ylabel("Reward", fontsize=12, fontweight="bold")
        axes[0, 0].set_title("Mean Reward by Model", fontsize=13, fontweight="bold")
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(
            [name.replace("ppo_car_racing_step_", "") for name in model_names],
            rotation=45,
            ha="right",
        )
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis="y")

        # Plot 2: Win Rate
        bars = axes[0, 1].bar(
            x_pos, win_rates, color=colors, alpha=0.8, edgecolor="black", linewidth=2
        )
        axes[0, 1].axhline(y=100, color="g", linestyle="--", linewidth=2, alpha=0.5)
        axes[0, 1].set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        axes[0, 1].set_title("Win Rate (Reward > 900)", fontsize=13, fontweight="bold")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(
            [name.replace("ppo_car_racing_step_", "") for name in model_names],
            rotation=45,
            ha="right",
        )
        axes[0, 1].set_ylim([0, 110])
        axes[0, 1].grid(alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Plot 3: Success vs Win
        width = 0.35
        axes[1, 0].bar(
            x_pos - width / 2,
            success_rates,
            width,
            label="Success (>0)",
            color="skyblue",
            alpha=0.8,
            edgecolor="black",
        )
        axes[1, 0].bar(
            x_pos + width / 2,
            win_rates,
            width,
            label="Win (>900)",
            color="orange",
            alpha=0.8,
            edgecolor="black",
        )
        axes[1, 0].set_ylabel("Percentage (%)", fontsize=12, fontweight="bold")
        axes[1, 0].set_title("Success vs Win Rate", fontsize=13, fontweight="bold")
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(
            [name.replace("ppo_car_racing_step_", "") for name in model_names],
            rotation=45,
            ha="right",
        )
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 110])
        axes[1, 0].grid(alpha=0.3, axis="y")

        # Plot 4: Relative Improvement
        if len(model_names) > 1:
            rewards_array = np.array(rewards_means)
            improvements = (rewards_array - rewards_array[0]) / (abs(rewards_array[0]) + 1e-6) * 100

            axes[1, 1].plot(
                range(len(model_names)),
                improvements,
                "o-",
                linewidth=3,
                markersize=10,
                color="darkblue",
                alpha=0.7,
            )
            axes[1, 1].axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.5)
            axes[1, 1].fill_between(range(len(model_names)), improvements, alpha=0.3)
            axes[1, 1].set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
            axes[1, 1].set_xlabel("Training Progression", fontsize=12, fontweight="bold")
            axes[1, 1].set_title(
                "Relative Improvement over Training", fontsize=13, fontweight="bold"
            )
            axes[1, 1].set_xticks(range(len(model_names)))
            axes[1, 1].set_xticklabels(
                [name.replace("ppo_car_racing_step_", "") for name in model_names],
                rotation=45,
                ha="right",
            )
            axes[1, 1].grid(alpha=0.3)

            for i, imp in enumerate(improvements):
                axes[1, 1].text(i, imp, f"{imp:.1f}%", ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plot_file = self.comparison_dir / "model_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"📊 Comparison plots saved to: {plot_file}")
        plt.close()

    # ------------------------------------------------------------------ #
    #  Distributions
    # ------------------------------------------------------------------ #
    def plot_distributions(self):
        """Generates reward distribution plots."""
        fig, axes = plt.subplots(1, len(self.all_results), figsize=(16, 5))
        fig.suptitle(
            "Reward Distributions by Model",
            fontsize=16,
            fontweight="bold",
        )

        if len(self.all_results) == 1:
            axes = [axes]

        for idx, (model_name, results) in enumerate(self.all_results.items()):
            rewards = [ep["total_reward"] for ep in results["episodes"]]
            stats = results["statistics"]

            axes[idx].hist(
                rewards, bins=15, color="steelblue", alpha=0.7, edgecolor="black"
            )
            axes[idx].axvline(
                x=stats["mean_reward"],
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {stats['mean_reward']:.1f}",
            )
            axes[idx].axvline(
                x=stats["median_reward"],
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {stats['median_reward']:.1f}",
            )
            axes[idx].axvline(
                x=900,
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Target: 900",
            )

            axes[idx].set_xlabel("Total Reward")
            axes[idx].set_ylabel("Frequency")
            short_name = model_name.replace("ppo_car_racing_step_", "")
            axes[idx].set_title(short_name, fontweight="bold")
            axes[idx].legend(fontsize=9)
            axes[idx].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plot_file = self.comparison_dir / "reward_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"📊 Distributions saved to: {plot_file}")
        plt.close()

    # ------------------------------------------------------------------ #
    #  CSV and Full Run
    # ------------------------------------------------------------------ #
    def save_comparison_csv(self):
        """Exports comparison to CSV."""
        df = self.create_comparison_dataframe()
        csv_file = self.comparison_dir / "model_comparison.csv"
        df.to_csv(csv_file, index=False)
        print(f"📄 Comparison exported to CSV: {csv_file}")
        return csv_file

    def run_full_comparison(self, model_names):
        """Runs the full comparative analysis."""
        print("\n" + "=" * 80)
        print("RL MODEL COMPARATIVE ANALYSIS")
        print("=" * 80)

        self.load_all_models(model_names)
        report = self.generate_comparison_report()

        report_file = self.comparison_dir / "comparison_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")

        print("\n📊 Generating plots...")
        self.plot_comparison()
        self.plot_distributions()
        self.save_comparison_csv()

        print("\n" + "=" * 80)
        print("✅ COMPARATIVE ANALYSIS COMPLETED")
        print("=" * 80 + "\n")

        return report


if __name__ == "__main__":
    # Model names as they appear in evaluation_results/
    model_names = [
        "ppo_car_racing_step_500000",
        "ppo_car_racing_step_1000000",
        "ppo_car_racing_step_2000000",
    ]

    analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")
    rep = analyzer.run_full_comparison(model_names)
    print(rep)