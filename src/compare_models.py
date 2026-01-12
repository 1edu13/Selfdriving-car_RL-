import json
import re
from pathlib import Path
from datetime import datetime
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ComparativeAnalysis:
    """
    Analyzes and compares evaluation results from multiple models.
    Generates comparative reports and visualizations.
    """

    def __init__(self, evaluation_results_dir: str = "evaluation_results"):
        self.eval_dir = Path(evaluation_results_dir)
        self.comparison_dir = Path("comparison_analysis")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.all_results = {}

    def load_model_results(self, model_name: str):
        """Loads results for a specific model."""
        results_file = self.eval_dir / model_name / "results.json"
        if not results_file.exists():
            print(f"⚠️  Not found: {results_file}")
            return None
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)

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

    # ------------------------------------------------------------------ #
    #  Data Processing
    # ------------------------------------------------------------------ #
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Creates a DataFrame with aggregated statistics."""
        data = []
        for model, res in self.all_results.items():
            s = res["statistics"]
            data.append({
                "Model": model,
                "Mean Reward": s["mean_reward"],
                "Std Dev": s["std_reward"],
                "Win Rate (%)": s["win_rate"],
                "Avg Steps": s["mean_length"],
                "Steering (std)": s.get("steering_std", 0),  # Fallback if not present
                "Throttle (mean)": s["throttle_mean"],
                "Brake (mean)": s["brake_mean"]
            })
        return pd.DataFrame(data)

    def get_all_episodes_dataframe(self) -> pd.DataFrame:
        """Creates a detailed DataFrame with ALL episodes from ALL models."""
        all_eps = []
        for model, res in self.all_results.items():
            for ep in res["episodes"]:
                flat_ep = ep.copy()
                flat_ep["Model"] = model
                # Add calculated metrics
                # Efficiency: Reward per step (proxy for Speed/Efficiency)
                flat_ep["efficiency"] = ep["total_reward"] / max(ep["episode_length"], 1)
                all_eps.append(flat_ep)
        return pd.DataFrame(all_eps)

    # ------------------------------------------------------------------ #
    #  Advanced Plotting Logic
    # ------------------------------------------------------------------ #
    def plot_correlations(self, df_all: pd.DataFrame):
        """A. Heatmap de correlaciones."""
        # Select numeric columns relevant for correlation
        cols = ["total_reward", "episode_length", "efficiency",
                "steering_std", "throttle_mean", "brake_mean"]

        # Rename for cleaner plot
        rename_map = {
            "total_reward": "Reward", "episode_length": "Steps",
            "efficiency": "Speed/Eff", "steering_std": "Steer Activity",
            "throttle_mean": "Throttle", "brake_mean": "Brake"
        }
        df_corr = df_all[cols].rename(columns=rename_map)

        corr = df_corr.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        # Add labels
        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)

        # Add text annotations
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black")

        ax.set_title("Metric Correlations (All Models)", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Correlation Coefficient")

        out_path = self.comparison_dir / "A_correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"📊 Heatmap saved to: {out_path}")

    def plot_radar_charts(self):
        """B. Radar chart de control (Profile)."""
        # Prepare data: Normalize metrics 0-1 across models to compare profiles
        df = self.create_comparison_dataframe()

        # Metrics to display on radar
        categories = ['Steering (Activity)', 'Throttle', 'Brake', 'Consistency', 'Efficiency']
        # Consistency = 1 / StdDev of Reward (Normalized)
        # Efficiency = Mean Reward / Avg Steps (Normalized)

        # Calculate derived metrics
        df['Consistency'] = 1 / (df['Std Dev'] + 1e-6)
        df['Efficiency'] = df['Mean Reward'] / df['Avg Steps']
        df['Steering (Activity)'] = df['Steering (std)']  # Using std as activity proxy

        # Normalize columns for the chart (Min-Max scaling)
        plot_df = pd.DataFrame()
        plot_df['Model'] = df['Model']
        for col in categories:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val == 0:
                plot_df[col] = 0.5  # Default if all same
            else:
                plot_df[col] = (df[col] - min_val) / (max_val - min_val)

        # Plotting
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

        for idx, row in plot_df.iterrows():
            values = row[categories].tolist()
            values += values[:1]  # Close the loop

            short_name = row['Model'].split('_step_')[-1]
            if "final" in row['Model']: short_name = "Final"

            ax.plot(angles, values, linewidth=2, linestyle='solid', label=short_name, color=colors[idx % len(colors)])
            ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_title("Driving Profile Comparison (Normalized)", fontsize=15, fontweight="bold", y=1.05)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        out_path = self.comparison_dir / "B_control_radar.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"📊 Radar chart saved to: {out_path}")

    def plot_learning_curve_log(self):
        """C. Learning curve (Reward vs Log Steps)."""
        steps = []
        rewards = []
        stds = []

        for name, res in self.all_results.items():
            # Extract steps from filename
            match = re.search(r'step_(\d+)', name)
            if match:
                s = int(match.group(1))
            elif "final" in name:
                # Estimate final steps (e.g., 2M) or ask user.
                # For plotting, let's assume max of others * 2 or manual 2M
                s = 2000000
            else:
                continue

            steps.append(s)
            rewards.append(res["statistics"]["mean_reward"])
            stds.append(res["statistics"]["std_reward"])

        # Sort by steps
        sorted_indices = np.argsort(steps)
        steps = np.array(steps)[sorted_indices]
        rewards = np.array(rewards)[sorted_indices]
        stds = np.array(stds)[sorted_indices]

        fig, ax = plt.subplots(figsize=(10, 6))

        # Log scale X
        ax.set_xscale('log')

        ax.plot(steps, rewards, 'o-', color='purple', linewidth=2, markersize=8)
        ax.fill_between(steps, rewards - stds, rewards + stds, color='purple', alpha=0.2, label='Std Dev')

        # Target Line
        ax.axhline(900, color='green', linestyle='--', label='Target (900)')

        ax.set_xlabel('Training Steps (Log Scale)', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title('Learning Curve (Log-Scale X)', fontsize=14, fontweight="bold")
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend()

        # Annotate points
        for s, r in zip(steps, rewards):
            ax.text(s, r + 20, f"{int(r)}", ha='center', va='bottom', fontsize=9)

        out_path = self.comparison_dir / "C_learning_curve_log.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"📊 Learning curve saved to: {out_path}")

    def plot_scatter_survival(self, df_all: pd.DataFrame):
        """D. Scatter plot: Episode Length vs Reward."""
        fig, ax = plt.subplots(figsize=(10, 7))

        models = df_all['Model'].unique()
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

        for i, model in enumerate(models):
            subset = df_all[df_all['Model'] == model]
            short_name = model.split('_step_')[-1]
            if "final" in model: short_name = "Final"

            ax.scatter(subset['episode_length'], subset['total_reward'],
                       label=short_name, alpha=0.7, s=60, edgecolors='w', color=colors[i % len(colors)])

        ax.set_xlabel('Episode Length (Steps)')
        ax.set_ylabel('Total Reward')
        ax.set_title('Survival vs Success (Scatter)', fontsize=14, fontweight="bold")
        ax.axhline(900, color='green', linestyle='--', alpha=0.5, label='Win Threshold')
        ax.grid(True, alpha=0.3)
        ax.legend(title="Model Checkpoint")

        out_path = self.comparison_dir / "D_scatter_survival.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"📊 Scatter plot saved to: {out_path}")

    # ------------------------------------------------------------------ #
    #  Main Pipeline
    # ------------------------------------------------------------------ #
    def run_full_comparison(self, model_names):
        print("\n" + "=" * 80)
        print("ADVANCED COMPARATIVE ANALYSIS")
        print("=" * 80)

        self.load_all_models(model_names)

        # Create datasets
        df_all = self.get_all_episodes_dataframe()

        # Generate Text Report (Keeping previous logic simpler here for brevity,
        # but you can include the full text report logic from previous version)
        print("Generating visualizations...")

        # 1. Standard Plots (from before)
        # self.plot_comparison() # You can keep the bar charts if you want

        # 2. NEW Advanced Plots
        self.plot_correlations(df_all)  # A
        self.plot_radar_charts()  # B
        self.plot_learning_curve_log()  # C
        self.plot_scatter_survival(df_all)  # D

        # Save CSV data
        csv_path = self.comparison_dir / "all_episodes_data.csv"
        df_all.to_csv(csv_path, index=False)
        print(f"📄 Raw data exported to: {csv_path}")

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETED")
        return "Analysis Done"


if __name__ == "__main__":
    # Example usage
    model_names = [
        "ppo_car_racing_step_500000",
        "ppo_car_racing_step_1000000",
        "ppo_car_racing_step_2000000",
    ]
    analyzer = ComparativeAnalysis()
    analyzer.run_full_comparison(model_names)