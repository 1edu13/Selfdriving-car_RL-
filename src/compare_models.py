import json
import re
from pathlib import Path
from datetime import datetime
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# import seaborn as sns  <-- Lo comento para evitar errores si no lo tienes instalado, el código usa matplotlib nativo

class ComparativeAnalysis:
    """
    Analyzes and compares evaluation results from multiple models.
    Supports extended comparison (e.g., 7 models).
    """

    def __init__(self, evaluation_results_dir: str = "evaluation_results"):
        self.eval_dir = Path(evaluation_results_dir)
        self.comparison_dir = Path("comparison_analysis")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)
        self.all_results = {}

        # Extended Color Palette for 7+ models
        # (Red, Orange, Yellow, Green, Cyan, Blue, Purple)
        self.colors = [
            "#FF6B6B",  # 200k (Redish)
            "#FFA07A",  # 500k (Orange)
            "#FFD700",  # 1M (Gold)
            "#90EE90",  # 1.5M (Light Green)
            "#4ECDC4",  # 2M (Teal)
            "#45B7D1",  # 2.5M (Sky Blue)
            "#9370DB"  # 3M (Purple)
        ]

    def load_model_results(self, model_name: str):
        """Loads results for a specific model."""
        results_file = self.eval_dir / model_name / "results.json"
        if not results_file.exists():
            print(f"⚠️  Not found: {results_file}")
            return None
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all_models(self, model_names):
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
        data = []
        for model, res in self.all_results.items():
            s = res["statistics"]
            data.append({
                "Model": model,
                "Mean Reward": s["mean_reward"],
                "Std Dev": s["std_reward"],
                "Win Rate (%)": s["win_rate"],
                "Avg Steps": s["mean_length"],
                "Steering (std)": s.get("steering_std", 0),
                "Throttle (mean)": s["throttle_mean"],
                "Brake (mean)": s["brake_mean"]
            })
        return pd.DataFrame(data)

    def get_all_episodes_dataframe(self) -> pd.DataFrame:
        all_eps = []
        for model, res in self.all_results.items():
            for ep in res["episodes"]:
                flat_ep = ep.copy()
                flat_ep["Model"] = model
                flat_ep["efficiency"] = ep["total_reward"] / max(ep["episode_length"], 1)
                all_eps.append(flat_ep)
        return pd.DataFrame(all_eps)

    # ------------------------------------------------------------------ #
    #  Helper: Step Parser
    # ------------------------------------------------------------------ #
    def _parse_steps(self, model_name):
        """Extracts step count from model name like 'model_0500k' or 'ppo_step_100'."""
        # Try k/m notation first (our new convention)
        if 'k' in model_name.lower():
            nums = re.findall(r'(\d+)k', model_name.lower())
            if nums: return int(nums[0]) * 1000
        if 'm' in model_name.lower():
            nums = re.findall(r'(\d+)m', model_name.lower())
            if nums: return int(nums[0]) * 1000000

        # Try standard number extraction
        nums = re.findall(r'\d+', model_name)
        if nums:
            val = int(nums[-1])
            # Heuristic: if small number < 5000, assumes it's 'k' notation without 'k'
            if val < 5000: return val * 1000
            return val

        return 0  # Fallback

    # ------------------------------------------------------------------ #
    #  Advanced Plotting
    # ------------------------------------------------------------------ #
    def plot_correlations(self, df_all: pd.DataFrame):
        """A. Heatmap de correlaciones."""
        cols = ["total_reward", "episode_length", "efficiency",
                "steering_std", "throttle_mean", "brake_mean"]
        rename_map = {
            "total_reward": "Reward", "episode_length": "Steps",
            "efficiency": "Speed/Eff", "steering_std": "Steer Activity",
            "throttle_mean": "Throttle", "brake_mean": "Brake"
        }
        df_corr = df_all[cols].rename(columns=rename_map)
        corr = df_corr.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(corr.columns)))
        ax.set_yticks(np.arange(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black")

        ax.set_title("Metric Correlations (All Models)", fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plot_file = self.comparison_dir / "A_correlation_heatmap.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"📊 Heatmap saved to: {plot_file}")

    def plot_radar_charts(self):
        """B. Radar chart de control (Profile)."""
        df = self.create_comparison_dataframe()
        categories = ['Steering (Activity)', 'Throttle', 'Brake', 'Consistency', 'Efficiency']

        df['Consistency'] = 1 / (df['Std Dev'] + 1e-6)
        df['Efficiency'] = df['Mean Reward'] / df['Avg Steps']
        df['Steering (Activity)'] = df['Steering (std)']

        # Normalize
        plot_df = pd.DataFrame()
        plot_df['Model'] = df['Model']
        for col in categories:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val == 0:
                plot_df[col] = 0.5
            else:
                plot_df[col] = (df[col] - min_val) / (max_val - min_val)

        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        # Use our extended palette
        palette = self.colors[:len(plot_df)]

        for idx, row in plot_df.iterrows():
            values = row[categories].tolist()
            values += values[:1]

            # Shorten name for legend
            short_name = row['Model'].replace("ppo_car_racing_step_", "")

            ax.plot(angles, values, linewidth=2, linestyle='solid', label=short_name, color=palette[idx % len(palette)])
            ax.fill(angles, values, color=palette[idx % len(palette)], alpha=0.05)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_title("Driving Profile Comparison (Normalized)", fontsize=15, fontweight="bold", y=1.05)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

        plot_file = self.comparison_dir / "B_control_radar.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"📊 Radar chart saved to: {plot_file}")

    def plot_learning_curve_log(self):
        """C. Learning curve (Reward vs Log Steps)."""
        steps = []
        rewards = []
        stds = []

        for name, res in self.all_results.items():
            s = self._parse_steps(name)
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

        # Line plot
        ax.plot(steps, rewards, 'o-', color='darkblue', linewidth=2, markersize=8)
        ax.fill_between(steps, rewards - stds, rewards + stds, color='blue', alpha=0.1, label='Std Dev')

        # Target Line
        ax.axhline(900, color='green', linestyle='--', label='Target (900)')

        ax.set_xlabel('Training Steps (Log Scale)', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title('Learning Curve (Log-Scale X)', fontsize=14, fontweight="bold")
        ax.grid(True, which="both", ls="-", alpha=0.4)
        ax.legend()

        # Annotate points with Step labels (e.g. "200k", "3M")
        for s, r in zip(steps, rewards):
            label = f"{int(s / 1000)}k" if s < 1000000 else f"{s / 1000000:.1f}M"
            ax.text(s, r + 40, label, ha='center', va='bottom', fontsize=9, fontweight='bold')

        plot_file = self.comparison_dir / "C_learning_curve_log.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"📊 Learning curve saved to: {plot_file}")

    def plot_scatter_survival(self, df_all: pd.DataFrame):
        """D. Scatter plot: Episode Length vs Reward."""
        fig, ax = plt.subplots(figsize=(12, 8))

        models = df_all['Model'].unique()
        palette = self.colors[:len(models)]

        for i, model in enumerate(models):
            subset = df_all[df_all['Model'] == model]
            short_name = model.replace("ppo_car_racing_step_", "")

            ax.scatter(subset['episode_length'], subset['total_reward'],
                       label=short_name, alpha=0.7, s=80, edgecolors='w', color=palette[i % len(palette)])

        ax.set_xlabel('Episode Length (Steps)')
        ax.set_ylabel('Total Reward')
        ax.set_title('Survival vs Success (Scatter)', fontsize=14, fontweight="bold")
        ax.axhline(900, color='green', linestyle='--', alpha=0.5, label='Win Threshold')
        ax.legend(title="Model Checkpoint")
        ax.grid(True, alpha=0.3)

        plot_file = self.comparison_dir / "D_scatter_survival.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"📊 Scatter plot saved to: {plot_file}")

    def plot_boxplots(self):
        """Box Plots for reward comparison."""
        fig, ax = plt.subplots(figsize=(14, 8))

        data_to_plot = []
        labels = []

        # Sort keys based on step parsing
        sorted_keys = sorted(self.all_results.keys(), key=lambda x: self._parse_steps(x))

        for model_name in sorted_keys:
            results = self.all_results[model_name]
            rewards = [ep["total_reward"] for ep in results["episodes"]]
            data_to_plot.append(rewards)

            short_name = model_name.replace("ppo_car_racing_step_", "")
            labels.append(short_name)

        box = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

        palette = self.colors[:len(labels)]
        for patch, color in zip(box['boxes'], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Total Reward")
        ax.set_title("Reward Stability & Variance", fontsize=16, fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.axhline(y=900, color="green", linestyle="--", linewidth=2, label="Target (900)")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plot_file = self.comparison_dir / "model_comparison_boxplot.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        print(f"📊 Boxplot saved to: {plot_file}")

    # ------------------------------------------------------------------ #
    #  Main Pipeline
    # ------------------------------------------------------------------ #
    def run_full_comparison(self, model_names):
        print("\n" + "=" * 80)
        print("ADVANCED COMPARATIVE ANALYSIS")
        print("=" * 80)

        self.load_all_models(model_names)
        df_all = self.get_all_episodes_dataframe()

        print("Generating visualizations...")
        self.plot_boxplots()  # Stability
        self.plot_correlations(df_all)  # A
        self.plot_radar_charts()  # B
        self.plot_learning_curve_log()  # C
        self.plot_scatter_survival(df_all)  # D

        # Save CSV data
        csv_path = self.comparison_dir / "all_episodes_data.csv"
        df_all.to_csv(csv_path, index=False)
        print(f"📄 Raw data exported to: {csv_path}")

        # Report textual simple
        report_path = self.comparison_dir / "comparison_report.txt"
        with open(report_path, "w") as f:
            f.write("Comparison completed.\n")
            f.write(f"Models analyzed: {len(model_names)}\n")
            f.write("Check plots in this folder.\n")

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETED")


if __name__ == "__main__":
    # Nombres EXACTOS de las carpetas que ya tienes generadas en evaluation_results
    existing_models = [
        "ppo_car_racing_step_500000",
        "ppo_car_racing_step_1000000",
        "ppo_car_racing_step_2000000",
    ]

    print("🚀 Iniciando prueba con modelos existentes...")
    analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")
    analyzer.run_full_comparison(existing_models)