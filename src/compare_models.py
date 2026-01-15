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
    Generates comparative reports and visualizations (Classic + Advanced).
    """

    def __init__(self, evaluation_results_dir="evaluation_results"):
        """
        Args:
            evaluation_results_dir (str): Directory where results are located.
        """
        self.eval_dir = Path(evaluation_results_dir)
        self.comparison_dir = Path("comparison_analysis")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = {}

        # Extended Color Palette (Red -> Violet) for 7 models
        self.colors = [
            "#FF6B6B",  # 200k (Soft Red)
            "#FFA07A",  # 500k (Orange)
            "#FFD700",  # 1M (Gold)
            "#D4E157",  # 1250k (Lime Green) - NUEVO
            "#90EE90",  # 1.5M (Light Green)
            "#4ECDC4",  # 2M (Teal)
            "#45B7D1",  # 2.5M (Sky Blue)
            "#9370DB"  # 3M (Purple)
        ]

    # ------------------------------------------------------------------ #
    #  Data Loading & Helpers
    # ------------------------------------------------------------------ #
    def load_model_results(self, model_name):
        """Loads results for a specific model."""
        results_file = self.eval_dir / model_name / "results.json"

        if not results_file.exists():
            print(f"⚠️  Not found: {results_file}")
            return None

        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        return results

    def load_all_models(self, model_names):
        """Loads results for all specified models."""
        print("\n📂 Loading model results...")

        for model_name in model_names:
            print(f"  Loading: {model_name}...", end=" ")
            results = self.load_model_results(model_name)

            if results:
                self.all_results[model_name] = results
                print("✅")
            else:
                print("❌")

        if not self.all_results:
            raise ValueError("No valid results were loaded")

        print(f"\n✓ {len(self.all_results)} models loaded successfully\n")

    def _parse_steps(self, model_name):
        """Extracts step count from model name for sorting."""
        if 'k' in model_name.lower():
            nums = re.findall(r'(\d+)k', model_name.lower())
            if nums: return int(nums[0]) * 1000
        if 'm' in model_name.lower():
            nums = re.findall(r'(\d+)m', model_name.lower())
            if nums: return int(nums[0]) * 1000000
        nums = re.findall(r'\d+', model_name)
        if nums:
            val = int(nums[-1])
            return val * 1000 if val < 5000 else val
        return 0

    # ------------------------------------------------------------------ #
    #  Data Processing
    # ------------------------------------------------------------------ #
    def create_comparison_dataframe(self):
        """Creates a DataFrame with statistics from all models."""
        comparison_data = []

        for model_name, results in self.all_results.items():
            stats = results['statistics']

            # --- CORRECCIÓN ---
            # Intentar obtener steering_std de stats, si no, calcularlo desde los episodios
            steering_std = stats.get('steering_std')
            if steering_std is None or steering_std == 0:
                # Recopilar todos los steering_std de los episodios
                all_stds = [ep.get('steering_std', 0) for ep in results.get('episodes', [])]
                # Calcular el promedio (evitando división por cero)
                steering_std = sum(all_stds) / len(all_stds) if all_stds else 0
            # ------------------

            comparison_data.append({
                'Model': model_name,
                'Mean Reward': stats['mean_reward'],
                'Std Dev': stats['std_reward'],
                'Min': stats['min_reward'],
                'Max': stats['max_reward'],
                'Win Rate (%)': stats['win_rate'],
                'Success Rate (%)': stats['success_rate'],
                'Avg Steps': stats['mean_length'],
                'Episodes': results['num_episodes'],
                # Usar el valor corregido
                'Steering (std)': steering_std,
                'Throttle (mean)': stats.get('throttle_mean', 0),
                'Brake (mean)': stats.get('brake_mean', 0)
            })

        df = pd.DataFrame(comparison_data)
        return df

    def get_all_episodes_dataframe(self) -> pd.DataFrame:
        """Creates a detailed DataFrame with ALL episodes."""
        all_eps = []
        for model, res in self.all_results.items():
            for ep in res["episodes"]:
                flat_ep = ep.copy()
                flat_ep["Model"] = model
                flat_ep["efficiency"] = ep["total_reward"] / max(ep["episode_length"], 1)
                all_eps.append(flat_ep)
        return pd.DataFrame(all_eps)

    # ------------------------------------------------------------------ #
    #  Text Report (Existing Structure, Translated)
    # ------------------------------------------------------------------ #
    def generate_comparison_report(self):
        """Generates a textual comparison report."""
        df = self.create_comparison_dataframe()

        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║              COMPARATIVE ANALYSIS OF MODELS - SELF-DRIVING CAR             ║
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
            stats = results['statistics']
            report += f"""
📌 {model_name.upper()}
{'─' * 80}
Evaluated Episodes:     {results['num_episodes']}
Device:                 {results.get('device', 'N/A')}
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
            # Sort models logic could be added here, currently taking insertion order
            first_stats = self.all_results[model_names[0]]['statistics']
            last_stats = self.all_results[model_names[-1]]['statistics']

            improvement = ((last_stats['mean_reward'] - first_stats['mean_reward'])
                           / abs(first_stats['mean_reward'] + 1e-6) * 100)

            report += f"""
{'═' * 80}
🚀 IMPROVEMENT ANALYSIS (First vs Last)
{'─' * 80}

Base Model:            {model_names[0]}
Final Model:           {model_names[-1]}

Base Reward:           {first_stats['mean_reward']:>10.2f}
Final Reward:          {last_stats['mean_reward']:>10.2f}
Absolute Improvement:  {last_stats['mean_reward'] - first_stats['mean_reward']:>10.2f}
Relative Improvement:  {improvement:>10.1f}%

Base Win Rate:         {first_stats['win_rate']:>10.1f}%
Final Win Rate:        {last_stats['win_rate']:>10.1f}%
Win Rate Improvement:  {last_stats['win_rate'] - first_stats['win_rate']:>10.1f}%

{'═' * 80}

"""
        return report

    def _format_table(self, df):
        """Formats a DataFrame as an ASCII table."""
        df_display = df.copy()
        # Keep only main columns for the text table
        cols_to_keep = ['Model', 'Mean Reward', 'Std Dev', 'Win Rate (%)', 'Avg Steps', 'Episodes']
        existing_cols = [c for c in cols_to_keep if c in df_display.columns]
        df_display = df_display[existing_cols]

        for col in df_display.columns:
            if col != 'Model' and col != 'Episodes':
                df_display[col] = df_display[col].round(2)

        return df_display.to_string(index=False)

    # ------------------------------------------------------------------ #
    #  EXISTING PLOTS (Originals - Translated)
    # ------------------------------------------------------------------ #
    def plot_comparison(self):
        """Generates comparison plots (Original 2x2 Bar Charts)."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Comparative Analysis of RL Models", fontsize=18, fontweight='bold')

        model_names = list(self.all_results.keys())
        # Sort for plotting
        model_names.sort(key=self._parse_steps)

        rewards_means = [self.all_results[m]['statistics']['mean_reward'] for m in model_names]
        rewards_stds = [self.all_results[m]['statistics']['std_reward'] for m in model_names]
        win_rates = [self.all_results[m]['statistics']['win_rate'] for m in model_names]
        success_rates = [self.all_results[m]['statistics']['success_rate'] for m in model_names]

        x_pos = np.arange(len(model_names))
        colors = self.colors[:len(model_names)]

        short_names = [name.replace('ppo_car_racing_step_', '').replace('.pth', '') for name in model_names]

        # Plot 1: Mean Reward
        axes[0, 0].bar(x_pos, rewards_means, yerr=rewards_stds, capsize=10,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        axes[0, 0].axhline(y=900, color='g', linestyle='--', linewidth=2, label='Target (900)')
        axes[0, 0].axhline(y=np.mean(rewards_means), color='r', linestyle=':', linewidth=2, label='Average')
        axes[0, 0].set_ylabel('Reward', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Mean Reward by Model', fontsize=13, fontweight='bold')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(short_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis='y')

        # Plot 2: Win Rate
        bars = axes[0, 1].bar(x_pos, win_rates, color=colors, alpha=0.8,
                              edgecolor='black', linewidth=2)
        axes[0, 1].axhline(y=100, color='g', linestyle='--', linewidth=2, alpha=0.5)
        axes[0, 1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Win Rate (Reward > 900)', fontsize=13, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(short_names, rotation=45, ha='right')
        axes[0, 1].set_ylim([0, 110])
        axes[0, 1].grid(alpha=0.3, axis='y')

        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Plot 3: Success vs Win
        width = 0.35
        axes[1, 0].bar(x_pos - width / 2, success_rates, width, label='Success (>0)',
                       color='skyblue', alpha=0.8, edgecolor='black')
        axes[1, 0].bar(x_pos + width / 2, win_rates, width, label='Win (>900)',
                       color='orange', alpha=0.8, edgecolor='black')
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Comparison: Success vs Win', fontsize=13, fontweight='bold')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(short_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 110])
        axes[1, 0].grid(alpha=0.3, axis='y')

        # Plot 4: Improvement
        if len(model_names) > 1:
            rewards_array = np.array(rewards_means)
            improvements = ((rewards_array - rewards_array[0]) / abs(rewards_array[0] + 1e-6) * 100)

            axes[1, 1].plot(range(len(model_names)), improvements, 'o-', linewidth=3,
                            markersize=10, color='darkblue', alpha=0.7)
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            axes[1, 1].fill_between(range(len(model_names)), improvements, alpha=0.3)
            axes[1, 1].set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Training Progression', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Relative Improvement Over Training', fontsize=13, fontweight='bold')
            axes[1, 1].set_xticks(range(len(model_names)))
            axes[1, 1].set_xticklabels(short_names, rotation=45, ha='right')
            axes[1, 1].grid(alpha=0.3)

            for i, imp in enumerate(improvements):
                axes[1, 1].text(i, imp, f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plot_file = self.comparison_dir / "model_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comparative plots saved to: {plot_file}")
        plt.close()

    def plot_distributions(self):
        """Generates reward distribution plots (Original Histograms)."""
        model_names = list(self.all_results.keys())
        model_names.sort(key=self._parse_steps)

        fig, axes = plt.subplots(1, len(model_names), figsize=(16, 5))
        fig.suptitle("Reward Distributions by Model", fontsize=16, fontweight='bold')

        if len(model_names) == 1:
            axes = [axes]

        for idx, model_name in enumerate(model_names):
            rewards = [ep['total_reward'] for ep in self.all_results[model_name]['episodes']]
            stats = self.all_results[model_name]['statistics']

            axes[idx].hist(rewards, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            axes[idx].axvline(x=stats['mean_reward'], color='red', linestyle='--',
                              linewidth=2, label=f'Mean: {stats["mean_reward"]:.1f}')
            axes[idx].axvline(x=stats['median_reward'], color='green', linestyle='--',
                              linewidth=2, label=f'Median: {stats["median_reward"]:.1f}')
            axes[idx].axvline(x=900, color='orange', linestyle='--',
                              linewidth=2, alpha=0.7, label='Target: 900')

            axes[idx].set_xlabel('Total Reward')
            axes[idx].set_ylabel('Frequency')
            short_name = model_name.replace('ppo_car_racing_step_', '').replace('.pth', '')
            axes[idx].set_title(f'{short_name}', fontweight='bold')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plot_file = self.comparison_dir / "reward_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Distributions saved to: {plot_file}")
        plt.close()

    # ------------------------------------------------------------------ #
    #  NEW ADVANCED PLOTS (Added)
    # ------------------------------------------------------------------ #
    def plot_correlations(self, df_all):
        """A. Correlation Heatmap."""
        cols = ["total_reward", "episode_length", "efficiency",
                "steering_std", "throttle_mean", "brake_mean"]
        map_cols = {"total_reward": "Reward", "episode_length": "Steps", "efficiency": "Speed/Eff",
                    "steering_std": "Steer Act.", "throttle_mean": "Throttle", "brake_mean": "Brake"}

        valid = [c for c in cols if c in df_all.columns]
        df_corr = df_all[valid].rename(columns=map_cols).corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(df_corr, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(np.arange(len(df_corr.columns)))
        ax.set_yticks(np.arange(len(df_corr.columns)))
        ax.set_xticklabels(df_corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(df_corr.columns)

        for i in range(len(df_corr)):
            for j in range(len(df_corr)):
                ax.text(j, i, f"{df_corr.iloc[i, j]:.2f}", ha="center", va="center")

        ax.set_title("Metric Correlations", fontweight="bold")
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "A_correlation_heatmap.png", dpi=300)
        plt.close()
        print("📊 Saved: A_correlation_heatmap.png")

    def plot_radar_charts(self):
        """B. Control Profile Radar Chart."""
        df = self.create_comparison_dataframe()

        # Fix column names for consistency
        df['Throttle'] = df['Throttle (mean)']
        df['Brake'] = df['Brake (mean)']
        df['Consistency'] = 1 / (df['Std Dev'] + 1e-6)
        df['Efficiency'] = df['Mean Reward'] / df['Avg Steps']
        df['Steering (Activity)'] = df['Steering (std)']

        categories = ['Steering (Activity)', 'Throttle', 'Brake', 'Consistency', 'Efficiency']

        plot_df = pd.DataFrame()
        plot_df['Model'] = df['Model']
        for col in categories:
            if col in df.columns:
                min_v, max_v = df[col].min(), df[col].max()
                plot_df[col] = (df[col] - min_v) / (max_v - min_v) if max_v > min_v else 0.5
            else:
                plot_df[col] = 0.0

        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        palette = self.colors[:len(plot_df)]

        for idx, row in plot_df.iterrows():
            values = row[categories].tolist()
            values += values[:1]
            short_name = row['Model'].replace("ppo_car_racing_step_", "")
            ax.plot(angles, values, linewidth=2, label=short_name, color=palette[idx % len(palette)])
            ax.fill(angles, values, color=palette[idx % len(palette)], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
        ax.set_title("Driving Profile Comparison (Normalized)", fontsize=15, fontweight="bold", y=1.05)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))

        plt.savefig(self.comparison_dir / "B_control_radar.png", dpi=300)
        plt.close()
        print("📊 Saved: B_control_radar.png")

    def plot_learning_curve_log(self):
        """C. Learning Curve (Log Scale)."""
        steps, rewards, stds = [], [], []

        for name, res in self.all_results.items():
            s = self._parse_steps(name)
            steps.append(s)
            rewards.append(res["statistics"]["mean_reward"])
            stds.append(res["statistics"]["std_reward"])

        idxs = np.argsort(steps)
        steps = np.array(steps)[idxs]
        rewards = np.array(rewards)[idxs]
        stds = np.array(stds)[idxs]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xscale('log')
        ax.plot(steps, rewards, 'o-', color='darkblue', linewidth=2)
        ax.fill_between(steps, rewards - stds, rewards + stds, color='blue', alpha=0.1)
        ax.axhline(900, color='green', linestyle='--', label='Target (900)')

        ax.set_xlabel('Training Steps (Log Scale)')
        ax.set_ylabel('Mean Reward')
        ax.set_title('Learning Curve (Log-Scale)', fontweight="bold")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend()

        for s, r in zip(steps, rewards):
            lbl = f"{int(s / 1000)}k" if s < 1000000 else f"{s / 1000000:.1f}M"
            ax.text(s, r + 40, lbl, ha='center', fontsize=9, fontweight='bold')

        plt.savefig(self.comparison_dir / "C_learning_curve_log.png", dpi=300)
        plt.close()
        print("📊 Saved: C_learning_curve_log.png")

    def plot_scatter_survival(self, df_all):
        """D. Scatter Plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        models = df_all['Model'].unique()

        for i, m in enumerate(models):
            sub = df_all[df_all['Model'] == m]
            short = m.replace("ppo_car_racing_step_", "")
            ax.scatter(sub['episode_length'], sub['total_reward'], label=short, alpha=0.7,
                       color=self.colors[i % len(self.colors)])

        ax.set_xlabel('Steps');
        ax.set_ylabel('Reward')
        ax.set_title('Survival vs Success', fontweight="bold")
        ax.legend()
        plt.savefig(self.comparison_dir / "D_scatter_survival.png", dpi=300)
        plt.close()
        print("📊 Saved: D_scatter_survival.png")

    def plot_boxplots(self):
        """E. Stability Boxplot."""
        fig, ax = plt.subplots(figsize=(12, 7))
        data, labels = [], []
        sorted_keys = sorted(self.all_results.keys(), key=self._parse_steps)

        for m in sorted_keys:
            data.append([ep["total_reward"] for ep in self.all_results[m]["episodes"]])
            labels.append(m.replace("ppo_car_racing_step_", ""))

        try:
            box = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        except:
            box = ax.boxplot(data, labels=labels, patch_artist=True)

        for patch, color in zip(box['boxes'], self.colors):
            patch.set_facecolor(color);
            patch.set_alpha(0.7)

        ax.set_ylabel("Total Reward");
        ax.set_title("Reward Stability (Box Plot)", fontweight="bold")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.comparison_dir / "model_comparison_boxplot.png", dpi=300)
        plt.close()
        print("📊 Saved: model_comparison_boxplot.png")

    # ------------------------------------------------------------------ #
    #  Exports
    # ------------------------------------------------------------------ #
    def save_comparison_csv(self):
        """Exports the comparison to CSV."""
        df = self.create_comparison_dataframe()

        csv_file = self.comparison_dir / "model_comparison.csv"
        df.to_csv(csv_file, index=False)

        print(f"📄 Comparison exported to CSV: {csv_file}")
        return csv_file

    def run_full_comparison(self, model_names):
        """Runs the full comparative analysis."""
        print("\n" + "=" * 80)
        print("COMPARATIVE ANALYSIS OF RL MODELS")
        print("=" * 80)

        # Load data
        self.load_all_models(model_names)

        # Generate text report
        report = self.generate_comparison_report()

        # Save report
        report_file = self.comparison_dir / "comparison_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📄 Report saved to: {report_file}")

        # Prepare data for new plots
        df_all = self.get_all_episodes_dataframe()

        # Generate plots
        print("\n📊 Generating plots...")
        self.plot_comparison()  # Classic 1 (Bars)
        self.plot_distributions()  # Classic 2 (Histograms)
        self.plot_correlations(df_all)  # New A
        self.plot_radar_charts()  # New B
        self.plot_learning_curve_log()  # New C
        self.plot_scatter_survival(df_all)  # New D
        self.plot_boxplots()  # New E

        # Export CSV
        self.save_comparison_csv()

        print("\n" + "=" * 80)
        print("✅ COMPARATIVE ANALYSIS COMPLETED")
        print("=" * 80 + "\n")

        return report


if __name__ == "__main__":
    # ========== CONFIGURATION ===========

    # Model names as they appear in the directories
    model_names = [
        "model_0200k",  # 500K steps
        "model_0500k",  # 1M steps
        "model_1000k",  # 2M steps
        "model_1250k",
        "model_1500k",  # 500K steps
        "model_2000k",  # 1M steps
        "model_2500k",  # 2M steps
        "model_3000k"
    ]

    # Run analysis
    analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")
    report = analyzer.run_full_comparison(model_names)

    # Show report
    # print(report)