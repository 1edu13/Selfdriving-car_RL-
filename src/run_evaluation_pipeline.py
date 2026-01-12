#!/usr/bin/env python3
"""
MASTER SCRIPT - Full Evaluation Pipeline
Self-Driving Car using Reinforcement Learning

Executes evaluation for all models and generates comparative reports.
"""

import sys
import os
from pathlib import Path

# Import evaluators
from evaluate_pro import RobustEvaluator
from compare_models import ComparativeAnalysis


def print_banner(title):
    """Prints a formatted banner."""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def main():
    """Runs the full evaluation pipeline."""

    print_banner("EVALUATION PIPELINE - SELF-DRIVING CAR RL")

    # ========== MODEL CONFIGURATION ==========
    # CHANGE THESE PATHS TO YOUR ACTUAL MODELS

    models_to_evaluate = {
        'ppo_car_racing_step_500000': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_step_491520.pth',
        'ppo_car_racing_step_1000000': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_step_1064960.pth',
        'ppo_car_racing_step_2000000': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_final.pth',
    }

    # ========== EVALUATION PARAMETERS ==========
    num_episodes = 30  # Episodes per model
    seed = 100  # Seed for reproducibility

    # ========== PHASE 1: EVALUATE INDIVIDUAL MODELS ==========

    print_banner("PHASE 1: Individual Model Evaluation")

    evaluation_results = {}

    for model_name, model_path in models_to_evaluate.items():
        print(f"\n📊 Evaluating: {model_name}")
        print(f"   File: {model_path}")
        print("-" * 80)

        # Verify file exists
        if not Path(model_path).exists():
            print(f"⚠️  WARNING: File not found {model_path}")
            print(f"   Please update the path in the script.")
            continue

        try:
            # Create evaluator
            evaluator = RobustEvaluator(
                model_path=model_path,
                num_episodes=num_episodes,
                seed=seed
            )

            # Run evaluation
            all_metrics, stats = evaluator.run()

            evaluation_results[model_name] = {
                'metrics': all_metrics,
                'stats': stats
            }

            print(f"\n✅ {model_name} evaluated successfully")

        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}:")
            print(f"   {str(e)}")
            continue

    if not evaluation_results:
        print("\n" + "!" * 80)
        print("ERROR: No models could be evaluated.")
        print("Please check the model paths in the configuration.")
        print("!" * 80)
        return False

    # ========== PHASE 2: COMPARATIVE ANALYSIS ==========

    print_banner("PHASE 2: Comparative Analysis")

    try:
        # Create analyzer
        analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")

        # Load evaluated models
        model_names = list(evaluation_results.keys())

        # Run full analysis
        report = analyzer.run_full_comparison(model_names)

        print(report)

    except Exception as e:
        print(f"\n❌ Error in comparative analysis:")
        print(f"   {str(e)}")
        return False

    # ========== FINAL SUMMARY ==========

    print_banner("✅ PIPELINE COMPLETED")

    print("\n📁 GENERATED FILES:\n")

    print("  INDIVIDUAL RESULTS:")
    print("  └─ evaluation_results/")
    for model_name in evaluation_results.keys():
        print(f"      ├─ {model_name}/")
        print(f"      │   ├─ results.json          (Detailed data)")
        print(f"      │   ├─ evaluation_plots.png  (Performance plots)")
        print(f"      │   ├─ report.txt            (Text report)")
        print(f"      │   └─ videos/               (Episode videos)")

    print("\n  COMPARATIVE ANALYSIS:")
    print("  └─ comparison_analysis/")
    print("      ├─ model_comparison.png         (Comparative plots)")
    print("      ├─ reward_distributions.png     (Distributions)")
    print("      ├─ comparison_report.txt        (Comparative report)")
    print("      └─ model_comparison.csv         (Data in CSV)")

    print("\n" + "=" * 80)
    print("📊 NEXT STEPS FOR YOUR PRESENTATION:")
    print("=" * 80)
    print("""
1. DOCUMENTATION:
   - Copy 'comparison_report.txt' to your documentation.
   - Include graphs ('model_comparison.png', 'reward_distributions.png').
   - Generate a PDF with the results.

2. PRESENTATION:
   - Use videos from 'evaluation_results/*/videos/' in your slides.
   - Include comparison plots.
   - Prepare a summary of key findings.

3. ANALYSIS:
   - Identify the model with the best performance.
   - Analyze stability (standard deviation).
   - Document lessons learned.

4. FUTURE IMPROVEMENTS:
   - Consider additional training if target (900) is not met.
   - Adjust hyperparameters based on results.
   - Experiment with different seeds for robustness.
    """)

    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)