#!/usr/bin/env python3
"""
MASTER SCRIPT - Full Evaluation Pipeline
Self-Driving Car using Reinforcement Learning
"""

import sys
from pathlib import Path

# Import evaluators
from evaluate_pro import RobustEvaluator
from compare_models import ComparativeAnalysis


def print_banner(title):
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def main():
    print_banner("EVALUATION PIPELINE - SELF-DRIVING CAR RL")

    # ========== CONFIGURATION ==========
    models_to_evaluate = {
        'ppo_car_racing_step_500000': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_step_491520.pth',
        'ppo_car_racing_step_1000000': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_step_1064960.pth',
        'ppo_car_racing_step_2000000': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_final.pth',
    }

    num_episodes = 30
    seed = 100

    # ========== PHASE 1: INDIVIDUAL EVALUATION ==========
    print_banner("PHASE 1: Individual Model Evaluation")

    evaluation_results = {}

    for model_name, model_path in models_to_evaluate.items():
        print(f"\n📊 Evaluating: {model_name}")
        if not Path(model_path).exists():
            print(f"⚠️  WARNING: File not found {model_path}")
            continue

        try:
            evaluator = RobustEvaluator(model_path, num_episodes, seed)
            all_metrics, stats = evaluator.run()
            evaluation_results[model_name] = True
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

    if not evaluation_results:
        print("❌ No models evaluated.")
        return False

    # ========== PHASE 2: COMPARATIVE ANALYSIS ==========
    print_banner("PHASE 2: Advanced Comparative Analysis")

    try:
        analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")
        analyzer.run_full_comparison(list(evaluation_results.keys()))
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        return False

    # ========== SUMMARY ==========
    print_banner("✅ PIPELINE COMPLETED")
    print("\n📁 NEW VISUALIZATIONS GENERATED (in comparison_analysis/):")
    print("   1. A_correlation_heatmap.png  (Metrics relationships)")
    print("   2. B_control_radar.png        (Driving style profile)")
    print("   3. C_learning_curve_log.png   (Training progress)")
    print("   4. D_scatter_survival.png     (Episode length vs Reward)")
    print("\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)