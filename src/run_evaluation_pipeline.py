#!/usr/bin/env python3
"""
MASTER SCRIPT - Full Evaluation Pipeline (7-Model Comparison)
Self-Driving Car using Reinforcement Learning
"""

import sys
from pathlib import Path

# Import evaluators
from evaluate_pro2 import RobustEvaluator
from compare_models import ComparativeAnalysis


def print_banner(title):
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def main():
    print_banner("EVALUATION PIPELINE - SELF-DRIVING CAR RL")

    # ========== CONFIGURATION ==========
    # PATH TO THE 8 MODELS

    models_to_evaluate = {
        # Initial Phase (Basic Learning)
        'model_0200k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_B\model_0200k.pth',

        # First Progress (Car no longer spins in circles)
        'model_0500k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_B\model_0500k.pth',

        # Midpoint (Rapid Improvement)
        'model_1000k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_B\model_1000k.pth',

        # Midpoint+ (Rapid Improvement)
        'model_1250k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_B\model_grass_1250k.pth',

        # Transition (Convergence)
        'model_1500k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_B\model_1500k.pth',  # Approx

        # Pre-Saturation (Previous "final" model, which was 2M)
        'model_2000k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_B\model_2000k.pth',

        # Knee Confirmation
        'model_2500k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T6_grass\model_2500k.pth',

        # Real Final (Saturation)
        'model_3000k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T6_grass\model_3000k.pth',
    }

    num_episodes = 30  # Episodes per model
    seed = 100  # Same seed for all

    # ========== PHASE 1: INDIVIDUAL EVALUATION ==========
    print_banner("PHASE 1: Individual Model Evaluation")

    evaluation_results = {}

    for model_key, model_path in models_to_evaluate.items():
        print(f"\n📊 Evaluating: {model_key}")
        print(f"   Path: {model_path}")

        if not Path(model_path).exists():
            print(f"⚠️  WARNING: File not found {model_path}")
            print(f"   Skipping this model...")
            continue

        try:
            # Use the key name (model_0200k) as folder identifier
            # to keep the order clean
            evaluator = RobustEvaluator(model_path, num_episodes, seed)
            # Force the model name to match our sorted key
            evaluator.model_name = model_key
            # Re-create directories with the new name
            evaluator.output_dir = Path("evaluation_results") / model_key
            evaluator.output_dir.mkdir(parents=True, exist_ok=True)
            evaluator.videos_dir = evaluator.output_dir / "videos"
            evaluator.videos_dir.mkdir(exist_ok=True)

            all_metrics, stats = evaluator.run()
            evaluation_results[model_key] = True

        except Exception as e:
            print(f"❌ Error evaluating {model_key}: {e}")
            continue

    if not evaluation_results:
        print("❌ No models evaluated. Check your paths.")
        return False

    # ========== PHASE 2: COMPARATIVE ANALYSIS ==========
    print_banner("PHASE 2: Advanced Comparative Analysis")

    try:
        # Pass the sorted list of keys so plots appear in order (200->3M)
        sorted_models = sorted(list(evaluation_results.keys()))

        analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")
        analyzer.run_full_comparison(sorted_models)

    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

    # ========== SUMMARY ==========
    print_banner("✅ PIPELINE COMPLETED")
    print(f"Evaluated {len(evaluation_results)} models successfully.")
    print("Check 'comparison_analysis/' for the new 7-model charts.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)