#!/usr/bin/env python3
"""
MASTER SCRIPT - Full Evaluation Pipeline (7-Model Comparison)
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
    # RUTAS DE LOS 7 MODELOS (AJUSTA LAS RUTAS REALES)
    # Sugerencia: Usa nombres de clave que indiquen los pasos para ordenamiento automático
    models_to_evaluate = {
        # Fase Inicial (Aprendizaje básico)
        'model_0200k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T4\model_0200k.pth',

        # Primer Progreso (El coche ya no gira en círculos)
        'model_0500k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T4\model_0500k.pth',  # Tu archivo anterior

        # Punto Medio (Mejora rápida)
        'model_1000k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T4\model_1000k.pth',  # Tu archivo anterior

        # Transición (Convergencia)
        'model_1500k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T4\model_1500k.pth',  # Aprox

        # Pre-Saturación (Tu modelo "final" anterior, que era 2M)
        'model_2000k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T4\model_2000k.pth',

        # Confirmación Rodilla
        'model_2500k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T4\model_2500k.pth',

        # Final Real (Saturación)
        'model_3000k': r'C:\Users\hmphu\PycharmProjects\Selfdriving-car_RL-\Models\models_T4\model_3000k.pth',
    }

    num_episodes = 30  # Episodios por modelo
    seed = 42  # Misma seed para todos

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
            # Usamos el nombre de la clave (model_0200k) como identificador de carpeta
            # para mantener el orden limpio
            evaluator = RobustEvaluator(model_path, num_episodes, seed)
            # Forzamos el nombre del modelo para que coincida con nuestra clave ordenada
            evaluator.model_name = model_key
            # Re-creamos directorios con el nuevo nombre
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
        # Pasamos la lista ordenada de claves para que los gráficos salgan en orden (200->3M)
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