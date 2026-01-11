#!/usr/bin/env python3
"""
SCRIPT MAESTRO - Pipeline Completo de Evaluación
Self-Driving Car using Reinforcement Learning

Ejecuta evaluación de todos los modelos y genera reportes comparativos.
"""

import sys
import os
from pathlib import Path

# Importar evaluadores
from evaluate_pro import RobustEvaluator
from compare_models import ComparativeAnalysis


def print_banner(title):
    """Imprime un banner formateado."""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width + "\n")


def main():
    """Ejecuta el pipeline completo de evaluación."""
    
    print_banner("PIPELINE DE EVALUACIÓN - SELF-DRIVING CAR RL")
    
    # ========== CONFIGURACIÓN DE MODELOS ==========
    # CAMBIAR ESTAS RUTAS A TUS MODELOS REALES
    
    models_to_evaluate = {
        'ppo_car_racing_step_500000': r'C:\Users\emped\OneDrive\Documentos\MIS COSAS\Yo\3 CURSO\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_step_491520.pth',
        'ppo_car_racing_step_1000000': r'C:\Users\emped\OneDrive\Documentos\MIS COSAS\Yo\3 CURSO\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_step_1064960.pth',
        'ppo_car_racing_step_2000000': r'C:\Users\emped\OneDrive\Documentos\MIS COSAS\Yo\3 CURSO\Selfdriving-car_RL-\Models\models_T3\ppo_car_racing_final.pth',
    }
    
    # ========== PARÁMETROS DE EVALUACIÓN ==========
    num_episodes = 30  # Episodios por modelo
    seed = 100         # Seed para reproducibilidad
    
    # ========== FASE 1: EVALUAR CADA MODELO ==========
    
    print_banner("FASE 1: Evaluación de Modelos Individuales")
    
    evaluation_results = {}
    
    for model_name, model_path in models_to_evaluate.items():
        print(f"\n📊 Evaluando: {model_name}")
        print(f"   Archivo: {model_path}")
        print("-" * 80)
        
        # Verificar que el archivo existe
        if not Path(model_path).exists():
            print(f"⚠️  ADVERTENCIA: No se encontró el archivo {model_path}")
            print(f"   Por favor, actualiza la ruta en el script.")
            continue
        
        try:
            # Crear evaluador
            evaluator = RobustEvaluator(
                model_path=model_path,
                num_episodes=num_episodes,
                seed=seed
            )
            
            # Ejecutar evaluación
            all_metrics, stats = evaluator.run()
            
            evaluation_results[model_name] = {
                'metrics': all_metrics,
                'stats': stats
            }
            
            print(f"\n✅ {model_name} evaluado exitosamente")
            
        except Exception as e:
            print(f"\n❌ Error evaluando {model_name}:")
            print(f"   {str(e)}")
            continue
    
    if not evaluation_results:
        print("\n" + "!" * 80)
        print("ERROR: No se pudo evaluar ningún modelo.")
        print("Por favor, verifica las rutas de los modelos en la configuración.")
        print("!" * 80)
        return False
    
    # ========== FASE 2: ANÁLISIS COMPARATIVO ==========
    
    print_banner("FASE 2: Análisis Comparativo")
    
    try:
        # Crear analizador
        analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")
        
        # Cargar modelos evaluados
        model_names = list(evaluation_results.keys())
        
        # Ejecutar análisis completo
        report = analyzer.run_full_comparison(model_names)
        
        print(report)
        
    except Exception as e:
        print(f"\n❌ Error en análisis comparativo:")
        print(f"   {str(e)}")
        return False
    
    # ========== RESUMEN FINAL ==========
    
    print_banner("✅ PIPELINE COMPLETADO")
    
    print("\n📁 ARCHIVOS GENERADOS:\n")
    
    print("  RESULTADOS INDIVIDUALES:")
    print("  └─ evaluation_results/")
    for model_name in evaluation_results.keys():
        print(f"      ├─ {model_name}/")
        print(f"      │   ├─ results.json          (Datos detallados)")
        print(f"      │   ├─ evaluation_plots.png  (Gráficos)")
        print(f"      │   ├─ report.txt            (Reporte textual)")
        print(f"      │   └─ videos/               (Videos de episodios)")
    
    print("\n  ANÁLISIS COMPARATIVO:")
    print("  └─ comparison_analysis/")
    print("      ├─ model_comparison.png         (Gráficos comparativos)")
    print("      ├─ reward_distributions.png     (Distribuciones)")
    print("      ├─ comparison_report.txt        (Reporte comparativo)")
    print("      └─ model_comparison.csv         (Datos en CSV)")
    
    print("\n" + "=" * 80)
    print("📊 PRÓXIMOS PASOS PARA TU PRESENTACIÓN:")
    print("=" * 80)
    print("""
1. DOCUMENTACIÓN:
   - Copia comparison_report.txt a tu documentación
   - Incluye las gráficas (model_comparison.png, reward_distributions.png)
   - Genera un PDF con los resultados

2. PRESENTACIÓN:
   - Usa los videos de evaluation_results/*/videos/ en tu presentación
   - Incluye los gráficos de comparativa
   - Prepara un resumen de hallazgos principales

3. ANÁLISIS:
   - Identifica qué modelo tiene mejor rendimiento
   - Analiza la estabilidad (desviación estándar)
   - Documenta lecciones aprendidas

4. MEJORAS FUTURAS:
   - Considera entrenamiento adicional si no alcanzas 900
   - Ajusta hyperparámetros basado en los resultados
   - Experimenta con diferentes seeds para robustez
    """)
    
    print("=" * 80 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
