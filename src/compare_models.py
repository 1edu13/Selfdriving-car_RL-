import json
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ComparativeAnalysis:
    """
    Analiza y compara resultados de evaluación de múltiples modelos.
    Genera reportes comparativos y visualizaciones.
    """

    def __init__(self, evaluation_results_dir: str = "evaluation_results"):
        """
        Args:
            evaluation_results_dir (str): Directorio donde están los resultados.
        """
        self.eval_dir = Path(evaluation_results_dir)
        self.comparison_dir = Path("comparison_analysis")
        self.comparison_dir.mkdir(parents=True, exist_ok=True)

        self.all_results = {}

    # ------------------------------------------------------------------ #
    #  Carga de resultados
    # ------------------------------------------------------------------ #
    def load_model_results(self, model_name: str):
        """Carga los resultados de un modelo específico."""
        results_file = self.eval_dir / model_name / "results.json"

        if not results_file.exists():
            print(f"⚠️  No se encontró: {results_file}")
            return None

        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        return results

    def load_all_models(self, model_names):
        """Carga resultados de todos los modelos especificados."""
        print("\n📂 Cargando resultados de modelos...")

        for model_name in model_names:
            print(f"  Cargando: {model_name}...", end=" ")
            results = self.load_model_results(model_name)
            if results is not None:
                self.all_results[model_name] = results
                print("✅")
            else:
                print("❌")

        if not self.all_results:
            raise ValueError("No se cargó ningún resultado válido")

        print(f"\n✓ {len(self.all_results)} modelos cargados exitosamente\n")

    # ------------------------------------------------------------------ #
    #  DataFrame comparativo
    # ------------------------------------------------------------------ #
    def create_comparison_dataframe(self) -> pd.DataFrame:
        """Crea un DataFrame con las estadísticas de todos los modelos."""
        comparison_data = []

        for model_name, results in self.all_results.items():
            stats = results["statistics"]
            comparison_data.append(
                {
                    "Modelo": model_name,
                    "Recompensa Media": stats["mean_reward"],
                    "Std Dev": stats["std_reward"],
                    "Min": stats["min_reward"],
                    "Max": stats["max_reward"],
                    "Tasa Victoria (%)": stats["win_rate"],
                    "Tasa Éxito (%)": stats["success_rate"],
                    "Pasos Promedio": stats["mean_length"],
                    "Episodios": results["num_episodes"],
                }
            )

        return pd.DataFrame(comparison_data)

    # ------------------------------------------------------------------ #
    #  Reporte textual
    # ------------------------------------------------------------------ #
    def _format_table(self, df: pd.DataFrame) -> str:
        """Formatea un DataFrame como tabla ASCII."""
        df_display = df.copy()
        for col in df_display.columns:
            if col not in ("Modelo", "Episodios"):
                df_display[col] = df_display[col].round(2)
        return df_display.to_string(index=False)

    def generate_comparison_report(self) -> str:
        """Genera un reporte textual comparativo."""
        df = self.create_comparison_dataframe()

        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║              ANÁLISIS COMPARATIVO DE MODELOS - SELF-DRIVING CAR            ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 RESUMEN EJECUTIVO
{'─' * 80}
Modelos Evaluados: {len(self.all_results)}
Fecha de Análisis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

┌────────────────────────────────────────────────────────────────────────────┐
│ TABLA COMPARATIVA DE RENDIMIENTO                                          │
└────────────────────────────────────────────────────────────────────────────┘

{self._format_table(df)}

╔════════════════════════════════════════════════════════════════════════════╗
║                          ANÁLISIS DETALLADO                               ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

        # Análisis por modelo
        for model_name, results in self.all_results.items():
            stats = results["statistics"]
            report += f"""
📌 {model_name.upper()}
{'─' * 80}
Episodios Evaluados:    {results['num_episodes']}
Device:                 {results['device']}
Seed:                   {results['seed']}

RENDIMIENTO:
  • Recompensa Media:   {stats['mean_reward']:>8.2f} ± {stats['std_reward']:>6.2f}
  • Rango:              [{stats['min_reward']:>7.1f}, {stats['max_reward']:>7.1f}]
  • Mediana:            {stats['median_reward']:>8.2f}
  • Tasa Victoria:      {stats['win_rate']:>8.1f}% (recompensa > 900)
  • Tasa Éxito:         {stats['success_rate']:>8.1f}% (recompensa > 0)

DURACIÓN:
  • Pasos Promedio:     {stats['mean_length']:>8.1f} ± {stats['std_length']:>6.1f}

CONTROL:
  • Dirección (mean):   {stats['steering_mean']:>8.4f}
  • Acelerador (mean):  {stats['throttle_mean']:>8.4f}
  • Freno (mean):       {stats['brake_mean']:>8.4f}
"""

        # Comparativa de mejora
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
🚀 ANÁLISIS DE MEJORA (Primero vs Último)
{'─' * 80}

Modelo Base:           {model_names[0]}
Modelo Final:          {model_names[-1]}

Recompensa Base:       {first['mean_reward']:>10.2f}
Recompensa Final:      {last['mean_reward']:>10.2f}
Mejora Absoluta:       {last['mean_reward'] - first['mean_reward']:>10.2f}
Mejora Relativa:       {improvement:>10.1f}%

Victoria Base:         {first['win_rate']:>10.1f}%
Victoria Final:        {last['win_rate']:>10.1f}%
Mejora en Victoria:    {last['win_rate'] - first['win_rate']:>10.1f}%

{'═' * 80}
"""

        report += """
💡 CONCLUSIONES Y RECOMENDACIONES
{'─' * 80}

1. RENDIMIENTO GENERAL:
   - Identificar el modelo con mayor recompensa media estable.
   - Los modelos entrenados más tiempo tienden a ser más estables.

2. ESTABILIDAD (Desviación Estándar):
   - Menor desviación = comportamiento más predecible.

3. TASA DE VICTORIA:
   - Objetivo: recompensa > 900.

4. DURACIÓN DE EPISODIOS:
   - Mayor duración indica que el agente se mantiene en pista.

5. PATRONES DE CONTROL:
   - Valores extremos de steering/brake pueden indicar conducción nerviosa.

╔════════════════════════════════════════════════════════════════════════════╗
║                          FIN DEL ANÁLISIS                                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"""
        return report

    # ------------------------------------------------------------------ #
    #  Gráficos
    # ------------------------------------------------------------------ #
    def plot_comparison(self):
        """Genera gráficos de comparación entre modelos."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Análisis Comparativo de Modelos RL", fontsize=18, fontweight="bold")

        model_names = list(self.all_results.keys())
        rewards_means = [self.all_results[m]["statistics"]["mean_reward"] for m in model_names]
        rewards_stds = [self.all_results[m]["statistics"]["std_reward"] for m in model_names]
        win_rates = [self.all_results[m]["statistics"]["win_rate"] for m in model_names]
        success_rates = [self.all_results[m]["statistics"]["success_rate"] for m in model_names]

        x_pos = np.arange(len(model_names))
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"][: len(model_names)]

        # Plot 1: Recompensa media
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
        axes[0, 0].axhline(y=900, color="g", linestyle="--", linewidth=2, label="Objetivo (900)")
        axes[0, 0].axhline(
            y=np.mean(rewards_means),
            color="r",
            linestyle=":",
            linewidth=2,
            label="Promedio",
        )
        axes[0, 0].set_ylabel("Recompensa", fontsize=12, fontweight="bold")
        axes[0, 0].set_title("Recompensa Media por Modelo", fontsize=13, fontweight="bold")
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(
            [name.replace("ppo_car_racing_step_", "") for name in model_names],
            rotation=45,
            ha="right",
        )
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3, axis="y")

        # Plot 2: Tasa de victoria
        bars = axes[0, 1].bar(
            x_pos, win_rates, color=colors, alpha=0.8, edgecolor="black", linewidth=2
        )
        axes[0, 1].axhline(y=100, color="g", linestyle="--", linewidth=2, alpha=0.5)
        axes[0, 1].set_ylabel("Porcentaje (%)", fontsize=12, fontweight="bold")
        axes[0, 1].set_title("Tasa de Victoria (Recompensa > 900)", fontsize=13, fontweight="bold")
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

        # Plot 3: Éxito vs Victoria
        width = 0.35
        axes[1, 0].bar(
            x_pos - width / 2,
            success_rates,
            width,
            label="Éxito (>0)",
            color="skyblue",
            alpha=0.8,
            edgecolor="black",
        )
        axes[1, 0].bar(
            x_pos + width / 2,
            win_rates,
            width,
            label="Victoria (>900)",
            color="orange",
            alpha=0.8,
            edgecolor="black",
        )
        axes[1, 0].set_ylabel("Porcentaje (%)", fontsize=12, fontweight="bold")
        axes[1, 0].set_title("Comparativa: Éxito vs Victoria", fontsize=13, fontweight="bold")
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(
            [name.replace("ppo_car_racing_step_", "") for name in model_names],
            rotation=45,
            ha="right",
        )
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 110])
        axes[1, 0].grid(alpha=0.3, axis="y")

        # Plot 4: Mejora relativa
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
            axes[1, 1].set_ylabel("Mejora (%)", fontsize=12, fontweight="bold")
            axes[1, 1].set_xlabel("Progresión del Entrenamiento", fontsize=12, fontweight="bold")
            axes[1, 1].set_title(
                "Mejora Relativa a lo Largo del Entrenamiento", fontsize=13, fontweight="bold"
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
        print(f"📊 Gráficos comparativos guardados en: {plot_file}")
        plt.close()

    # ------------------------------------------------------------------ #
    #  Distribuciones
    # ------------------------------------------------------------------ #
    def plot_distributions(self):
        """Genera gráficos de distribución de recompensas."""
        fig, axes = plt.subplots(1, len(self.all_results), figsize=(16, 5))
        fig.suptitle(
            "Distribuciones de Recompensa por Modelo",
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
                label=f"Media: {stats['mean_reward']:.1f}",
            )
            axes[idx].axvline(
                x=stats["median_reward"],
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Mediana: {stats['median_reward']:.1f}",
            )
            axes[idx].axvline(
                x=900,
                color="orange",
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                label="Objetivo: 900",
            )

            axes[idx].set_xlabel("Recompensa Total")
            axes[idx].set_ylabel("Frecuencia")
            short_name = model_name.replace("ppo_car_racing_step_", "")
            axes[idx].set_title(short_name, fontweight="bold")
            axes[idx].legend(fontsize=9)
            axes[idx].grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plot_file = self.comparison_dir / "reward_distributions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"📊 Distribuciones guardadas en: {plot_file}")
        plt.close()

    # ------------------------------------------------------------------ #
    #  CSV y pipeline completo
    # ------------------------------------------------------------------ #
    def save_comparison_csv(self):
        """Exporta la comparación como CSV."""
        df = self.create_comparison_dataframe()
        csv_file = self.comparison_dir / "model_comparison.csv"
        df.to_csv(csv_file, index=False)
        print(f"📄 Comparación exportada a CSV: {csv_file}")
        return csv_file

    def run_full_comparison(self, model_names):
        """Ejecuta el análisis comparativo completo."""
        print("\n" + "=" * 80)
        print("ANÁLISIS COMPARATIVO DE MODELOS RL")
        print("=" * 80)

        self.load_all_models(model_names)
        report = self.generate_comparison_report()

        report_file = self.comparison_dir / "comparison_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📄 Reporte guardado en: {report_file}")

        print("\n📊 Generando gráficos...")
        self.plot_comparison()
        self.plot_distributions()
        self.save_comparison_csv()

        print("\n" + "=" * 80)
        print("✅ ANÁLISIS COMPARATIVO COMPLETADO")
        print("=" * 80 + "\n")

        return report


if __name__ == "__main__":
    # Nombres de los modelos tal como están en evaluation_results/
    model_names = [
        "ppo_car_racing_step_500000",
        "ppo_car_racing_step_1000000",
        "ppo_car_racing_step_2000000",
    ]

    analyzer = ComparativeAnalysis(evaluation_results_dir="evaluation_results")
    rep = analyzer.run_full_comparison(model_names)
    print(rep)
