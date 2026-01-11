import os
import json
from pathlib import Path
from datetime import datetime

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import Agent
from utils import make_env, get_device


class RobustEvaluator:
    """
    Evaluador profesional para modelos PPO en CarRacing-v2.
    Captura métricas detalladas, videos y genera reportes.
    """

    def __init__(self, model_path: str, num_episodes: int = 30, seed: int = 100):
        """
        Args:
            model_path (str): Ruta al archivo .pth del modelo.
            num_episodes (int): Número de episodios a evaluar.
            seed (int): Seed para reproducibilidad.
        """
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.seed = seed
        self.device = get_device()
        self.model_name = os.path.basename(model_path).replace(".pth", "")

        # Directorios de salida
        self.output_dir = Path("evaluation_results") / self.model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir = self.output_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print("Evaluador RL - Self-Driving Car")
        print(f"{'='*60}")
        print(f"Modelo: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Episodios: {self.num_episodes}")
        print(f"Seed: {self.seed}")
        print(f"Directorio de salida: {self.output_dir}")
        print(f"{'='*60}\n")

    # --------------------------------------------------------------------- #
    #  Carga del agente
    # --------------------------------------------------------------------- #
    def load_agent(self) -> Agent:
        """Carga el modelo entrenado."""
        # Crear ambiente dummy para obtener las formas correctas
        dummy_env = make_env(
            "CarRacing-v2", seed=self.seed, idx=0,
            capture_video=False, run_name="dummy"
        )()
        dummy_envs = gym.vector.SyncVectorEnv([lambda: dummy_env])

        # Inicializar y cargar agente
        agent = Agent(dummy_envs).to(self.device)
        agent.load_state_dict(torch.load(self.model_path, map_location=self.device))
        agent.eval()

        dummy_env.close()
        return agent

    # --------------------------------------------------------------------- #
    #  Evaluación de un episodio
    # --------------------------------------------------------------------- #
    def evaluate_episode(self, agent: Agent, episode_num: int,
                         capture_video: bool = True) -> dict:
        """
        Ejecuta un episodio y captura métricas detalladas.

        Returns:
            dict: Métricas del episodio.
        """
        run_name = f"eval_{self.model_name}_ep{episode_num:03d}"

        if capture_video:
            env = make_env(
                "CarRacing-v2",
                seed=self.seed + episode_num,
                idx=0,
                capture_video=True,
                run_name=run_name,
            )()
        else:
            env = make_env(
                "CarRacing-v2",
                seed=self.seed + episode_num,
                idx=0,
                capture_video=False,
                run_name=run_name,
            )()

        obs, _ = env.reset(seed=self.seed + episode_num)
        obs = torch.as_tensor(np.array(obs)).float().unsqueeze(0).to(self.device)

        episode_reward = 0.0
        episode_length = 0
        done = False
        actions_taken = []
        rewards_per_step = []

        with torch.no_grad():
            while not done:
                # Acción determinística: media de la política
                hidden = agent.network(obs / 255.0)
                action_mean = agent.actor_mean(hidden)

                next_obs, reward, terminated, truncated, _ = env.step(
                    action_mean.cpu().numpy()[0]
                )

                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                actions_taken.append(action_mean.cpu().numpy()[0])
                rewards_per_step.append(reward)

                obs = torch.as_tensor(np.array(next_obs)).float().unsqueeze(0).to(self.device)

        env.close()

        # Compilar métricas
        metrics = {
            "episode_num": episode_num,
            "total_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "avg_reward_per_step": float(
                episode_reward / max(episode_length, 1)
            ),
            "max_reward_step": float(max(rewards_per_step)) if rewards_per_step else 0.0,
            "min_reward_step": float(min(rewards_per_step)) if rewards_per_step else 0.0,
        }

        if actions_taken:
            actions_arr = np.array(actions_taken)
            metrics.update(
                {
                    "steering_mean": float(actions_arr[:, 0].mean()),
                    "steering_std": float(actions_arr[:, 0].std()),
                    "throttle_mean": float(actions_arr[:, 1].mean()),
                    "brake_mean": float(actions_arr[:, 2].mean()),
                }
            )
        else:
            metrics.update(
                {
                    "steering_mean": 0.0,
                    "steering_std": 0.0,
                    "throttle_mean": 0.0,
                    "brake_mean": 0.0,
                }
            )

        return metrics

    # --------------------------------------------------------------------- #
    #  Evaluación completa
    # --------------------------------------------------------------------- #
    def run_full_evaluation(self):
        """Ejecuta la evaluación completa de todos los episodios."""
        print("Cargando modelo...")
        agent = self.load_agent()

        all_metrics = []
        print(f"\nEjecutando {self.num_episodes} episodios...\n")

        for ep in range(self.num_episodes):
            capture_video = (ep % 5 == 0)  # capturar vídeo cada 5 episodios
            metrics = self.evaluate_episode(agent, ep, capture_video=capture_video)
            all_metrics.append(metrics)

            reward_str = f"{metrics['total_reward']:7.1f}"
            length_str = f"{metrics['episode_length']:4d}"
            status = "📹" if capture_video else "  "

            print(
                f"[{ep+1:2d}/{self.num_episodes}] "
                f"Reward: {reward_str} | Steps: {length_str} | {status}"
            )

        return all_metrics

    # --------------------------------------------------------------------- #
    #  Estadísticas agregadas
    # --------------------------------------------------------------------- #
    @staticmethod
    def compute_statistics(all_metrics):
        """Calcula estadísticas agregadas."""
        rewards = [m["total_reward"] for m in all_metrics]
        lengths = [m["episode_length"] for m in all_metrics]

        stats = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "median_reward": float(np.median(rewards)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "win_rate": float(
                np.sum([1 for r in rewards if r > 900]) / len(rewards) * 100
            ),
            "success_rate": float(
                np.sum([1 for r in rewards if r > 0]) / len(rewards) * 100
            ),
            "steering_mean": float(np.mean([m["steering_mean"] for m in all_metrics])),
            "throttle_mean": float(np.mean([m["throttle_mean"] for m in all_metrics])),
            "brake_mean": float(np.mean([m["brake_mean"] for m in all_metrics])),
        }
        return stats

    # --------------------------------------------------------------------- #
    #  Guardar resultados
    # --------------------------------------------------------------------- #
    def save_results(self, all_metrics, stats):
        """Guarda los resultados en JSON."""
        results = {
            "model": self.model_name,
            "evaluation_date": datetime.now().isoformat(),
            "num_episodes": self.num_episodes,
            "seed": self.seed,
            "device": str(self.device),
            "statistics": stats,
            "episodes": all_metrics,
        }

        results_file = self.output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ Resultados guardados en: {results_file}")
        return results_file

    # --------------------------------------------------------------------- #
    #  Gráficos
    # --------------------------------------------------------------------- #
    def generate_plots(self, all_metrics, stats):
        """Genera gráficos para visualización."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Evaluación: {self.model_name}", fontsize=16, fontweight="bold")

        rewards = [m["total_reward"] for m in all_metrics]
        lengths = [m["episode_length"] for m in all_metrics]
        episodes = list(range(1, len(rewards) + 1))

        # Plot 1: Recompensas por episodio
        axes[0, 0].plot(episodes, rewards, "b-o", alpha=0.7)
        axes[0, 0].axhline(
            y=stats["mean_reward"],
            color="r",
            linestyle="--",
            label=f"Media: {stats['mean_reward']:.1f}",
        )
        axes[0, 0].axhline(y=900, color="g", linestyle="--", label="Objetivo (900)")
        axes[0, 0].set_xlabel("Episodio")
        axes[0, 0].set_ylabel("Recompensa Total")
        axes[0, 0].set_title("Recompensa por Episodio")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

        # Plot 2: Histograma de recompensas
        axes[0, 1].hist(
            rewards, bins=15, color="steelblue", edgecolor="black", alpha=0.7
        )
        axes[0, 1].axvline(
            x=stats["mean_reward"],
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Media: {stats['mean_reward']:.1f}",
        )
        axes[0, 1].set_xlabel("Recompensa Total")
        axes[0, 1].set_ylabel("Frecuencia")
        axes[0, 1].set_title("Distribución de Recompensas")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

        # Plot 3: Longitud de episodios
        axes[1, 0].plot(episodes, lengths, "g-o", alpha=0.7)
        axes[1, 0].axhline(
            y=stats["mean_length"],
            color="r",
            linestyle="--",
            label=f"Media: {stats['mean_length']:.0f}",
        )
        axes[1, 0].set_xlabel("Episodio")
        axes[1, 0].set_ylabel("Pasos")
        axes[1, 0].set_title("Longitud de Episodio")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # Plot 4: Estadísticas de control
        control_actions = ["Steering", "Throttle", "Brake"]
        control_values = [
            stats["steering_mean"],
            stats["throttle_mean"],
            stats["brake_mean"],
        ]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        axes[1, 1].bar(
            control_actions,
            control_values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
        )
        axes[1, 1].set_ylabel("Valor Promedio")
        axes[1, 1].set_title("Promedio de Acciones de Control")
        axes[1, 1].grid(alpha=0.3, axis="y")
        axes[1, 1].set_ylim([-0.5, 1.0])

        plt.tight_layout()
        plot_file = self.output_dir / "evaluation_plots.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        print(f"📊 Gráficos guardados en: {plot_file}")
        plt.close()

    # --------------------------------------------------------------------- #
    #  Reporte de texto
    # --------------------------------------------------------------------- #
    def generate_report(self, stats):
        """Genera un reporte de texto formateado."""
        report = f"""
╔════════════════════════════════════════════════════════════════════╗
║                  REPORTE DE EVALUACIÓN - MODELO RL                ║
╚════════════════════════════════════════════════════════════════════╝

📋 INFORMACIÓN DEL MODELO
{'─' * 64}
Nombre del Modelo:     {self.model_name}
Archivo:               {self.model_path}
Fecha de Evaluación:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dispositivo:           {self.device}

📊 CONFIGURACIÓN DE EVALUACIÓN
{'─' * 64}
Episodios Evaluados:   {self.num_episodes}
Seed:                  {self.seed}
Ambiente:              CarRacing-v2
Video Capturado:       Sí (cada 5 episodios)

📈 ESTADÍSTICAS DE RENDIMIENTO
{'─' * 64}
Recompensa Media:      {stats['mean_reward']:>10.2f} ± {stats['std_reward']:.2f}
Recompensa Mínima:     {stats['min_reward']:>10.2f}
Recompensa Máxima:     {stats['max_reward']:>10.2f}
Recompensa Mediana:    {stats['median_reward']:>10.2f}

Tasa de Victoria (>900):  {stats['win_rate']:>6.1f}%
Tasa de Éxito (>0):       {stats['success_rate']:>6.1f}%

Pasos Promedio:        {stats['mean_length']:>10.1f} ± {stats['std_length']:.1f}

🎮 ACCIONES DE CONTROL PROMEDIO
{'─' * 64}
Dirección (Steering):  {stats['steering_mean']:>10.4f}
Acelerador (Throttle): {stats['throttle_mean']:>10.4f}
Freno (Brake):         {stats['brake_mean']:>10.4f}

💾 ARCHIVOS GENERADOS
{'─' * 64}
✓ results.json              - Datos detallados (JSON)
✓ evaluation_plots.png      - Gráficos de rendimiento
✓ videos/                   - Videos de episodios seleccionados
✓ report.txt                - Este reporte

╔════════════════════════════════════════════════════════════════════╗
║                          FIN DEL REPORTE                          ║
╚════════════════════════════════════════════════════════════════════╝
"""
        report_file = self.output_dir / "report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)

        print(report)
        print(f"\n📄 Reporte guardado en: {report_file}")

    # --------------------------------------------------------------------- #
    #  Pipeline completo
    # --------------------------------------------------------------------- #
    def run(self):
        """Ejecuta el pipeline completo de evaluación."""
        all_metrics = self.run_full_evaluation()
        stats = self.compute_statistics(all_metrics)
        self.save_results(all_metrics, stats)
        self.generate_plots(all_metrics, stats)
        self.generate_report(stats)

        print(f"\n{'='*60}")
        print("✅ EVALUACIÓN COMPLETADA")
        print(f"{'='*60}\n")

        return all_metrics, stats


if __name__ == "__main__":
    # ========== CONFIGURACIÓN MANUAL ==========
    modelo_500k = r"path/to/ppo_car_racing_step_512000.pth"
    modelo_1m = r"path/to/ppo_car_racing_step_1024000.pth"
    modelo_2m = r"path/to/ppo_car_racing_step_2048000.pth"

    # Descomenta el modelo que quieras evaluar:
    # evaluator = RobustEvaluator(modelo_500k, num_episodes=30)
    # evaluator = RobustEvaluator(modelo_1m, num_episodes=30)
    evaluator = RobustEvaluator(modelo_2m, num_episodes=30)

    evaluator.run()
