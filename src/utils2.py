import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, FrameStack


class GrassPenaltyWrapper(gym.Wrapper):
    """
    Wrapper que penaliza si el agente pisa la hierba.
    Versión 3.0:
    - Lógica ROI (Region of Interest): Solo mira la zona del coche.
    - Evita falsos positivos por el césped del fondo.
    """

    def __init__(self, env, grass_penalty=0.8, max_off_track=50):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.max_off_track = max_off_track
        self.off_track_frames = 0
        self.episode_steps = 0
        self.debug_printed = False

    def reset(self, **kwargs):
        """Reinicia los contadores al empezar un nuevo episodio."""
        self.off_track_frames = 0
        self.episode_steps = 0
        self.debug_printed = False
        return self.env.reset(**kwargs)

    def step(self, action):
        self.episode_steps += 1

        # 1. Paso normal del entorno
        obs, rew, terminated, truncated, info = self.env.step(action)

        # --- PERIODO DE GRACIA (ZOOM INICIAL) ---
        if self.episode_steps < 60:
            return obs, rew, terminated, truncated, info

        # 2. DEFINIR ZONA CRÍTICA (ROI)
        # El coche está siempre centrado abajo.
        # Recortamos solo un rectángulo alrededor del coche para ignorar el paisaje.
        # Vertical: 60 a 80 (Justo delante y sobre el coche)
        # Horizontal: 38 a 58 (El ancho de la carretera central)
        roi = obs[60:80, 38:58, :]

        # 3. DETECTAR HIERBA EN LA ROI
        # Usamos la lógica de color dominante sobre el recorte (roi), no sobre toda la obs
        is_green = (
                (roi[:, :, 1] > roi[:, :, 0] + 10) &  # Verde > Rojo
                (roi[:, :, 1] > roi[:, :, 2] + 10) &  # Verde > Azul
                (roi[:, :, 1] > 100)  # Brillo mínimo
        )

        # Ratio de hierba EN LA ZONA DEL COCHE
        green_ratio = np.mean(is_green)

        # 4. PENALIZACIÓN
        # Ahora podemos ser más estrictos con el umbral (0.4) porque solo miramos la carretera.
        # Si el 40% de la zona del coche es verde, es que te has salido.
        if green_ratio > 0.40:
            rew -= self.grass_penalty
            self.off_track_frames += 1

            # Debug visual en consola (solo la primera vez que se sale en el episodio)
            if not self.debug_printed and self.off_track_frames > 5:
                print(f"⚠️  SALIDA DE PISTA DETECTADA (Step {self.episode_steps})")
                self.debug_printed = True

            # Muerte súbita
            if self.off_track_frames > self.max_off_track:
                # print(f"💀 Muerte súbita (Step {self.episode_steps})")
                terminated = True
                info['off_track_timeout'] = True
        else:
            # Si vuelve a la pista, reseteamos el contador
            if self.off_track_frames > 0:
                self.off_track_frames = 0
                self.debug_printed = False  # Permitir imprimir de nuevo

        info['grass_ratio'] = green_ratio

        return obs, rew, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video, run_name, apply_grass_penalty=False):
    """
    Función para crear y configurar el entorno.
    """

    def thunk():
        env = gym.make(env_id, render_mode="rgb_array")

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos_T4/{run_name}")

        # --- APLICAR WRAPPER ---
        if apply_grass_penalty:
            env = GrassPenaltyWrapper(env)
        # -----------------------

        env = GrayScaleObservation(env, keep_dim=False)
        env = FrameStack(env, 4)

        env.action_space.seed(seed + idx)
        return env

    return thunk


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")