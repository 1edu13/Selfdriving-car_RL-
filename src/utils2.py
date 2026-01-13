import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, FrameStack


class GrassPenaltyWrapper(gym.Wrapper):
    """
    Wrapper que penaliza si el agente pisa la hierba.
    Funciona sobre la imagen RGB original (0-255).
    """

    def __init__(self, env, grass_penalty=0.8, max_off_track=50):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.max_off_track = max_off_track
        self.off_track_frames = 0

    def step(self, action):
        # 1. Obtenemos la observación ORIGINAL (RGB 96x96x3, valores 0-255)
        obs, rew, terminated, truncated, info = self.env.step(action)

        # 2. Detectar hierba usando Numpy sobre la imagen RGB
        # En CarRacing, la hierba es verde brillante.
        # Filtro: Canal Verde > 150 Y Rojo < 100 Y Azul < 100 (aprox)
        # obs[:, :, 0] es R, [:, :, 1] es G, [:, :, 2] es B
        is_green = (obs[:, :, 1] > 150) & (obs[:, :, 0] < 100) & (obs[:, :, 2] < 100)

        # Ratio de hierba en la imagen (cuántos píxeles son verdes vs total)
        green_ratio = np.mean(is_green)

        # 3. Aplicar penalización y lógica de terminación
        # Si más del 25% de la pantalla es hierba, consideramos que está fuera
        if green_ratio > 0.25:
            rew -= self.grass_penalty  # Penalización fuerte
            self.off_track_frames += 1

            # Si lleva demasiados frames consecutivos fuera, terminamos el episodio
            # Esto evita que el agente pierda tiempo dando vueltas en el campo
            if self.off_track_frames > self.max_off_track:
                terminated = True
                info['off_track_timeout'] = True
        else:
            self.off_track_frames = 0

        # Guardamos el ratio para logs si fuera necesario
        info['grass_ratio'] = green_ratio

        return obs, rew, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video, run_name, apply_grass_penalty=False):
    """
    Función para crear y configurar el entorno.
    Args:
        apply_grass_penalty (bool): Si es True, activa el wrapper de hierba.
    """

    def thunk():
        # Inicializar entorno con render_mode="rgb_array" (necesario para ver colores)
        env = gym.make(env_id, render_mode="rgb_array")

        # Wrapper para grabar video (solo en el entorno 0)
        if capture_video and idx == 0:
            # Guarda en la carpeta videos_T4
            env = gym.wrappers.RecordVideo(env, f"videos_T5/{run_name}")

        # --- APLICAR WRAPPER DE HIERBA AQUÍ ---
        # Debe ir ANTES de GrayScaleObservation para tener acceso a los colores
        if apply_grass_penalty:
            env = GrassPenaltyWrapper(env)
        # --------------------------------------

        # 1. Conversión a Escala de Grises
        # (96, 96, 3) -> (96, 96)
        env = GrayScaleObservation(env, keep_dim=False)

        # 2. Stack de Frames
        # Apila 4 frames para dar contexto temporal (velocidad)
        env = FrameStack(env, 4)

        env.action_space.seed(seed + idx)
        return env

    return thunk


def get_device():
    """Devuelve cpu o cuda según disponibilidad."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")