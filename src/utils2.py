import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import GrayScaleObservation, FrameStack


class GrassPenaltyWrapper(gym.Wrapper):
    """
    Wrapper que penaliza si el agente pisa la hierba.
    Versión corregida para colores de CarRacing-v2.
    """

    def __init__(self, env, grass_penalty=0.8, max_off_track=50):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.max_off_track = max_off_track
        self.off_track_frames = 0
        self.debug_printed = False  # Para no llenar la consola

    def step(self, action):
        # 1. Obtenemos la observación ORIGINAL (RGB 96x96x3, valores 0-255)
        obs, rew, terminated, truncated, info = self.env.step(action)

        # 2. Detectar hierba (CORREGIDO)
        # En lugar de valores fijos < 100, buscamos que el Verde sea el canal dominante.
        # Condición: Verde > Rojo + 10  Y  Verde > Azul + 10  Y  Verde > 100 (brillo mínimo)
        is_green = (
            (obs[:, :, 1] > obs[:, :, 0] + 10) &
            (obs[:, :, 1] > obs[:, :, 2] + 10) &
            (obs[:, :, 1] > 100)
        )

        # Ratio de hierba en la imagen (0.0 a 1.0)
        green_ratio = np.mean(is_green)

        # 3. Aplicar penalización y lógica de terminación
        # Si más del 30% de la pantalla es hierba (ajustado de 0.25 a 0.30 para ser seguro)
        if green_ratio > 0.30:
            rew -= self.grass_penalty
            self.off_track_frames += 1

            # Debug: Avisar la primera vez que detecta hierba para confirmar que funciona
            if not self.debug_printed and self.off_track_frames > 5:
                print(f"DEBUG: Hierba detectada (Ratio: {green_ratio:.2f}). Contador: {self.off_track_frames}")
                self.debug_printed = True

            # Si lleva demasiados frames consecutivos fuera, terminamos el episodio
            if self.off_track_frames > self.max_off_track:
                print(f"💀 Muerte súbita: Demasiado tiempo en la hierba ({self.off_track_frames} frames).")
                terminated = True
                info['off_track_timeout'] = True
        else:
            # Si vuelve a la carretera, reseteamos el contador
            if self.off_track_frames > 0:
                self.off_track_frames = 0
                self.debug_printed = False # Permitir imprimir de nuevo en la siguiente salida

        info['grass_ratio'] = green_ratio

        return obs, rew, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video, run_name, apply_grass_penalty=False):
    """
    Función para crear y configurar el entorno.
    """

    def thunk():
        # Inicializar entorno con render_mode="rgb_array"
        env = gym.make(env_id, render_mode="rgb_array")

        # Wrapper para grabar video
        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(env, f"videos_T4/{run_name}")

        # --- APLICAR WRAPPER DE HIERBA AQUÍ ---
        # Debe ir ANTES de GrayScaleObservation para tener acceso a los colores RGB
        if apply_grass_penalty:
            print(f"🌿 GrassPenaltyWrapper ACTIVO en entorno {idx}")
            env = GrassPenaltyWrapper(env)
        # --------------------------------------

        # 1. Conversión a Escala de Grises
        env = GrayScaleObservation(env, keep_dim=False)

        # 2. Stack de Frames
        env = FrameStack(env, 4)

        env.action_space.seed(seed + idx)
        return env

    return thunk


def get_device():
    """Devuelve cpu o cuda según disponibilidad."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")