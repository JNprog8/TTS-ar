"""Motor de inferencia TTS multi-modelo con soporte de arquetipos demográficos reales
(masculino, femenino, anciano, anciana, jóvenes) y modulación de estados emocionales/síntomas
(neutral, molesto, adolorido, preocupado)."""
import io
import wave
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from piper import PiperVoice, SynthesisConfig

from config import (
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_W,
    PIPER_CONFIG_PATH,
    PIPER_MALE_CONFIG_PATH,
    PIPER_MALE_MODEL_PATH,
    PIPER_MODEL_PATH,
    VOICES_CATALOG_CSV,
)

# Arquetipos acústicos por ID de voz (género y grupo etario)
PERSONA_PROFILES = {
    0: {"name": "daniela", "gender": "female", "model_type": "female", "pitch_base": 0.0, "speed_base": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
    1: {"name": "martin", "gender": "male", "model_type": "male", "pitch_base": 0.0, "speed_base": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
    2: {"name": "marta", "gender": "female_elderly", "model_type": "female", "pitch_base": -1.0, "speed_base": 0.88, "noise_scale": 0.85, "noise_w": 1.15},
    3: {"name": "roberto", "gender": "male_elderly", "model_type": "male", "pitch_base": -1.8, "speed_base": 0.85, "noise_scale": 0.90, "noise_w": 1.25},
    4: {"name": "sofia", "gender": "female_young", "model_type": "female", "pitch_base": 1.2, "speed_base": 1.05, "noise_scale": 0.667, "noise_w": 0.75},
    5: {"name": "lucas", "gender": "male_young", "model_type": "male", "pitch_base": 0.8, "speed_base": 1.05, "noise_scale": 0.667, "noise_w": 0.75},
}

# Modificadores acústicos por estado emocional / síntoma del paciente
EMOTION_MODIFIERS = {
    "neutral": {"speed_factor": 1.0, "noise_scale_mult": 1.0, "noise_w_mult": 1.0, "pitch_offset": 0.0},
    "pain": {"speed_factor": 0.82, "noise_scale_mult": 1.35, "noise_w_mult": 1.45, "pitch_offset": -0.6},
    "worried": {"speed_factor": 1.15, "noise_scale_mult": 1.20, "noise_w_mult": 0.90, "pitch_offset": 1.3},
    "annoyed": {"speed_factor": 1.08, "noise_scale_mult": 1.05, "noise_w_mult": 0.75, "pitch_offset": -0.5},
    "custom": {"speed_factor": 1.0, "noise_scale_mult": 1.0, "noise_w_mult": 1.0, "pitch_offset": 0.0},
}


class TTSEngine:
    def __init__(self):
        self.female_voice = self._load_female_model()
        self.male_voice = self._load_male_model()
        self.catalog = pd.read_csv(VOICES_CATALOG_CSV).set_index("voice_id")

    def _load_female_model(self) -> PiperVoice:
        if not PIPER_MODEL_PATH.exists() or not PIPER_CONFIG_PATH.exists():
            import sys
            sys.path.insert(0, str(VOICES_DIR))
            from download_voices import ensure_voices
            ensure_voices()
        return PiperVoice.load(
            str(PIPER_MODEL_PATH),
            config_path=str(PIPER_CONFIG_PATH),
        )

    def _load_male_model(self) -> PiperVoice:
        if not PIPER_MALE_MODEL_PATH.exists() or not PIPER_MALE_CONFIG_PATH.exists():
            import sys
            sys.path.insert(0, str(VOICES_DIR))
            from download_voices import ensure_voices
            ensure_voices()
        if PIPER_MALE_MODEL_PATH.exists() and PIPER_MALE_CONFIG_PATH.exists():
            return PiperVoice.load(
                str(PIPER_MALE_MODEL_PATH),
                config_path=str(PIPER_MALE_CONFIG_PATH),
            )
        return self.female_voice

    def is_valid_voice(self, voice_id: int) -> bool:
        return voice_id in self.catalog.index

    def list_voices(self) -> pd.DataFrame:
        return self.catalog.reset_index()

    def synthesize(
        self,
        text: str,
        voice_id: int = 0,
        speed: float = 1.0,
        style_strength: float = 1.0,
        emotion: str = "neutral",
        pitch_shift: float | None = None,
    ) -> bytes:
        persona = PERSONA_PROFILES.get(voice_id, PERSONA_PROFILES[0])
        emotion_mod = EMOTION_MODIFIERS.get(emotion, EMOTION_MODIFIERS["neutral"])

        # Seleccionar modelo según el género
        model = self.male_voice if persona["model_type"] == "male" else self.female_voice

        # 1. Calcular velocidad efectiva de habla
        effective_speed = speed * persona["speed_base"] * emotion_mod["speed_factor"]
        length_scale = 1.0 / max(0.4, min(2.5, effective_speed))

        # 2. Calcular variabilidad tímbrica y duración fonética (VITS)
        noise_scale = persona["noise_scale"] * emotion_mod["noise_scale_mult"] * max(0.2, min(2.0, style_strength))
        noise_w_scale = persona["noise_w"] * emotion_mod["noise_w_mult"] * max(0.2, min(2.0, style_strength))

        syn_config = SynthesisConfig(
            speaker_id=None,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=True,
            volume=1.0,
        )

        # 3. Síntesis base con el modelo correspondiente (Femenino o Masculino)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            model.synthesize_wav(
                text=text,
                wav_file=wav_file,
                syn_config=syn_config,
            )

        buffer.seek(0)

        # 4. Modulación tonal demográfica y emocional (Pitch Shift ultrarrápido)
        total_pitch_shift = persona["pitch_base"] + emotion_mod["pitch_offset"]
        if pitch_shift is not None:
            total_pitch_shift += pitch_shift

        if abs(total_pitch_shift) > 0.05:
            wav_tensor, sr = torchaudio.load(buffer)
            wav_tensor = torchaudio.functional.pitch_shift(
                wav_tensor,
                sample_rate=sr,
                n_steps=total_pitch_shift,
            )
            out_buf = io.BytesIO()
            torchaudio.save(out_buf, wav_tensor, sr, format="wav")
            out_buf.seek(0)
            return out_buf.read()

        return buffer.read()


engine = TTSEngine()
