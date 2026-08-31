"""Motor de inferencia TTS multi-modelo con soporte de arquetipos demográficos reales
(masculino, femenino, anciano, anciana, jóvenes), clonación tímbrica del locutor argentino
y modulación de estados emocionales/síntomas (neutral, molesto, adolorido, preocupado)."""
import io
import wave
from pathlib import Path

import pandas as pd
import torch
import torchaudio
import torchaudio.functional as AF
from piper import PiperVoice, SynthesisConfig

from config import (
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_W,
    PIPER_CONFIG_PATH,
    PIPER_MALE_CONFIG_PATH,
    PIPER_MALE_MODEL_PATH,
    PIPER_MODEL_PATH,
    VOICES_CATALOG_CSV,
    VOICES_DIR,
)

# Rutas del convertidor de tono OpenVoice y embeddings del dataset argentino
CONVERTER_CONFIG = VOICES_DIR / "converter" / "config.json"
CONVERTER_CKPT = VOICES_DIR / "converter" / "checkpoint.pth"
MALE_AR_SE = VOICES_DIR / "embeddings" / "male_ar_speaker.pt"
MALE_BASE_SE = VOICES_DIR / "embeddings" / "male_base_speaker.pt"

# Arquetipos acústicos reales por ID de voz
PERSONA_PROFILES = {
    0: {"name": "daniela", "gender": "female", "model_type": "female", "speed_base": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
    1: {"name": "martin", "gender": "male", "model_type": "male", "speed_base": 1.0, "noise_scale": 0.667, "noise_w": 0.8},
    2: {"name": "marta", "gender": "female_elderly", "model_type": "female", "speed_base": 0.88, "noise_scale": 0.68, "noise_w": 0.82},
    3: {"name": "roberto", "gender": "male_elderly", "model_type": "male", "speed_base": 0.86, "noise_scale": 0.68, "noise_w": 0.82},
    4: {"name": "sofia", "gender": "female_young", "model_type": "female", "speed_base": 1.06, "noise_scale": 0.667, "noise_w": 0.78},
    5: {"name": "lucas", "gender": "male_young", "model_type": "male", "speed_base": 1.06, "noise_scale": 0.667, "noise_w": 0.78},
}

# Modificadores acústicos por estado emocional / síntoma del paciente
EMOTION_MODIFIERS = {
    "neutral": {"speed_factor": 1.0, "noise_scale_mult": 1.0, "noise_w_mult": 1.0, "pitch_offset": 0.0},
    "pain": {"speed_factor": 0.85, "noise_scale_mult": 1.15, "noise_w_mult": 1.20, "pitch_offset": 0.0},
    "worried": {"speed_factor": 1.10, "noise_scale_mult": 1.10, "noise_w_mult": 0.90, "pitch_offset": 0.0},
    "annoyed": {"speed_factor": 1.05, "noise_scale_mult": 1.05, "noise_w_mult": 0.80, "pitch_offset": 0.0},
    "custom": {"speed_factor": 1.0, "noise_scale_mult": 1.0, "noise_w_mult": 1.0, "pitch_offset": 0.0},
}


class TTSEngine:
    def __init__(self):
        self.female_voice = self._load_female_model()
        self.male_voice = self._load_male_model()
        self.catalog = pd.read_csv(VOICES_CATALOG_CSV).set_index("voice_id")
        self.tone_converter, self.src_se, self.tgt_se = self._init_tone_converter()

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

    def _init_tone_converter(self):
        """Inicializa el convertidor de color de tono para clonación del locutor argentino."""
        try:
            if CONVERTER_CONFIG.exists() and CONVERTER_CKPT.exists() and MALE_AR_SE.exists() and MALE_BASE_SE.exists():
                from openvoice.api import ToneColorConverter
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                converter = ToneColorConverter(str(CONVERTER_CONFIG), device=device)
                converter.load_ckpt(str(CONVERTER_CKPT))
                src_se = torch.load(MALE_BASE_SE, map_location=device)
                tgt_se = torch.load(MALE_AR_SE, map_location=device)
                return converter, src_se, tgt_se
        except Exception as e:
            print(f"[AVISO] ToneColorConverter no inicializado ({e}). Se usará síntesis directa.")
        return None, None, None

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

        # 1. Seleccionar el modelo neuronal base (Femenino o Masculino)
        model = self.male_voice if persona["model_type"] == "male" else self.female_voice

        # 2. Calcular velocidad efectiva de habla (duración prosódica natural)
        effective_speed = speed * persona["speed_base"] * emotion_mod["speed_factor"]
        length_scale = 1.0 / max(0.4, min(2.5, effective_speed))

        # 3. Variabilidad tímbrica VITS sin distorsiones
        noise_scale = persona["noise_scale"] * emotion_mod["noise_scale_mult"] * max(0.4, min(1.6, style_strength))
        noise_w_scale = persona["noise_w"] * emotion_mod["noise_w_mult"] * max(0.4, min(1.6, style_strength))

        syn_config = SynthesisConfig(
            speaker_id=None,
            length_scale=length_scale,
            noise_scale=noise_scale,
            noise_w_scale=noise_w_scale,
            normalize_audio=True,
            volume=1.0,
        )

        # 4. Síntesis base con motor fonético
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            model.synthesize_wav(
                text=text,
                wav_file=wav_file,
                syn_config=syn_config,
            )

        buffer.seek(0)

        # 5. Clonación del color de tono al locutor masculino argentino
        if persona["model_type"] == "male" and self.tone_converter is not None:
            try:
                wav_converted, sr = self.tone_converter.convert(
                    buffer,
                    src_se=self.src_se,
                    tgt_se=self.tgt_se,
                    tau=0.3,
                )
                out_buf = io.BytesIO()
                torchaudio.save(out_buf, wav_converted, sr, format="wav")
                out_buf.seek(0)
                buffer = out_buf
            except Exception as e:
                print(f"[AVISO] Error en conversión de tono: {e}")
                buffer.seek(0)

        # 6. Pitch shift opcional si fue solicitado explícitamente
        if pitch_shift is not None and abs(pitch_shift) > 0.05:
            wav_tensor, sr = torchaudio.load(buffer)
            wav_tensor = AF.pitch_shift(
                wav_tensor,
                sample_rate=sr,
                n_steps=pitch_shift,
                n_fft=2048,
                win_length=1024,
                hop_length=256,
            )
            out_buf = io.BytesIO()
            torchaudio.save(out_buf, wav_tensor, sr, format="wav")
            out_buf.seek(0)
            return out_buf.read()

        return buffer.read()


engine = TTSEngine()
