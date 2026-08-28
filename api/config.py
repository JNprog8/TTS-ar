"""Configuración de rutas y constantes del servicio de TTS (Piper TTS / VITS)."""
from pathlib import Path

# Directorio de voces y artefactos ONNX de Piper
VOICES_DIR = Path(__file__).parent.parent / "voices"
PIPER_MODEL_PATH = VOICES_DIR / "piper_ar.onnx"
PIPER_CONFIG_PATH = VOICES_DIR / "piper_ar.onnx.json"

PIPER_MALE_MODEL_PATH = VOICES_DIR / "piper_male.onnx"
PIPER_MALE_CONFIG_PATH = VOICES_DIR / "piper_male.onnx.json"

VOICES_CATALOG_CSV = VOICES_DIR / "voices_catalog.csv"

SAMPLE_RATE = 22050
MAX_TEXT_LENGTH = 1000

MIN_SPEED, MAX_SPEED = 0.5, 2.0

DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_W = 0.8
