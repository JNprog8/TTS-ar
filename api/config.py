"""Configuración centralizada de rutas, modelos y constantes del motor TTS."""
from pathlib import Path

# Directorios principales
API_DIR = Path(__file__).resolve().parent
ROOT_DIR = API_DIR.parent
VOICES_DIR = ROOT_DIR / "voices"

# Modelos ONNX de Piper (Femenino y Masculino)
PIPER_MODEL_PATH = VOICES_DIR / "piper_ar.onnx"
PIPER_CONFIG_PATH = VOICES_DIR / "piper_ar.onnx.json"

PIPER_MALE_MODEL_PATH = VOICES_DIR / "piper_male.onnx"
PIPER_MALE_CONFIG_PATH = VOICES_DIR / "piper_male.onnx.json"

# Catálogo de arquetipos
VOICES_CATALOG_CSV = VOICES_DIR / "voices_catalog.csv"

# Convertidor de Color de Tono (OpenVoice) y Embeddings de Hablantes
CONVERTER_DIR = VOICES_DIR / "converter"
CONVERTER_CONFIG_PATH = CONVERTER_DIR / "config.json"
CONVERTER_CKPT_PATH = CONVERTER_DIR / "checkpoint.pth"

EMBEDDINGS_DIR = VOICES_DIR / "embeddings"
MALE_AR_SE_PATH = EMBEDDINGS_DIR / "male_ar_speaker.pt"
MALE_BASE_SE_PATH = EMBEDDINGS_DIR / "male_base_speaker.pt"

# Parámetros acústicos globales
SAMPLE_RATE = 22050
MAX_TEXT_LENGTH = 1000

MIN_SPEED, MAX_SPEED = 0.4, 2.5
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_W = 0.8
