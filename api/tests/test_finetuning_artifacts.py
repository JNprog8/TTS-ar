"""Tests de verificación de integridad para los artefactos de Piper TTS."""
import json
from pathlib import Path
import pandas as pd
import pytest
from piper import PiperVoice

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
VOICES_DIR = ROOT_DIR / "voices"


def test_piper_model_and_config_exist():
    """Verifica que el modelo ONNX y el archivo de configuración JSON de Piper existan."""
    onnx_path = VOICES_DIR / "piper_ar.onnx"
    json_path = VOICES_DIR / "piper_ar.onnx.json"

    assert onnx_path.exists(), f"El modelo {onnx_path} no existe"
    assert json_path.exists(), f"La configuración {json_path} no existe"

    # Verificar que el JSON de configuración sea válido y contenga idioma es_AR
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    assert "espeak" in config or "audio" in config, "Configuración de Piper inválida"
    if "espeak" in config:
        assert "es" in config["espeak"].get("voice", ""), "La voz debe ser en español"


def test_voices_catalog_integrity():
    """Verifica que el catálogo de voces tenga las columnas requeridas."""
    catalog_path = VOICES_DIR / "voices_catalog.csv"
    assert catalog_path.exists(), "voices_catalog.csv no existe"

    df = pd.read_csv(catalog_path)
    assert set(["voice_id", "gender", "name"]).issubset(df.columns)
    assert len(df) >= 1, "Debe haber al menos 1 voz registrada"


def test_piper_voice_loadable():
    """Verifica que el modelo Piper ONNX se pueda cargar e inicializar correctamente."""
    onnx_path = VOICES_DIR / "piper_ar.onnx"
    json_path = VOICES_DIR / "piper_ar.onnx.json"
    voice = PiperVoice.load(str(onnx_path), config_path=str(json_path))
    assert voice is not None
    assert voice.config.sample_rate in (22050, 24000, 16000)
