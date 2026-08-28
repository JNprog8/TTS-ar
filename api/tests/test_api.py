"""Tests unitarios para los endpoints de la API TTS con emociones y perfiles de pacientes."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

# Asegurar que api/ esté en el path de importación
API_DIR = Path(__file__).resolve().parent.parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

with patch("piper.PiperVoice.load") as mock_load, \
     patch("pandas.read_csv") as mock_csv:
    mock_voice = MagicMock()
    mock_voice.config.num_speakers = 1
    mock_voice.config.sample_rate = 22050
    mock_load.return_value = mock_voice
    mock_csv.return_value = pd.DataFrame([
        {"voice_id": 0, "gender": "female", "name": "daniela", "description": "Paciente Femenina Adulta"},
        {"voice_id": 1, "gender": "male", "name": "martin", "description": "Paciente Masculino Adulto"},
        {"voice_id": 2, "gender": "female_elderly", "name": "marta", "description": "Paciente Femenina Anciana"},
        {"voice_id": 3, "gender": "male_elderly", "name": "roberto", "description": "Paciente Masculino Anciano"},
    ])
    from fastapi.testclient import TestClient
    from config import MAX_TEXT_LENGTH
    from main import app, engine


@pytest.fixture(autouse=True)
def setup_catalog():
    mock_df = pd.DataFrame([
        {"voice_id": 0, "gender": "female", "name": "daniela", "description": "Paciente Femenina Adulta"},
        {"voice_id": 1, "gender": "male", "name": "martin", "description": "Paciente Masculino Adulto"},
        {"voice_id": 2, "gender": "female_elderly", "name": "marta", "description": "Paciente Femenina Anciana"},
        {"voice_id": 3, "gender": "male_elderly", "name": "roberto", "description": "Paciente Masculino Anciano"},
    ]).set_index("voice_id")
    engine.catalog = mock_df


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_list_voices_returns_demographic_profiles(client):
    """GET /audio/voices debe devolver la lista de arquetipos de pacientes con género y descripción."""
    response = client.get("/audio/voices")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4
    assert data[0]["id"] == 0
    assert data[0]["name"] == "daniela"
    assert "Femenina Adulta" in data[0]["description"]
    assert data[3]["id"] == 3
    assert "Masculino Anciano" in data[3]["description"]


def test_tts_with_emotion_success(client):
    """POST /audio/tts con emoción (ej. pain) debe pasar correctamente al engine y retornar WAV."""
    fake_wav_bytes = b"RIFF....WAVEfmt ...."
    with patch.object(engine, "synthesize", return_value=fake_wav_bytes) as mock_synth:
        response = client.post(
            "/audio/tts",
            json={
                "id": 2,
                "text": "Ay doctor, me duele el pecho.",
                "speed": 1.0,
                "style_strength": 1.0,
                "emotion": "pain"
            }
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content == fake_wav_bytes
        mock_synth.assert_called_once_with(
            text="Ay doctor, me duele el pecho.",
            voice_id=2,
            speed=1.0,
            style_strength=1.0,
            emotion="pain",
            pitch_shift=None,
        )


def test_tts_empty_text_returns_400(client):
    response = client.post(
        "/audio/tts",
        json={"id": 0, "text": "   ", "speed": 1.0, "style_strength": 1.0}
    )
    assert response.status_code == 400
    assert "vacío" in response.json()["detail"]


def test_tts_invalid_voice_returns_400(client):
    response = client.post(
        "/audio/tts",
        json={"id": 999, "text": "Hola doctor", "speed": 1.0, "style_strength": 1.0}
    )
    assert response.status_code == 400
    assert "No existe la voz" in response.json()["detail"]


def test_tts_text_too_long_returns_400(client):
    long_text = "a" * (MAX_TEXT_LENGTH + 1)
    response = client.post(
        "/audio/tts",
        json={"id": 0, "text": long_text, "speed": 1.0, "style_strength": 1.0}
    )
    assert response.status_code == 400
    assert f"supera los {MAX_TEXT_LENGTH} caracteres" in response.json()["detail"]
