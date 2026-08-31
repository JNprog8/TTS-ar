"""API REST de TTS argentino para simulación de pacientes médicos (Piper TTS / VITS / OpenVoice)."""
import io
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from config import API_DIR, MAX_TEXT_LENGTH
from schemas import EmotionType, ErrorResponse, TTSRequest, VoiceInfo
from tts_engine import engine

logger = logging.getLogger("api")

app = FastAPI(
    title="TTS Argentino — Simulador de Paciente Médico",
    description="API de síntesis de voz en español argentino para pacientes virtuales (géneros, grupos etarios y estados emocionales)",
    version="2.0",
)

TEMPLATE_PATH = API_DIR / "templates" / "index.html"


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """Evita registros 404 de favicon en navegadores."""
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index_demo() -> HTMLResponse:
    """Reproductor y probador web interactivo con selector de paciente y estados emocionales."""
    if TEMPLATE_PATH.exists():
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>TTS Argentino API</h1><p>Visite <a href='/docs'>/docs</a> para la documentación OpenAPI.</p>")


@app.get("/audio/voices", response_model=List[VoiceInfo])
def list_voices() -> List[VoiceInfo]:
    """Catálogo de arquetipos de pacientes (género, edad y descripción) para selección por el LLM."""
    catalog = engine.list_voices()
    return [
        VoiceInfo(
            id=row.voice_id,
            gender=row.gender,
            name=row.name,
            description=row.description if "description" in catalog.columns else f"Voz {row.voice_id}",
        )
        for row in catalog.itertuples()
    ]


def _synthesize_audio_response(
    voice_id: int,
    text: str,
    speed: float,
    style_strength: float,
    emotion: str,
    pitch_shift: float | None = None,
) -> StreamingResponse:
    """Lógica común de validación y síntesis para GET y POST."""
    if not text.strip():
        raise HTTPException(status_code=400, detail="El campo 'text' no puede estar vacío")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"El texto supera los {MAX_TEXT_LENGTH} caracteres")
    if not engine.is_valid_voice(voice_id):
        raise HTTPException(status_code=400, detail=f"No existe la voz con id={voice_id}")

    wav_bytes = engine.synthesize(
        text=text,
        voice_id=voice_id,
        speed=speed,
        style_strength=style_strength,
        emotion=emotion,
        pitch_shift=pitch_shift,
    )

    headers = {
        "Content-Disposition": 'inline; filename="synthesis.wav"',
        "Accept-Ranges": "bytes",
    }

    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers=headers,
    )


@app.post(
    "/audio/tts",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Audio sintetizado del paciente en formato WAV (streaming)",
        },
        400: {"model": ErrorResponse, "description": "Parámetros inválidos"},
    },
)
def text_to_speech_post(request: TTSRequest) -> StreamingResponse:
    """Contrato principal de integración para el LLM y pipeline STT/Whisper."""
    return _synthesize_audio_response(
        voice_id=request.id,
        text=request.text,
        speed=request.speed,
        style_strength=request.style_strength,
        emotion=request.emotion.value if isinstance(request.emotion, EmotionType) else str(request.emotion),
        pitch_shift=request.pitch_shift,
    )


@app.get(
    "/audio/tts",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"audio/wav": {}},
            "description": "Audio sintetizado para reproducción directa en navegador",
        },
        400: {"model": ErrorResponse, "description": "Parámetros inválidos"},
    },
)
def text_to_speech_get(
    id: int = 0,
    text: str = "Ay doctor, me duele mucho la panza.",
    speed: float = 1.0,
    style_strength: float = 1.0,
    emotion: EmotionType = EmotionType.NEUTRAL,
    pitch_shift: float = 0.0,
) -> StreamingResponse:
    """Endpoint GET para pruebas y reproducción directa en navegador."""
    return _synthesize_audio_response(
        voice_id=id,
        text=text,
        speed=speed,
        style_strength=style_strength,
        emotion=emotion.value if isinstance(emotion, EmotionType) else str(emotion),
        pitch_shift=pitch_shift,
    )
