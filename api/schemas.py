"""Contratos de entrada/salida de la API de TTS con soporte de sentimientos/emociones del paciente."""
from enum import Enum
from pydantic import BaseModel, Field


class EmotionType(str, Enum):
    NEUTRAL = "neutral"      # Paciente en estado neutro / de control
    ANNOYED = "annoyed"      # Paciente molesto / quejoso / irritado
    PAIN = "pain"            # Paciente adolorido / con dolor físico / quejumbroso
    WORRIED = "worried"      # Paciente preocupado / ansioso / asustado
    CUSTOM = "custom"        # Control manual de parámetros


class TTSRequest(BaseModel):
    id: int = Field(0, description="Identificador de la voz/arquetipo del paciente (ver GET /audio/voices)")
    text: str = Field(..., min_length=1, description="Texto que dice el paciente")

    speed: float = Field(1.0, ge=0.4, le=2.5, description="Velocidad base de habla (default: 1.0)")
    style_strength: float = Field(1.0, ge=0.0, le=2.0, description="Intensidad de estilo y energía tímbrica (default: 1.0)")

    emotion: EmotionType = Field(
        EmotionType.NEUTRAL,
        description="Sentimiento/estado del paciente: neutral, annoyed, pain, worried, custom",
    )
    pitch_shift: float | None = Field(
        None,
        ge=-12.0,
        le=12.0,
        description="Ajuste manual de tono en semitonos (opcional, ej. -3.0 para más grave, +2.0 para más agudo)",
    )


class VoiceInfo(BaseModel):
    id: int
    gender: str
    name: str | None = None
    description: str | None = None


class ErrorResponse(BaseModel):
    detail: str
