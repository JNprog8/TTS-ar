# AGENTS.md — TTS Argentino

Reglas compartidas del repo, leídas por cualquier herramienta agentic
(Antigravity, Claude Code, Cursor). Overrides específicos de Antigravity
(enrutamiento de modelos, terminal policy) van en `GEMINI.md`, no acá —
así este archivo sigue siendo portable si el equipo suma otra herramienta.

## Qué es este repo

API de TTS en español argentino basada en **Piper TTS / VITS** con
fonemización nativa `espeak-ng` (`es_AR`) sobre el dataset
`ylacombe/google-argentinian-spanish` + una API FastAPI de alto rendimiento
en ONNX Runtime que sirve el modelo resultante. Consumidor real:
un pipeline externo STT → texto, LLM externo → elige `id` de voz según
género/tono, `POST /audio/tts` → audio.

## Estructura

- `finetuning/` — preparación de datos y notebook de entrenamiento Piper VITS (`prepare_dataset_piper.py`, `train_piper.py`, `finetune_piper_ar.ipynb`).
- `voices/` — modelo ONNX (`piper_ar.onnx`), configuración JSON (`piper_ar.onnx.json`) y catálogo de voces (`voices_catalog.csv`).
- `api/` — servicio FastAPI (`config.py`, `schemas.py`, `tts_engine.py`, `main.py`).

## Reglas críticas (no negociables sin confirmación explícita del usuario)

1. No cambiar el contrato de `POST /audio/tts` (nombres de campos, códigos
   de error) sin aprobación explícita — el LLM/STT externos ya integran
   contra este contrato.
2. Control de habla = `speed` (mapea a `length_scale`) + `style_strength`
   (mapea a `noise_scale` / `noise_w` en VITS).
3. No commitear `voices/*.onnx`, datasets crudos, ni audios de prueba — son
   artefactos binarios, van a `.gitignore`.

## Convenciones de código

- Identificadores en inglés; comentarios y docstrings en español, breves
  (1-3 líneas), explicando el "por qué", no el "qué".
- Type hints en firmas públicas, `pydantic` para validación en la API,
  `logging` en vez de `print` dentro de `api/`.
- Un archivo, una responsabilidad.