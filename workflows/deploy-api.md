# deploy-api

Levanta la API localmente, la valida con un smoke test, y deja todo listo
para deploy.

1. `pip install -r api/requirements.txt`.
2. Confirmar que `voices/f5tts_ar_finetuned.safetensors`, `voices/vocab_ar.txt`
   y al menos un `voices/ref_voice_<id>.wav` existen (si no, correr
   `/run-finetune` primero).
3. `uvicorn main:app --app-dir api --host 0.0.0.0 --port 8000`.
4. Smoke test: `GET /audio/voices` debe devolver al menos una voz;
   `POST /audio/tts` con un texto corto y un id válido debe devolver 200
   y un WAV no vacío; con un id inválido debe devolver 400.
5. Si todo pasa, reportar el resumen de smoke tests corridos y sus
   resultados — no asumir que "levantó" significa que funciona.
