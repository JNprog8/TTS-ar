# add-voice

Agrega un nuevo arquetipo o voz de paciente al catálogo de voces.

1. Identificar el perfil demográfico o modelo Piper correspondiente (femenino, masculino o multihablante).
2. Registrar la nueva voz en `voices/voices_catalog.csv` especificando `voice_id`, `gender`, `name` y `description`.
3. Si requiere modulación acústica específica (tono, velocidad, formantes), configurar su perfil en `PERSONA_PROFILES` dentro de `api/tts_engine.py`.
4. Validar ejecutando `GET /audio/voices` y sintetizando una muestra de prueba.
5. Ejecutar `python -m pytest api/tests` para asegurar consistencia.
