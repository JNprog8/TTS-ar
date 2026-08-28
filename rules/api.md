# Reglas — API

Aplican a cualquier trabajo dentro de `api/`.

- `schemas.py` es el contrato público. Cualquier campo nuevo va con
  `Field(..., description=...)` y default explícito si es opcional.
- Toda validación de negocio (texto vacío, id inexistente, límites) va en
  `main.py` antes de llamar a `engine`, nunca dentro de `tts_engine.py` —
  el engine asume inputs ya validados.
- `TTSEngine` carga el modelo una sola vez al importar el módulo (patrón
  singleton). No instanciar `TTSEngine()` de nuevo dentro de un endpoint.
- Todo endpoint nuevo devuelve `HTTPException` con `status_code` y `detail`
  explícitos para los casos 400 — nunca dejar que una excepción del modelo
  llegue como 500 genérico sin contexto.
- Tests de endpoints van en `api/tests/`, uno por endpoint como mínimo,
  usando `TestClient` de FastAPI (no levantar un servidor real para testear).
