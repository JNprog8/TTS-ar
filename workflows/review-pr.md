# review-pr

Pensado para ejecutarse vía GitHub agent (MCP) sobre un PR abierto.

1. Traer el diff completo del PR.
2. Si el diff toca `api/schemas.py` o el training loop del notebook,
   marcar el PR como "necesita revisión de Claude/Gemini Pro" y no aprobar
   automáticamente (ver regla en `.agent/rules/git.md`).
3. Correr lint (`ruff check api/`) y tests (`pytest api/tests/`) sobre la
   rama del PR.
4. Comentar en el PR: resultado de lint/tests, y si el diff rompe alguna
   regla crítica de `AGENTS.md` (contrato de `/audio/tts`, `temperature`,
   binarios commiteados).
5. Solo mergear si lint+tests pasan y ninguna regla crítica fue violada.
