# GEMINI.md — Overrides de Antigravity para TTS Argentino

Este archivo solo contiene lo que es específico de Antigravity. Las reglas
generales del proyecto están en `AGENTS.md`; en caso de conflicto, lo que
diga acá gana.

## Modo de agente recomendado

- `api/` y `finetuning/*.ipynb`: **Agent-assisted** (no Autopilot). Son los
  dos artefactos que definen el contrato externo y la calidad de la voz.
- Workflows, docs, tests nuevos: Autopilot está bien, es boilerplate de bajo riesgo.
- Terminal Policy: Auto para comandos de lectura/test (`pytest`, `ruff`,
  `git status`); pedir confirmación para cualquier `git push` o `pip install`.

## Enrutamiento de agentes por tipo de tarea

| Tarea | Agente | Por qué |
|---|---|---|
| Boilerplate: schema nuevo, endpoint simple, docstrings, tests repetitivos | **Gemini 3 Flash** | Rápido y barato para trabajo mecánico |
| Feature con lógica de negocio, entender `tts_engine.py` + `main.py` + `config.py` en conjunto | **Gemini 3 Pro** | Contexto amplio, razona bien sobre varios archivos a la vez |
| Debugging de síntesis ONNX o fonemización espeak-ng | **Claude** | Mejor para debugging fonético multi-paso |
| Abrir/revisar PRs, correr CI, comentar diffs, mergear | **GitHub agent** (MCP) vía skill `github-pr-flow` | Tarea async ligada a Git/CI |