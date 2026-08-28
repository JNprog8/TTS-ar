# Reglas — Git y PRs

Aplican al GitHub agent y a cualquier agente que abra commits/PRs.

- Mensajes de commit en español, formato `tipo: descripción breve`
  (`fix:`, `feat:`, `docs:`, `refactor:`).
- Un PR = un cambio lógico. No mezclar cambios de `api/` con cambios del
  notebook de finetuning en el mismo PR.
- Todo PR que toque `api/schemas.py` o el training loop del notebook
  necesita revisión de Claude o Gemini Pro antes de mergear (ver `GEMINI.md`,
  sección "Protocolo de handoff").
- La descripción del PR incluye siempre: qué cambia, por qué, y cómo se
  probó (comando de test corrido o muestra de audio escuchada).
