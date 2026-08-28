---
description: Usar cuando una tarea implica debuggear el entrenamiento de finetune_f5tts_ar.ipynb — pérdidas que no bajan, audio de validación que suena mal, problemas con el vocabulario extendido, o clips de referencia que no clonan bien. Se activa por contexto, no hace falta invocarla por nombre.
name: f5tts-finetune-review
---

# Revisión de finetuning de F5-TTS

## Cuándo se activa
Cambios o errores relacionados con las celdas 4-9 del notebook: preparación
de datos, extensión de vocabulario, entrenamiento (LoRA o completo), o
validación/exportación.

## Checklist antes de tocar el training

1. ¿El problema es de datos (transcripciones mal alineadas, vocabulario
   incompleto) o del modelo (loss, hiperparámetros)? Revisar primero los
   `missing_chars` reportados en la celda 6 antes de tocar el training loop.
2. Si el audio de validación no pronuncia bien tildes/eñe: syntoma clásico
   de vocabulario no extendido correctamente, o de haber entrenado con el
   checkpoint base sin pasar por `EXTENDED_CKPT_PATH`.
3. Si el timbre clonado no se parece al hablante: revisar primero la calidad
   del clip de referencia (celda 3) — F5-TTS es muy sensible a ruido de
   fondo o cortes de frase en el audio de referencia, más que a hiperparámetros.
4. Si el acento sigue sonando neutro/no-argentino tras el finetuning: es
   esperable con pocas epochs o poco dataset — reportar como "necesita más
   datos/epochs", no como bug. Comparar LoRA vs finetuning completo si el
   dataset es grande y hay GPU disponible.

## Qué reportar al terminar

Modo usado (LoRA o completo), epochs corridos, caracteres agregados al
vocabulario, y qué voice_ids se validaron auditivamente — no solo "listo,
corrió sin errores".
