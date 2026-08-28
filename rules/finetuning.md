# Reglas — Finetuning

Aplican a cualquier trabajo dentro de `finetuning/`.

- Piper TTS (VITS) utiliza fonemización nativa con `espeak-ng` (`es_AR`).
- El dataset de entrenamiento se procesa con `prepare_dataset_piper.py` para generar el formato estándar `metadata.csv` (`audio_file|speaker_id|text`).
- Los modelos exportados deben guardarse en formato ONNX (`.onnx` + `.onnx.json`) en `voices/`.
- Validar siempre los modelos exportados ejecutando la suite de tests en `api/tests/`.
