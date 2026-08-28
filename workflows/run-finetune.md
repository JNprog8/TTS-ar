# run-finetune

Prepara el dataset y entrena/finetunea modelos Piper TTS (VITS) en español argentino.

1. Preparar el dataset argentino de Google ejecutando `python finetuning/prepare_dataset_piper.py`.
2. Verificar que se genere `finetuning/data/piper_dataset/metadata.csv` y los audios en `wavs/`.
3. Ejecutar el entrenamiento con `python finetuning/train_piper.py` o mediante el notebook interactivo `finetuning/finetune_piper_ar.ipynb`.
4. Exportar el modelo resultante a formato ONNX en `voices/piper_ar.onnx` y `voices/piper_ar.onnx.json`.
5. Ejecutar `python -m pytest api/tests` para validar la integración del nuevo modelo con la API.
