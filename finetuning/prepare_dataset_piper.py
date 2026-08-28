"""Prepara el dataset 'ylacombe/google-argentinian-spanish' para entrenamiento y finetuning con Piper TTS (VITS).

Estructura de salida:
  finetuning/data/piper_dataset/
    ├── wavs/
    │   ├── audio_000000.wav (22.05 kHz, mono)
    │   └── ...
    ├── metadata.csv (formato: filename|speaker_id|transcript)
    └── speaker_map.json
"""

import json
import os
from pathlib import Path
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "finetuning" / "data" / "piper_dataset"
WAVS_DIR = DATA_DIR / "wavs"
METADATA_FILE = DATA_DIR / "metadata.csv"
SPEAKER_MAP_FILE = DATA_DIR / "speaker_map.json"

TARGET_SR = 22050


def prepare_piper_dataset():
    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    print("Descargando / cargando dataset 'ylacombe/google-argentinian-spanish'...")
    ds = load_dataset("ylacombe/google-argentinian-spanish", split="train")

    print(f"Total de registros: {len(ds)}")

    # Mapeo de hablantes a IDs numéricos contiguos (0..N-1)
    unique_speakers = sorted(list(set(ds["speaker_id"])))
    speaker_to_id = {spk: idx for idx, spk in enumerate(unique_speakers)}
    print(f"Hablantes detectados: {len(unique_speakers)}")

    with open(SPEAKER_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(speaker_to_id, f, indent=2, ensure_ascii=False)

    metadata_lines = []

    print("Procesando audios y generando metadatos...")
    for idx, item in enumerate(tqdm(ds)):
        audio_dict = item["audio"]
        audio_array = audio_dict["array"]
        orig_sr = audio_dict["sampling_rate"]
        text = item.get("text") or item.get("transcription", "")
        text = text.strip()
        if not text:
            continue

        # Resamplear a 22050 Hz si difiere
        if orig_sr != TARGET_SR:
            audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=TARGET_SR)

        wav_filename = f"audio_{idx:06d}.wav"
        wav_path = WAVS_DIR / wav_filename

        sf.write(str(wav_path), audio_array, TARGET_SR, subtype="PCM_16")

        spk_id = speaker_to_id[item["speaker_id"]]
        # Formato estándar de Piper: audio_path|speaker_id|text
        metadata_lines.append(f"{wav_filename}|{spk_id}|{text}")

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines) + "\n")

    print(f"\n[OK] Dataset de Piper preparado exitosamente en {DATA_DIR}")
    print(f"  - Audios guardados: {len(metadata_lines)} en {WAVS_DIR}")
    print(f"  - Metadatos guardados: {METADATA_FILE}")


if __name__ == "__main__":
    prepare_piper_dataset()
