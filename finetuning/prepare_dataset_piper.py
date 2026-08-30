"""Prepara el dataset 'ylacombe/google-argentinian-spanish' para entrenamiento y finetuning con Piper TTS (VITS).

Estructura de salida:
  finetuning/data/piper_dataset/
    ├── wavs/
    │   ├── audio_000000.wav (22.05 kHz, mono PCM_16)
    │   └── ...
    ├── metadata.csv (formato: filename|speaker_id|transcript)
    ├── train.csv
    ├── val.csv
    └── speaker_map.json
"""

import argparse
import json
import os
import random
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "finetuning" / "data" / "piper_dataset"
WAVS_DIR = DATA_DIR / "wavs"
RAW_DATA_DIR = ROOT_DIR / "finetuning" / "data" / "ar_rioplatense_raw"
METADATA_FILE = DATA_DIR / "metadata.csv"
TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "val.csv"
SPEAKER_MAP_FILE = DATA_DIR / "speaker_map.json"

TARGET_SR = 22050


def parse_args():
    parser = argparse.ArgumentParser(description="Preparación del dataset para Piper TTS.")
    parser.add_argument("--max_samples", type=int, default=None, help="Límite de muestras a procesar (para pruebas rápidas)")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Proporción de validación (default: 0.05)")
    parser.add_argument("--force_download", action="store_true", help="Forzar descarga desde HuggingFace ignorando datos locales")
    return parser.parse_args()


def process_local_raw_dataset(max_samples: int | None = None, val_ratio: float = 0.05):
    """Procesa el dataset desde la copia local raw usando Torchaudio."""
    raw_metadata_path = RAW_DATA_DIR / "metadata.csv"
    if not raw_metadata_path.exists():
        return False

    print(f"Detectado dataset local en: {RAW_DATA_DIR}")
    print("Leyendo metadatos locales...")

    lines = []
    with open(raw_metadata_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or (idx == 0 and ("audio_file" in line or "filename" in line)):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                audio_path_str = parts[0].strip()
                text = parts[1].strip()
                if text:
                    lines.append((audio_path_str, text))

    if max_samples:
        lines = lines[:max_samples]

    print(f"Total de audios a procesar: {len(lines)}")
    WAVS_DIR.mkdir(parents=True, exist_ok=True)

    speaker_to_id = {"speaker_0": 0}
    with open(SPEAKER_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(speaker_to_id, f, indent=2, ensure_ascii=False)

    metadata_lines = []
    resamplers = {}

    for idx, (raw_audio_path_str, text) in enumerate(tqdm(lines, desc="Resampleando y formateando audios")):
        raw_path = Path(raw_audio_path_str)
        if not raw_path.is_absolute():
            raw_path = RAW_DATA_DIR / raw_path
        if not raw_path.exists():
            raw_path = RAW_DATA_DIR / "wavs" / Path(raw_audio_path_str).name
        if not raw_path.exists():
            continue

        try:
            wav_tensor, orig_sr = torchaudio.load(str(raw_path))
            if wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

            if orig_sr != TARGET_SR:
                if orig_sr not in resamplers:
                    resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, TARGET_SR)
                wav_tensor = resamplers[orig_sr](wav_tensor)

            # Normalizar amplitud a 0.95 pico
            max_val = torch.max(torch.abs(wav_tensor))
            if max_val > 1e-6:
                wav_tensor = (wav_tensor / max_val) * 0.95

            wav_filename = f"audio_{idx:06d}.wav"
            out_wav_path = WAVS_DIR / wav_filename
            torchaudio.save(str(out_wav_path), wav_tensor, TARGET_SR, encoding="PCM_S", bits_per_sample=16)

            metadata_lines.append(f"{wav_filename}|0|{text}")
        except Exception as e:
            print(f"Error procesando {raw_path}: {e}")
            continue

    _save_splits(metadata_lines, val_ratio)
    return True


def process_huggingface_dataset(max_samples: int | None = None, val_ratio: float = 0.05):
    """Descarga y procesa desde HuggingFace Hub."""
    from datasets import load_dataset

    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    print("Descargando / cargando dataset 'ylacombe/google-argentinian-spanish' desde HuggingFace...")
    ds = load_dataset("ylacombe/google-argentinian-spanish", split="train")

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Total de registros: {len(ds)}")

    unique_speakers = sorted(list(set(ds["speaker_id"])))
    speaker_to_id = {spk: idx for idx, spk in enumerate(unique_speakers)}
    print(f"Hablantes detectados: {len(unique_speakers)}")

    with open(SPEAKER_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(speaker_to_id, f, indent=2, ensure_ascii=False)

    metadata_lines = []
    resamplers = {}

    for idx, item in enumerate(tqdm(ds, desc="Procesando audios")):
        audio_dict = item["audio"]
        audio_array = audio_dict["array"]
        orig_sr = audio_dict["sampling_rate"]
        text = item.get("text") or item.get("transcription", "")
        text = text.strip()
        if not text:
            continue

        wav_tensor = torch.from_numpy(audio_array).float()
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        elif wav_tensor.shape[0] > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

        if orig_sr != TARGET_SR:
            if orig_sr not in resamplers:
                resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, TARGET_SR)
            wav_tensor = resamplers[orig_sr](wav_tensor)

        max_val = torch.max(torch.abs(wav_tensor))
        if max_val > 1e-6:
            wav_tensor = (wav_tensor / max_val) * 0.95

        wav_filename = f"audio_{idx:06d}.wav"
        wav_path = WAVS_DIR / wav_filename
        torchaudio.save(str(wav_path), wav_tensor, TARGET_SR, encoding="PCM_S", bits_per_sample=16)

        spk_id = speaker_to_id[item["speaker_id"]]
        metadata_lines.append(f"{wav_filename}|{spk_id}|{text}")

    _save_splits(metadata_lines, val_ratio)


def _save_splits(metadata_lines: list[str], val_ratio: float):
    random.seed(42)
    shuffled = list(metadata_lines)
    random.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio))
    val_lines = shuffled[:val_count]
    train_lines = shuffled[val_count:]

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines) + "\n")

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"\n[OK] Dataset preparado exitosamente en {DATA_DIR}")
    print(f"  - Total muestras: {len(metadata_lines)}")
    print(f"  - Entrenamiento: {len(train_lines)} ({TRAIN_FILE.name})")
    print(f"  - Validación:    {len(val_lines)} ({VAL_FILE.name})")
    print(f"  - Metadatos:     {METADATA_FILE}")


def main():
    args = parse_args()
    if not args.force_download and process_local_raw_dataset(args.max_samples, args.val_ratio):
        return
    process_huggingface_dataset(args.max_samples, args.val_ratio)


if __name__ == "__main__":
    main()
