"""Prepara el dataset 'ylacombe/google-argentinian-spanish' (femenino o masculino)
para entrenamiento y finetuning con Piper TTS (VITS).

Estructura de salida:
  finetuning/data/piper_male_dataset/ (o piper_dataset/)
    ├── wavs/
    │   ├── audio_000000.wav (22.05 kHz, mono PCM_16)
    │   └── ...
    ├── metadata.csv (formato: filename|speaker_id|transcript)
    ├── train.csv
    ├── val.csv
    └── speaker_map.json
"""

import argparse
import io
import json
import os
import random
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT_DIR / "finetuning" / "data"
RAW_DATA_DIR = ROOT_DIR / "finetuning" / "data" / "ar_rioplatense_raw"
TARGET_SR = 22050


def parse_args():
    parser = argparse.ArgumentParser(description="Preparación del dataset para Piper TTS.")
    parser.add_argument("--gender", type=str, default="female", choices=["female", "male"], help="Subconjunto a preparar (female o male)")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directorio de salida personalizado")
    parser.add_argument("--max_samples", type=int, default=None, help="Límite de muestras a procesar")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Proporción de validación (default: 0.05)")
    parser.add_argument("--force_download", action="store_true", help="Forzar descarga desde HuggingFace ignorando datos locales")
    return parser.parse_args()


def process_local_raw_dataset(target_dir: Path, max_samples: int | None = None, val_ratio: float = 0.05):
    """Procesa el dataset desde la copia local raw femenina usando Torchaudio."""
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
    wavs_dir = target_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    speaker_to_id = {"speaker_0": 0}
    with open(target_dir / "speaker_map.json", "w", encoding="utf-8") as f:
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
            out_wav_path = wavs_dir / wav_filename
            torchaudio.save(str(out_wav_path), wav_tensor, TARGET_SR, encoding="PCM_S", bits_per_sample=16)

            metadata_lines.append(f"{wav_filename}|0|{text}")
        except Exception as e:
            print(f"Error procesando {raw_path}: {e}")
            continue

    _save_splits(target_dir, metadata_lines, val_ratio)
    return True


def process_huggingface_dataset(gender: str, target_dir: Path, max_samples: int | None = None, val_ratio: float = 0.05):
    """Descarga y procesa desde HuggingFace Hub para el género indicado ('female' o 'male')."""
    from datasets import Audio, load_dataset

    wavs_dir = target_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDescargando dataset 'ylacombe/google-argentinian-spanish' (split: {gender}) desde HuggingFace...")
    ds = load_dataset("ylacombe/google-argentinian-spanish", gender, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print(f"Total de registros a procesar: {len(ds)}")

    unique_speakers = sorted(list(set(ds["speaker_id"]))) if "speaker_id" in ds.column_names else ["speaker_0"]
    speaker_to_id = {spk: idx for idx, spk in enumerate(unique_speakers)}
    print(f"Hablantes detectados: {len(unique_speakers)}")

    with open(target_dir / "speaker_map.json", "w", encoding="utf-8") as f:
        json.dump(speaker_to_id, f, indent=2, ensure_ascii=False)

    metadata_lines = []
    resamplers = {}

    for idx, item in enumerate(tqdm(ds, desc=f"Procesando audios ({gender})")):
        audio_bytes = item["audio"]["bytes"]
        text = item.get("text") or item.get("transcription", "")
        text = text.strip()
        if not text or not audio_bytes:
            continue

        try:
            wav_tensor, orig_sr = torchaudio.load(io.BytesIO(audio_bytes))
            if wav_tensor.shape[0] > 1:
                wav_tensor = wav_tensor.mean(dim=0, keepdim=True)

            if orig_sr != TARGET_SR:
                if orig_sr not in resamplers:
                    resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, TARGET_SR)
                wav_tensor = resamplers[orig_sr](wav_tensor)

            max_val = torch.max(torch.abs(wav_tensor))
            if max_val > 1e-6:
                wav_tensor = (wav_tensor / max_val) * 0.95

            wav_filename = f"audio_{idx:06d}.wav"
            wav_path = wavs_dir / wav_filename
            torchaudio.save(str(wav_path), wav_tensor, TARGET_SR, encoding="PCM_S", bits_per_sample=16)

            spk_name = item.get("speaker_id", "speaker_0")
            spk_id = speaker_to_id.get(spk_name, 0)
            metadata_lines.append(f"{wav_filename}|{spk_id}|{text}")
        except Exception as e:
            print(f"Error procesando muestra {idx}: {e}")
            continue

    _save_splits(target_dir, metadata_lines, val_ratio)


def _save_splits(target_dir: Path, metadata_lines: list[str], val_ratio: float):
    random.seed(42)
    shuffled = list(metadata_lines)
    random.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio))
    val_lines = shuffled[:val_count]
    train_lines = shuffled[val_count:]

    meta_file = target_dir / "metadata.csv"
    train_file = target_dir / "train.csv"
    val_file = target_dir / "val.csv"

    with open(meta_file, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines) + "\n")

    with open(train_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    print(f"\n[OK] Dataset preparado exitosamente en {target_dir}")
    print(f"  - Total muestras: {len(metadata_lines)}")
    print(f"  - Entrenamiento: {len(train_lines)} ({train_file.name})")
    print(f"  - Validación:    {len(val_lines)} ({val_file.name})")
    print(f"  - Metadatos:     {meta_file}")


def main():
    args = parse_args()
    if args.output_dir:
        target_dir = args.output_dir
    else:
        target_dir = DATA_ROOT / ("piper_male_dataset" if args.gender == "male" else "piper_dataset")

    target_dir.mkdir(parents=True, exist_ok=True)

    if args.gender == "female" and not args.force_download and process_local_raw_dataset(target_dir, args.max_samples, args.val_ratio):
        return

    process_huggingface_dataset(args.gender, target_dir, args.max_samples, args.val_ratio)


if __name__ == "__main__":
    main()
