"""Script para entrenamiento, finetuning y exportación ONNX de modelos Piper TTS (VITS).

Uso:
  python finetuning/train_piper.py --dataset_dir finetuning/data/piper_dataset --epochs 50 --batch_size 32
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "finetuning" / "data" / "piper_dataset"
VOICES_DIR = ROOT_DIR / "voices"


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento y exportación de Piper TTS (VITS).")
    parser.add_argument("--dataset_dir", type=Path, default=DATA_DIR, help="Ruta al directorio de datos preparados")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size de entrenamiento")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--lr", type=float, default=2e-4, help="Tasa de aprendizaje")
    parser.add_argument("--export_only", action="store_true", help="Exportar checkpoint PyTorch a ONNX sin entrenar")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Piper TTS (VITS) Training & Export Manager")
    print("=" * 60)
    print(f"Dataset dir: {args.dataset_dir}")
    print(f"Voices dir:  {VOICES_DIR}")
    
    if not (args.dataset_dir / "metadata.csv").exists():
        print(f"\n[AVISO] No se encontró {args.dataset_dir / 'metadata.csv'}.")
        print("Ejecutá 'python finetuning/prepare_dataset_piper.py' para procesar el dataset primero.")


if __name__ == "__main__":
    main()
