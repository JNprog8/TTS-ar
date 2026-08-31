"""Pipeline integral de entrenamiento, finetuning y exportación ONNX para modelos Piper TTS (VITS).
Optimizado para aceleración GPU NVIDIA CUDA (RTX 4060) y fonemización en español argentino (es_AR).

Uso:
  python finetuning/train_piper.py --dataset_dir finetuning/data/piper_dataset --epochs 10 --batch_size 16
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
from tqdm import tqdm

# Asegurar UTF-8 en salida estándar
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "finetuning" / "data" / "piper_dataset"
VOICES_DIR = ROOT_DIR / "voices"
RUNS_DIR = ROOT_DIR / "runs" / "piper_ar"

# Constantes acústicas estándar Piper VITS (22.05 kHz)
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0.0
F_MAX = 8000.0
NUM_SYMBOLS = 256
EMBEDDING_DIM = 192
SEGMENT_SIZE = 16384  # 64 frames mel * 256 hop = 16384 muestras de audio (~0.74s)


class MelSpectrogramExtractor(nn.Module):
    """Extractor diferenciable y estable de Mel-Espectrogramas en GPU."""
    def __init__(self, sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 2:
            pass  # [B, T]
        elif wav.ndim == 3:
            wav = wav.squeeze(1)  # [B, 1, T] -> [B, T]
        mel = self.mel_transform(wav)
        # Dynamic range compression logarítmica acotada
        mel = torch.log(torch.clamp(mel, min=1e-5, max=1e4))
        return mel


class PiperDataset(Dataset):
    """Dataset para entrenamiento y validación de Piper VITS."""
    def __init__(self, csv_path: Path, wavs_dir: Path, phoneme_id_map: dict, max_wav_len: int = 22050 * 10):
        self.wavs_dir = wavs_dir
        self.phoneme_id_map = phoneme_id_map
        self.max_wav_len = max_wav_len
        self.items = []

        if not csv_path.exists():
            print(f"[AVISO] Archivo {csv_path} no encontrado.")
            return

        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 3:
                    wav_file, spk_id, text = parts[0].strip(), int(parts[1].strip()), parts[2].strip()
                    wav_path = self.wavs_dir / wav_file
                    if wav_path.exists():
                        self.items.append((wav_path, spk_id, text))

        print(f"Dataset cargado desde {csv_path.name}: {len(self.items)} muestras válidas.")

    def __len__(self):
        return len(self.items)

    def _text_to_phoneme_ids(self, text: str) -> list[int]:
        tokens = [1]  # BOS
        for char in text.lower():
            if char in self.phoneme_id_map:
                tokens.append(self.phoneme_id_map[char][0])
            else:
                tokens.append(3)  # Espacio / pad
        tokens.append(2)  # EOS
        return tokens

    def __getitem__(self, idx):
        wav_path, spk_id, text = self.items[idx]
        wav, sr = torchaudio.load(str(wav_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)  # [T]

        # Asegurar longitud mínima de SEGMENT_SIZE
        if wav.shape[0] < SEGMENT_SIZE:
            wav = F.pad(wav, (0, SEGMENT_SIZE - wav.shape[0]))
        elif wav.shape[0] > self.max_wav_len:
            wav = wav[:self.max_wav_len]

        token_ids = self._text_to_phoneme_ids(text)
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        return token_tensor, wav, spk_id


def collate_piper_batch(batch):
    """Collate con padding dinámico para secuencias de texto y audios."""
    tokens_list, wavs_list, spk_list = zip(*batch)

    token_lengths = torch.tensor([len(t) for t in tokens_list], dtype=torch.long)
    wav_lengths = torch.tensor([len(w) for w in wavs_list], dtype=torch.long)

    max_token_len = max(len(t) for t in tokens_list)
    max_wav_len = max(len(w) for w in wavs_list)

    padded_tokens = torch.zeros(len(tokens_list), max_token_len, dtype=torch.long)
    for i, t in enumerate(tokens_list):
        padded_tokens[i, :len(t)] = t

    padded_wavs = torch.zeros(len(wavs_list), max_wav_len, dtype=torch.float32)
    for i, w in enumerate(wavs_list):
        padded_wavs[i, :len(w)] = w

    speaker_ids = torch.tensor(spk_list, dtype=torch.long)
    return padded_tokens, token_lengths, padded_wavs, wav_lengths, speaker_ids


def rand_slice_segments(x_wav: torch.Tensor, z_lat: torch.Tensor, segment_size: int = SEGMENT_SIZE, hop_length: int = HOP_LENGTH):
    """Extrae aleatoriamente rebanadas temporales coordinadas de audio y espacio latente."""
    b, t_wav = x_wav.shape
    t_frames = segment_size // hop_length
    z_len = z_lat.shape[-1]

    wav_slices = []
    z_slices = []

    for i in range(b):
        cur_w_len = x_wav[i].shape[-1]
        max_start_w = max(0, cur_w_len - segment_size)
        start_w = random.randint(0, max_start_w) if max_start_w > 0 else 0
        end_w = start_w + segment_size

        wav_slices.append(x_wav[i, start_w:end_w])

        start_frame = start_w // hop_length
        end_frame = start_frame + t_frames

        if end_frame <= z_len:
            z_slices.append(z_lat[i, :, start_frame:end_frame])
        else:
            z_slice = z_lat[i, :, -t_frames:] if z_len >= t_frames else F.pad(z_lat[i], (0, t_frames - z_len))
            z_slices.append(z_slice)

    return torch.stack(wav_slices, dim=0), torch.stack(z_slices, dim=0)


# -------------------------------------------------------------
# Módulos Neuronales VITS (Encoder, Posterior, HiFi-GAN)
# -------------------------------------------------------------

class WNResBlock(nn.Module):
    """Residual Block estilo WaveNet."""
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.normal_(self.conv1.weight, 0.0, 0.02)
        nn.init.normal_(self.conv2.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.conv1(x))
        h = self.conv2(h)
        return x + h


class TextEncoder(nn.Module):
    """Encoder de texto fonético VITS."""
    def __init__(self, num_symbols: int = NUM_SYMBOLS, out_channels: int = EMBEDDING_DIM, hidden_channels: int = 192):
        super().__init__()
        self.embedding = nn.Embedding(num_symbols, hidden_channels)
        self.conv_layers = nn.ModuleList([
            WNResBlock(hidden_channels, dilation=1),
            WNResBlock(hidden_channels, dilation=2),
            WNResBlock(hidden_channels, dilation=4),
            WNResBlock(hidden_channels, dilation=8),
        ])
        self.proj_mean = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self.proj_log_std = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        nn.init.normal_(self.proj_mean.weight, 0.0, 0.02)
        nn.init.normal_(self.proj_log_std.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor):
        h = self.embedding(x).transpose(1, 2)
        for layer in self.conv_layers:
            h = layer(h)
        stats_mean = self.proj_mean(h)
        stats_log_std = torch.clamp(self.proj_log_std(h), min=-6.0, max=3.0)
        return stats_mean, stats_log_std, h


class PosteriorEncoder(nn.Module):
    """Posterior Encoder de espectrograma Mel a espacio latente z ~ N(mu, sigma)."""
    def __init__(self, in_channels: int = N_MELS, out_channels: int = EMBEDDING_DIM, hidden_channels: int = 192):
        super().__init__()
        self.pre_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.res_blocks = nn.ModuleList([
            WNResBlock(hidden_channels, dilation=1),
            WNResBlock(hidden_channels, dilation=2),
            WNResBlock(hidden_channels, dilation=4),
        ])
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, kernel_size=1)
        nn.init.normal_(self.pre_conv.weight, 0.0, 0.02)
        nn.init.normal_(self.proj.weight, 0.0, 0.02)

    def forward(self, mel: torch.Tensor):
        h = self.pre_conv(mel)
        for block in self.res_blocks:
            h = block(h)
        stats = self.proj(h)
        mean, log_std = torch.chunk(stats, 2, dim=1)
        log_std = torch.clamp(log_std, min=-6.0, max=3.0)
        z = mean + torch.randn_like(mean) * torch.exp(log_std)
        return z, mean, log_std


class HiFiGANDecoder(nn.Module):
    """Generador HiFi-GAN que sintetiza audio crudo desde representaciones latentes z."""
    def __init__(self, in_channels: int = EMBEDDING_DIM):
        super().__init__()
        self.pre_conv = nn.Conv1d(in_channels, 256, kernel_size=7, padding=3)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(128, 64, kernel_size=16, stride=8, padding=4),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
        ])
        self.resblocks = nn.ModuleList([
            WNResBlock(128, dilation=1),
            WNResBlock(64, dilation=1),
            WNResBlock(32, dilation=1),
            WNResBlock(16, dilation=1),
        ])
        self.post_conv = nn.Conv1d(16, 1, kernel_size=7, padding=3)

        nn.init.normal_(self.pre_conv.weight, 0.0, 0.02)
        for up in self.ups:
            nn.init.normal_(up.weight, 0.0, 0.02)
        nn.init.normal_(self.post_conv.weight, 0.0, 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.pre_conv(z)
        for up, res in zip(self.ups, self.resblocks):
            h = F.leaky_relu(h, 0.1)
            h = up(h)
            h = res(h)
        h = F.leaky_relu(h, 0.1)
        audio = torch.tanh(self.post_conv(h))
        return audio


class MultiPeriodDiscriminator(nn.Module):
    """Discriminador multi-período."""
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, (5, 1), (3, 1), padding=(2, 0)),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 1, (3, 1), 1, padding=(1, 0)),
            ) for _ in periods
        ])
        self.periods = periods

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs = [], []
        for period, disc in zip(self.periods, self.discriminators):
            b, c, t = y.shape
            if t % period != 0:
                n_pad = period - (t % period)
                y_p = F.pad(y, (0, n_pad), "reflect")
                y_hat_p = F.pad(y_hat, (0, n_pad), "reflect")
            else:
                y_p, y_hat_p = y, y_hat
            y_p = y_p.view(b, c, -1, period)
            y_hat_p = y_hat_p.view(b, c, -1, period)
            y_d_rs.append(disc(y_p))
            y_d_gs.append(disc(y_hat_p))
        return y_d_rs, y_d_gs


class PiperVITSModel(nn.Module):
    """Modelo VITS para síntesis y finetuning."""
    def __init__(self, num_symbols: int = NUM_SYMBOLS):
        super().__init__()
        self.text_encoder = TextEncoder(num_symbols=num_symbols)
        self.posterior_encoder = PosteriorEncoder()
        self.decoder = HiFiGANDecoder()

    def forward(self, tokens: torch.Tensor, token_lengths: torch.Tensor, mel: torch.Tensor = None):
        stats_mean, stats_log_std, h_text = self.text_encoder(tokens, token_lengths)
        if mel is not None:
            z, z_mean, z_log_std = self.posterior_encoder(mel)
            return stats_mean, stats_log_std, z_mean, z_log_std, z
        else:
            z = stats_mean + torch.randn_like(stats_mean) * torch.exp(stats_log_std) * 0.667
            audio_hat = self.decoder(z)
            return audio_hat


# -------------------------------------------------------------
# Pipeline de Entrenamiento
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento y Finetuning de Piper TTS en español argentino.")
    parser.add_argument("--dataset_dir", type=Path, default=DATA_DIR, help="Ruta al directorio piper_dataset")
    parser.add_argument("--output_dir", type=Path, default=RUNS_DIR, help="Ruta para guardar checkpoints")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (ideal 16 para 8GB VRAM)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_step", type=int, default=5, help="Guardar checkpoint cada N épocas")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Dispositivo (cuda / cpu)")
    parser.add_argument("--export_onnx", action="store_true", default=True, help="Exportar a ONNX al finalizar")
    parser.add_argument("--export_only", action="store_true", default=False, help="Solo exportar el checkpoint especificado en --resume a ONNX sin entrenar")
    parser.add_argument("--export_target", type=Path, default=None, help="Ruta personalizada para exportar el modelo ONNX final")
    parser.add_argument("--resume", type=Path, default=None, help="Ruta a checkpoint para reanudar o exportar")
    return parser.parse_args()


def export_model_to_onnx(model: nn.Module, onnx_out_path: Path):
    """Exporta el modelo entrenado a formato ONNX."""
    print(f"\n[ONNX] Exportando modelo a {onnx_out_path}...")
    model.eval()
    model.cpu()

    class InferenceWrapper(nn.Module):
        def __init__(self, vits_model):
            super().__init__()
            self.vits = vits_model

        def forward(self, input_ids: torch.Tensor, input_lengths: torch.Tensor, scales: torch.Tensor):
            noise_scale = scales[0]
            batch_size, max_len = input_ids.shape
            seq_range = torch.arange(max_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            mask = (seq_range < input_lengths.unsqueeze(1)).float().unsqueeze(1)
            stats_mean, stats_log_std, _ = self.vits.text_encoder(input_ids, input_lengths)
            stats_mean = stats_mean * mask
            stats_log_std = stats_log_std * mask
            z = stats_mean + torch.randn_like(stats_mean) * torch.exp(stats_log_std) * noise_scale
            audio = self.vits.decoder(z)
            return audio

    wrapper = InferenceWrapper(model)
    dummy_input_ids = torch.randint(1, 50, (1, 32), dtype=torch.long)
    dummy_lengths = torch.tensor([32], dtype=torch.long)
    dummy_scales = torch.tensor([0.667, 1.0, 0.8], dtype=torch.float32)

    onnx_out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import onnx
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_lengths, dummy_scales),
            str(onnx_out_path),
            input_names=["input", "input_lengths", "scales"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 1: "phonemes"},
                "input_lengths": {0: "batch_size"},
                "output": {0: "batch_size", 2: "audio_samples"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"[OK] ONNX exportado exitosamente ({onnx_out_path.stat().st_size / (1024*1024):.2f} MB)")
    except (ImportError, Exception) as e:
        print(f"[AVISO] Error exportando a formato ONNX: {e}")
        print(f"        El checkpoint PyTorch fue guardado correctamente en {onnx_out_path.parent}.")


def train(args):
    if args.export_only:
        if not args.resume or not args.resume.exists():
            print(f"[ERROR] Se requiere --resume apuntando a un checkpoint existente para exportar.")
            return
        device = torch.device(args.device)
        model = PiperVITSModel(num_symbols=NUM_SYMBOLS).to(device)
        print(f"[ONNX] Cargando checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        target_path = args.export_target if args.export_target else (args.output_dir / "piper_vits_finetuned.onnx")
        export_model_to_onnx(model, target_path)
        return

    print("=" * 65)
    print(" Piper TTS (VITS) Finetuning — Español Argentino (es_AR)")
    print("=" * 65)
    print(f"Dataset:    {args.dataset_dir}")
    print(f"Salida:     {args.output_dir}")
    print(f"Device:     {args.device}")
    print(f"Épocas:     {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr}")

    if args.device == "cuda":
        print(f"GPU:        {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB VRAM)")

    config_base_path = VOICES_DIR / "piper_ar.onnx.json"
    phoneme_id_map = {}
    if config_base_path.exists():
        with open(config_base_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            phoneme_id_map = cfg.get("phoneme_id_map", {})

    train_dataset = PiperDataset(args.dataset_dir / "train.csv", args.dataset_dir / "wavs", phoneme_id_map)
    val_dataset = PiperDataset(args.dataset_dir / "val.csv", args.dataset_dir / "wavs", phoneme_id_map)

    if len(train_dataset) == 0:
        print("[ERROR] Dataset de entrenamiento vacío. Ejecutá prepare_dataset_piper.py primero.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_piper_batch,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_piper_batch,
        num_workers=0,
        pin_memory=(args.device == "cuda"),
    )

    device = torch.device(args.device)
    model = PiperVITSModel(num_symbols=NUM_SYMBOLS).to(device)
    disc = MultiPeriodDiscriminator().to(device)
    mel_extractor = MelSpectrogramExtractor().to(device)

    optim_g = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)
    optim_d = torch.optim.AdamW(disc.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=0.01)

    # Scheduler con decaimiento coseno para convergencia suave y prevención de sobreajuste
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optim_g, T_max=args.epochs, eta_min=1e-5)
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optim_d, T_max=args.epochs, eta_min=1e-5)

    start_epoch = 1
    best_val_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and args.resume.exists():
        print(f"Reanudando desde checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optim_g.load_state_dict(ckpt["optim_g"])
        start_epoch = ckpt.get("epoch", 1) + 1
        best_val_loss = ckpt.get("val_loss_mel", float("inf"))

    print("\nIniciando bucle de finetuning...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        disc.train()
        total_loss_g, total_loss_d, total_loss_mel = 0.0, 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Época [{epoch}/{start_epoch + args.epochs - 1}]")
        for tokens, token_lengths, wavs, wav_lengths, _ in pbar:
            tokens, token_lengths = tokens.to(device), token_lengths.to(device)
            wavs = wavs.to(device)

            with torch.no_grad():
                mel_real = mel_extractor(wavs)

            # Forward VITS
            stats_m, stats_logs, z_m, z_logs, z = model(tokens, token_lengths, mel=mel_real)

            # Extraer rebanadas temporales coordinadas
            wav_slice_real, z_slice = rand_slice_segments(wavs, z, segment_size=SEGMENT_SIZE)

            # ---------------------
            # 1. Entrenar Generador
            # ---------------------
            optim_g.zero_grad()
            wav_slice_hat = model.decoder(z_slice).squeeze(1)

            # Mel Reconstruction Loss sobre rebanadas
            mel_slice_real = mel_extractor(wav_slice_real)
            mel_slice_hat = mel_extractor(wav_slice_hat)
            loss_mel = F.l1_loss(mel_slice_hat, mel_slice_real) * 40.0

            # KL Divergence Loss
            stats_m_exp = F.interpolate(stats_m, size=z_m.shape[-1], mode="linear", align_corners=False)
            stats_logs_exp = F.interpolate(stats_logs, size=z_logs.shape[-1], mode="linear", align_corners=False)

            var_p = torch.exp(2.0 * stats_logs_exp)
            var_q = torch.exp(2.0 * z_logs)
            loss_kl = torch.mean(stats_logs_exp - z_logs + (var_q + (z_m - stats_m_exp) ** 2) / (2.0 * var_p) - 0.5) * 1.0

            # Adversarial Loss
            _, y_d_gs = disc(wav_slice_real.unsqueeze(1), wav_slice_hat.unsqueeze(1))
            loss_adv_g = sum(torch.mean((y_dg - 1.0) ** 2) for y_dg in y_d_gs)

            loss_g = loss_mel + loss_kl + loss_adv_g
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim_g.step()

            # -------------------------
            # 2. Entrenar Discriminador
            # -------------------------
            optim_d.zero_grad()
            y_d_rs, y_d_gs = disc(wav_slice_real.unsqueeze(1), wav_slice_hat.detach().unsqueeze(1))
            loss_d = sum(torch.mean((y_dr - 1.0) ** 2) + torch.mean(y_dg ** 2) for y_dr, y_dg in zip(y_d_rs, y_d_gs))

            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            optim_d.step()

            total_loss_g += loss_g.item()
            total_loss_d += loss_d.item()
            total_loss_mel += loss_mel.item()

            pbar.set_postfix({
                "Loss_G": f"{loss_g.item():.2f}",
                "Mel": f"{loss_mel.item():.2f}",
                "Loss_D": f"{loss_d.item():.2f}",
            })

        scheduler_g.step()
        scheduler_d.step()

        avg_g = total_loss_g / len(train_loader)
        avg_mel = total_loss_mel / len(train_loader)
        avg_d = total_loss_d / len(train_loader)

        # -----------------------------
        # 3. Validación Anti-Overfitting
        # -----------------------------
        model.eval()
        val_mel_total = 0.0
        with torch.no_grad():
            for v_tokens, v_lengths, v_wavs, v_wlengths, _ in val_loader:
                v_tokens, v_lengths, v_wavs = v_tokens.to(device), v_lengths.to(device), v_wavs.to(device)
                v_mel = mel_extractor(v_wavs)
                _, _, _, _, v_z = model(v_tokens, v_lengths, mel=v_mel)
                v_wav_slice, v_z_slice = rand_slice_segments(v_wavs, v_z, segment_size=SEGMENT_SIZE)
                v_wav_hat = model.decoder(v_z_slice).squeeze(1)
                val_mel_total += F.l1_loss(mel_extractor(v_wav_hat), mel_extractor(v_wav_slice)).item() * 40.0
        avg_val_mel = val_mel_total / max(1, len(val_loader))

        print(f"--> Época {epoch} Finalizada | Train_Mel = {avg_mel:.4f} | Val_Mel = {avg_val_mel:.4f} | Loss_D = {avg_d:.4f}")

        # Guardar mejor modelo según métrica de validación (Anti-Overfitting)
        if avg_val_mel < best_val_loss:
            best_val_loss = avg_val_mel
            best_ckpt_path = args.output_dir / "piper_vits_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "val_loss_mel": best_val_loss,
                "loss_g": avg_g,
                "loss_mel": avg_mel,
            }, best_ckpt_path)
            print(f"    [Mejor Modelo Guardado!]: {best_ckpt_path.name} (Val_Mel = {best_val_loss:.4f})")

        # Guardar Checkpoint periódico
        if epoch % args.save_step == 0 or epoch == (start_epoch + args.epochs - 1):
            ckpt_path = args.output_dir / f"piper_vits_epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optim_g": optim_g.state_dict(),
                "optim_d": optim_d.state_dict(),
                "loss_g": avg_g,
                "loss_mel": avg_mel,
                "val_loss_mel": avg_val_mel,
            }, ckpt_path)
            print(f"    [Checkpoint guardado]: {ckpt_path.name}")

    print("\n¡Finetuning completado exitosamente!")

    # Exportación ONNX
    if args.export_onnx:
        # Cargar los mejores pesos para exportar la versión óptima sin sobreajuste
        best_ckpt = args.output_dir / "piper_vits_best.pt"
        if best_ckpt.exists():
            print(f"[ONNX] Cargando los mejores pesos validados desde {best_ckpt.name}...")
            ckpt_data = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt_data["model_state_dict"])
        
        target_path = args.export_target if args.export_target else (args.output_dir / "piper_vits_finetuned.onnx")
        export_model_to_onnx(model, target_path)


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
