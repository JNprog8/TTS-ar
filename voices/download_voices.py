"""Descarga y verifica automáticamente los modelos de voz (Piper ONNX) y el convertidor tímbrico (OpenVoice).
Útil para nuevos desarrolladores, despliegues CI/CD y entornos Docker.
"""

from pathlib import Path
import shutil
import urllib.request

VOICES_DIR = Path(__file__).resolve().parent
CONVERTER_DIR = VOICES_DIR / "converter"

MODELS = [
    {
        "name": "Voz Femenina (Daniela - es_AR high)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "target": VOICES_DIR / "piper_ar.onnx",
    },
    {
        "name": "Config Femenina (Daniela - es_AR JSON)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
        "target": VOICES_DIR / "piper_ar.onnx.json",
    },
    {
        "name": "Voz Masculina (Claude - es_MX high / Latino)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx",
        "target": VOICES_DIR / "piper_male.onnx",
    },
    {
        "name": "Config Masculina (Claude - es_MX JSON)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
        "target": VOICES_DIR / "piper_male.onnx.json",
    },
    {
        "name": "OpenVoice Converter Config",
        "url": "https://huggingface.co/myshell-ai/OpenVoice/raw/main/checkpoints/converter/config.json",
        "target": CONVERTER_DIR / "config.json",
    },
    {
        "name": "OpenVoice Converter Checkpoint",
        "url": "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints/converter/checkpoint.pth",
        "target": CONVERTER_DIR / "checkpoint.pth",
    },
]


def ensure_voices():
    """Verifica y descarga los artefactos necesarios."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    CONVERTER_DIR.mkdir(parents=True, exist_ok=True)

    for m in MODELS:
        target = m["target"]
        if not target.exists() or target.stat().st_size == 0:
            print(f"Descargando {m['name']}...")
            req = urllib.request.Request(m["url"], headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(target, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"  [OK] Guardado en {target.name} ({target.stat().st_size} bytes)")
        else:
            print(f"  [OK] {target.name} ya existe.")


if __name__ == "__main__":
    print("=" * 60)
    print(" Verificando modelos y convertidores de voz...")
    print("=" * 60)
    ensure_voices()
    print("\n¡Todos los modelos de voz están listos para usar!")
