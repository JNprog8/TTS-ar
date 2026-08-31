"""Descarga automáticamente los modelos ONNX y configuraciones de Piper TTS si no están presentes.
Útil para nuevos colaboradores o entornos de despliegue (CI/CD / Docker).
"""
from pathlib import Path
import shutil
import urllib.request

VOICES_DIR = Path(__file__).resolve().parent

MODELS = [
    {
        "name": "Voz Femenina (Daniela - es_AR high)",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx",
        "onnx_target": VOICES_DIR / "piper_ar.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_AR/daniela/high/es_AR-daniela-high.onnx.json",
        "json_target": VOICES_DIR / "piper_ar.onnx.json",
    },
    {
        "name": "Voz Masculina (Claude - es_MX high / Latino)",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx",
        "onnx_target": VOICES_DIR / "piper_male.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_MX/claude/high/es_MX-claude-high.onnx.json",
        "json_target": VOICES_DIR / "piper_male.onnx.json",
    },
]


def ensure_voices():
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        # Descargar ONNX
        if not m["onnx_target"].exists():
            print(f"Descargando {m['name']} ONNX...")
            with urllib.request.urlopen(m["onnx_url"]) as response, open(m["onnx_target"], "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"  [OK] Guardado en {m['onnx_target'].name} ({m['onnx_target'].stat().st_size} bytes)")
        else:
            print(f"  [OK] {m['onnx_target'].name} ya existe.")

        # Descargar JSON
        if not m["json_target"].exists():
            print(f"Descargando {m['name']} Config JSON...")
            with urllib.request.urlopen(m["json_url"]) as response, open(m["json_target"], "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"  [OK] Guardado en {m['json_target'].name}")
        else:
            print(f"  [OK] {m['json_target'].name} ya existe.")


if __name__ == "__main__":
    print("=" * 60)
    print("Verificando / Descargando modelos de voz Piper TTS...")
    print("=" * 60)
    ensure_voices()
    print("\n¡Todos los modelos de voz están listos para usar!")
