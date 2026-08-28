# TTS Argentino — Piper TTS (VITS) + FastAPI

Servicio de síntesis de voz (TTS) en español argentino de alta fidelidad, ultrarrápido y liviano, impulsado por **Piper TTS (VITS)** con fonemización nativa `espeak-ng` (`es_AR`) y servido mediante una API REST en **FastAPI** con **ONNX Runtime**.

---

## 1. Por qué Piper TTS (VITS)

- **Fonemización Nativa Real (`espeak-ng es_AR`)**: Traduce el texto directamente a fonemas IPA en español rioplatense, garantizando vocales puras, consonantes correctas (`rr`, `r`, `ll`/`y` con yeísmo) y **cero acento extranjero o sesgo del inglés**.
- **Ultrarrápido y Liviano**: Síntesis en menos de 100 ms por frase en CPU/GPU gracias a ONNX Runtime. El modelo pesa solo ~100 MB.
- **Contrato API Inviolable**: Mantiene intacto el contrato `POST /audio/tts` para clientes externos y LLMs.

---

## 2. Arquitectura del Proyecto

```
tts-ar-project/
├── finetuning/
│   ├── prepare_dataset_piper.py   # procesa dataset y genera metadata.csv
│   ├── train_piper.py             # script de entrenamiento VITS
│   └── finetune_piper_ar.ipynb    # notebook interactivo (Colab/Jupyter)
├── voices/
│   ├── piper_ar.onnx              # modelo VITS en formato ONNX
│   ├── piper_ar.onnx.json         # configuración fonética y de audio
│   └── voices_catalog.csv         # catálogo de voces disponibles
├── api/
│   ├── config.py                  # rutas y constantes del motor Piper
│   ├── schemas.py                 # modelos Pydantic de request/response
│   ├── tts_engine.py              # wrapper del motor Piper TTS (ONNX)
│   ├── main.py                    # app FastAPI + Playground interactivo
│   └── requirements.txt
└── README.md
```

---

## 3. Parámetros del Contrato `POST /audio/tts`

- **`id`** (int): Identificador de la voz a usar (consultable en `GET /audio/voices`).
- **`text`** (str): Texto en español a sintetizar.
- **`speed`** (float, 0.5–2.0): Velocidad de habla (mapea a `length_scale = 1.0 / speed`).
- **`style_strength`** (float 0.0–1.0): Variabilidad tímbrica y expresividad de VITS (mapea a `noise_scale` y `noise_w`).

---

## 4. Instalación y Uso Local en Windows

### Paso 1: Activar entorno virtual
```powershell
.venv\Scripts\Activate.ps1
```

### Paso 2: Instalar dependencias
```powershell
pip install -r api/requirements.txt
```

### Paso 3: Levantar la API con Uvicorn
```powershell
uvicorn main:app --app-dir api --host 0.0.0.0 --port 8000
```

* **Playground Web interactivo**: [http://localhost:8000/](http://localhost:8000/)
* **Documentación Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Paso 4: Ejecutar tests unitarios
```powershell
python -m pytest api/tests
```
