# Guía de Uso y Despliegue en Windows — TTS Argentino (Piper TTS / VITS)

Esta guía detalla paso a paso cómo activar el entorno, iniciar el servicio de **FastAPI**, acceder al **Playground Web interactivo** y documentación **Swagger**, y consumir los endpoints de síntesis de audio en español argentino.

---

## 1. Prerrequisitos en Windows

* **Sistema Operativo**: Windows 10 / 11 (64-bit).
* **Python**: 3.10, 3.11 o 3.12 instalado.
* **FFmpeg**: Opcional, pero recomendado en el `PATH` para manipulación de audio.

---

## 2. Activar el Entorno Virtual

Abrí una terminal de **PowerShell** en la raíz del proyecto (`C:\workspace\TTS-ar`) y ejecutá:

```powershell
# Permitir ejecución de scripts en la sesión actual (si es necesario)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activar el entorno virtual
.venv\Scripts\Activate.ps1
```

*(Cuando el entorno está activo verás el prefijo `(.venv)` al inicio de la línea de comandos).*

---

## 3. Levantar el Servicio FastAPI

Con el entorno virtual activado, ejecutá desde la raíz del proyecto:

```powershell
uvicorn main:app --app-dir api --host 0.0.0.0 --port 8000
```

Al iniciar, el motor cargará el modelo Piper ONNX y mostrará:
```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

## 4. Interfaces Web Disponibles

| Interfaz | URL | Descripción |
|---|---|---|
| **Playground Web (Reproductor)** | [http://localhost:8000/](http://localhost:8000/) | Interfaz visual interactiva con selector de voces, sliders de velocidad/estilo y reproductor de audio HTML5 integrado |
| **Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | Documentación OpenAPI interactiva para probar endpoints directamente |
| **ReDoc UI** | [http://localhost:8000/redoc](http://localhost:8000/redoc) | Documentación técnica alternativa de la API |

---

## 5. Endpoints de la API

### A. `GET /` — Playground Web con Reproductor Integrado
Abrí **[http://localhost:8000](http://localhost:8000)** en tu navegador (Chrome, Edge, Firefox) para escribir cualquier texto y escuchar la síntesis en tiempo real.

---

### B. `GET /audio/voices` — Catálogo de Voces
Devuelve la lista de voces disponibles para que un LLM o cliente elija el `id`.

* **URL**: `http://localhost:8000/audio/voices`
* **Método**: `GET`
* **Respuesta de ejemplo (JSON)**:
```json
[
  { "id": 0, "gender": "female", "name": "daniela" }
]
```

---

### C. `POST /audio/tts` — Síntesis de Audio (Contrato Principal)
Contrato estándar utilizado para integraciones backend / pipelines STT → LLM → TTS.

* **URL**: `http://localhost:8000/audio/tts`
* **Método**: `POST`
* **Headers**: `Content-Type: application/json`
* **Cuerpo de la Petición (JSON)**:
```json
{
  "id": 0,
  "text": "Che, ¿cómo andás? Te confirmo que la reunión es a las cinco de la tarde.",
  "speed": 1.0,
  "style_strength": 0.8
}
```
* **Parámetros**:
  - `id` *(int, requerido)*: Identificador de la voz.
  - `text` *(string, requerido)*: Texto a sintetizar.
  - `speed` *(float, opcional, default 1.0)*: Velocidad de habla (rango `0.5` a `2.0`).
  - `style_strength` *(float, opcional, default 1.0)*: Variabilidad de estilo/ruido fonético (`0.0` a `1.0`).
* **Respuesta**: Archivo de audio binario `audio/wav` (StreamingResponse).

---

## 6. Pipeline de Finetuning y Datos con Piper

* **Preparar el dataset en formato Piper**:
  ```powershell
  python finetuning/prepare_dataset_piper.py
  ```
* **Entrenar / Finetunear modelo VITS**:
  ```powershell
  python finetuning/train_piper.py
  ```
* **Ejecutar tests unitarios**:
  ```powershell
  python -m pytest api/tests
  ```
