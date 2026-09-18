# TTS Argentino (TTS-ar) — Microservicio de Síntesis Vocal para MedSim

Servicio de síntesis de voz (Text-to-Speech) en español rioplatense argentino de alta fidelidad, ultra bajo retardo y optimizado para producción. Diseñado e integrado nativamente como el motor vocal de simulación clínica para **MedSim**.

Impulsado por **Piper TTS (VITS)** con fonemización nativa `espeak-ng` (`es_AR`), modulación tímbrica **OpenVoice** y ejecución de inferencia acelerada con **ONNX Runtime**.

---

## 1. Características Principales

- **Fonemización Nativa Rioplatense (`espeak-ng es_AR`)**: Transcribe el texto directamente a fonemas IPA en español rioplatense, garantizando vocales puras, yeísmo correcto (`ll`/`y`), cadencia porteña y **cero acento extranjero o sesgo del inglés**.
- **Inferencia Ultrarrápida (<100 ms)**: Diseñado para interacción médica en tiempo real. Ejecuta sobre CPU con un consumo de recursos mínimo (~100 MB de RAM), sin requerir GPUs costosas.
- **Clonación y Modulación Tímbrica (OpenVoice)**: Ajusta el tono, edad, gravedad y resonancia vocal de los pacientes virtuales sobre la marcha.
- **Soporte de Emociones y Afecciones Clínicas**: Modulación de velocidad, entonación y textura para reflejar dolor físico (`pain`), queja/irritación (`annoyed`), angustia/ansiedad (`worried`) y estados de control (`neutral`).
- **Contrato REST para MedSim**: Compatible de forma directa con el cliente `TTSService` del backend de MedSim (`POST /audio/tts`).
- **Playground Web Integrado**: Interfaz visual interactiva en la raíz (`/`) para auditar voces y entonaciones directamente desde el navegador.

---

## 2. Catálogo de Arquetipos de Pacientes

El microservicio incluye un catálogo preconfigurado de perfiles demográficos para asignar a los distintos casos clínicos de MedSim:

| ID | Nombre | Género | Grupo Etario | Descripción / Caso Clínico Típico |
|:---:|---|---|---|---|
| **0** | **Daniela** | Femenino | Adulta (30-50 años) | Tono natural, articulación clara, ideal para consultas generales y guardia. |
| **1** | **Martín** | Masculino | Adulto (30-50 años) | Voz masculina estándar argentina, natural y expresiva. |
| **2** | **Marta** | Femenino | Adulta mayor (>65 años) | Modulación con menor velocidad y tono maduro para pacientes geriátricos. |
| **3** | **Roberto** | Masculino | Adulto mayor (>65 años) | Voz grave y pausada para pacientes añosos o con patologías crónicas. |
| **4** | **Sofía** | Femenino | Joven (18-25 años) | Tono ágil y juvenil para simulación de pacientes universitarios/jóvenes. |
| **5** | **Lucas** | Masculino | Joven (18-25 años) | Cadencia informal y dinámica para consultas juveniles o deportivas. |

### Estados Emocionales Soportados (`emotion`)
- `neutral`: Conversación basal, anamnesis estándar.
- `pain`: Voz quebrada o quejumbrosa con velocidad reducida, simulando dolor agudo o cólico.
- `annoyed`: Entonación más cortante, volumen proyectado y queja persistente.
- `worried`: Cadencia ansiosa, ligeramente acelerada o titubeante.
- `custom`: Permite control manual granular mediante `pitch_shift` y `style_strength`.

---

## 3. Integración con MedSim (Docker Compose)

En producción o entornos contenerizados, **TTS-ar** y **MedSim** conviven en la misma red puente de Docker (`medsim_medsim_net`).

```
+--------------------------------------------------------------+
| Docker Network: medsim_medsim_net                            |
|                                                              |
|   +-------------------+              +-------------------+   |
|   |    MedSim App     |              |    TTS-ar (tts)   |   |
|   | (Backend FastAPI) |              | (Motor Síntesis)  |   |
|   |                   |  POST /audio |                   |   |
|   | TTS_API_URL=      | -----------> | Expone: 8000      |   |
|   | http://tts:8000   |   /tts       | Alias DNS: tts    |   |
|   +-------------------+              +-------------------+   |
|             |                                  |             |
+-------------|----------------------------------|-------------+
              | (Host: 8000)                     | (Host: 8001)
     http://localhost:8000              http://localhost:8001
```

### Configuración en el `.env` de MedSim
Para que MedSim utilice este servicio en Docker, en `MedSim/.env` debe figurar:
```env
TTS_API_URL=http://tts:8000
TTS_API_KEY=local
TTS_MODEL_ID=piper-vits
TTS_LANGUAGE=es
TTS_VOICE_ID=1
TTS_SPEED=1.0
TTS_TEMPERATURE=0.5
```

---

## 4. Despliegue Rápido con Docker

### Paso 1: Configurar variables (opcional)
Si deseas modificar los puertos o la red, copia el archivo de ejemplo:
```bash
cp .env.example .env
```

### Paso 2: Construir y Levantar el Servicio
```bash
docker compose up -d --build
```
> Durante el build, el contenedor descargará y validará automáticamente los modelos ONNX y checkpoints necesarios mediante `voices/download_voices.py`.

### Paso 3: Verificar el Despliegue
Comprueba que el contenedor responda correctamente a la prueba de salud:
```bash
curl -f http://localhost:8001/audio/voices
```

Para seguir los logs en tiempo real:
```bash
docker compose logs -f tts
```

Para detener el servicio:
```bash
docker compose down
```

---

## 5. Especificación de la API REST

### `POST /audio/tts` (Contrato Principal)
Genera el audio sintetizado del paciente en formato binario WAV PCM (16-bit, 22050 Hz).

**Headers:**
```http
Content-Type: application/json
```

**Payload:**
```json
{
  "id": 1,
  "text": "Hola doctor, me duele muchísimo el pecho y me cuesta respirar hondo.",
  "speed": 1.0,
  "style_strength": 1.0,
  "emotion": "pain",
  "pitch_shift": 0.0
}
```

**Parámetros:**
- `id` *(int, requerido)*: ID del paciente/voz (0 a 5).
- `text` *(string, requerido)*: Texto a sintetizar (máx. 1000 caracteres).
- `speed` *(float, default: 1.0)*: Velocidad de habla (rango permitido: 0.4 a 2.5).
- `style_strength` *(float, default: 1.0)*: Variabilidad expresiva tímbrica (0.0 a 2.0).
- `emotion` *(string, default: "neutral")*: `neutral`, `annoyed`, `pain`, `worried` o `custom`.
- `pitch_shift` *(float, opcional)*: Desplazamiento de tono en semitonos (-12.0 a 12.0).

**Respuesta Exitosa (200 OK):**
- `Content-Type: audio/wav`
- `Content-Disposition: inline; filename="synthesis.wav"`

**Ejemplo con cURL:**
```bash
curl -X POST "http://localhost:8001/audio/tts" \
     -H "Content-Type: application/json" \
     -d '{"id": 0, "text": "Buen día doctor, vengo por el control.", "emotion": "neutral"}' \
     --output prueba.wav
```

---

### `GET /audio/tts` (Reproducción Directa)
Permite generar audio pasando parámetros por query string, ideal para pruebas directas en etiquetas `<audio>` HTML o reproductores.

```http
GET /audio/tts?id=1&text=Hola%20doctor&speed=1.0&emotion=pain
```

---

### `GET /audio/voices` (Catálogo de Voces)
Retorna la lista de voces y arquetipos disponibles en formato JSON.

```json
[
  {
    "id": 0,
    "gender": "female",
    "name": "Daniela",
    "description": "Femenina adulta argentina (base es_AR high)"
  },
  {
    "id": 1,
    "gender": "male",
    "name": "Martín",
    "description": "Masculino adulto argentino (clonado)"
  }
]
```

---

### `GET /` (Playground Web Interactivo)
Accediendo desde el navegador a `http://localhost:8001/`, dispones de un entorno gráfico para probar cualquier frase con los distintos pacientes, ajustar velocidades y escuchar el resultado al instante.

---

## 6. Instalación y Uso Local (Desarrollo sin Docker)

Si deseas ejecutar o modificar el motor localmente en tu sistema operativo:

### Requisitos Previos
- Python 3.10 o 3.11 instalado.
- `ffmpeg` y `espeak-ng` en el PATH del sistema:
  - **Windows**: `winget install eSpeak-NG.eSpeak-NG` y `winget install Gyan.FFmpeg`.
  - **Ubuntu/Debian**: `sudo apt-get install -y espeak-ng ffmpeg libsndfile1`.

### Pasos de Instalación
```powershell
# 1. Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1   # En Linux: source .venv/bin/activate

# 2. Instalar dependencias
pip install --upgrade pip
pip install -r api/requirements.txt

# 3. Descargar modelos y convertidores de voz
python voices/download_voices.py

# 4. Iniciar el servidor de desarrollo
uvicorn main:app --app-dir api --host 0.0.0.0 --port 8000 --reload
```

---

## 7. Pruebas Automatizadas

El proyecto cuenta con una batería de tests unitarios y de integración con `pytest`:

```powershell
# Ejecutar suite de pruebas
python -m pytest api/tests -v
```

Los tests validan:
1. El contrato de la API y códigos de estado HTTP (200, 400, 422).
2. La validez de los encabezados WAV retornados y frecuencias de muestreo.
3. El preprocesamiento de textos y sanitización contra caracteres inválidos.
4. La integridad de los artefactos ONNX y checkpoints de modelos.

---

## 8. Arquitectura del Código

```
TTS-ar/
├── api/
│   ├── config.py                 # Constantes, rutas de modelos y configuraciones
│   ├── main.py                   # Endpoints FastAPI y ciclo de vida
│   ├── openvoice/                # Módulos de clonación y adaptación de tono
│   ├── schemas.py                # Modelos de validación Pydantic
│   ├── templates/
│   │   └── index.html            # UI del Playground web
│   ├── tests/                    # Tests de integración y endpoints
│   └── tts_engine.py             # Orquestador del motor Piper + OpenVoice
├── voices/
│   ├── converter/                # Checkpoints de OpenVoice (checkpoint.pth)
│   ├── download_voices.py        # Script de descarga y verificación de artefactos
│   ├── piper_ar.onnx.json        # Configuración fonética y de audio es_AR
│   ├── piper_male.onnx.json      # Configuración de voz masculina
│   └── voices_catalog.csv        # Mapeo y metadatos de los arquetipos
├── finetuning/                   # Pipelines de entrenamiento y notebooks
├── Dockerfile                    # Receta de construcción de la imagen de producción
├── docker-compose.yml            # Orquestación y enlace con MedSim
├── .dockerignore                 # Exclusiones de build
└── README.md                     # Documentación de producción
```

---

## 9. Licencia y Créditos
- **Motor Base**: [Piper TTS](https://github.com/rhasspy/piper) por Michael Hansen (Licencia MIT / GPL).
- **Modelos Fonéticos**: Voces en español rioplatense entrenadas sobre datasets de dominio público y fonemizadas con [eSpeak NG](https://github.com/espeak-ng/espeak-ng).
- **Adaptación Tímbrica**: Basado en la arquitectura [OpenVoice](https://github.com/myshell-ai/OpenVoice) (MyShell AI).
