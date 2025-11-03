# Verificador de Identidad por Imagen

Sistema de verificación facial que identifica al usuario establecido utilizando un problema de clasificación binaria. El sistema utiliza embeddings faciales preentrenados y un clasificador para determinar si una imagen contiene el rostro del usuario ("yo") o no ("no-yo").

## Tecnologías Utilizadas

- **Python 3.10+**
- **FastAPI**: Framework para la API REST
- **facenet-pytorch**: Para detección de rostros (MTCNN) y extracción de embeddings (InceptionResnetV1)
- **scikit-learn**: Para el clasificador (LogisticRegression) y escalado
- **Pillow**: Para procesamiento de imágenes
- **numpy**: Para operaciones numéricas
- **joblib**: Para guardar/cargar modelos
- **uvicorn**: Servidor ASGI para FastAPI
- **python-multipart**: Para manejo de archivos en FastAPI

## Instalación

### Requisitos Previos

- Python 3.10 o superior
- uv (gestor de paquetes)

### Instalación de Dependencias

Para desarrollo (incluye dependencias de dev):
```bash
uv sync --group dev
```

Para producción:
```bash
uv sync --group default
```

Para añadir nuevas dependencias:
```bash
uv add [dependencia] --group dev
```

## Configuración

Crear un archivo `.env` en la raíz del proyecto con las siguientes variables:

```env
MODEL_PATH=models/model.joblib
SCALER_PATH=models/scaler.joblib
THRESHOLD=0.75
PORT=33001
MAX_MB=10
```

## Ejecución

### Producción
```bash
uv run uvicorn app.main:app --reload --port 33001
```


## Cómo Funciona el Sistema

El sistema implementa un verificador binario de identidad facial que responde a la pregunta "¿soy yo?" para imágenes enviadas.

### Pipeline de Procesamiento

1. **Recepción de Imagen**: La API recibe una imagen vía POST al endpoint `/verify`
2. **Validación**: Se valida el tipo de archivo (solo JPEG/PNG) y tamaño máximo
3. **Detección de Rostro**: Utiliza MTCNN para detectar y recortar el rostro principal
4. **Extracción de Embedding**: El rostro recortado pasa por InceptionResnetV1 para obtener un vector de 512 dimensiones
5. **Escalado**: El embedding se escala usando StandardScaler entrenado
6. **Clasificación**: Un modelo LogisticRegression predice la probabilidad de ser "yo"
7. **Umbral**: Si la probabilidad >= umbral (0.75 por defecto), se considera "yo"; sino "no-yo" o "desconocido"
8. **Respuesta**: Devuelve JSON con decisión, score, umbral y tiempo de procesamiento

### Arquitectura

- **app.py**: Configuración principal de FastAPI
- **main.py**: Punto de entrada de la aplicación
- **router/prediction_router.py**: Endpoint `/verify`
- **services/classifier_service.py**: Servicio de clasificación
- **services/validator_service.py**: Servicio de validación
- **model/face_model.py**: Modelo de rostro (singleton) con MTCNN y clasificador
- **utils/**: Utilidades para logging y construcción de respuestas
- **exceptions/**: Excepciones personalizadas

## API

### Endpoint `/verify`

**Método**: POST  
**Content-Type**: multipart/form-data  
**Parámetro**: `file` (imagen JPEG o PNG)

#### Ejemplo de Solicitud
```bash
curl -F "image=@selfie.jpg" http://localhost:33001/verify
```

#### Respuesta Exitosa (200)
```json
{
    "success": true,
    "data": {
        "model_version": "me-verifier-v1",
        "is_me": true,
        "score": 1.0,
        "threshold": 0.7,
        "timing_ms": 1307.69
    },
    "metadata": {
        "request_id": "7ea3db19-469a-4d9f-bb80-e0732a791f09",
        "timestamp": "2025-11-02T22:11:49.942865-03:00"
    }
}
```

#### Respuesta de Error (400/422/500)
```json
{
  "error": "Descripción del error",
  "code": 400
}
```

## Proceso de Entrenamiento

El sistema requiere entrenamiento previo con fotos del usuario y fotos de "no-yo".

### 1. Recolección de Datos

- **"Yo"**: 40-50 fotos propias con variaciones de luz, ángulos, expresiones
- **"No-yo"**: 200-400 fotos de otras personas

Guardar en:
- `data/raw/me/photos/` (fotos propias)
- `data/raw/not_me/photos/` (fotos de otros)

### 2. Preprocesamiento

```bash
uv run .\scripts\crop_faces.py
```

- Detecta rostros usando MTCNN
- Recorta y normaliza a 160x160 píxeles
- Guarda recortes en `data/cropped/me/` y `data/cropped/not_me/`

### 3. Extracción de Embeddings

```bash
uv run .\scripts\embeddings.py
```

- Pasa cada recorte por InceptionResnetV1
- Genera embeddings de 512 dimensiones
- Guarda en `data/embeddings/embeddings.npy` y `data/embeddings/labels.npy`

### 4. Entrenamiento del Modelo

```bash
uv run .\scripts\train.py
```

- Entrena LogisticRegression en los embeddings
- Aplica StandardScaler
- Guarda modelo en `models/model.joblib` y scaler en `models/scaler.joblib`
- Genera métricas en `models/metrics.json`

### 5. Evaluación

```bash
uv run .\scripts\evaluate.py
```

- Calcula matriz de confusión, AUC, curva ROC
- Busca umbral óptimo
- Guarda reportes en `reports/`

## Pruebas

### Ejecutar Pruebas Unitarias
```bash
uv run pytest
```

### Pruebas con Postman/curl

- Enviar imagen propia: debería devolver `is_me: true`
- Enviar imagen ajena: debería devolver `is_me: false`
- Enviar archivo inválido: error 400
- Enviar imagen sin rostro: error 422

## Logging y Monitoreo

- Logs en formato JSON con latencia, resultado, tamaño de imagen
- Archivo de log de inferencias: `data/inference_log.csv`
- Métricas del modelo: `models/metrics.json`

## Consideraciones Éticas y de Privacidad

- Las imágenes se procesan localmente, no se almacenan
- Solo se extraen embeddings, no se guardan las imágenes originales
- El modelo es personal, no generalizable a otros usuarios
- Implementar controles de acceso en producción

## Mejoras Futuras

- Soporte para múltiples rostros por imagen
- Modelo más robusto con data augmentation
- API de batch processing
- Integración con bases de datos para usuarios múltiples
- Optimización de latencia con GPU
- Autenticación JWT para endpoints