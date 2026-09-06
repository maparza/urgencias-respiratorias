# Predicción de Urgencias Respiratorias en Chile

Proyecto de machine learning que estima si la **semana siguiente** presentará alta demanda de atenciones de urgencia por causas respiratorias en una región de Chile. El modelo se publica mediante una API construida con FastAPI, empaquetada con Docker y desplegada en Google Cloud Run.

## Demo

- **API pública:** [https://urgencias-respiratorias-136562563739.southamerica-west1.run.app](https://urgencias-respiratorias-136562563739.southamerica-west1.run.app)
- **Documentación Swagger:** [https://urgencias-respiratorias-136562563739.southamerica-west1.run.app/docs](https://urgencias-respiratorias-136562563739.southamerica-west1.run.app/docs)

> La disponibilidad de la demo depende de que el servicio de Cloud Run permanezca activo.

## Objetivo

Construir un clasificador interpretable que, a partir de la región, la semana estadística y el número de atenciones respiratorias observado durante las tres semanas anteriores, anticipe si la demanda de la semana siguiente alcanzará un nivel alto.

El proyecto cubre el flujo completo:

```text
Datos abiertos en Parquet
        ↓
Exploración y preparación
        ↓
Ingeniería de variables temporales
        ↓
Entrenamiento y evaluación temporal
        ↓
Exportación del Pipeline a modelo.pkl
        ↓
API FastAPI → Docker → Google Cloud Run
```

## Fuente de datos

Los datos provienen del conjunto **Atenciones de urgencias de causas respiratorias por semana epidemiológica**, publicado por el Ministerio de Salud de Chile en el Portal de Datos Abiertos:

- [Ficha oficial del conjunto de datos en datos.gob.cl](https://datos.gob.cl/es/dataset/atenciones-de-urgencia-causas-respiratorias)

El conjunto contiene totales de consultas y hospitalizaciones por causas respiratorias provenientes del Sistema de Atención Diaria de Urgencias (SADU). Para este proyecto se utiliza el recurso en formato **Parquet**, almacenado localmente como:

```text
data/at_urg_respiratorio_semanal.parquet
```

Parquet permite conservar los tipos de datos y realizar lecturas columnares eficientes, con menor tamaño y tiempos de carga que formatos de texto como CSV.

## Preparación de los datos

El análisis y el entrenamiento se realizan en `exploracion.ipynb`.

### 1. Selección de la causa

Se filtran los registros con:

```python
OrdenCausa == 3
```

Este valor corresponde a **`TOTAL CAUSA SISTEMA RESPIRATORIO`**.

### 2. Agregación semanal y regional

Después de eliminar regiones nulas y normalizar nombres, se suma `NumTotal` por:

```text
Anio + SemanaEstadistica + RegionGlosa
```

El resultado contiene una observación por año, semana epidemiológica y región.

### 3. Variables predictoras

Los datos se ordenan cronológicamente dentro de cada región y se crean las siguientes variables:

| Variable | Descripción |
|---|---|
| `SemanaEstadistica` | Semana epidemiológica del año. |
| `RegionGlosa` | Nombre de la región. |
| `lag_1` | Total de atenciones de la semana anterior. |
| `lag_2` | Total de atenciones de dos semanas atrás. |
| `lag_3` | Total de atenciones de tres semanas atrás. |
| `promedio_3_semanas` | Promedio de `lag_1`, `lag_2` y `lag_3`. |

### 4. Variable objetivo

Para cada región se calcula el percentil 75 histórico de `NumTotal`. La variable binaria `AltaDemanda` indica si el total observado en la **semana siguiente** es igual o superior a ese umbral regional:

```text
AltaDemanda = 1  si NumTotal_siguiente >= percentil 75 de la región
AltaDemanda = 0  en caso contrario
```

Las primeras observaciones sin suficientes rezagos y las últimas sin una semana siguiente conocida se excluyen del entrenamiento.

## Modelo

El estimador final es un `Pipeline` de scikit-learn. Esto garantiza que las mismas transformaciones aplicadas durante el entrenamiento se reutilicen al realizar inferencias:

- Variables numéricas: `StandardScaler`.
- Variable categórica `RegionGlosa`: `OneHotEncoder(handle_unknown="ignore")`.
- Clasificador: `LogisticRegression(max_iter=1000, class_weight="balanced")`.

El uso de `class_weight="balanced"` compensa el desbalance entre semanas de demanda normal y alta demanda.

### Separación temporal

Para respetar el orden cronológico y evitar una partición aleatoria poco realista:

| Conjunto | Periodo |
|---|---|
| Entrenamiento | 2014–2024 |
| Prueba | 2025–2026 |

### Resultados

Métricas aproximadas obtenidas sobre el conjunto temporal de prueba:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.741 |
| Recall — clase 1 (`AltaDemanda`) | 0.84 |
| F1-score — clase 1 (`AltaDemanda`) | 0.70 |
| ROC-AUC | 0.816 |

Matriz de confusión obtenida en la evaluación:

```text
[[626, 283],
 [ 81, 418]]
```

El recall de 0.84 indica que el modelo identifica aproximadamente el 84 % de las semanas de alta demanda presentes en el conjunto de prueba. Las cifras pueden variar al actualizar los datos o volver a entrenar el modelo.

## Artefacto entrenado

El `Pipeline` completo se serializa con `pickle` en:

```text
modelo.pkl
```

El archivo incluye el preprocesamiento y el clasificador. Conviene mantener compatibles las versiones de Python y scikit-learn utilizadas en el entrenamiento y en producción.

## API

La API se implementa en `app.py` con FastAPI.

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Comprueba que el servicio está disponible y enlaza la documentación. |
| `POST` | `/predict` | Predice si la semana siguiente tendrá alta demanda. |
| `GET` | `/docs` | Interfaz Swagger interactiva. |

### Respuesta del endpoint raíz

```json
{
  "mensaje": "API de Predicción de Urgencias Respiratorias",
  "documentacion": "/docs"
}
```

### Ejemplo de predicción

Solicitud:

```json
{
  "SemanaEstadistica": 23,
  "RegionGlosa": "Región Metropolitana de Santiago",
  "lag_1": 32000,
  "lag_2": 30000,
  "lag_3": 28000
}
```

Respuesta:

```json
{
  "alta_demanda": false,
  "probabilidad": 0.19332109814386017
}
```

La API calcula `promedio_3_semanas` a partir de los tres rezagos recibidos. `probabilidad` corresponde a la probabilidad estimada de la clase positiva. El resultado depende del modelo almacenado en `modelo.pkl`.

Ejemplo con `curl`:

```bash
curl -X POST \
  "https://urgencias-respiratorias-136562563739.southamerica-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "SemanaEstadistica": 23,
    "RegionGlosa": "Región Metropolitana de Santiago",
    "lag_1": 32000,
    "lag_2": 30000,
    "lag_3": 28000
  }'
```

## Estructura del repositorio

```text
urgencias-respiratorias/
├── app.py                              # API FastAPI
├── exploracion.ipynb                  # EDA, preparación y entrenamiento
├── modelo.pkl                         # Pipeline entrenado
├── requirements.txt                   # Dependencias de Python
├── Dockerfile                         # Imagen del servicio
├── .dockerignore                      # Exclusiones del contexto de Docker
├── .gitignore                         # Archivos excluidos de Git
├── README.md
└── data/
    └── at_urg_respiratorio_semanal.parquet  # Datos locales, no versionados
```

## Ejecución local

### Requisitos

- Python 3.12 recomendado.
- `pip`.
- El archivo `modelo.pkl` en la raíz del proyecto.

### Crear el entorno e instalar dependencias

Desde la raíz del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Windows, la activación del entorno virtual se realiza con:

```powershell
.venv\Scripts\activate
```

### Iniciar la API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Luego abre:

- API: [http://127.0.0.1:8080](http://127.0.0.1:8080)
- Swagger: [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)

## Ejecución con Docker

Construir la imagen:

```bash
docker build -t urgencias-respiratorias .
```

Ejecutar el contenedor:

```bash
docker run --rm -p 8080:8080 urgencias-respiratorias
```

La documentación quedará disponible en [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs).

## Despliegue en Google Cloud Run

El servicio está desplegado en la región `southamerica-west1` (Santiago) dentro del proyecto de Google Cloud `urgencias-respiratorias-epa`.

Con Google Cloud CLI instalado, una cuenta autenticada y facturación habilitada:

```bash
gcloud config set project urgencias-respiratorias-epa

gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com

gcloud run deploy urgencias-respiratorias \
  --source . \
  --region southamerica-west1 \
  --allow-unauthenticated
```

Cloud Run construye el contenedor desde el `Dockerfile`, publica una nueva revisión y dirige el tráfico al servicio.

## Datos y control de versiones

La carpeta `data/` se ignora mediante `.gitignore` y `.dockerignore` porque los archivos de origen pueden ser grandes y ya se encuentran disponibles en el portal oficial. Para reproducir el entrenamiento:

1. Descarga el recurso Parquet desde datos.gob.cl.
2. Guárdalo como `data/at_urg_respiratorio_semanal.parquet`.
3. Ejecuta las celdas de `exploracion.ipynb` en orden.
4. Verifica las métricas y vuelve a exportar `modelo.pkl`.

No subas datos sensibles, credenciales de Google Cloud ni archivos de configuración con secretos al repositorio.

## Limitaciones

- La predicción depende de la calidad, cobertura y oportunidad del reporte de atenciones de urgencia.
- El percentil 75 es una definición relativa de alta demanda: representa un nivel alto respecto del historial de cada región, no un umbral de capacidad hospitalaria.
- El modelo utiliza estacionalidad, región y rezagos recientes, pero no incorpora variables como clima, contaminación, circulación viral, población o disponibilidad de camas.
- Las observaciones de 2025–2026 pueden estar sujetas a actualizaciones en la fuente.
- Un cambio en los patrones de demanda puede degradar el desempeño y requerir reentrenamiento.
- `pickle` no debe cargarse desde fuentes no confiables y puede presentar incompatibilidades entre versiones de scikit-learn.
- La salida es una estimación estadística y no reemplaza criterios clínicos ni decisiones de gestión sanitaria.

## Posibles mejoras

- Calcular y guardar los umbrales regionales usando exclusivamente el periodo de entrenamiento en cada evaluación.
- Incorporar validación temporal progresiva (*walk-forward validation*).
- Ajustar el umbral de clasificación según el costo de falsos negativos y falsos positivos.
- Evaluar calibración de probabilidades y curvas precision-recall.
- Comparar con modelos no lineales y métodos de *gradient boosting*.
- Agregar variables meteorológicas, ambientales, epidemiológicas y demográficas.
- Automatizar la actualización de datos, el reentrenamiento y el monitoreo de deriva.
- Incorporar pruebas unitarias, validación del contrato de la API y CI/CD.

---

Este proyecto tiene fines educativos y demuestra un flujo reproducible de datos abiertos, machine learning, API y despliegue en la nube.
