# src/

Código fuente del chatbot (**en desarrollo**).

**Scripts disponibles**

| Archivo | Función |
|---|---|
| `eda_utils.py` | Carga de datos públicos, estilo de gráficos y utilidades del EDA (`notebooks/01_eda.ipynb`). |
| `preparar_datos_publicos.py` | Genera los archivos públicos de `data/` a partir de los archivos de trabajo privados: corpus completo, batería sin textos protegidos, metadatos, registro de verificación e inventario. |

 Módulos previstos, en el orden del pipeline:

| Módulo | Función |
|---|---|
| `ingesta` | Inventario, limpieza y versionado del corpus |
| `chunking` | Segmentación por unidades semánticas |
| `embeddings` | Representación vectorial de fragmentos y consultas |
| `retrieval` | Búsqueda top-k por similitud coseno (y BM25 como referencia) |
| `seleccion` | Filtro por umbral τ y abstención de nivel 1 |
| `generacion` | Construcción del prompt y llamada al LLM |
| `validacion` | Verificación del formato JSON y de las fuentes citadas |
| `registro` | Escritura del log por consulta |
| `whatsapp` | Integración con la API oficial de WhatsApp Business Platform |
| `acceso` | Control de participantes (números cifrados con hash) |

El mismo núcleo (`ingesta` → `registro`) se usa en la evaluación offline y en el piloto; WhatsApp solo transporta mensajes.
