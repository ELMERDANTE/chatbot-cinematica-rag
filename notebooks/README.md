# notebooks/

| Notebook | Estado | Contenido |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | ✅ Ejecutado | Análisis exploratorio del corpus v5.1 y de la batería v4: composición, longitudes, origen, partición, posición de la clave, valor de g, cobertura del corpus, control de filtración y embudo de fuentes. |
| [`02_feature_engineering.ipynb`](02_feature_engineering.ipynb) | ✅ Ejecutado | Feature engineering del buscador (normalización, palabras vacías, raíces), variantes B0–V3 con un ejemplo real, Recall@k, validación cruzada de 5 particiones, holdout de validación con IC y p, gráfico Recall@k vs k y verificación de cero leakage. |
| [`03_variantes_datos.ipynb`](03_variantes_datos.ipynb) | ✅ Ejecutado | Variantes de datos con el buscador fijo V1 (sin enrutador): cómo cambia cada ficha, ejemplo real, holdout con IC y p, Recall@k vs k, validación cruzada, D2 construida solo con datos del corpus, sensibilidad al filtro anti-fuga, prueba adicional del enrutador retirado y verificación de cero leakage. |
| [`04_importancia_variables.ipynb`](04_importancia_variables.ipynb) | ✅ Ejecutado | Importancia de variables del buscador: ablación de los campos de la ficha (CV + holdout con IC y p) y modelo explicativo del fallo (regresión logística y bosque aleatorio, AUC con CV agrupada, importancia por permutación en validación). |
| [`05_optimizacion_hiperparametros.ipynb`](05_optimizacion_hiperparametros.ipynb) | ✅ Ejecutado | Avance del sprint de punta a punta hasta la optimización de hiperparámetros: datos y partición, preprocesamiento, modelo base, rejilla de 576 configuraciones con CV de 5 particiones, CV anidada, importancia de los hiperparámetros, comparación final en validación con IC y p, y verificación de cero leakage. |
| Diagnóstico inicial | Previsto | Subtemas con menor rendimiento. |
| Métricas de la evaluación técnica | Previsto | Recall@k, MRR, corrección, fidelidad y abstención. |
| Estudio educativo | Previsto | ANCOVA y tamaño del efecto. |
| Encuesta | Previsto | Alfa de Cronbach y descriptivos por dimensión. |

Ningún cuaderno contiene datos personales de estudiantes. Para reproducirlos:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_variantes_datos.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_importancia_variables.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_optimizacion_hiperparametros.ipynb
```
