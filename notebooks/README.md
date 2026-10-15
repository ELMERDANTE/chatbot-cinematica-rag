# notebooks/

| Notebook | Estado | Contenido |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | ✅ Ejecutado | Análisis exploratorio del corpus v5.1 y de la batería v4: composición, longitudes, origen, partición, posición de la clave, valor de g, cobertura del corpus, control de filtración y embudo de fuentes. |
| Diagnóstico inicial | Previsto | Subtemas con menor rendimiento. |
| Métricas de la evaluación técnica | Previsto | Recall@k, MRR, corrección, fidelidad y abstención. |
| Estudio educativo | Previsto | ANCOVA y tamaño del efecto. |
| Encuesta | Previsto | Alfa de Cronbach y descriptivos por dimensión. |

Ningún cuaderno contiene datos personales de estudiantes. Para reproducir el EDA:

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
```
