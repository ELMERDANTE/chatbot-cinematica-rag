# experiments/

Ejecuciones registradas de la evaluación técnica.

| Carpeta | Contenido |
|---|---|
| [`feature_engineering/`](feature_engineering/README.md) | Feature engineering y experimentos A/B del componente de recuperación: línea base BM25 y 3 variantes, validación cruzada + holdout, gráfico Recall@k vs k y verificación de cero leakage |

Cada ejecución se guarda en `runs/<run_id>/` con:

- copia de la configuración usada y su hash (`config.yaml`);
- registro JSONL de todas las consultas (`log.jsonl`, según `evaluation/log_schema.json`);
- resumen de métricas (`metricas.json`).

Los runs de `feature_engineering/runs/` se publican porque no contienen datos personales ni enunciados protegidos (las preguntas de libros se registran solo con su identificador). La carpeta `experiments/runs/` de nivel superior queda reservada para ejecuciones con el LLM y no se publica.

El conjunto de evaluación final se ejecuta **una sola vez** por configuración finalista.
