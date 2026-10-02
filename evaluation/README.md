# evaluation/

Protocolo de evaluación técnica del sistema.

## Comparaciones

- **Línea base:** mismo LLM y misma plantilla, sin RAG (contrasta HE1 y HE2).
- **Modelos:** al menos dos LLM de distinto costo, bajo el mismo protocolo.
- **Recuperación:** búsqueda densa frente a BM25.

## Métricas

| Dimensión | Métricas |
|---|---|
| Recuperación | Recall@k (principal), Precision@k, MRR, tasa de recuperación fallida |
| Generación | Corrección (principal), exactitud conceptual, corrección matemática, fidelidad, afirmaciones no sustentadas, calidad socrática, abstención correcta/indebida |
| Operación | Latencia p50/p95, tokens, costo por consulta, disponibilidad |

## Regla de selección de configuración

Entre las configuraciones que cumplen en validación las restricciones de fidelidad, calidad pedagógica y latencia, se elige la de **menor costo por consulta** cuya corrección no sea inferior a la mejor en más de δ puntos porcentuales (δ se fija antes de ver los resultados de validación).

## Pruebas estadísticas

McNemar (corrección), Wilcoxon de rangos con signo (rúbrica) e intervalos bootstrap, sobre las mismas preguntas.

## Archivos

- `log_schema.json`: campos que se registran en cada consulta.
