# data/bateria/

Banco controlado de 400 preguntas de cinemática para la evaluación técnica (no publicado).

## Partición (preliminar)

| Conjunto | n | Uso |
|---|---|---|
| Desarrollo | 200 | Ajuste de prompt, chunking y k |
| Validación | 80 | Calibrar τ y k; aplicar la regla de selección; fijar umbrales |
| Evaluación final | 120 | Una sola ejecución con la configuración congelada |

Estratificación por subtema y por tipo de pregunta: conceptual, numérica, error frecuente, datos incompletos, ambigua y fuera de dominio (10–15 % no respondibles).

## Esquema

`ejemplo_esquema.csv` muestra las columnas con **dos filas ficticias** de ejemplo.
