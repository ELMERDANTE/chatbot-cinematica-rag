# experiments/

Ejecuciones registradas de la evaluación técnica. Cada ejecución se guarda en `runs/<run_id>/` (no publicado) con:

- copia de la configuración usada y su hash;
- registro JSONL de todas las consultas;
- resumen de métricas.

Convención de nombre: `AAAAMMDD-<conjunto>-<config>`, por ejemplo `20270115-validacion-v0.3`.

El conjunto de evaluación final se ejecuta **una sola vez** por configuración finalista.
