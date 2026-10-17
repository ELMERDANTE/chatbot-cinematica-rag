# data/bateria/

Batería de evaluación **v4**: 400 preguntas de cinemática para la evaluación técnica del chatbot.

| Archivo | Contenido |
|---|---|
| `bateria_v4_publica.jsonl` | Las 400 preguntas: fuente, enunciado, alternativas, clave, solución de referencia, fichas del corpus esperadas, partición y estado de verificación. Para las 50 preguntas tomadas de libros con derechos de autor (Lumbreras y material escolar), `enunciado` y `alternativas` se publican vacíos y `fuente` da la referencia exacta. |
| `bateria_v4_metadatos.csv` | Una fila por pregunta, sin textos: subtema, componente, nivel, origen, partición, clave, valor de g, verificación, modificaciones, fichas esperadas y longitud del enunciado. |
| `registro_verificacion.csv` | Cada pregunta modificada respecto de su fuente (figura descrita, dato agregado, alternativa corregida) y cada pregunta anulada o excluida, con su motivo. |
| `ejemplo_esquema.csv` | Esquema preliminar (histórico). |

## Diseño

| | Teoría | Práctica básica | Práctica intermedia | Práctica avanzada | No respondible | Total |
|---|---|---|---|---|---|---|
| Cada subtema (MRU, MRUV, MVCL, MPCL, MCU) | 18 | 18 | 18 | 18 | 8 | 80 |
| **Total** | 90 | 90 | 90 | 90 | 40 | **400** |

**Partición** (estratificada, semilla fija, familias sin partir):

| Conjunto | n | Uso |
|---|---|---|
| Desarrollo | 200 | Ajuste de prompt, fragmentación y k |
| Validación | 80 | Calibrar τ y k; fijar umbrales |
| Evaluación final | 120 | Una sola ejecución con la configuración congelada |

## Origen y verificación

| Origen | Preguntas | Verificación |
|---|---|---|
| Academia Ágora (material propio de la academia) | 230 | Clave confirmada por el docente |
| Material escolar de MCU (adaptado al formato de Ágora) | 28 | Clave confirmada por el docente |
| *Física esencial* (Lumbreras) | 22 | Clave impresa en el libro o cotejada con el libro físico |
| Generadas por el investigador con asistencia de IA | 120 | Pendiente de revisión docente (81 de teoría y 39 no respondibles) |

El proceso completo de selección está en `data/eda/inventario_fuentes.csv` y en el EDA (`notebooks/01_eda.ipynb`): de **890** preguntas originales revisadas a **694** verificadas y **280** seleccionadas.

**Las preguntas de libros con derechos de autor** se pueden verificar con la referencia exacta (capítulo, sección y número). Las páginas escaneadas están disponibles para el jurado a solicitud del investigador.

**Versión congelada (15-10-2026; revisión de redacción 16-10-2026)**

| Archivo | SHA-256 |
|---|---|
| Batería completa (privada) | `e4da76fb741cda723f9685e467992a28187be96102418ae692df09b728cf5870` |
| `bateria_v4_publica.jsonl` | `78edfa6ebe3dda2d629083e6149f162cc4b4bef7ee9ba315a8eecc9a623f73f0` |

*Revisión del 16-10-2026:* en las 120 preguntas generadas, el campo `fuente` pasó de nombrar la herramienta a decir «Generada con asistencia de IA, revisión docente pendiente». No cambió ningún enunciado, alternativa, clave, solución ni partición. Huellas anteriores: completa `8b78283b…`, pública `1cdb1a41…`.

Cada ejecución de la evaluación registra este hash en su configuración.
