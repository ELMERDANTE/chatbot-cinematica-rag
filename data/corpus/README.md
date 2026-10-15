# data/corpus/

Base de conocimiento del sistema RAG — **versión v5.1 (136 fichas)**.

| Archivo | Contenido |
|---|---|
| `processed/corpus_v5_1.jsonl` | Una ficha por línea: `id`, `tipo` (teoría / procedimiento / ejemplo), `nivel`, `titulo`, contenido, `ecuaciones`, `condiciones`, `errores_frecuentes`, `fuente` (libro, capítulo y páginas) e `index_text` (texto que se indexa). |
| `processed/corpus_v5_1_lectura.md` | La misma información en formato de lectura. |
| `manifest.json` | Inventario con el hash SHA-256 de cada ficha y del archivo completo. |

| Subtema | Teoría | Procedimiento | Ejemplo | Total |
|---|---|---|---|---|
| MRU | 15 | 8 | 7 | 30 |
| MRUV | 16 | 7 | 8 | 31 |
| MVCL | 14 | 4 | 7 | 25 |
| MPCL | 15 | 6 | 7 | 28 |
| MCU | 12 | 4 | 6 | 22 |

## Origen y derechos

- **Contenido propio:** las fichas son **síntesis propias**, redactadas a partir de *Física esencial* (Lumbreras Editores) y *Física: principios con aplicaciones* (Giancoli, 6.ª ed.). No reproducen texto de los libros; los citan con capítulo y páginas en el campo `fuente`.
- **Ejemplos resueltos:** son problemas originales.
- **Fichas nuevas en la v5.1:** se agregaron 3 fichas para cubrir vacíos detectados en el material de la academia.

## Regla de segmentación

Una ficha = una **unidad semántica completa** (55–146 palabras): una definición, una fórmula con el significado de sus variables, un procedimiento o un ejemplo con su solución. Nunca se separa una fórmula de su significado ni un problema de su solución. Cada ficha se indexa como un fragmento, sin volver a partirla.

## Validación

- **Estado:** todas las fichas tienen `estado_validacion = pendiente`.
- **Validación prevista:** el corpus se valida por juicio de expertos independientes del investigador (exactitud, claridad, pertinencia y nivel) antes de congelarse.
