# data/

| Carpeta | Contenido | ¿Se publica? |
|---|---|---|
| `corpus/processed/` | Corpus v5.1: 136 fichas de teoría, procedimientos y ejemplos resueltos (síntesis propias). | **Sí**, completo. |
| `corpus/raw/` | Escaneos de los libros de referencia (Lumbreras, Giancoli). | No: derechos de autor. |
| `bateria/` | Batería de evaluación v4 (400 preguntas): versión pública, metadatos y registro de verificación. | **Sí**, con una excepción: el enunciado de las 50 preguntas tomadas de libros con derechos de autor se reemplaza por su referencia exacta. |
| `eda/` | Inventario de fuentes revisadas (embudo de selección). | **Sí**. |

## Reglas de separación (control de fuga de información)

1. Ninguna pregunta de la batería ni ítem del pre-test/post-test entra al corpus (ni siquiera con otros datos).
2. Las variantes de un mismo problema (misma familia) van al mismo conjunto de la partición.
3. El conjunto de evaluación final (120 preguntas) se ejecuta una sola vez, con la configuración congelada.
4. Nunca se guardan datos personales de estudiantes en esta carpeta.

> **Nota sobre la publicación de la batería.** El conjunto de evaluación final es público para que el jurado pueda verificarlo. Por eso, en la tesis se declara que los resultados corresponden a la versión de la batería congelada en esta fecha (SHA-256 en `bateria/README.md`).
