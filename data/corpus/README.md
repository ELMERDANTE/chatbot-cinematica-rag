# data/corpus/

Base de conocimiento del sistema RAG.

## Estructura prevista (no publicada)

```
corpus/
├── raw/          ← materiales originales (PDF, imágenes)
├── processed/    ← unidades semánticas transcritas y verificadas (JSONL)
└── manifest.json ← inventario: id, fuente, subtema, tipo, hash SHA-256
```

## Regla de segmentación (preliminar)

Un fragmento corresponde a una **unidad semántica completa**: una definición, una fórmula con el significado de sus variables y unidades, un procedimiento o un ejemplo resuelto con su solución. Nunca se separa una fórmula de su significado ni un problema de su solución.

## Validación

El corpus se valida por juicio de expertos independientes del investigador (exactitud, claridad, pertinencia y nivel) antes de congelarse como `corpus_v1`.
