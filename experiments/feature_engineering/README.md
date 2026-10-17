# Feature engineering y experimentos A/B del componente de recuperación

**Proyecto:** chatbot educativo de cinemática basado en LLM + RAG (MIA-UNI)
**Autor:** Elmer Dante Rojas Zuta
**Datos:** corpus v5.1 (136 fichas) · batería v4 (400 preguntas; se usan las 360 respondibles)

## 1. Qué se compara y por qué

El chatbot responde en dos pasos: primero **recupera** fichas del corpus (la «R» de RAG) y luego el LLM **genera** la respuesta con esas fichas. Si la recuperación falla, el LLM no recibe el contenido correcto. Por eso esta etapa optimiza la recuperación sin usar el LLM: no tiene costo de API y cada variante se evalúa en segundos.

**Pregunta del experimento:** ¿qué cambio en el buscador recupera más fichas correctas entre las primeras 5?

**Métrica principal:** Recall@5, la proporción de las fichas esperadas de cada pregunta (anotadas a mano, 1–3 por pregunta) que aparecen entre las 5 primeras recuperadas. Se fija k = 5 porque es el número de fichas que recibirá el LLM; el EDA mostró que 5 fichas ocupan unos 800–1200 tokens.

**Métricas secundarias:**
- Recall@1, Recall@3 y Recall@10;
- MRR@10: qué tan arriba aparece la primera ficha correcta;
- nDCG@5 y Precision@5;
- Fallo@5: preguntas sin ninguna ficha correcta entre las 5 primeras;
- acierto de subtema@1: la primera ficha es del subtema correcto;
- costo y tiempo: construcción del índice y latencia por consulta.

### Variantes: un cambio por vez

| Variante | Se construye sobre | Único cambio |
|---|---|---|
| **B0 (línea base)** | — | BM25 con normalización mínima: minúsculas, sin tildes y separación en palabras |
| **V1** | B0 | Preprocesamiento: se quitan las palabras vacías y se reducen las palabras a su raíz (stemming Snowball en español) |
| **V2** | V1 | Puntuación híbrida: BM25 + similitud semántica LSA (TF-IDF + SVD), con las puntuaciones normalizadas y combinadas con peso α |
| **V3** | V1 | Enrutador de subtema: un clasificador estima P(subtema \| pregunta) y suma β·P al puntaje de las fichas de ese subtema |

V3 se construye sobre V1, y no sobre V2, porque V2 no superó a V1 en la validación cruzada de desarrollo. V3 responde al diagnóstico de B0 y V1: las preguntas de práctica describen una situación sin nombrar el concepto, y BM25 mezcla fichas de subtemas distintos. Por ejemplo, las «vueltas» de una pregunta de MRU atraen fichas de MCU.

**Sustituto de los embeddings:** LSA se usa como sustituto offline de un modelo de embeddings, porque el entorno de trabajo no tiene acceso a Hugging Face. En la siguiente etapa se reemplazará por un modelo multilingüe con la misma interfaz (`src/rag/recuperacion.py`).

## 2. Tabla estándar (holdout de validación, 72 preguntas respondibles)

| Variante | **Recall@5** (IC 95 %) | Δ vs B0 (IC 95 %; p) | Recall@1 | Recall@10 | MRR@10 | nDCG@5 | Fallo@5 | Acierto de subtema@1 | Índice (ms) | Consulta (ms, media / p95) | Costo API |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **B0** BM25 base | 0,377 (0,273–0,479) | — | 0,146 | 0,488 | 0,285 | 0,287 | 0,583 | 0,708 | 14 | 0,8 / 1,8 | 0 |
| **V1** + raíces | 0,444 (0,340–0,549) | +0,067 (−0,014; +0,150; p = 0,10) | 0,125 | 0,553 | 0,314 | 0,315 | 0,486 | 0,694 | 13 | 0,4 / 0,7 | 0 |
| **V2** + híbrido LSA | 0,417 (0,312–0,521) | +0,039 (−0,032; +0,111; p = 0,29) | 0,139 | 0,572 | 0,319 | 0,312 | 0,528 | 0,736 | 52 | 2,1 / 5,8 | 0 |
| **V3** + enrutador | **0,470** (0,361–0,572) | **+0,093** (+0,009; +0,181; **p = 0,03**) | **0,188** | **0,644** | **0,376** | **0,358** | **0,458** | **0,931** | 13 \* | 1,2 / 1,5 | 0 |

- **Δ e IC:** son diferencias **pareadas** sobre las mismas 72 preguntas, con IC bootstrap de 5000 remuestreos.
- **Comparación con su base:**
  - V2 vs V1: −0,028 (p = 0,25);
  - V3 vs V1: +0,025 (−0,021; +0,065; p = 0,26).
- **\* Tiempo de índice de V3:** no incluye el entrenamiento del enrutador, que se hace una sola vez fuera del índice y toma unos 0,1 s.
- **Costo:** ninguna variante usa API; el costo es solo de cómputo local, del orden de 1–2 milisegundos por consulta. Los tiempos varían algo entre ejecuciones (dependen de la carga de la máquina); las métricas de recuperación no cambian.
- **Archivos:** tabla completa en [`resultados/tabla_principal.csv`](resultados/tabla_principal.csv); métricas por run en `runs/<run_id>/metricas.json`.

### Gráfico clave: Recall@k vs k

![Recall@k vs k](resultados/recall_vs_k.png)

- **V3 domina desde k = 4** y su ventaja crece con k: Recall@10 = 0,64, frente a 0,49 de la línea base.
- **k = 1 a 3:** las variantes casi no se distinguen. El problema no es la primera posición, sino que la ficha correcta queda más abajo.

### Estabilidad: validación cruzada de 5 particiones en desarrollo (180 preguntas)

| Variante | Recall@5 media ± de | Recall@10 | MRR@10 | nDCG@5 |
|---|---|---|---|---|
| B0 | 0,434 ± 0,111 | 0,535 | 0,379 | 0,361 |
| V1 | 0,465 ± 0,104 | 0,592 | 0,369 | 0,364 |
| V2 | 0,459 ± 0,120 | 0,595 | 0,364 | 0,360 |
| V3 | **0,531 ± 0,086** | **0,665** | **0,395** | **0,403** |

El orden de las variantes es el mismo en la validación cruzada y en el holdout. V3 es la mejor en las dos y tiene la menor dispersión entre particiones. Detalle por partición: [`resultados/cv_desarrollo.csv`](resultados/cv_desarrollo.csv).

### Resultados por segmento (Recall@5 en validación)

| Segmento | B0 | V1 | V2 | V3 |
|---|---|---|---|---|
| Teoría (18) | 0,84 | 0,89 | 0,89 | 0,91 |
| Práctica (54) | 0,22 | 0,30 | 0,26 | **0,32** |
| Cobertura parcial (4) | 0,00 | 0,00 | 0,00 | 0,00 |
| Solo preguntas con texto público (63) | 0,39 | 0,45 | 0,41 | 0,47 |

Por subtema y por nivel: [`resultados/recall5_por_subtema.csv`](resultados/recall5_por_subtema.csv) y [`resultados/recall5_por_nivel.csv`](resultados/recall5_por_nivel.csv). MRUV y MVCL son los subtemas más difíciles (Recall@5 entre 0,21 y 0,27).

## 3. Conclusiones

1. **Configuración elegida para continuar:** V3, es decir, BM25 + raíces + enrutador de subtema con β = 2,0.
   - Es la única variante con mejora significativa frente a la línea base (+9,3 puntos de Recall@5; p = 0,03).
   - Sube el acierto de subtema de 69–74 % a 93 %.
   - Mantiene el tiempo de consulta en torno a 1 ms.
2. **Límite de la evidencia:** la mejora de V3 sobre V1 es consistente (+6,6 puntos en la validación cruzada y +2,5 en el holdout), pero en el holdout **no es significativa** (p = 0,26). Con 72 preguntas, el IC de una diferencia pareada mide entre ±4 y ±9 puntos, así que solo se detectan con seguridad mejoras grandes.
3. **LSA no aporta (V2):** empeora ligeramente a V1. El LSA se ajusta solo con 136 fichas y no aprende relaciones semánticas útiles; se sustituirá por embeddings preentrenados.
4. **El cuello de botella ya no es el subtema, sino la ficha dentro del subtema:** en 29 de las 33 preguntas en las que V3 falla, la primera ficha sí es del subtema correcto. Hay dos causas:
   - **Anotación estrecha:** cada pregunta de práctica tiene anotadas 1–3 fichas de teoría, pero los procedimientos («Cómo resolver…») y los ejemplos resueltos del mismo tipo también serían útiles para el LLM y no cuentan como acierto. Por eso Recall@5 subestima la utilidad real del contexto. Se revisará la anotación en desarrollo.
   - **Cobertura parcial:** las 4 preguntas de validación sin ficha específica fallan en todas las variantes. Confirma el hallazgo del EDA: el corpus v6 necesita 4–5 fichas nuevas.
5. **Teoría vs práctica:** la recuperación ya funciona bien en teoría (0,91). El margen de mejora está en práctica (0,32), que es donde el corpus v6 añadirá unos 120 ejemplos resueltos de la academia.

## 4. Feature set y pipeline

```
pregunta ──► normalizar ──► tokens ──► [V1+] quitar palabras vacías y dígitos ──► raíz Snowball
                                              │
       corpus (136 fichas) ──► mismo preprocesamiento ──► BM25 (k1 = 1,5; b = 0,75) ──┐
                                              └─► [V2] TF-IDF ──► SVD (LSA) ─────────┤──► normalización min-max ──► puntaje
       corpus + preguntas de ENTRENAMIENTO ──► [V3] TF-IDF + regresión logística ──► P(subtema) · β ──┘
                                                                                      └──► top-k fichas
```

**Archivos:** `src/rag/preprocesamiento.py`, `src/rag/recuperacion.py`, `src/rag/enrutador.py` y `src/rag/metricas.py`.

**Qué se ajusta y con qué datos («fit» solo en entrenamiento):**

| Componente | Qué aprende | Se ajusta con | Nunca ve |
|---|---|---|---|
| Normalización, palabras vacías, raíces | Nada (reglas fijas) | — | — |
| BM25 (IDF, longitudes) | Estadísticas de términos | **Solo el corpus** | Ninguna pregunta |
| TF-IDF + SVD del híbrido (V2) | Vocabulario y espacio latente | **Solo el corpus** | Ninguna pregunta |
| α y dimensiones del LSA (V2) | Hiperparámetros | Particiones de entrenamiento de la CV; para el holdout, todo desarrollo | Validación y evaluación final |
| Enrutador de subtema (V3) | TF-IDF + regresión logística | Corpus + **preguntas de entrenamiento** de cada partición; para el holdout, desarrollo | La partición de prueba de la CV, validación y evaluación final |
| β del enrutador (V3) | Hiperparámetro | CV **anidada** de 3 particiones dentro del entrenamiento | Ídem |

- **Grillas:** α ∈ {0,3; 0,5; 0,7}, dimensiones ∈ {50; 100} y β ∈ {0,5; 1; 2; 4}.
- **Valores finales elegidos en desarrollo:** α = 0,5, 50 dimensiones y β = 2,0. β = 2 se eligió en 4 de las 5 particiones de la CV y es un valor interior de la grilla.

## 5. Partición y confirmación de cero leakage

**Partición de la batería (fijada antes de cualquier experimento, semilla 20261015):**
- **Tamaños:** 200 de desarrollo, 80 de validación y 120 de evaluación final. Sin contar las no respondibles: 180, 72 y 108.
- **Estratificación exacta:** cada subtema aporta 40/16/24. Cada estrato (teoría, práctica básica, intermedia y avanzada, y no respondible) se reparte exactamente 50/20/30 %.
- **Agrupación por familia:** las 5 familias de preguntas parecidas (variantes de un mismo problema) están enteras en un solo conjunto.

**Validación:** el script [`verificar_fugas.py`](verificar_fugas.py) comprueba todo lo anterior sobre los datos y los logs, y guarda el resultado en [`resultados/verificacion_fugas.json`](resultados/verificacion_fugas.json). Resultado: **todo_ok = true**.

| Verificación | Resultado |
|---|---|
| Preguntas copiadas en el corpus | Ninguna. Similitud de Jaccard máxima entre una pregunta y una ficha: 0,45; ninguna supera 0,5 |
| Partición 200/80/120, estratos y subtemas | Exacta: desviación 0,0 respecto de 50/20/30 % |
| Familias partidas entre conjuntos | 0 de 395 |
| Familias compartidas entre entrenamiento y prueba en la CV de 5 particiones | 0 en cada partición; cada partición de prueba tiene 36 preguntas y entre 19,4 % y 22,2 % de cada subtema |
| Consultas de evaluación final en los logs | 0: las 288 consultas registradas (4 runs × 72) son de validación |
| Enunciados de libros protegidos en los logs | 0 (`consulta_texto = null`) |
| Ajuste con preguntas de prueba | Ninguno: el índice se ajusta solo con el corpus; el enrutador y los hiperparámetros, solo con entrenamiento (sección 4) |

**La evaluación final sigue sellada.** Se ejecutará una sola vez, con la configuración y el umbral de abstención congelados, cuando el sistema completo (recuperación + LLM) esté listo.

## 6. Validación: validación cruzada + holdout

1. **Validación cruzada en desarrollo:**
   - Método: `StratifiedGroupKFold(5)`, con y = subtema y grupos = familia.
   - Objetivo: estimar la estabilidad de cada variante y elegir los hiperparámetros sin mirar validación.
   - Los hiperparámetros se eligen dentro de cada partición de entrenamiento; en V3, con una CV anidada de 3 particiones.
2. **Holdout de validación:** cada variante se ejecuta **una vez** con su configuración final, elegida con todo desarrollo. Esta es la comparación oficial de la tabla estándar.
3. **Pruebas estadísticas:**
   - IC 95 % bootstrap de cada métrica (2000 remuestreos);
   - diferencias pareadas bootstrap (5000 remuestreos) con p bilateral.

## 7. Registro de experimentos (logs)

Cada variante genera un run en `runs/fe-<variante>-<hash de configuración>/`:

| Archivo | Contenido |
|---|---|
| `config.yaml` | Configuración completa de la variante: preprocesamiento, base, hiperparámetros, semilla, versiones de corpus y batería, partición evaluada |
| `log.jsonl` | Un registro por consulta, según `evaluation/log_schema.json`: `run_id`, `config_hash`, `commit`, fecha, `consulta_id`, top-10 con puntaje y rango, métricas y latencia |
| `metricas.json` | Métricas agregadas con IC |

Resumen global con hiperparámetros por partición: [`resultados/resumen.json`](resultados/resumen.json).

## 8. Cómo reproducir

```bash
pip install -r requirements.txt
python experiments/feature_engineering/ejecutar.py        # CV, holdout, runs y tablas (unos minutos)
python experiments/feature_engineering/graficar.py        # resultados/recall_vs_k.png
python experiments/feature_engineering/verificar_fugas.py # resultados/verificacion_fugas.json (sale con error si algo falla)
```

- **Con la batería completa:** si existe la copia privada (`data/bateria/privado/bateria_v4_completa.jsonl`, excluida del repositorio), se usan las 360 preguntas respondibles. Los resultados de esta carpeta se obtuvieron así.
- **Con la batería pública:** desde un clon limpio, el script usa la batería pública, que omite los enunciados de las preguntas tomadas de libros (50 de 400). Los números cambian ligeramente; como referencia, la fila «solo preguntas con texto público» de la sección 2.

## 9. Limitaciones y siguientes pasos

- **Tamaño:** 72 preguntas en el holdout dan IC amplios (±10 puntos). La evaluación final (108 respondibles) y el corpus v6 darán estimaciones más precisas.
- **Embeddings:** el LSA es un sustituto offline. La comparación definitiva entre lo léxico y lo semántico se hará con un modelo multilingüe de embeddings.
- **Anotación:** se revisará la anotación de fichas esperadas para incluir los procedimientos y ejemplos pertinentes, y se medirá también «Recall@5 amplio».
- **Siguiente experimento:** comparar la generación del **LLM solo** con la del **LLM + RAG (V3)** en preguntas de desarrollo, midiendo corrección, fidelidad y costo por consulta.
