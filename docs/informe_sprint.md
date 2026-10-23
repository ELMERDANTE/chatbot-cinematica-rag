# Informe de sprint: EDA, feature engineering y variantes de datos

**Proyecto:** *Desarrollo y evaluación de un chatbot educativo basado en IA generativa y RAG para reforzar el aprendizaje de cinemática preuniversitaria en una academia de Chachapoyas*
**Autor:** Elmer Dante Rojas Zuta · Proyecto de Investigación II (MIA-UNI)
**Rama:** `variantes-datos`

## 1. Contexto

- **Objetivo del proyecto:** un tutor de cinemática por WhatsApp que, antes de responder, **recupera** fichas de una base de conocimiento validada y **genera** una respuesta pedagógica con un LLM (RAG). La pregunta técnica de la tesis es cuánto mejora la recuperación la corrección de las respuestas frente al mismo LLM sin recuperación.
- **Componente de este sprint:** la **recuperación**, porque si el buscador no entrega la ficha correcta, el LLM no puede responder bien. Se evalúa sin LLM, con costo de API cero.
- **Datos:**

| Conjunto | Contenido | Papel |
|---|---|---|
| Corpus v5.1 | 136 fichas: teoría, procedimientos y ejemplos resueltos | Lo que el chatbot consulta |
| Batería v4 | 400 preguntas (70 % material real de la academia Ágora y libros) con clave, solución y fichas esperadas | El examen. Partición 200 / 80 / 120 (desarrollo / validación / evaluación final) |
| Reserva de Ágora | 106 problemas reales verificados que **no** están en la batería | Material para la variante de datos D2 |

- **Métrica central:** **Recall@5**, la proporción de las fichas esperadas de cada pregunta que aparece entre las 5 que se entregarán al LLM.
- **Métricas secundarias:** Recall@k, MRR@10, nDCG@5, Fallo@5, acierto de subtema, tiempo y costo.

### Resumen del EDA (detalle en [`results/eda/EDA_resumen.md`](../results/eda/EDA_resumen.md))

- **Corpus homogéneo:** 55–146 palabras por ficha, así que **una ficha = un fragmento**. Con k = 5, el contexto es de unos 800–1200 tokens.
- **Batería balanceada:** 80 preguntas por subtema; en cada uno, 18 de teoría, 18 por nivel de práctica y 8 no respondibles. La partición estratificada es exacta, con familias de preguntas parecidas sin partir.
- **Trazabilidad:** 890 preguntas revisadas = 694 verificadas + 196 no verificadas; se seleccionaron 280.
- **Sin filtración** entre batería y corpus.
- **Hallazgo accionable:** 11 preguntas tratan temas sin ficha específica (cobertura parcial).

### Resumen del feature engineering (detalle en [`experiments/feature_engineering/`](../experiments/feature_engineering/README.md))

- **Transformaciones del texto:** normalización, palabras vacías y raíces (Snowball), TF-IDF/LSA y un enrutador de subtema (TF-IDF + regresión logística).
- **Regla de ajuste:** todo se ajusta solo con el corpus o con preguntas de entrenamiento.
- **Variante adoptada, V3** (raíces + enrutador): Recall@5 de 0,377 → **0,470** (p = 0,03).
- **Diagnóstico:** el buscador ya acierta el subtema el 93 % de las veces. Lo que falla es la ficha exacta en problemas de práctica (Recall@5 de 0,32). **El cuello de botella son los datos.**

## 2. Línea base

| Etapa | Línea base | Recall@5 (validación) |
|---|---|---|
| Feature engineering | B0: BM25 con normalización mínima | 0,377 |
| **Este sprint (variantes de datos)** | **D0: buscador V3 sobre el corpus v5.1 sin cambios** | **0,470** |

En este sprint el buscador queda **fijo** (V3) y solo cambian **los datos**, como pide la tarea.

## 3. Experimentos A/B: un cambio por variante, en los datos

| Variante | Qué cambia en los datos | Por qué |
|---|---|---|
| **D0** baseline | Nada: corpus v5.1 (título + contenido + ecuaciones + error frecuente) | Referencia |
| **D1** | Se quitan los campos estructurados (ecuaciones, condiciones, error frecuente): solo título + contenido | ¿Esos campos ayudan a buscar o meten ruido? |
| **D2** | Se agregan al texto indexado de cada ficha hasta 5 **problemas reales de la reserva de Ágora** que la usan (expansión de documentos) | Los problemas de práctica describen situaciones con el vocabulario de la academia («móvil», «tarda», «lanzado»), que el corpus casi no tenía |

**Regla de D2:** la expansión solo se usa para buscar; al LLM se le entrega la ficha original. Los problemas que se parecen a cualquier pregunta de la batería (Jaccard > 0,45) se descartan, y fueron 32 de 106.

## 4. Resultados comparables

**Tabla (holdout de validación, 72 preguntas respondibles):**

| Variante | **Recall@5** (IC 95 %) | Δ vs D0 (p) | Recall@5 práctica | Recall@5 teoría | Recall@10 | MRR@10 | Fallo@5 | Latencia por consulta | Costo API |
|---|---|---|---|---|---|---|---|---|---|
| D0 baseline | 0,470 (0,361–0,572) | — | 0,32 | 0,91 | 0,644 | 0,376 | 46 % | ≈ 1 ms | 0 |
| D1 sin campos | 0,491 (0,380–0,595) | +0,021 (p = 0,57) | 0,37 | 0,85 | 0,637 | 0,384 | 44 % | ≈ 1 ms | 0 |
| **D2 + problemas Ágora** | **0,611** (0,514–0,715) | **+0,141 (p = 0,002)** | **0,52** | 0,89 | **0,755** | **0,539** | **32 %** | ≈ 1 ms | 0 |

**Gráfico único: Recall@k vs k**

![Recall@k vs k](../experiments/variantes_datos/resultados/recall_vs_k.png)

- **Mejor variante:** D2 está por encima en **todos** los valores de k, de 1 a 10.
- **Dónde mejora más:** en práctica (+0,19, p = 0,002) y en los subtemas más débiles: MRUV de 0,27 a 0,57 y MVCL de 0,21 a 0,46.
- **Validación cruzada en desarrollo:** D2 supera a D0 en las 5 particiones (Recall@5 de 0,531 → 0,620 en promedio).
- **Con el buscador más simple** (BM25 sin raíces ni enrutador), D2 también mejora: 0,377 → 0,553.

## 5. Validación

- **Split:** estratificado (subtema × componente × nivel) y **agrupado por familia**, con semilla fija (20261015) y partición congelada con su huella SHA-256.
- **Validación cruzada:** `StratifiedGroupKFold(5)` en desarrollo. Los hiperparámetros (β del enrutador) se eligen con CV anidada solo en entrenamiento.
- **Holdout:** validación, una ejecución por variante. La **evaluación final no se ha usado**.
- **Cero leakage, verificado por script** ([`verificar_fugas.py`](../experiments/variantes_datos/verificar_fugas.py) → `todo_ok = true`):
  - ningún problema agregado en D2 está en la batería, y la similitud máxima es 0,448;
  - no hay familias partidas;
  - los logs solo contienen preguntas de validación.
- **Sensibilidad:** con un filtro más estricto (Jaccard ≤ 0,35), D2 sigue mejorando de forma significativa (+0,106, p = 0,01).

## 6. Conclusión y decisión

- **Se adopta D2: corpus v5.1 expandido con problemas reales de Ágora**, con el buscador V3.
  - **Justificación técnica:** es la mayor mejora obtenida hasta ahora (+14 puntos de Recall@5, p = 0,002). Es estable en la validación cruzada y ataca justo la debilidad detectada en práctica.
  - **Costo y viabilidad:** costo de API cero y latencia de alrededor de 1 ms. No aumenta el contexto del LLM, porque la expansión solo sirve para buscar. Usa material propio de la academia que ya estaba verificado.
- **Se descarta D1:** no mejora de forma significativa y empeora la teoría. Los campos estructurados se mantienen.
- **Lección del sprint:** mejorar **los datos** rindió más que mejorar el algoritmo: +14 puntos frente a +9 del mejor cambio de buscador.

## 7. Reproducibilidad

```bash
pip install -r requirements.txt
python src/preparar_expansion.py --privado <carpeta_privada>   # solo si se reconstruye la expansión (requiere la reserva privada)
python experiments/variantes_datos/ejecutar.py                 # experimentos (≈ 30 s)
python experiments/variantes_datos/graficar.py                 # gráfico
python experiments/variantes_datos/verificar_fugas.py          # verificación de cero leakage
jupyter nbconvert --to notebook --execute notebooks/03_variantes_datos.ipynb
```

| Qué | Dónde |
|---|---|
| Logs por consulta y configuración de cada variante | `experiments/variantes_datos/runs/dv-<variante>-<hash>/` (`config.yaml`, `log.jsonl`, `metricas.json`) |
| Tablas y gráfico | `experiments/variantes_datos/resultados/` |
| Datos de la variante D2 | `data/corpus/variantes/expansion_agora.jsonl`: fuente y similitud de cada problema |
| Resultados desde un clon limpio | Con la batería pública (sin los enunciados de libros), Recall@5: D0 = 0,474 y D2 = 0,635. La conclusión es la misma |
| Huellas de los datos | Corpus: `data/corpus/manifest.json` (SHA-256 por ficha). Batería: `data/bateria/README.md` (SHA-256 de la versión congelada) |

## 8. Riesgos y próximos pasos

- **Riesgo de sesgo optimista:** los problemas de la academia se repiten por tipos, y parte de la mejora de D2 viene de esa semejanza. Con un filtro muy estricto (Jaccard ≤ 0,30) el efecto desaparece. Se confirmará **una sola vez** en la evaluación final, que sigue sellada.
- **Riesgo de drift:** si la academia cambia su material o su forma de redactar los problemas, la expansión debe actualizarse.
- **Sesgo de la batería:** la teoría y las no respondibles son en su mayoría generadas y están pendientes de revisión docente.
- **Próximo sprint:**
  1. Integrar D2 como **corpus v6**, agregando las fichas faltantes (sombras, vectores, cruce de un río).
  2. Reemplazar el LSA por embeddings.
  3. Iniciar el experimento de generación: **LLM solo frente a LLM + RAG**.
