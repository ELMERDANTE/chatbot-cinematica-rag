# Variantes de datos: experimentos A/B sobre el corpus

**Proyecto:** chatbot educativo de cinemática basado en LLM + RAG (MIA-UNI)
**Autor:** Elmer Dante Rojas Zuta
**Informe de sprint completo:** [`docs/informe_sprint.md`](../../docs/informe_sprint.md)
**Notebook ejecutado:** [`notebooks/03_variantes_datos.ipynb`](../../notebooks/03_variantes_datos.ipynb)

> **Versión corregida.** Atiende dos observaciones de la docente:
> 1. Se retiró el enrutador de subtema.
> 2. La variante D2 ya no agrega datos externos al corpus.
>
> La versión anterior de D2 agregaba problemas de la reserva de Ágora. Eso no es una variante comparable, porque mete información que D0 y D1 no tienen. Esa ampliación se trata aparte, como candidata para el corpus v6 (sección 4).

## 1. Qué se compara

El avance de feature engineering mostró que el buscador falla sobre todo en **problemas de práctica**. Las fichas están escritas en lenguaje de teoría y los problemas en lenguaje de examen.

**Buscador fijo: V1** (BM25 + palabras vacías + raíces), **sin el enrutador de subtema**. Su aporte frente a V1 no era significativo y agregaba un clasificador entrenado aparte, que depende de los 5 subtemas de la batería.

**Las tres variantes usan las mismas 136 fichas del corpus v5.1.** Solo cambia el texto de cada ficha que se indexa:

| Variante | Único cambio | Por qué |
|---|---|---|
| **D0 · baseline** | Corpus v5.1 tal cual: título + contenido + relaciones + condiciones + error frecuente | Punto de partida |
| **D1** | Se indexa solo título + contenido, sin los campos estructurados | Saber si esos campos ayudan a buscar o meten ruido |
| **D2** | A cada ficha de teoría o procedimiento se le agrega el **enunciado de los ejemplos resueltos del propio corpus** de su mismo subtema con los que se vincula | Dar a las fichas lenguaje de problema **sin agregar datos nuevos** |

**Cómo se construye D2:**
- **Vínculo:** cada uno de los 35 ejemplos resueltos se vincula con sus *n* fichas de teoría o procedimiento más parecidas del mismo subtema. El parecido lo calcula BM25 entre textos del corpus, sin usar preguntas ni etiquetas de la batería.
- **Elección de *n*:** se probó *n* ∈ {1, 2, 3} con la CV de 5 particiones en desarrollo, antes de mirar validación. El resultado fue 0,487 / 0,476 / 0,488 → *n* = 3 ([`eleccion_n_D2.csv`](resultados/eleccion_n_D2.csv)).
- **Regla anti-fuga del proyecto:** se excluye todo ejemplo cuya similitud de Jaccard con **cualquier** pregunta de la batería supere 0,45. Quedan **22 de 35 ejemplos**, que se agregan 66 veces.
  - Las similitudes se calculan con [`src/preparar_ejemplos_d2.py`](../../src/preparar_ejemplos_d2.py) y se publican solo como números en [`data/corpus/variantes/ejemplos_jaccard.json`](../../data/corpus/variantes/ejemplos_jaccard.json).
- **Uso del texto agregado:** solo sirve para **buscar**. Al LLM se le entrega la ficha original.

## 2. Tabla estándar (holdout de validación, 72 preguntas respondibles)

| Variante | **Recall@5** (IC 95 %) | Δ vs D0 (IC 95 %; p) | Recall@5 práctica | Recall@5 teoría | Recall@1 | Recall@10 | MRR@10 | Fallo@5 | Palabras indexadas por ficha | Costo API |
|---|---|---|---|---|---|---|---|---|---|---|
| **D0** corpus v5.1 | 0,444 (0,340–0,549) | — | 0,30 | 0,89 | 0,125 | 0,553 | 0,314 | 0,486 | 102 | 0 |
| **D1** sin campos | 0,451 (0,347–0,556) | +0,007 (−0,049; +0,062; p = 0,92) | 0,33 | 0,81 | 0,139 | 0,539 | 0,321 | 0,486 | 87 | 0 |
| **D2** + enunciados de ejemplos | 0,486 (0,375–0,597) | +0,042 (−0,021; +0,104; p = 0,21) | 0,35 | 0,89 | 0,208 | 0,581 | 0,390 | 0,472 | 121 | 0 |

- **Δ e IC:** diferencias pareadas sobre las mismas 72 preguntas, con bootstrap de 5000 remuestreos.
- **Tiempo de búsqueda:** < 1 ms por consulta en las tres variantes.

### Gráfico clave: Recall@k vs k

![Recall@k vs k](resultados/recall_vs_k.png)

D2 queda algo por encima de D0 en todos los valores de k, pero la diferencia en k = 5 no es significativa.

### Estabilidad: validación cruzada de 5 particiones en desarrollo (180 preguntas)

| Variante | Recall@5 media ± de | Recall@10 | Particiones en que supera a D0 |
|---|---|---|---|
| D0 | 0,465 ± 0,104 | 0,592 | — |
| D1 | 0,459 ± 0,114 | 0,608 | 2 de 5 |
| D2 | 0,488 ± 0,100 | 0,633 | 3 de 5 |

### Por subtema (Recall@5 en validación)

| | MRU | MRUV | MVCL | MPCL | MCU |
|---|---|---|---|---|---|
| D0 | 0,50 | 0,21 | 0,21 | 0,60 | 0,67 |
| D2 | 0,54 | 0,21 | 0,25 | 0,60 | 0,80 |

MRUV y caída libre (MVCL) siguen siendo los subtemas más débiles. El corpus casi no tiene ejemplos de esos tipos de problema que pasen el filtro.

### Sensibilidad: ¿qué pasa sin la regla anti-fuga?

| Filtro de Jaccard | Ejemplos usados | Enunciados agregados | Recall@5 | Δ vs D0 (IC 95 %) | p |
|---|---|---|---|---|---|
| **Sin filtro** | 35 | 105 | 0,549 | +0,104 (+0,028; +0,188) | **0,008** |
| ≤ 0,45 (regla del proyecto) | 22 | 66 | 0,486 | +0,042 (−0,021; +0,104) | 0,21 |
| ≤ 0,40 | 19 | 57 | 0,486 | +0,042 (−0,007; +0,097) | 0,12 |
| ≤ 0,35 | 9 | 27 | 0,431 | −0,014 (−0,076; +0,049) | 0,72 |

**Lectura:**
- Sin filtro, D2 «mejora» 10 puntos y de forma significativa.
- Esa mejora viene de trasladar a las fichas **13 ejemplos que se parecen a preguntas de la batería**. El EDA ya había advertido que 17 preguntas comparten estructura con algún ejemplo del corpus.
- Con la regla aplicada, la mejora no es significativa.

**Es la misma trampa involuntaria que tenía la versión anterior de D2, y por eso la regla anti-fuga se fija antes de medir.**

### Prueba adicional: el enrutador de subtema retirado

Archivo: [`resultados/prueba_enrutador.csv`](resultados/prueba_enrutador.csv). β se eligió con CV anidada solo en desarrollo.

| Corpus | Sin enrutador (V1) | Con enrutador (V3) | Δ (IC 95 %; p) | Particiones de CV que gana |
|---|---|---|---|---|
| D0 | 0,444 | 0,470 | +0,025 (−0,021; +0,065; p = 0,26) | 4 de 5 |
| D2 | 0,486 | 0,532 | +0,046 (+0,005; +0,097; p = 0,03) | 5 de 5 |

Sobre el corpus adoptado (D0), el enrutador no aporta de forma significativa. Sobre D2 sí aparece una diferencia significativa, aunque en el límite. Se reporta con transparencia, pero no cambia la decisión: D2 no se adopta, y retirar el enrutador fue una decisión de simplicidad y generalidad.

## 3. Decisión

- **No se adopta ninguna variante.** Ni D1 ni D2 superan a D0 de forma significativa. **Se mantiene el corpus v5.1 (D0) con el buscador V1.**
- **Se descarta D1:** quitar los campos estructurados no mejora y empeora la teoría (0,89 → 0,81).
- **D2 queda como resultado informativo.** Mejora el ranking (Recall@1 de 0,125 a 0,208, MRR de 0,31 a 0,39), pero no Recall@5. Con los datos actuales no hay suficientes ejemplos «limpios» para dar a las fichas el lenguaje de los problemas.
- **Conclusión del sprint:** con el mismo corpus, las variantes no superan a D0. Mejorar la práctica requiere **datos nuevos**: el corpus v6.

## 4. Fuera de las variantes: ampliación del corpus con problemas de Ágora (candidata para el corpus v6)

- **Qué es:** agregar a las fichas problemas reales de la **reserva** de Ágora, que son problemas con clave verificada que no entraron a la batería.
- **Construcción:** [`src/preparar_expansion.py`](../../src/preparar_expansion.py) → [`data/corpus/variantes/expansion_agora.jsonl`](../../data/corpus/variantes/expansion_agora.jsonl). Se agregaron 73 problemas a 48 fichas y se descartaron 32 de 106 por parecido con la batería.
- **Resultado exploratorio:** con el buscador V1, el Recall@5 sube de 0,444 a 0,576 (p = 0,01).
- **Por qué no es una variante:** mete información que D0 y D1 no tienen, así que no compara formas de preparar los mismos datos.
- **Riesgo de optimismo:** la reserva sale de la misma fuente que la batería. En la sensibilidad, con el filtro ≤ 0,30 la mejora desaparece.
- **Tratamiento:** se evaluará como una **nueva versión del corpus (v6)**, construida y congelada antes de medirla, con el examen final sellado y, si es posible, con preguntas nuevas independientes.

## 5. Validación y cero leakage

- **Misma partición y validación que el avance anterior:**
  - partición con semilla 20261015;
  - CV `StratifiedGroupKFold(5)`, estratificada por subtema y agrupada por familia;
  - *n* de D2 elegido solo con CV en desarrollo;
  - holdout de validación, ejecutado una vez por variante.
- **Verificación automática:** [`verificar_fugas.py`](verificar_fugas.py) → [`resultados/verificacion_fugas.json`](resultados/verificacion_fugas.json) con **todo_ok = true**.
  - Los enunciados que agrega D2 son todos del corpus, y su similitud máxima con la batería es 0,441.
  - La partición es exacta y no hay familias partidas.
  - Ninguna partición de la CV comparte familias.
  - Los logs contienen solo preguntas de validación.
  - No hay textos protegidos.

## 6. Reproducibilidad

```bash
pip install -r requirements.txt
python experiments/variantes_datos/ejecutar.py        # elección de n + CV + holdout + sensibilidad + prueba del enrutador
python experiments/variantes_datos/graficar.py        # resultados/recall_vs_k.png
python experiments/variantes_datos/verificar_fugas.py # resultados/verificacion_fugas.json
```

- **Runs:** cada variante deja `runs/dv-<variante>-<hash>/` con su `config.yaml`, el `log.jsonl` por consulta y su `metricas.json`.
- **Batería:** se usa la batería completa (privada). Desde un clon limpio se usa la pública, que omite los enunciados de libros, y los números cambian ligeramente. Con ella:
  - la CV elige *n* = 1;
  - Recall@5 da D0 = 0,452, D1 = 0,444 y D2 = 0,452 (63 preguntas de validación).

  La conclusión es la misma: ninguna variante supera a D0.
