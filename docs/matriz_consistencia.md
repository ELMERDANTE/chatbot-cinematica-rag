# Matriz de consistencia

**Tesis:** *Desarrollo y evaluación de un chatbot educativo basado en IA generativa y RAG para reforzar el aprendizaje de cinemática preuniversitaria en una academia de Chachapoyas*
**Autor:** Elmer Dante Rojas Zuta · Proyecto de Investigación II (MIA-UNI)
**Fuente:** proyecto de investigación, capítulos I–IV (secciones 1.1.3, 2, 3.3, 4 y tabla 4.10). Los enunciados se transcriben tal como figuran en el proyecto.

## 1. Matriz resumida

| Problema | Objetivo | Hipótesis | Variables | Indicadores | Metodología |
|---|---|---|---|---|---|
| **PG.** ¿En qué medida el uso del chatbot mejora el rendimiento en cinemática respecto del refuerzo convencional? | **OG y OE4.** Determinar el efecto del chatbot en el rendimiento, controlando el pre-test | **HG.** Post-test del grupo experimental significativamente mayor que el del grupo de comparación, controlando el pre-test | VI: condición de grupo · VD: rendimiento en cinemática · Covariable: rendimiento inicial | Puntaje de post-test (0–20) · puntaje de pre-test (0–20) | Cuasi-experimental de dos grupos con pre/post · ANCOVA · tamaño del efecto |
| **PE1.** Dificultades y necesidades de acompañamiento | **OE1.** Diagnosticar dificultades y necesidades | No aplica (diagnóstico) | Dificultades por subtema · acceso tecnológico · hábitos de consulta | Aciertos por subtema · acceso a smartphone y datos | Prueba diagnóstica por subtemas · encuesta · análisis descriptivo |
| **PE2.** Arquitectura adecuada y su calidad técnica | **OE2.** Diseñar, implementar y validar técnicamente el prototipo | No aplica (desarrollo); criterios de éxito fijados en validación | Configuración del sistema · calidad de recuperación · calidad de respuesta · desempeño operativo | Recall@k · corrección · fidelidad · abstención · latencia · costo | Evaluación técnica con banco controlado y partición desarrollo / validación / final |
| **PE3.** Aporte de la recuperación de información | **OE3.** Comparar el sistema con RAG frente al mismo LLM sin RAG | **HE1** (corrección) y **HE2** (afirmaciones no sustentadas) | VI: configuración (con RAG / sin RAG) · VD: calidad de respuesta | Proporción de respuestas correctas · tasa de afirmaciones no sustentadas | Comparación pareada · McNemar · Wilcoxon · rúbrica a ciegas · kappa |
| **PE4.** Satisfacción, utilidad, uso y su relación con la ganancia | **OE5.** Describir satisfacción, utilidad y uso; explorar su relación con la ganancia | **HE3** (satisfacción) y **HE4** (exploratoria) | Satisfacción y utilidad percibida · intensidad de uso · ganancia | Likert 1–5 por dimensión · n.º de consultas, días activos · ganancia normalizada | Encuesta · registros · entrevistas · Wilcoxon · regresión · análisis temático |

## 2. Matriz desarrollada

### 2.1 Problema, objetivo e hipótesis generales

| | |
|---|---|
| **Problema general** | ¿En qué medida el uso de un chatbot educativo basado en IA generativa y RAG, accesible a través de WhatsApp, mejora el rendimiento en cinemática de los estudiantes preuniversitarios de la Academia Ágora de Chachapoyas, en comparación con el refuerzo académico convencional? |
| **Objetivo general** | Desarrollar un chatbot educativo basado en inteligencia artificial generativa y RAG, accesible vía WhatsApp, y evaluar su calidad técnica y su efecto en el rendimiento en cinemática de los estudiantes preuniversitarios de la Academia Ágora de Chachapoyas, en comparación con el refuerzo académico convencional. |
| **Hipótesis general (HG)** | Los estudiantes de la Academia Ágora que utilizan el chatbot educativo basado en IA generativa y RAG obtienen un puntaje de post-test de cinemática significativamente mayor que los estudiantes que reciben refuerzo académico convencional, controlando el puntaje de pre-test. |
| **Variables** | Independiente: condición de grupo (experimental / comparación). Dependiente principal: rendimiento en cinemática. Covariable: rendimiento inicial. |
| **Indicadores** | Puntaje de post-test (escala 0–20), con cuatro ítems por subtema (MRU, MRUV, caída libre, tiro parabólico y movimiento circular). Puntaje de pre-test (0–20). |
| **Instrumento** | Prueba de cinemática en dos formas paralelas (A y B) de 20 ítems, con el orden contrabalanceado. |
| **Análisis** | Equivalencia inicial (diferencias estandarizadas) y ANCOVA del post-test con el pre-test como covariable, con tamaño del efecto. |

### 2.2 Problemas, objetivos e hipótesis específicos

| Problema específico | Objetivo específico | Hipótesis específica | Variables e indicadores | Técnica / instrumento y análisis |
|---|---|---|---|---|
| **PE1.** ¿Cuáles son las principales dificultades conceptuales y procedimentales en cinemática y las necesidades de acompañamiento extracurricular de los estudiantes de la Academia Ágora? | **OE1.** Diagnosticar las principales dificultades conceptuales y procedimentales en cinemática y las necesidades de acompañamiento extracurricular de los estudiantes de la Academia Ágora, mediante una prueba diagnóstica organizada por subtemas y una encuesta inicial aplicadas en el ciclo 2026-II. | No aplica: tiene carácter diagnóstico. | **Dificultades:** aciertos por subtema. **Acompañamiento:** acceso a teléfono y datos, hábitos de consulta fuera del aula. | Prueba diagnóstica por subtemas y cuestionario inicial. Análisis descriptivo. |
| **PE2.** ¿Qué arquitectura basada en un LLM y RAG, accesible por WhatsApp, permite ofrecer tutoría socrática en cinemática con respuestas correctas, fieles a una base de conocimiento validada y con costo y latencia viables? | **OE2.** Diseñar, implementar y validar técnicamente un prototipo de chatbot que integre un LLM con una arquitectura RAG, una base de conocimiento de cinemática validada por docentes, tutoría socrática, una política de abstención y acceso mediante WhatsApp. | No aplica: tiene carácter de desarrollo tecnológico. Los criterios de éxito se fijan en validación. | **Calidad de recuperación:** Recall@k (principal), Precision@k, MRR, tasa de recuperación fallida. **Calidad de respuesta:** corrección, fidelidad, calidad socrática, abstención correcta. **Desempeño operativo:** latencia p50/p95, tokens, costo por consulta, disponibilidad. | Banco controlado de 400 preguntas, partido en desarrollo / validación / evaluación final (200 / 80 / 120). Registros del sistema. Configuración versionada. |
| **PE3.** ¿En qué medida la recuperación de información mejora la corrección y la fidelidad de las respuestas del sistema respecto del mismo LLM sin recuperación? | **OE3.** Comparar la calidad de las respuestas del sistema con RAG frente al mismo LLM sin recuperación de información, mediante un banco controlado de preguntas de cinemática, métricas de recuperación y generación, y una rúbrica experta aplicada a ciegas. | **HE1.** Sobre un mismo conjunto controlado de preguntas de cinemática, el sistema con RAG produce una proporción de respuestas correctas significativamente mayor que el mismo LLM sin recuperación de información, con idéntica plantilla pedagógica.<br>**HE2.** En las mismas condiciones, el sistema con RAG presenta una tasa de afirmaciones no sustentadas significativamente menor que el mismo LLM sin recuperación de información. | **VI:** configuración del sistema (LLM con RAG / mismo LLM sin RAG; modelos de distinto costo). **VD:** corrección (principal), exactitud conceptual, corrección matemática, fidelidad, afirmaciones no sustentadas, calidad socrática. | Rúbrica experta a ciegas con dos evaluadores (kappa ponderado). McNemar para corrección y afirmaciones no sustentadas; Wilcoxon de rangos con signo para los puntajes de la rúbrica; intervalos bootstrap. |
| **PE4.** ¿Cómo valoran los estudiantes el chatbot en términos de satisfacción y utilidad percibida, cómo lo utilizan y qué relación existe entre la intensidad de uso y la ganancia académica? | **OE5.** Describir la satisfacción, la utilidad percibida y los patrones de uso del chatbot en el grupo experimental, y explorar la relación entre la intensidad de uso y la ganancia académica. | **HE3.** La satisfacción y la utilidad percibida del chatbot por los estudiantes del grupo experimental presentan una mediana significativamente superior al punto neutro de la escala.<br>**HE4 (exploratoria).** En el grupo experimental, una mayor intensidad de uso del chatbot se asocia con una mayor ganancia normalizada, controlando el puntaje de pre-test. | **Satisfacción y utilidad:** Likert 1–5 en claridad, utilidad percibida, facilidad de uso, confianza y satisfacción global. **Intensidad de uso:** n.º de consultas, días activos, sesiones de resolución guiada. **Ganancia normalizada.** | Cuestionario de satisfacción validado por juicio de expertos (alfa de Cronbach), registros del sistema y entrevista semiestructurada. Wilcoxon para una muestra (HE3), regresión lineal con el pre-test como covariable (HE4) y análisis temático. |

**Nota:** el OE4 corresponde al problema general y a la HG (sección 2.1). El OE1 y el OE2 no se asocian a hipótesis estadísticas por su naturaleza diagnóstica y de desarrollo.

### 2.3 Variables moderadoras y de control

| Variable | Indicadores | Fuente |
|---|---|---|
| Asistencia | % de asistencia | Registro de asistencia |
| Acceso tecnológico | Acceso a smartphone y datos | Encuesta inicial |
| Aula | Aula, docente y horario | Registro de asignación |

## 3. Metodología

| Aspecto | Descripción |
|---|---|
| **Enfoque** | Mixto, con predominio cuantitativo. |
| **Tipo** | Aplicada y tecnológica. |
| **Nivel** | Explicativo, con un componente evaluativo de la calidad técnica y pedagógica del prototipo. |
| **Diseño** | Dos componentes secuenciales: (1) **evaluación técnica** controlada del sistema, con banco de preguntas y comparación con RAG frente a sin RAG; (2) **estudio educativo cuasi-experimental** de dos grupos con pre-test y post-test, con la configuración del sistema ya congelada. |
| **Población** | Estudiantes del programa preuniversitario de la Academia Ágora matriculados en física en el ciclo regular 2027-I. |
| **Muestra** | No probabilística y censal respecto de dos aulas (≈ 50 estudiantes por grupo). El diagnóstico (OE1) se aplica a los estudiantes preuniversitarios del ciclo 2026-II, que no participan en la intervención. |
| **Técnicas e instrumentos** | Prueba de cinemática (formas paralelas A/B), cuestionarios (inicial, de satisfacción y de control), entrevista semiestructurada, rúbrica de evaluación experta y registros del sistema. |
| **Análisis** | ANCOVA (HG); McNemar, Wilcoxon y bootstrap (HE1, HE2); Wilcoxon para una muestra (HE3); regresión lineal (HE4); análisis temático. |

## 4. Avance en el repositorio por objetivo

| Objetivo | Qué está hecho en el repositorio | Dónde |
|---|---|---|
| **OE2** (prototipo y validación técnica) | Base de conocimiento (corpus v5.1, 136 fichas) y banco controlado (batería v4, 400 preguntas con partición congelada) | `data/`, rama `eda` |
| **OE2**: calidad de recuperación | EDA de los datos; feature engineering y variantes de datos del buscador; importancia de variables | Ramas `eda`, `feature-engineering`, `variantes-datos` / `sin-enrutador`, `importancia-variables` |
| **OE3** (con RAG frente a sin RAG) | Diseño del experimento, plantilla pedagógica y esquema de registro listos; ejecución pendiente | `prompts/`, `evaluation/`, `docs/figuras/flujo_experimental.png` |
| OE1, OE4 y OE5 | Pendientes: dependen del trabajo de campo (2026-II y 2027-I) | — |
