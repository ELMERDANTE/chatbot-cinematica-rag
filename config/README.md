# config/

Configuración versionada del sistema. Todos los parámetros que afectan el comportamiento del chatbot se definen en **un único archivo**; su hash identifica la configuración de cada ejecución.

- `config.example.yaml`: plantilla con los valores iniciales del diseño.
  - Los parámetros marcados como `preliminar` son propuestas del diseño.
  - Los marcados como `calibrar` se fijan **solo** con los conjuntos de desarrollo y validación.

**Regla:** antes de ejecutar el conjunto de evaluación final, la configuración se congela (se registra con fecha y hash) y no se modifica después.
