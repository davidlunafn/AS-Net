
# Estructura y Guía para el Artículo Científico de AS-Net

Este documento sirve como una guía para estructurar tu artículo de investigación, organizando el material que hemos generado.

---

### **Abstract (Resumen)**

*   **Contexto:** El monitoreo de la biodiversidad a través de la ecoacústica se ve obstaculizado por el ruido ambiental, que degrada el rendimiento de los sistemas de análisis automático.
*   **Problema:** Los modelos de separación de fuentes existentes, a menudo diseñados para el habla humana, no están optimizados para las características únicas de las señales bioacústicas.
*   **Solución Propuesta:** Se presenta AS-Net, una nueva arquitectura de red neuronal de dominio del tiempo, basada en TCNs, diseñada para la separación de biofonía aviar.
*   **Metodología:** El modelo fue entrenado con datos sintéticos (cantos de _Parus major_ + ruido rosa) y evaluado rigurosamente en un conjunto de prueba con ruido de lluvia real. Se utilizó un protocolo de evaluación dual, midiendo tanto la calidad de la señal (SDR, SIR) como la utilidad funcional a través del rendimiento de un clasificador BirdNET.
*   **Resultados:** AS-Net demostró una mejora robusta y consistente en la calidad de la señal en todos los niveles de ruido, con una notable supresión de interferencias (hasta +9.86 dB de mejora en SIR). Funcionalmente, el modelo fue capaz de restaurar el rendimiento de detección de especies a un nivel casi perfecto (>95% F1-Score) en condiciones de ruido moderado, donde el clasificador de otro modo fallaría.
*   **Conclusión:** AS-Net es una herramienta eficaz y robusta que mejora significativamente la viabilidad del análisis bioacústico automatizado en entornos ruidosos, representando una contribución práctica para el campo de la ecoacústica computacional.

---

## 1. Introducción

*   **Párrafo 1: El Campo.** Comienza con el contexto general de la ecoacústica como herramienta para el monitoreo de la biodiversidad a gran escala (puedes basarte en la **Sección 1.1** de `docs/planning.md`).
*   **Párrafo 2: El Problema.** Introduce el concepto de paisaje sonoro (biofonía, geofonía, antropofonía) y el "problema del cóctel bioacústico". Explica cómo el ruido enmascara las señales de interés y degrada el rendimiento de los análisis automáticos (usa la **Sección 1.2** de `docs/planning.md`).
*   **Párrafo 3: La Brecha en la Solución.** Argumenta por qué las soluciones existentes no son óptimas. Menciona el "desajuste de dominio" entre el habla humana y las vocalizaciones de las aves (características espectro-temporales diferentes). (Usa la **Sección 2.3** de `docs/planning.md`).
*   **Párrafo 4: Objetivos y Contribución.** Presenta formalmente el objetivo de tu trabajo: introducir y validar AS-Net. Resume tus contribuciones clave: (1) la arquitectura AS-Net, (2) la metodología de evaluación dual, y (3) la validación en un escenario realista con ruido de lluvia. (Usa y adapta la **Sección 1.3 y 1.4** de `docs/planning.md`).

---

## 2. Materiales y Métodos

*Esta sección describe CÓMO hiciste la investigación.*

### 2.1. Arquitectura del Modelo: AS-Net

*   Describe la arquitectura general de AS-Net como un modelo de extremo a extremo, en el dominio del tiempo, basado en una estructura de codificador-separador-decodificador.
*   **AQUÍ INSERTAS LA FIGURA:** `as_net_architecture_final.pdf`.
*   Describe cada componente principal, usando los detalles del diagrama:
    *   **Encoder:** `Conv1D` con sus parámetros (kernel, stride, canales).
    *   **Separation Module:** Pila de 8 `TCN Blocks` con sus parámetros (kernel, dilatación, dropout).
    *   **Masking Module:** `Conv1D` seguida de una activación `Sigmoid` para generar las máscaras.
    *   **Decoder:** `ConvTranspose1D` para reconstruir la señal en el dominio del tiempo.

### 2.2. Conjuntos de Datos

*   **Datos de Entrenamiento y Validación:** Describe la creación del conjunto de datos sintético. Especie: _Parus major_ (Carbonero Común). Ruido: **ruido rosa**. Niveles de SNR: -15dB a 15dB. División: 80/10/10.
*   **Datos de Prueba (Generalización):** Describe la creación del conjunto de datos para la evaluación final. Ruido: **grabaciones de lluvia real**. Esto es un punto fuerte, ¡resáltalo!

### 2.3. Procedimiento de Entrenamiento

*   **AQUÍ INSERTAS LA TABLA:** Copia y pega la "Tabla de Hiperparámetros de Entrenamiento" que te generé anteriormente.
*   Menciona brevemente que el modelo se entrenó durante 50 épocas y que la convergencia fue monitoreada usando una pérdida de validación.

### 2.4. Protocolo de Evaluación

*   Explica el enfoque de evaluación dual.
*   **Métricas Objetivas:** Define brevemente SI-SDR, SDR y SIR como métricas de calidad de la señal.
*   **Métricas Funcionales:** Explica que se usó un clasificador BirdNET pre-entrenado. Define el F1-Score como la métrica para evaluar el rendimiento de la detección en tres tipos de audio: limpio (techo de rendimiento), mezclado (línea de base) y separado (resultado del modelo).

---

## 3. Resultados

*Esta sección presenta los hallazgos de forma objetiva, sin interpretarlos aún.*

### 3.1. Convergencia del Modelo

*   Menciona que el modelo convergió exitosamente.
*   **AQUÍ INSERTAS LA FIGURA:** `learning_curves.png`.
*   Señala el hallazgo clave de la gráfica: "El modelo muestra una excelente capacidad de generalización, evidenciada por una pérdida de validación que se mantuvo consistentemente por debajo de la pérdida de entrenamiento, indicando la ausencia de sobreajuste".

### 3.2. Resultados de la Evaluación Objetiva

*   **AQUÍ INSERTAS LA TABLA:** Copia y pega la tabla de resultados finales (`results_analysis.md`).
*   **AQUÍ INSERTAS LAS FIGURAS:** `evaluation_distribution_box.png` y `sdr_sir_improvement_bar.png`.
*   Describe neutralmente los resultados. Ej: "Como se muestra en la Tabla X y la Figura Y, el modelo AS-Net logró una mejora positiva y consistente en todas las métricas de calidad de señal y en todos los niveles de SNR. La mejora del SIR fue particularmente pronunciada, con una media de X dB".

### 3.3. Resultados de la Evaluación Funcional

*   **AQUÍ INSERTAS LA FIGURA:** `f1_score_comparison.png`.
*   Describe numéricamente el hallazgo principal. Ej: "La Figura Z ilustra el impacto del modelo en la tarea de detección. A -5dB de SNR, el F1-Score mejoró de 0.77 en la mezcla a 0.90 en la señal separada. A partir de 0dB, el rendimiento de la señal separada (F1 > 0.95) fue virtualmente indistinguible del de la señal limpia (F1 = 0.96)".

---

## 4. Discusión

*Esta es la sección más importante, donde interpretas el "¿y qué?" de tus resultados.*

*   **Párrafo 1: Resumen de Hallazgos.** Reitera brevemente tus resultados principales: AS-Net funciona, es robusto y es útil en la práctica.
*   **Párrafo 2: La Importancia de la Robustez.** Discute por qué el rendimiento consistente en diferentes niveles de SNR (mostrado en las métricas objetivas) es importante. Significa que el modelo es fiable.
*   **Párrafo 3: El Impacto Práctico.** Este es el punto clave. Discute el significado de los resultados del F1-Score. Argumenta que tu modelo no es solo un ejercicio técnico, sino que **rescata la capacidad de análisis** en condiciones de ruido donde antes era imposible. El hecho de que puedas hacer que un audio a 0dB se comporte casi como un audio limpio es la conclusión más potente.
*   **Párrafo 4: Limitaciones y Futuro.** Menciona las limitaciones (entrenamiento con ruido sintético, una sola especie). Luego, usa esto como un puente para proponer **trabajo futuro**: entrenar con ruidos más variados, abordar el problema de múltiples especies, y la implementación en hardware de campo (usa la **Sección 6.3** de `docs/planning.md`).

---

## 5. Conclusión

*   Un párrafo breve y contundente. Vuelve a empezar desde lo general a lo particular.
*   "La ecoacústica automática es prometedora pero está limitada por el ruido. En este trabajo, presentamos y validamos AS-Net, un modelo de separación de fuentes que demostró ser robusto y eficaz. El modelo no solo mejora la calidad de la señal, sino que restaura el rendimiento de tareas ecológicas posteriores, representando un paso práctico y significativo hacia el monitoreo de la biodiversidad a gran escala en entornos del mundo real."

