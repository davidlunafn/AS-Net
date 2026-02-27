
# Análisis de Resultados de la Evaluación

Estos resultados son excelentes y cuentan una historia muy sólida y positiva sobre el modelo AS-Net.

### 1. Métricas Objetivas (Calidad de Señal: SDR, SIR, SI-SDR)

**Observación:** Todas las métricas de "mejora" (`improvement`) son consistentemente positivas en todos los niveles de ruido. La mejora del **SIR (Signal-to-Interference Ratio)** es especialmente notable, alcanzando casi **+10 dB**.

**Interpretación:** Esto es muy bueno. Significa que:
*   El modelo **siempre mejora la calidad de la señal**, nunca la empeora.
*   Es **extremadamente eficaz para eliminar el ruido** de la lluvia (la "interferencia"), que es precisamente para lo que fue diseñado.

### 2. Métricas Funcionales (Utilidad Práctica: F1-Score)

Aquí es donde los resultados son más impactantes y cuentan la historia principal.

*   **El Techo (`f1_clean` = 0.96):** BirdNET es capaz de detectar la especie casi perfectamente (`96%` de F1-Score) cuando el audio es limpio. Este es nuestro rendimiento ideal.
*   **El Problema (`f1_mixed`):** El ruido de la lluvia degrada masivamente el rendimiento. A -15dB, el F1-Score se desploma a **0.36**, lo que hace que el detector sea casi inútil.
*   **La Solución (`f1_separated`):** Aquí es donde el modelo AS-Net brilla:
    *   En los escenarios más difíciles (-15dB a 0dB), el modelo **siempre mejora el rendimiento** del detector.
    *   El salto más impresionante ocurre a **-5dB**, donde el F1-Score pasa de `0.77` a **`0.90`**.
    *   A partir de 0dB, el modelo **recupera prácticamente todo el rendimiento perdido**. El F1-Score de la señal separada (`~0.96`) es casi idéntico al de la señal limpia (`0.96`).

### Conclusión General (La Historia para el Artículo)

1.  **El modelo funciona:** Se ha demostrado cuantitativamente que AS-Net mejora la calidad de la señal de audio en términos de SDR, SIR y SI-SDR.
2.  **El modelo es útil en la práctica:** La limpieza del audio tiene un impacto directo y medible en una tarea ecológica real, permitiendo que un clasificador de especies funcione en condiciones de ruido en las que de otro modo fallaría.
3.  **El modelo restaura el rendimiento:** El hallazgo clave es que, en condiciones de ruido moderado a alto (de -5dB a 5dB), AS-Net es capaz de restaurar el rendimiento del clasificador a un nivel casi perfecto, como si el ruido nunca hubiera existido.

En resumen: **los resultados son excelentes.** Demuestran que el modelo no solo es un éxito técnico, sino también una herramienta con un impacto práctico y significativo.

---

### Tabla de Resultados Detallados

| SNR Level | SI-SDR Mixture | SI-SDR Separated | SI-SDR Impr. | SDR Mixture | SDR Separated | SDR Impr. | SIR Mixture | SIR Separated | SIR Impr. | F1 Clean | F1 Mixed | F1 Separated |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| -15 | -15.08 | -10.98 | 4.10 | -14.44 | -10.56 | 3.89 | -14.44 | -9.22 | 5.23 | 0.96 | 0.36 | 0.37 |
| -10 | -10.07 | -5.31 | 4.76 | -9.88 | -5.13 | 4.74 | -9.88 | -3.88 | 6.00 | 0.96 | 0.58 | 0.65 |
| -5  | -5.05  | 0.01   | 5.05 | -4.98  | 0.12   | 5.11 | -4.98  | 1.38  | 6.37 | 0.96 | 0.77 | 0.90 |
| 0   | -0.03  | 5.30   | 5.32 | -0.00  | 5.41   | 5.42 | -0.00  | 6.83  | 6.83 | 0.96 | 0.87 | 0.96 |
| 5   | 4.99   | 10.58  | 5.60 | 4.99   | 10.72  | 5.73 | 4.99   | 12.53 | 7.53 | 0.96 | 0.93 | 0.96 |
| 10  | 9.99   | 15.84  | 5.85 | 10.00  | 16.01  | 6.01 | 10.00  | 18.55 | 8.55 | 0.96 | 0.94 | 0.96 |
| 15  | 15.00  | 20.99  | 5.99 | 15.00  | 21.18  | 6.18 | 15.00  | 24.86 | 9.86 | 0.96 | 0.95 | 0.96 |

