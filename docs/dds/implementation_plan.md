# Plan de Implementación Detallado: AS-Net

Este documento describe el plan de implementación paso a paso para el proyecto AS-Net, desde la configuración inicial hasta la evaluación final del modelo. Se utilizará una arquitectura hexagonal para organizar el código.

## Fase 1: Configuración del Entorno y Generación de Datos

**Objetivo:** Preparar el entorno de desarrollo y crear el conjunto de datos sintético para el entrenamiento y la evaluación.

**Pasos:**

1.  **Configurar el entorno virtual (Realizado):**
    *   Usar `uv` para crear un entorno virtual.
    *   Instalar las dependencias necesarias: `torch`, `torchaudio`, `librosa`, `scikit-learn`, `numpy`.
    *   Crear un archivo `requirements.txt`.

2.  **Implementar la generación de datos:**
    *   **Dominio:**
        *   Crear modelos de dominio en `src/as_net/domain/models` para representar `AudioSignal`, `Spectrogram`, `BioacousticSource`, `Noise`, y `MixedSignal`.
        *   Definir un servicio de dominio en `src/as_net/domain/services` para la lógica de mezcla de señales.
    *   **Aplicación:**
        *   Crear un servicio de aplicación en `src/as_net/app/services` para orquestar la generación del conjunto de datos. Este servicio utilizará el servicio de dominio para mezclar las señales.
    *   **Adaptadores:**
        *   Implementar un adaptador de conducción (driving) en `src/as_net/adapters/driving` para iniciar el proceso de generación de datos (por ejemplo, un script de CLI).
        *   Implementar adaptadores conducidos (driven) en `src/as_net/adapters/driven` para leer los datos de audio de Xeno-Canto y Wytham Great Tit Song Dataset, y para guardar el conjunto de datos sintético en el directorio `data/processed`.

3.  **Generar el conjunto de datos:**
    *   Ejecutar el script de generación de datos para crear las mezclas de audio de 20 segundos con diferentes niveles de SNR (-5, 0, 5, 10, 15 dB). Cada muestra de 20 segundos contendrá entre 1 y 5 vocalizaciones de aves limpias colocadas al azar.
    *   Dividir el conjunto de datos en conjuntos de entrenamiento, validación y prueba.

## Fase 2: Implementación del Modelo AS-Net

**Objetivo:** Implementar la arquitectura de la red neuronal AS-Net.

**Pasos:**

1.  **Definir la arquitectura del modelo:**
    *   **Dominio:**
        *   Crear un modelo de dominio en `src/as_net/domain/models` para representar la arquitectura de AS-Net, incluyendo el codificador, el módulo de separación (TCN) y el decodificador.
    *   **Aplicación:**
        *   Crear un servicio de aplicación en `src/as_net/app/services` para construir el modelo AS-Net.

2.  **Implementar los componentes del modelo:**
    *   **Infraestructura:**
        *   Implementar los componentes de la red neuronal (capas convolucionales, TCN, etc.) en `src/as_net/infrastructure/models` utilizando PyTorch.

## Fase 3: Entrenamiento y Evaluación del Modelo

**Objetivo:** Entrenar el modelo AS-Net y evaluar su rendimiento.

**Pasos:**

1.  **Implementar el bucle de entrenamiento:**
    *   **Aplicación:**
        *   Crear un servicio de aplicación en `src/as_net/app/services` para gestionar el bucle de entrenamiento. Este servicio será responsable de cargar los datos, entrenar el modelo, y guardar los checkpoints del modelo en el directorio `models`.
    *   **Adaptadores:**
        *   Implementar un adaptador de conducción (driving) en `src/as_net/adapters/driving` para iniciar el proceso de entrenamiento (por ejemplo, un script de CLI).

2.  **Implementar la evaluación del modelo:**
    *   **Aplicación:**
        *   Crear un servicio de aplicación en `src/as_net/app/services` para evaluar el modelo. Este servicio calculará las métricas SDR y SIR, y también realizará la evaluación funcional utilizando el modelo BirdNET.
    *   **Adaptadores:**
        *   Implementar un adaptador de conducción (driving) en `src/as_net/adapters/driving` para iniciar el proceso de evaluación (por ejemplo, un script de CLI).
        *   Implementar un adaptador conducido (driven) en `src/as_net/adapters/driven` para cargar el modelo BirdNET pre-entrenado.

3.  **Realizar el análisis comparativo:**
    *   Implementar y entrenar los modelos de referencia (NMF, U-Net, Conv-TasNet).
    *   Evaluar los modelos de referencia utilizando el mismo protocolo de evaluación que para AS-Net.
    *   Comparar los resultados y generar la tabla de resultados.

## Fase 4: Documentación y Publicación

**Objetivo:** Documentar el proyecto y publicar los resultados.

**Pasos:**

1.  **Escribir la documentación:**
    *   Actualizar el `README.md` con instrucciones detalladas sobre cómo usar el código.
    *   Crear notebooks de Jupyter en el directorio `notebooks` para demostrar cómo usar el modelo y visualizar los resultados.

2.  **Publicar el código y los resultados:**
    *   Publicar el código fuente en un repositorio de GitHub.
    *   Publicar el conjunto de datos sintético y los modelos pre-entrenados.
    *   Escribir un artículo de investigación con los resultados del proyecto.

## Explicación del Ruido Rosa, Mezcla y Aleatoriedad

### Ruido Rosa

El ruido rosa es un tipo de señal de ruido que tiene una densidad espectral de potencia que es inversamente proporcional a la frecuencia. Esto significa que tiene la misma potencia en todas las octavas. En otras palabras, la potencia del ruido rosa disminuye a medida que aumenta la frecuencia. Esto lo diferencia del ruido blanco, que tiene una densidad espectral de potencia plana en todas las frecuencias.

En este proyecto, se utiliza ruido rosa para simular el ruido de fondo combinado de fuentes geofónicas (como el viento) y antropofónicas (como el tráfico distante). El ruido rosa es un sustituto más naturalista y perceptualmente más uniforme para muchos tipos de ruido ambiental que el ruido blanco.

El ruido rosa se genera utilizando un algoritmo que filtra el ruido blanco. El filtro aplica una pendiente de -3 dB por octava al espectro de potencia del ruido blanco.

### Mezcla de Señales

La mezcla de señales es el proceso de combinar dos o más señales de audio en una sola señal. En este proyecto, se mezclan las señales de biofonía (vocalizaciones de aves) con el ruido rosa para crear el conjunto de datos sintético.

La mezcla se realiza aditivamente, lo que significa que las muestras de las señales de audio se suman. Sin embargo, antes de la mezcla, se escala la señal de ruido para lograr una relación señal-ruido (SNR) específica.

La SNR es una medida de la potencia de la señal en relación con la potencia del ruido. Se expresa en decibelios (dB). Una SNR alta significa que la señal es mucho más fuerte que el ruido, mientras que una SNR baja significa que la señal es más débil que el ruido.

La fórmula para calcular el factor de escala para el ruido es la siguiente:

```
scaling_factor = sqrt(bioacoustic_power / (10^(snr / 10) * noise_power))
```

Donde:

*   `bioacoustic_power` es la potencia media de la señal de biofonía.
*   `noise_power` es la potencia media de la señal de ruido.
*   `snr` es la relación señal-ruido deseada en dB.

### Aleatoriedad

La aleatoriedad juega un papel importante en la generación del conjunto de datos para garantizar que el modelo no aprenda patrones espurios.

La aleatoriedad se introduce en los siguientes pasos:

*   **Selección de archivos de audio:** Se selecciona aleatoriamente un archivo de audio de biofonía del conjunto de datos de origen para cada llamada.
*   **Número de llamadas:** Se selecciona aleatoriamente un número de llamadas (entre 1 y 5) para agregar a cada muestra de 20 segundos.
*   **Posición de las llamadas:** Se selecciona aleatoriamente una posición dentro de la muestra de 20 segundos para insertar cada llamada.

Esta aleatoriedad ayuda a crear un conjunto de datos diverso y realista que es más probable que generalice a datos no vistos.

## Diagrama de la Metodología

```
+---------------------------------------+
|                                       |
|  Fase 1: Generación de Datos          |
|                                       |
+---------------------------------------+
|                                       |
|   +-------------------------------+   |
|   | Obtener Biofonía (Aves)       |   |
|   +-------------------------------+   |
|   | Generar Ruido (Rosa)          |   |
|   +-------------------------------+   |
|   | Mezclar a diferentes SNRs     |   |
|   +-------------------------------+   |
|   | Dividir en Train/Val/Test     |   |
|   +-------------------------------+   |
|                                       |
+------------------+--------------------+
                   |                     
                   v                     
+------------------+--------------------+
|                                       |
|  Fase 2: Implementación del Modelo    |
|                                       |
+---------------------------------------+
|                                       |
|   +-------------------------------+   |
|   | Definir Arquitectura AS-Net   |   |
|   +-------------------------------+   |
|   | Implementar en PyTorch        |   |
|   +-------------------------------+   |
|                                       |
+------------------+--------------------+
                   |                     
                   v                     
+------------------+--------------------+
|                                       |
|  Fase 3: Entrenamiento y Evaluación   |
|                                       |
+---------------------------------------+
|                                       |
|   +-------------------------------+   |
|   | Entrenar AS-Net               |   |
|   +-------------------------------+   |
|   | Evaluar (SDR, SIR)            |   |
|   +-------------------------------+   |
|   | Evaluar Funcionalmente (BirdNET)|   |
|   +-------------------------------+   |
|   | Entrenar y Evaluar Referencias|   |
|   +-------------------------------+   |
|   | Comparar Resultados           |   |
|   +-------------------------------+   |
|                                       |
+------------------+--------------------+
                   |                     
                   v                     
+------------------+--------------------+
|                                       |
|  Fase 4: Documentación y Publicación  |
|                                       |
+---------------------------------------+
|                                       |
|   +-------------------------------+   |
|   | Escribir Documentación        |   |
|   +-------------------------------+   |
|   | Publicar Código y Modelos     |   |
|   +-------------------------------+   |
|   | Escribir Paper                |   |
|   +-------------------------------+   |
|                                       |
+---------------------------------------+
```

## Diagrama de la Red AS-Net

```
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   Señal de Audio Mezclada (Forma de Onda 1D)                                                       |
|                                                                                                    |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------+-------------------------------------------------+
|                                                                                                    |
|   Codificador (Encoder)                                                                            |
|   - Capa Convolucional 1D (Kernel Grande, Stride Pequeño)                                          |
|   - Función de Activación (ReLU)                                                                   |
|                                                                                                    |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------+-------------------------------------------------+
|                                                                                                    |
|   Módulo de Separación (Separation Module)                                                         |
|   - Pila de Bloques de Red Convolucional Temporal (TCN)                                            |
|     - Cada bloque TCN contiene:                                                                    |
|       - Capa Convolucional 1D con Dilatación Creciente                                             |
|       - Normalización                                                                              |
|       - Función de Activación                                                                      |
|       - Conexión Residual                                                                          |
|                                                                                                    |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------+-------------------------------------------------+
|                                                                                                    |
|   Máscaras de Separación (Separation Masks)                                                        |
|   - Una máscara para cada fuente (Biofonía y Ruido)                                                |
|                                                                                                    |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------+-------------------------------------------------+
|                                                                                                    |
|   Multiplicación de Máscaras                                                                       |
|   - Multiplicar las máscaras con la salida del codificador                                         |
|                                                                                                    |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------+-------------------------------------------------+
|                                                                                                    |
|   Decodificador (Decoder)                                                                          |
|   - Capa Convolucional Transpuesta 1D                                                              |
|                                                                                                    |
+--------------------------------------------------+-------------------------------------------------+
                                                   |
                                                   v
+--------------------------------------------------+-------------------------------------------------+
|                                                                                                    |
|   Señales de Audio Separadas (Forma de Onda 1D)                                                    |
|   - Biofonía                                                                                       |
|   - Ruido                                                                                          |
|                                                                                                    |
+----------------------------------------------------------------------------------------------------+
```
