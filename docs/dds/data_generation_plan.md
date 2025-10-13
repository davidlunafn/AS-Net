# Plan de Implementación Detallado: Generación de Datos

Este documento describe el plan de implementación para la generación del conjunto de datos sintético para el proyecto AS-Net.

## 1. Acceso a los Datos

*   **Ubicación de los datos:** Los datos de audio sin procesar se encuentran en `/Volumes/SSD DL/osfstorage-archive`.
*   **Acceso a los datos:** Se implementará un adaptador conducido (driven) en `src/as_net/adapters/driven` para leer los archivos de audio de este directorio. Se debe tener en cuenta que esta ruta está fuera del directorio del proyecto, por lo que el código debe ser lo suficientemente flexible para manejar rutas de archivo absolutas.

## 2. Implementación de la Generación de Datos

*   **Dominio:**
    *   Crear modelos de dominio en `src/as_net/domain/models` para representar `AudioSignal`, `Spectrogram`, `BioacousticSource`, `Noise`, y `MixedSignal`.
    *   Definir un servicio de dominio en `src/as_net/domain/services` para la lógica de mezcla de señales. Este servicio tomará una lista de señales de audio de biofonía, una señal de ruido, y un nivel de SNR, y devolverá la señal mezclada.

*   **Aplicación:**
    *   Crear un servicio de aplicación en `src/as_net/app/services` para orquestar la generación del conjunto de datos. Este servicio será responsable de:
        *   Leer los archivos de audio de biofonía del directorio de datos sin procesar.
        *   Generar ruido rosa.
        *   Para cada muestra en el conjunto de datos:
            *   Crear una muestra de audio de 20 segundos.
            *   Seleccionar aleatoriamente entre 1 y 5 vocalizaciones de biofonía.
            *   Colocar las vocalizaciones en posiciones aleatorias dentro de la muestra de 20 segundos.
            *   Mezclar las vocalizaciones con el ruido rosa a los niveles de SNR especificados (-5, 0, 5, 10, 15 dB).
        *   Guardar las muestras generadas en el directorio `data/processed`.

*   **Adaptadores:**
    *   Implementar un adaptador de conducción (driving) en `src/as_net/adapters/driving` para iniciar el proceso de generación de datos (por ejemplo, un script de CLI). Este script tomará como argumentos la ruta al directorio de datos sin procesar y la ruta al directorio de salida.

## 3. Estructura del Conjunto de Datos Generado

El conjunto de datos generado se guardará en el directorio `data/processed` y tendrá la siguiente estructura:

```
data/
  processed/
    train/
      -5dB/
      0dB/
      5dB/
      10dB/
      15dB/
    val/
      -5dB/
      0dB/
      5dB/
      10dB/
      15dB/
    test/
      -5dB/
      0dB/
      5dB/
      10dB/
      15dB/
```

Cada subdirectorio contendrá los archivos de audio de 20 segundos generados.
