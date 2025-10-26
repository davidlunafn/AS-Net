# Plan de Implementación: Generación de Conjunto de Datos de Prueba con Lluvia Real

Este documento describe el plan para generar un conjunto de datos de prueba adicional para el proyecto AS-Net. El objetivo de este conjunto de datos es evaluar la capacidad de generalización del modelo AS-Net, que fue entrenado con ruido rosa sintético, a un escenario más realista utilizando grabaciones de lluvia real como fuente de ruido.

## 1. Requisitos

*   **Ruido Realista:** Utilizar grabaciones de lluvia real en lugar de ruido rosa sintético.
*   **Configurabilidad:** Permitir la configuración de las carpetas de entrada (para las grabaciones de lluvia y los cantos de aves) y la carpeta de salida para los datos generados.
*   **Múltiples Vocalizaciones:** Generar mezclas que contengan entre 1 y 5 vocalizaciones de aves, similar al conjunto de datos de entrenamiento.
*   **Guardado de Etiquetas:** Guardar las señales de biofonía limpias (etiquetas) junto con las mezclas y el ruido, para poder realizar una evaluación objetiva.
*   **Procesamiento de Audio de Lluvia:** Las grabaciones de lluvia tienen una duración de 1 minuto y deben ser divididas en 3 segmentos de 20 segundos cada una.
*   **Niveles de SNR:** Generar mezclas con los mismos niveles de SNR que el conjunto de datos de entrenamiento: -5, 0, 5, 10, y 15 dB.
*   **Consistencia de Frecuencia de Muestreo:** Detectar la frecuencia de muestreo de todas las grabaciones de entrada (lluvia y aves) y remuestrearlas a la frecuencia de muestreo utilizada durante el entrenamiento del modelo para asegurar la compatibilidad.

## 2. Implementación

La implementación se centrará en extender la funcionalidad existente de generación de datos.

*   **Servicio de Aplicación (`data_generation.py`):**
    *   Se modificará o se creará una nueva función en el servicio de generación de datos para orquestar este nuevo proceso.
    *   Esta función aceptará como parámetros las rutas a los directorios de entrada (aves y lluvia) y el directorio de salida.
    *   **Carga de Audio de Lluvia:** Implementar la lógica para leer los archivos de audio de lluvia desde la carpeta de entrada especificada.
    *   **División de Audio:** Para cada archivo de lluvia de 1 minuto, se dividirá en tres segmentos no superpuestos de 20 segundos.
    *   **Remuestreo:** Antes de cualquier procesamiento, se verificará la frecuencia de muestreo de cada archivo de audio (tanto de aves como de lluvia) y, si no coincide con la frecuencia de muestreo objetivo del proyecto (ej. 16kHz), se remuestreará utilizando `librosa.resample`.
    *   **Lógica de Mezcla:** Se reutilizará la lógica de mezcla existente del dominio para combinar los cantos de las aves con los segmentos de lluvia en los niveles de SNR especificados.
    *   **Guardado de Archivos:** El servicio guardará los archivos generados en la estructura de directorios especificada en la salida.

*   **Adaptador de Conducción (CLI):**
    *   Se creará un nuevo script de CLI o se modificará el existente para permitir al usuario iniciar este proceso de generación de datos de prueba.
    *   El script aceptará los siguientes argumentos:
        *   `--bird-calls-dir`: Directorio con las vocalizaciones de aves limpias.
        *   `--rain-audio-dir`: Directorio con las grabaciones de lluvia.
        *   `--output-dir`: Directorio donde se guardará el nuevo conjunto de datos.
        *   `--target-sr`: La frecuencia de muestreo a la que se deben convertir todos los audios.

## 3. Estructura del Conjunto de Datos de Salida

El conjunto de datos generado se guardará en la carpeta de salida especificada, siguiendo una estructura similar a la del conjunto de datos procesado principal, pero dentro de una carpeta `test_rain`:

```
<output_dir>/
  test_rain/
    -5dB/
      mixed/
      bio/
      noise/
    0dB/
      mixed/
      bio/
      noise/
    5dB/
      mixed/
      bio/
      noise/
    10dB/
      mixed/
      bio/
      noise/
    15dB/
      mixed/
      bio/
      noise/
```

*   `mixed/`: Contendrá los archivos de audio mezclados (ave + lluvia).
*   `bio/`: Contendrá las señales de biofonía limpias correspondientes a cada mezcla.
*   `noise/`: Contendrá los segmentos de ruido de lluvia correspondientes a cada mezcla.

Este plan permitirá la creación de un conjunto de datos de prueba robusto y realista para validar la generalización del modelo AS-Net.
