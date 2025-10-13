# Un Marco de Aprendizaje Profundo para la Separación de Biofonía de Paisajes Sonoros de Fuentes Mixtas: Un Estudio de Caso con Vocalizaciones Aviares Sintéticas

## 1\. Introducción: El Imperativo de la Separación de Fuentes en la Ecoacústica Computacional

### 1.1. El Paisaje Sonoro como Sistema de Información Ecológica

La ecoacústica se ha consolidado como una disciplina científica emergente que investiga los patrones y procesos ecológicos a través del análisis de los sonidos ambientales.<sup>1</sup> Este campo ofrece un paradigma no invasivo y altamente escalable para el monitoreo de ecosistemas, permitiendo la evaluación de la biodiversidad y la salud del hábitat a escalas espaciales y temporales sin precedentes.<sup>5</sup> La premisa fundamental de la ecoacústica es la "hipótesis del paisaje sonoro", que postula que el ambiente acústico es una amalgama de tres fuentes principales de sonido: la **biofonía**, que comprende los sonidos producidos por organismos vivos no humanos; la **geofonía**, que incluye los sonidos generados por procesos geofísicos como el viento, la lluvia o el movimiento del agua; y la **antropofonía**, que abarca los sonidos generados por actividades humanas.<sup>5</sup> La composición, diversidad y dinámica temporal de estas tres fuentes sonoras actúan como indicadores indirectos (proxies) de la estructura, función y salud de un ecosistema.<sup>1</sup>

El avance tecnológico y la reducción de costos de las Unidades de Grabación Autónoma (ARUs, por sus siglas en inglés) han catalizado una revolución en la recopilación de datos ecológicos, dando lugar a la era del Monitoreo Acústico Pasivo (PAM, por sus siglas en inglés) a gran escala.<sup>5</sup> Los investigadores ahora pueden desplegar redes de sensores que registran continuamente el ambiente acústico durante meses o incluso años, generando volúmenes de datos masivos que se miden en petabytes.<sup>5</sup> Esta avalancha de datos representa una oportunidad sin precedentes para la ecología y la conservación, pero simultáneamente presenta un desafío analítico formidable. La capacidad de recopilar datos ha superado con creces la capacidad de analizarlos de manera eficiente y extraer información ecológicamente relevante. Este desequilibrio ha creado un escenario que puede describirse como "rico en datos, pero pobre en información", donde vastos archivos de audio permanecen subutilizados debido a la falta de herramientas computacionales robustas y escalables para su procesamiento e interpretación.

### 1.2. El "Problema del Cóctel" Bioacústico

El principal obstáculo para la automatización del análisis de datos de PAM es el fenómeno del enmascaramiento de la señal, comúnmente conocido como el "problema del cóctel" bioacústico.<sup>12</sup> En los entornos naturales, las vocalizaciones de las especies de interés (la biofonía objetivo) rara vez ocurren en aislamiento. Por el contrario, están constantemente superpuestas y enmascaradas por una cacofonía de señales concurrentes, que incluyen los cantos de otras especies, el ruido del viento y la lluvia (geofonía), y el ruido de fondo del tráfico, la agricultura o la industria (antropofonía).<sup>14</sup> Este solapamiento acústico constituye el cuello de botella fundamental que limita la eficacia de los algoritmos de análisis automatizado.

La presencia de estas señales extrañas y de ruido de fondo degrada significativamente el rendimiento de las tareas bioacústicas esenciales. Los algoritmos diseñados para la detección y clasificación automática de especies, la estimación de la densidad poblacional a partir de la actividad vocal, el análisis del comportamiento a través de la sintaxis del canto y la localización de individuos en el espacio ven su precisión drásticamente reducida en condiciones de baja relación señal-ruido (SNR).<sup>16</sup> Esta degradación del rendimiento socava directamente la utilidad del PAM como herramienta para la investigación ecológica y la gestión de la conservación. Por lo tanto, la capacidad de separar computacionalmente una fuente de sonido objetivo de una grabación de fuentes mixtas no es simplemente una mejora incremental, sino una tecnología habilitadora crítica. La separación de fuentes eficaz funciona como un paso de preprocesamiento indispensable que promete mejorar la calidad de los datos de entrada para todos los análisis posteriores, desbloqueando así el verdadero potencial de los conjuntos de datos acústicos a gran escala.<sup>10</sup>

Es crucial reconocer que la definición de "ruido" en el dominio bioacústico es inherentemente dependiente del contexto, una distinción fundamental con respecto al procesamiento del habla humana. En las aplicaciones de mejora del habla, el "ruido" se define generalmente como cualquier señal que no sea la voz humana. En la bioacústica, sin embargo, la señal de "ruido" podría ser la vocalización de otra especie (biofonía no objetivo), el sonido del viento (geofonía) o el zumbido de una carretera cercana (antropofonía).<sup>15</sup> Si un investigador está estudiando los cantos de las aves, el coro de los insectos se considera ruido, y viceversa. Esta ambigüedad y la diversidad de las fuentes de interferencia exigen el desarrollo de modelos de separación de fuentes que sean altamente especializados y adaptables, lo que justifica la creación de nuevas arquitecturas en lugar de la simple reutilización de modelos preexistentes entrenados para el habla humana.

### 1.3. Planteamiento del Problema y Objetivos de la Investigación

El problema central que aborda esta propuesta es que los algoritmos de separación de fuentes existentes, muchos de los cuales fueron desarrollados y optimizados para el habla humana o la música, demuestran un rendimiento subóptimo cuando se aplican a señales bioacústicas. Las vocalizaciones de muchas especies, en particular las aves, poseen características espectro-temporales únicas, como modulaciones de frecuencia rápidas, estructuras armónicas complejas y un contenido de energía significativo en rangos de frecuencia mucho más altos que los del habla humana.<sup>13</sup> Los modelos existentes no están diseñados para manejar esta diversidad acústica, lo que resulta en una separación deficiente y la introducción de artefactos que pueden confundir los análisis posteriores.

En respuesta a este desafío, esta investigación se propone los siguientes objetivos:

- **Objetivo Principal:** Desarrollar y validar una nueva arquitectura de red neuronal profunda, diseñada específicamente para la tarea de separar la biofonía aviar del ruido de fondo de banda ancha y no biológico.
- **Objetivos Secundarios:**
  - Crear un conjunto de datos sintético, estandarizado y reproducible para el entrenamiento y la evaluación rigurosa de los modelos de separación de fuentes bioacústicas. Este conjunto de datos permitirá un control preciso sobre la relación señal-ruido (SNR) y proporcionará las señales de origen "limpias" necesarias para una evaluación cuantitativa.
  - Establecer un protocolo de evaluación de doble enfoque que mida no solo la calidad objetiva de la señal separada (utilizando métricas estándar como SDR y SIR), sino también la utilidad funcional de la separación en una tarea ecológica realista y posterior, como la detección automática de especies.
  - Realizar una evaluación comparativa exhaustiva del modelo propuesto frente a métodos establecidos del estado del arte para cuantificar su rendimiento, su contribución al campo y su superioridad demostrable.

### 1.4. Contribución y Alcance

Esta investigación contribuirá con un nuevo modelo de aprendizaje profundo de código abierto, diseñado y optimizado para la separación de fuentes bioacústicas. Al centrarse en un conjunto de datos sintético, el proyecto proporcionará una metodología robusta y reproducible que aborda directamente el desafío fundamental de la evaluación en la separación de fuentes: la ausencia de señales de origen de referencia (ground-truth) en las grabaciones de campo del mundo real.<sup>21</sup>

El componente de evaluación funcional, que vincula directamente el rendimiento de la separación con la eficacia de una tarea de análisis ecológico posterior, representa un avance metodológico significativo. Este enfoque va más allá de las métricas abstractas de calidad de la señal para cuantificar el beneficio práctico y tangible de la separación de fuentes para las aplicaciones de monitoreo de la biodiversidad. El alcance de este trabajo se limita deliberadamente a un escenario de una única fuente bioacústica (una especie de ave) y un modelo de ruido simplificado (ruido rosa) para establecer una prueba de concepto sólida y una línea de base rigurosa. Los resultados y la metodología desarrollados aquí sentarán las bases para futuras investigaciones sobre paisajes sonoros más complejos y de múltiples especies.

## 2\. Estado del Arte: Una Revisión Crítica de las Técnicas de Separación de Fuentes

Esta sección presenta una revisión exhaustiva de los métodos existentes para la separación de fuentes de audio, evaluando críticamente sus fortalezas y debilidades, particularmente en el contexto del análisis bioacústico. Esta revisión establece el fundamento teórico para el modelo propuesto e identifica las brechas de conocimiento que este trabajo pretende llenar, siguiendo una clara trayectoria evolutiva desde los métodos de procesamiento de señales clásicos hasta las arquitecturas de aprendizaje profundo de vanguardia.

### 2.1. Métodos Fundamentales de Procesamiento de Señales y Estadísticos

Los primeros enfoques para la reducción de ruido y la separación de fuentes se basaban en principios de procesamiento de señales y suposiciones estadísticas sobre la naturaleza de la señal y el ruido.

- **Sustracción Espectral:** Esta es una de las técnicas más antiguas y computacionalmente más simples. Su principio de funcionamiento consiste en estimar el espectro de potencia del ruido a partir de segmentos de la grabación donde se asume que la señal de interés está ausente (pausas o silencio). Esta estimación del espectro de ruido se promedia y luego se resta del espectro de la señal ruidosa en cada trama de análisis.<sup>22</sup>
  - _Limitaciones:_ A pesar de su simplicidad, la sustracción espectral tiene limitaciones significativas. Es muy propensa a introducir un tipo de distorsión conocido como "ruido musical", que consiste en artefactos tonales aleatorios que pueden ser perceptualmente más molestos que el ruido original.<sup>16</sup> Además, su rendimiento se degrada drásticamente en presencia de ruido no estacionario, es decir, ruido cuyas características estadísticas cambian con el tiempo, lo cual es la norma en los entornos ecológicos. Los estudios han demostrado que su eficacia para los sonidos de animales es inconsistente, en parte porque los algoritmos a menudo se entrenan y optimizan principalmente para el habla humana.<sup>26</sup>
- **Técnicas de Separación Ciega de Fuentes (BSS):** Este grupo de métodos intenta separar las fuentes basándose en suposiciones estadísticas sobre las señales de origen, sin conocimiento previo del proceso de mezcla.
  - **Análisis de Componentes Independientes (ICA):** El ICA asume que las señales de origen son estadísticamente independientes entre sí y no gaussianas. El algoritmo busca una matriz de desmezcla lineal que, al ser aplicada a las señales observadas, maximiza la independencia estadística de las señales de salida estimadas.<sup>27</sup>
  - **Factorización de Matrices No Negativas (NMF):** La NMF opera sobre una representación de tiempo-frecuencia no negativa de la señal (como un espectrograma de potencia) y la descompone en el producto de dos matrices de bajo rango: una matriz de "bases" espectrales y una matriz de "activaciones" temporales. La idea es que cada fuente puede ser representada por un conjunto de bases espectrales y sus correspondientes patrones de activación a lo largo del tiempo.<sup>31</sup>
  - _Limitaciones:_ Tanto el ICA como la NMF enfrentan serios desafíos en el contexto del monitoreo acústico pasivo. Su rendimiento es óptimo en escenarios "determinados" o "sobredeterminados", donde el número de sensores (micrófonos) es mayor o igual al número de fuentes. Sin embargo, la gran mayoría de las grabaciones de PAM son monocanal (un solo micrófono), lo que crea un problema "infradeterminado" matemáticamente mal planteado.<sup>14</sup> En estas condiciones, el rendimiento de ICA y NMF disminuye significativamente, especialmente con bajos niveles de SNR y cuando las señales de las fuentes se solapan considerablemente en el tiempo y la frecuencia, como es común en las mezclas bioacústicas.<sup>14</sup>

### 2.2. La Revolución del Aprendizaje Profundo en la Separación de Fuentes de Audio

En la última década, se ha producido un cambio de paradigma desde los métodos basados en características diseñadas a mano hacia modelos de aprendizaje profundo que aprenden representaciones de extremo a extremo directamente a partir de los datos.<sup>10</sup> Estos modelos han superado consistentemente a los métodos tradicionales en una amplia gama de tareas de separación de audio, demostrando una robustez y un rendimiento muy superiores, especialmente en escenarios acústicos complejos.<sup>35</sup>

#### 2.2.1. Modelos en el Dominio de Tiempo-Frecuencia (T-F)

Estos modelos operan sobre representaciones espectro-temporales del audio, como el espectrograma, y su objetivo es, por lo general, estimar una "máscara" que, al ser aplicada multiplicativamente al espectrograma de la mezcla, aísla el espectrograma de la fuente de interés.

- **Arquitectura U-Net:** Originaria del campo de la segmentación de imágenes médicas, la U-Net es una red neuronal totalmente convolucional que ha sido adaptada con gran éxito para la separación de fuentes de audio. Su arquitectura simétrica de codificador-decodificador, con "conexiones de salto" (skip connections) que conectan capas del codificador con sus contrapartes en el decodificador, es excepcionalmente eficaz para aprender funciones de enmascaramiento complejas. Estas conexiones permiten que la información de características de bajo nivel (detalles finos) del codificador se propague directamente al decodificador, ayudando a reconstruir con mayor precisión la estructura espectral de las fuentes separadas.<sup>10</sup>
- **BioCPPNet:** Un ejemplo destacado de la adaptación de la arquitectura U-Net al dominio bioacústico es la Bioacoustic Cocktail Party Problem Network (BioCPPNet). Este modelo se distingue por ser una arquitectura ligera y modular que opera directamente sobre la forma de onda de audio sin procesar. Logra esto mediante la incorporación de un codificador frontal (que puede ser aprendido o diseñado a mano) que transforma la forma de onda en una representación 2D, la cual es luego procesada por un núcleo U-Net 2D. Este enfoque híbrido demuestra el potencial de adaptar arquitecturas bien establecidas para las especificidades de las señales bioacústicas, como la diversidad de frecuencias y la necesidad de eficiencia computacional.<sup>13</sup>
- **Mecanismos de Atención:** Investigaciones más recientes han comenzado a integrar mecanismos de atención en las arquitecturas U-Net. La atención permite al modelo ponderar dinámicamente la importancia de diferentes características en el tiempo y la frecuencia, lo que le ayuda a modelar mejor las dependencias a largo plazo y a centrarse en las partes más informativas de la señal, mejorando así el rendimiento de la separación.<sup>40</sup>

#### 2.2.2. Modelos en el Dominio del Tiempo (de Extremo a Extremo)

Estos modelos representan la vanguardia actual en la separación de fuentes de audio. Operan directamente sobre la forma de onda de audio sin procesar, lo que les permite modelar conjuntamente la magnitud y la fase de la señal. Los métodos en el dominio T-F a menudo descartan la información de fase o utilizan la fase de la mezcla original para la reconstrucción, lo cual es una aproximación subóptima que puede introducir artefactos. Los modelos en el dominio del tiempo superan esta limitación.

- **Conv-TasNet:** Esta arquitectura fue un hito en la separación de fuentes de habla. Es una estructura 1D totalmente convolucional que consta de tres etapas: un codificador convolucional 1D aprendido que transforma segmentos cortos de la forma de onda en un espacio de características; un módulo de separación que utiliza una pila de redes convolucionales temporales (TCNs) con convoluciones dilatadas para capturar dependencias temporales a largo plazo y estimar las máscaras para cada fuente; y un decodificador convolucional 1D aprendido que reconstruye las formas de onda de las fuentes separadas.<sup>44</sup> Conv-TasNet demostró un rendimiento que superaba incluso a las máscaras de tiempo-frecuencia ideales, estableciendo un nuevo estado del arte.<sup>45</sup>
- **Adaptaciones para Bioacústica (LiTasNeT):** El modelo LiTasNeT (Lite TasNet) es un excelente ejemplo de la adaptación de Conv-TasNet para el dominio bioacústico, específicamente para la separación de sonidos de aves. Los autores introdujeron un esquema de compartición de parámetros dentro de los bloques convolucionales para reducir drásticamente el tamaño del modelo y su complejidad computacional. Esto da como resultado un modelo "ligero" que es adecuado para aplicaciones en tiempo real en dispositivos con recursos limitados, como las ARUs en el campo.<sup>48</sup> Este trabajo subraya una tendencia crítica en la investigación aplicada: la necesidad de equilibrar el rendimiento del modelo con la eficiencia computacional para la implementación en el mundo real. Existe una tensión inherente entre la complejidad del modelo, que a menudo se correlaciona con un mayor rendimiento, y la viabilidad de su despliegue en hardware de campo con restricciones de energía y procesamiento. La investigación que no solo busca mejorar las métricas de rendimiento, sino que también considera la eficiencia (tamaño del modelo, velocidad de inferencia), es de particular importancia para la ecoacústica práctica.

### 2.3. Identificación de Brechas de Conocimiento y Oportunidades

A pesar de los notables avances, persisten varias brechas y desafíos clave en la aplicación del aprendizaje profundo a la separación de fuentes bioacústicas.

- **Escasez de Datos Etiquetados:** El mayor obstáculo para el aprendizaje profundo supervisado en bioacústica es la crónica falta de conjuntos de datos grandes, diversos y de alta calidad con etiquetas precisas. La anotación manual de grabaciones de campo es un proceso extremadamente lento, costoso y que requiere de una gran experiencia taxonómica. Esta escasez de datos limita severamente la capacidad de entrenar modelos robustos y generalizables.<sup>50</sup> Esta limitación proporciona una fuerte justificación para el uso de datos sintéticos, que ofrecen un entorno controlado y escalable para el desarrollo y la evaluación comparativa rigurosa de modelos.
- **Desajuste de Dominio:** Existe un desajuste fundamental entre el dominio del habla humana y el dominio bioacústico. Los modelos entrenados en habla humana, que típicamente se muestrea a 8 o 16 kHz y tiene un rango de frecuencia relativamente limitado, no están optimizados para las características de las vocalizaciones animales. Muchos sonidos de aves, por ejemplo, contienen componentes de alta frecuencia y modulaciones rápidas que se pierden o se representan de forma inadecuada a bajas tasas de muestreo.<sup>12</sup> Esto exige modificaciones arquitectónicas específicas para manejar tasas de muestreo más altas y patrones espectro-temporales diferentes.
- **Deficiencias en la Evaluación:** El campo depende en gran medida de métricas objetivas de calidad de la señal, como el SDR y el SIR. Si bien estas métricas son útiles, no hay garantía de que una mejora en el SDR se traduzca en un beneficio tangible para las tareas ecológicas posteriores.<sup>21</sup> Un modelo podría lograr un SDR alto al eliminar ruido, pero en el proceso podría distorsionar características acústicas sutiles que son cruciales para que un clasificador de especies identifique correctamente la vocalización. Esta desconexión entre la fidelidad a nivel de señal y la utilidad funcional es una brecha metodológica crítica que esta propuesta busca abordar explícitamente.

**Tabla 1: Análisis Comparativo de Modelos de Separación de Fuentes del Estado del Arte**

| Método/Arquitectura | Dominio Operativo | Principio Fundamental | Fortalezas | Limitaciones Clave para la Bioacústica |
| --- | --- | --- | --- | --- |
| **Sustracción Espectral** | Tiempo-Frecuencia (T-F) | Estimación y resta del espectro de ruido. | Computacionalmente simple y rápido. | Introduce "ruido musical"; bajo rendimiento con ruido no estacionario; optimizado para el habla humana.<sup>26</sup> |
| --- | --- | --- | --- | --- |
| **NMF** | Tiempo-Frecuencia (T-F) | Descomposición en bases espectrales y activaciones temporales. | No supervisado; interpretabilidad de las bases. | Requiere múltiples canales para un buen rendimiento; bajo rendimiento en escenarios monocanal infradeterminados y con solapamiento.<sup>14</sup> |
| --- | --- | --- | --- | --- |
| **ICA** | Tiempo | Maximización de la independencia estadística de las fuentes. | No supervisado; bien fundamentado matemáticamente. | Requiere múltiples canales; asume independencia lineal, lo que a menudo no se cumple en mezclas convolutivas del mundo real.<sup>29</sup> |
| --- | --- | --- | --- | --- |
| **U-Net** | Tiempo-Frecuencia (T-F) | Aprendizaje de una función de enmascaramiento en el dominio T-F mediante una red convolucional codificador-decodificador. | Excelente rendimiento en la estimación de máscaras; captura de características multi-escala. | Descarta la información de fase; la resolución T-F implica un compromiso entre la resolución temporal y la de frecuencia.<sup>13</sup> |
| --- | --- | --- | --- | --- |
| **BioCPPNet** | Híbrido (Onda -> T-F -> Onda) | Utiliza un núcleo U-Net sobre una representación aprendida de la forma de onda. | Modular; diseñado para ser ligero y específico para bioacústica; opera sobre la forma de onda de entrada. | Aún depende de una representación T-F interna; la calidad de la reconstrucción está limitada por el decodificador.<sup>13</sup> |
| --- | --- | --- | --- | --- |
| **Conv-TasNet** | Tiempo | Mapeo de extremo a extremo de la forma de onda de la mezcla a las formas de onda de las fuentes mediante convoluciones 1D. | Modela conjuntamente magnitud y fase; captura dependencias temporales a largo plazo; alto rendimiento. | Alta complejidad computacional; no optimizado para las altas frecuencias y patrones de las señales bioacústicas.<sup>45</sup> |
| --- | --- | --- | --- | --- |

## 3\. Marco de Investigación y Metodología Propuestos

Esta sección detalla el enfoque metodológico que se seguirá para abordar los objetivos de la investigación. Se describe un protocolo riguroso y reproducible para la generación del conjunto de datos, el diseño de la arquitectura del modelo de red neuronal, y el procedimiento de entrenamiento.

### 3.1. Protocolo de Generación de Paisajes Sonoros Sintéticos

La utilización de un conjunto de datos sintético es una piedra angular de esta metodología. Permite un control experimental preciso sobre las condiciones de la señal y el ruido, y, lo que es más importante, proporciona las señales de origen limpias (ground-truth) que son indispensables para una evaluación cuantitativa y objetiva del rendimiento de la separación, un requisito que es casi imposible de cumplir con grabaciones de campo.<sup>21</sup>

#### 3.1.1. Obtención de Biofonía de Alta Fidelidad

- **Especie Modelo:** Se ha seleccionado el carbonero común (_Parus major_) como la especie modelo para esta investigación. Esta elección se basa en varias consideraciones clave: su canto está excepcionalmente bien estudiado en la literatura ornitológica; su estructura vocal es compleja, con una sintaxis que incluye notas, frases y estrofas, pero a la vez es lo suficientemente estructurada para ser modelable; y, fundamentalmente, existen grandes repositorios públicos de grabaciones de alta calidad, limpias y anotadas.<sup>53</sup>
- **Fuentes de Datos:** Las vocalizaciones limpias de individuos únicos de carbonero común se obtendrán principalmente del **Wytham Great Tit Song Dataset**. Este conjunto de datos es un recurso sin precedentes, que contiene más de un millón de unidades acústicas individuales de más de 400 aves, con metadatos detallados sobre la identidad del individuo, su historia de vida y el contexto de la grabación.<sup>53</sup> Para aumentar aún más la diversidad de ejemplares y asegurar la cobertura de la variabilidad intraespecífica en los dialectos y tipos de canto, esta colección se complementará con grabaciones de alta calidad (calidad "A") de la base de datos de ciencia ciudadana **Xeno-Canto**.<sup>57</sup> Se seleccionarán únicamente grabaciones que contengan un solo individuo vocalizando y que tengan un ruido de fondo mínimo.

#### 3.1.2. Modelado de Geofonía y Antropofonía

- **Modelo de Ruido:** Para simular el ruido de fondo combinado de fuentes geofónicas (como el viento) y antropofónicas (como el tráfico distante), se utilizará ruido rosa. A diferencia del ruido blanco, que tiene una densidad espectral de potencia plana en todas las frecuencias, el ruido rosa tiene una densidad espectral de potencia que es inversamente proporcional a la frecuencia (). Esto significa que tiene más energía en las frecuencias bajas que en las altas, lo que lo convierte en un sustituto mucho más naturalista y perceptualmente más uniforme para muchos tipos de ruido ambiental. El uso de ruido rosa proporciona una fuente de interferencia estandarizada, reproducible y más realista que el ruido blanco.

#### 3.1.3. Creación de Mezclas y Estructura del Conjunto de Datos

- **Procedimiento:** El proceso de creación de mezclas sintéticas seguirá un protocolo automatizado. Cada muestra de audio sintético tendrá una duración de 20 segundos. Dentro de cada muestra de 20 segundos, se insertarán aleatoriamente entre 1 y 5 vocalizaciones limpias de carbonero común en posiciones aleatorias. A continuación, se generará un segmento de ruido rosa de 20 segundos. Finalmente, las vocalizaciones y el ruido se mezclarán aditivamente.
- **Control de la Relación Señal-Ruido (SNR):** Un aspecto crucial del diseño experimental es la capacidad de evaluar la robustez del modelo en diferentes condiciones de ruido. Para ello, las mezclas se generarán en un rango controlado de relaciones señal-ruido (SNR). Específicamente, se crearán subconjuntos de datos para los siguientes niveles de SNR: **\-5 dB** (señal muy enmascarada por el ruido), **0 dB** (señal y ruido con igual potencia), **5 dB**, **10 dB** y **15 dB** (señal relativamente limpia). Este rango simula un espectro realista de condiciones de grabación, desde entornos muy ruidosos hasta condiciones casi ideales, lo que permitirá un análisis detallado del rendimiento del modelo en función del nivel de interferencia.<sup>14</sup>
- **División del Conjunto de Datos:** El conjunto de datos completo, que consistirá en las mezclas sintéticas junto con sus componentes originales correspondientes (la biofonía limpia y el ruido rosa), se dividirá en tres subconjuntos mutuamente excluyentes: entrenamiento (80%), validación (10%) y prueba (10%). Se garantizará que las vocalizaciones de origen utilizadas para crear las mezclas en un subconjunto no aparezcan en los otros, para evitar cualquier fuga de datos y asegurar una evaluación imparcial de la capacidad de generalización del modelo.

### 3.2. Una Nueva Arquitectura de Red Neuronal para la Separación de Biofonía

Se propone una nueva arquitectura de red neuronal de extremo a extremo, denominada **Red de Separación Aviar (AS-Net)**. El diseño de AS-Net se inspira en los éxitos de arquitecturas de dominio temporal como Conv-TasNet <sup>45</sup> y en las modificaciones específicas para bioacústica vistas en modelos como BioCPPNet <sup>13</sup> y LiTasNeT.<sup>48</sup> El objetivo es crear un modelo que aproveche el poder de las redes convolucionales en el dominio del tiempo, pero con modificaciones estructurales para optimizarlo para las características únicas del canto de las aves.

- **Plano Arquitectónico:**
  - **Codificador/Decodificador:** La arquitectura empleará un codificador y un decodificador convolucionales 1D, similares a los de Conv-TasNet. Sin embargo, se realizarán modificaciones clave: el codificador utilizará un tamaño de zancada (stride) más pequeño y un tamaño de núcleo (kernel) más grande en comparación con los modelos de habla estándar. Un stride más pequeño aumenta la resolución temporal de la representación interna, lo cual es crucial para capturar las rápidas modulaciones de frecuencia y los breves elementos (notas) del canto de las aves. Un kernel más grande permite al codificador capturar patrones acústicos en ventanas de tiempo más largas en la primera capa. Estas modificaciones están diseñadas específicamente para abordar el desajuste de dominio entre el habla humana y el canto de las aves.<sup>13</sup>
  - **Módulo de Separación:** El núcleo del modelo será un módulo de separación basado en una pila de bloques de Redes Convolucionales Temporales (TCN). Cada bloque TCN consistirá en una serie de capas convolucionales 1D con factores de dilatación que aumentan exponencialmente. Esta estructura de dilatación permite que el campo receptivo de la red crezca exponencialmente con la profundidad, lo que le permite modelar dependencias temporales a muy largo plazo de manera eficiente. Esto es fundamental para capturar la estructura sintáctica del canto de las aves, como la relación entre notas para formar frases y la repetición de frases para formar estrofas, tal como se describe en la literatura ornitológica.<sup>55</sup>
- **Justificación:** Se elige un enfoque en el dominio del tiempo para preservar la información de fase, que a menudo se descarta en los métodos basados en espectrogramas. La fase puede contener información crucial para la reconstrucción de alta fidelidad de las complejas estructuras armónicas y los transitorios rápidos presentes en las vocalizaciones aviares. La arquitectura AS-Net busca combinar el poder de modelado de largo alcance de las TCN con un front-end optimizado para las características espectro-temporales del canto de las aves.

### 3.3. Entrenamiento e Implementación del Modelo

- **Función de Pérdida:** El modelo se entrenará utilizando el paradigma de **Entrenamiento Invariante a la Permutación (PIT)**.<sup>13</sup> Dado que el orden de las fuentes de salida de un modelo de separación es arbitrario, el PIT resuelve el problema de la correspondencia de etiquetas comparando cada salida permutada con las etiquetas de referencia y calculando la pérdida en la permutación que produce el error más bajo. La función de pérdida principal será la **Relación Señal-Distorsión Invariante a la Escala (SI-SDR)** negativa, una métrica estándar y robusta para evaluar el rendimiento de la separación en el dominio del tiempo.<sup>20</sup> Adicionalmente, se podría incluir un término de pérdida auxiliar, como la pérdida L1 en los espectrogramas, para fomentar una mejor reconstrucción de la estructura armónica de la señal, como se ha demostrado que es beneficioso en trabajos anteriores.<sup>12</sup>
- **Régimen de Entrenamiento:** El modelo se entrenará utilizando el optimizador Adam, con un programa de reducción de la tasa de aprendizaje que disminuye la tasa de aprendizaje cuando la pérdida de validación se estanca. Todo el código se implementará en Python utilizando el framework de aprendizaje profundo PyTorch. Se utilizarán bibliotecas de código abierto estándar en la comunidad, como librosa para el preprocesamiento de audio y scikit-learn para la implementación de los modelos de referencia.<sup>61</sup> El código fuente completo del modelo, los scripts de generación de datos y los protocolos de evaluación se harán públicos para garantizar la reproducibilidad y fomentar la investigación futura.

**Tabla 2: Estructura del Conjunto de Datos de Paisajes Sonoros Sintéticos**

| Nivel de SNR (dB) | Nº de Vocalizaciones Únicas de _P. major_ | Nº de Mezclas de Entrenamiento | Nº de Mezclas de Validación | Nº de Mezclas de Prueba | Duración Total (horas) |
| --- | --- | --- | --- | --- | --- |
| **\-5** | 1,000 | 8,000 | 1,000 | 1,000 | ~55 |
| --- | --- | --- | --- | --- | --- |
| **0** | 1,000 | 8,000 | 1,000 | 1,000 | ~55 |
| --- | --- | --- | --- | --- | --- |
| **5** | 1,000 | 8,000 | 1,000 | 1,000 | ~55 |
| --- | --- | --- | --- | --- | --- |
| **10** | 1,000 | 8,000 | 1,000 | 1,000 | ~55 |
| --- | --- | --- | --- | --- | --- |
| **15** | 1,000 | 8,000 | 1,000 | 1,000 | ~55 |
| --- | --- | --- | --- | --- | --- |
| **Total** | 5,000 (únicas en total) | 40,000 | 5,000 | 5,000 | ~277 |
| --- | --- | --- | --- | --- | --- |

## 4\. Un Protocolo de Evaluación de Doble Enfoque

Para evaluar de manera exhaustiva el rendimiento del modelo propuesto, se ha diseñado un marco de evaluación de doble enfoque. Este protocolo va más allá de las métricas de ingeniería de señales tradicionales para medir el impacto práctico del modelo en una aplicación ecológica realista. Este enfoque dual permite una comprensión más holística y matizada del rendimiento, evaluando tanto la fidelidad de la reconstrucción de la señal como su utilidad funcional.

### 4.1. Evaluación del Rendimiento Objetivo con Métricas Estándar

Esta primera fase de la evaluación se centra en cuantificar la calidad de la señal de audio separada en comparación con la señal de origen limpia original (ground-truth). Para ello, se utilizarán métricas estándar de la literatura de evaluación de separación de fuentes de audio (BSS Eval).

- **Métricas:**
  - **Relación Señal-Distorsión (SDR):** Esta es la métrica más común para evaluar la calidad general de la separación. Mide la relación de potencia entre la señal objetivo reconstruida y el error de distorsión, que incluye todos los tipos de artefactos: interferencia residual de otras fuentes, ruido añadido por el algoritmo y distorsión de la propia señal objetivo. Un SDR más alto indica una mejor calidad de separación general.<sup>21</sup>
  - **Relación Señal-Interferencia (SIR):** Esta métrica se centra específicamente en cuantificar el grado de supresión de las otras fuentes de la mezcla. Mide la relación de potencia entre la señal objetivo y la interferencia residual de las otras fuentes en la salida separada. Un SIR alto indica que el modelo ha eliminado eficazmente las fuentes no deseadas.<sup>21</sup>
- **Procedimiento:** Las métricas SDR y SIR se calcularán para cada mezcla en el conjunto de datos de prueba. Los resultados se agruparán y promediarán para cada nivel de SNR por separado. Esto permitirá crear un perfil de rendimiento detallado que muestre cómo se comporta el modelo en un rango de condiciones de ruido, desde muy adversas hasta favorables. El uso de estas métricas estándar es crucial, ya que permite la comparación directa de los resultados obtenidos con los publicados en la vasta literatura sobre separación de fuentes, tanto en el habla como en la música y la bioacústica.<sup>65</sup>

### 4.2. Evaluación del Rendimiento Funcional mediante la Eficacia de una Tarea Posterior

Esta segunda fase de la evaluación es una contribución metodológica clave de esta propuesta. Su objetivo es trascender las métricas abstractas de calidad de la señal para medir la utilidad práctica del proceso de separación. La hipótesis central es que una mejor separación de fuentes debería conducir a un mejor rendimiento en una tarea de análisis automatizado posterior que dependa de la calidad de la señal de entrada.

- **Tarea Posterior:** La tarea seleccionada es la detección automática de vocalizaciones de carbonero común en clips de audio cortos (3 segundos). Esta es una tarea fundamental y omnipresente en el monitoreo bioacústico.
- **Modelo de Detección:** Para realizar esta tarea, se utilizará un modelo **BirdNET** pre-entrenado y disponible públicamente. BirdNET es un clasificador de sonidos de aves de última generación, basado en aprendizaje profundo, que ha sido entrenado con un conjunto de datos masivo que abarca miles de especies.<sup>67</sup> Es una herramienta ampliamente adoptada y respetada en la comunidad de investigación en ecología y conservación, lo que la convierte en una elección ideal y realista para simular un flujo de trabajo de análisis del mundo real.
- **Conjuntos de Datos Experimentales:** El detector BirdNET se ejecutará en cuatro conjuntos de datos paralelos, todos derivados del conjunto de prueba, para permitir una comparación controlada:
  - **Vocalizaciones Limpias (Límite Superior):** Las señales de carbonero común originales y sin ruido. El rendimiento del detector en estos datos establece el límite superior teórico o el "techo" de rendimiento, representando un escenario ideal sin interferencias.
  - **Mezclas Ruidosas (Línea de Base):** Las mezclas sintéticas originales (vocalización + ruido rosa). El rendimiento en estos datos establece la línea de base o el "suelo" de rendimiento, representando el escenario sin ningún preprocesamiento de separación de fuentes.
  - **Señales Separadas por AS-Net:** Las señales de salida producidas por el modelo propuesto (AS-Net) cuando se le alimenta con las mezclas ruidosas.
  - **Señales Separadas por Modelos de Referencia:** Las señales de salida producidas por los modelos del estado del arte (NMF, U-Net, Conv-TasNet) que se utilizarán para la evaluación comparativa (descritos en la Sección 5).
- **Métricas de Evaluación:** La salida de BirdNET para cada clip de 3 segundos es una puntuación de confianza (entre 0 y 1) que indica la probabilidad de que la especie objetivo esté presente. Al aplicar un umbral de confianza (por ejemplo, 0.5, como se recomienda en la literatura <sup>72</sup>), estas puntuaciones se pueden convertir en predicciones binarias (presencia/ausencia). Estas predicciones se compararán con la verdad fundamental (la presencia o ausencia real de una vocalización en cada clip) para calcular las siguientes métricas de clasificación estándar:
  - **Precisión:** La proporción de detecciones positivas que fueron correctas (verdaderos positivos / (verdaderos positivos + falsos positivos)).
  - **Recall (Sensibilidad):** La proporción de vocalizaciones reales que fueron detectadas correctamente (verdaderos positivos / (verdaderos positivos + falsos negativos)).
  - **Puntuación F1:** La media armónica de la precisión y el recall (), que proporciona una única medida equilibrada del rendimiento de la detección.

Este protocolo de evaluación dual permite capturar una imagen multidimensional del rendimiento de un modelo. Es teóricamente posible que un modelo logre un SDR alto, por ejemplo, mediante una supresión de ruido muy agresiva que alise la señal, pero que en el proceso elimine características acústicas sutiles (como armónicos de alta frecuencia o transitorios rápidos) que son vitales para que el detector posterior funcione correctamente. En tal caso, el modelo tendría un SDR alto pero una puntuación F1 baja. Por el contrario, otro modelo podría tener un SDR ligeramente más bajo debido a la presencia de artefactos menores, pero podría preservar las características discriminatorias clave de la vocalización, lo que resultaría en una puntuación F1 más alta. Este protocolo está diseñado precisamente para capturar estas compensaciones cruciales entre la fidelidad de la señal y la preservación de la información.

Además, este marco de evaluación proporciona una forma de cuantificar el "costo del ruido" en términos funcionales. Al comparar la puntuación F1 obtenida en los datos limpios con la obtenida en las mezclas ruidosas, se puede establecer una medida precisa y cuantitativa de cuánto degrada un nivel de SNR específico el rendimiento de un detector de última generación. Esto, a su vez, permite expresar el beneficio del modelo de separación propuesto en términos prácticos e intuitivos. En lugar de simplemente informar de una mejora de X dB en el SDR, se podrá afirmar, por ejemplo, que "a 0 dB de SNR, el ruido reduce el rendimiento de detección de BirdNET en un 50%, y la aplicación de nuestro modelo AS-Net como paso de preprocesamiento recupera el 80% de ese rendimiento perdido". Esta forma de presentar los resultados es mucho más impactante y comprensible para los ecólogos, biólogos de la conservación y gestores de la vida silvestre, que son los usuarios finales de estas tecnologías.

## 5\. Análisis Comparativo y Benchmarking

Para validar la novedad y la eficacia de la red AS-Net propuesta, su rendimiento se comparará rigurosamente con una selección de modelos de referencia. Estos modelos han sido cuidadosamente elegidos para representar diferentes épocas y enfoques metodológicos en la separación de fuentes, lo que permitirá situar el rendimiento de AS-Net en el contexto del estado del arte y cuantificar su avance.

### 5.1. Selección de Modelos de Referencia

- **Referencia 1 (Método Estadístico): Factorización de Matrices No Negativas (NMF).** Se seleccionará la NMF como representante de los métodos estadísticos tradicionales y no supervisados. Es un punto de referencia común en la literatura de separación de audio y servirá para establecer una línea de base que demuestre las ganancias de rendimiento obtenidas por los enfoques de aprendizaje profundo.<sup>31</sup> Se implementará una versión monocanal de NMF que opera sobre el espectrograma de potencia de la mezcla.
- **Referencia 2 (Aprendizaje Profundo en Dominio T-F): U-Net Estándar.** Se implementará y entrenará una arquitectura U-Net estándar, similar a las que se han utilizado con éxito en la separación de fuentes musicales y otras tareas de audio.<sup>37</sup> Este modelo operará sobre el espectrograma de magnitud de la mezcla y aprenderá a predecir una máscara para la fuente de biofonía. Esta referencia permitirá una comparación directa del enfoque de dominio temporal propuesto (AS-Net) con una alternativa potente y ampliamente utilizada en el dominio de la frecuencia.
- **Referencia 3 (Aprendizaje Profundo en Dominio Temporal): Conv-TasNet.** Se implementará y entrenará la arquitectura original de Conv-TasNet.<sup>45</sup> Este modelo representa el estado del arte en la separación de habla en el dominio del tiempo y es el ancestro conceptual más directo de la arquitectura AS-Net propuesta. La comparación con Conv-TasNet es crucial, ya que permitirá medir directamente la mejora obtenida gracias a las modificaciones arquitectónicas específicas para bioacústica introducidas en AS-Net (por ejemplo, los cambios en el codificador y la configuración del módulo de separación).

### 5.2. Comparación del Rendimiento e Interpretación

Todos los modelos, incluyendo la AS-Net propuesta y los tres modelos de referencia, se entrenarán y evaluarán bajo condiciones idénticas. Se utilizará el mismo conjunto de datos sintético, los mismos protocolos de división de datos (entrenamiento, validación, prueba) y las mismas funciones de pérdida y optimizadores (cuando sea aplicable) para garantizar una comparación justa y rigurosa.

Se generará una tabla de resultados exhaustiva que comparará el rendimiento de todos los modelos en todos los niveles de SNR. Esta comparación se basará en el conjunto completo de métricas definidas en el protocolo de evaluación dual (Sección 4). El análisis de los resultados se centrará en varios aspectos clave:

- **Superioridad General del Rendimiento:** El análisis determinará si la red AS-Net propuesta supera consistentemente a los modelos de referencia en las métricas objetivas (SDR, SIR) y, lo que es más importante, en la métrica funcional (puntuación F1 de la tarea posterior).
- **Robustez al Ruido:** Se evaluará cómo se degrada el rendimiento de cada modelo a medida que disminuye el SNR. Se espera que los modelos de aprendizaje profundo, y en particular AS-Net, muestren una mayor robustez y una degradación más gradual en comparación con el método NMF.
- **Correlación entre Métricas:** Se analizará la relación entre las métricas de calidad de la señal (SDR/SIR) y la métrica de rendimiento funcional (puntuación F1). Esto permitirá investigar si un SDR más alto se traduce siempre en una mejor detección de especies, o si existen casos en los que un modelo preserva mejor las características discriminatorias a pesar de tener un SDR ligeramente inferior.
- **Eficiencia Computacional:** Se realizará una comparación de la complejidad de los modelos, incluyendo el número de parámetros entrenables y el costo computacional (tiempo de inferencia por segundo de audio). Este análisis es fundamental para evaluar la viabilidad de cada modelo para su despliegue en aplicaciones de campo con recursos limitados, abordando la tensión entre rendimiento y eficiencia.

**Tabla 3: Matriz de Evaluación Exhaustiva para la Comparación de Modelos**

| Métrica de Evaluación | Nivel de SNR (dB) | NMF | U-Net (T-F) | Conv-TasNet (Tiempo) | AS-Net (Propuesta) |
| --- | --- | --- | --- | --- | --- |
| **Parámetros del Modelo (Millones)** | \-  | ~N/A | XX.X | XX.X | XX.X |
| --- | --- | --- | --- | --- | --- |
| **SDR (dB)** | \-5 |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 0   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 5   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 10  |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 15  |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| **SIR (dB)** | \-5 |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 0   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 5   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 10  |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 15  |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
| **Puntuación F1 (Tarea Posterior)** | \-5 |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 0   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 5   |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 10  |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
|     | 15  |     |     |     |     |
| --- | --- | --- | --- | --- | --- |

_Nota: Las celdas vacías se llenarán con los resultados empíricos obtenidos durante la investigación. El número de parámetros para NMF es no aplicable (N/A) en el mismo sentido que para las redes neuronales._

Esta tabla encapsulará la contribución empírica central de la investigación. Permitirá una comparación visual inmediata y multifacética entre el modelo propuesto y las líneas de base en todas las condiciones experimentales. Un revisor podrá determinar instantáneamente no solo si el modelo propuesto es mejor, sino también en qué medida, bajo qué condiciones de ruido y con qué eficiencia computacional. Esta presentación estructurada de los resultados será la principal pieza de evidencia para respaldar las afirmaciones de la propuesta sobre la novedad y la superioridad del rendimiento.

## 6\. Contribuciones Previstas, Limitaciones y Direcciones Futuras

Esta sección final resume el impacto esperado de la investigación, reconoce de forma transparente sus limitaciones inherentes y traza un camino claro para el trabajo futuro, situando este proyecto dentro de una agenda de investigación a más largo plazo.

### 6.1. Resumen de los Resultados Esperados y las Contribuciones Científicas

Se anticipa que esta investigación producirá varios resultados y contribuciones significativas para el campo de la ecoacústica computacional:

- **Un Modelo de Separación de Fuentes Optimizado:** El resultado principal será un modelo de aprendizaje profundo validado y de alto rendimiento (AS-Net), específicamente diseñado para la tarea de separar la biofonía aviar del ruido de fondo. El código fuente del modelo y los pesos pre-entrenados se publicarán bajo una licencia de código abierto para que la comunidad investigadora pueda utilizarlos y desarrollarlos.
- **Una Metodología de Evaluación Novedosa:** Esta investigación proporcionará la primera comparación cuantitativa y rigurosa del rendimiento de los modelos de separación de fuentes utilizando una combinación de métricas a nivel de señal (SDR/SIR) y métricas funcionales basadas en tareas (rendimiento de un clasificador de especies). Este protocolo de evaluación dual establece un nuevo estándar sobre cómo se debe medir la "eficacia" de los modelos de preprocesamiento en bioacústica, yendo más allá de la simple fidelidad de la señal para cuantificar la utilidad práctica.
- **Un Recurso para la Comunidad:** El conjunto de datos sintético generado, junto con los scripts para su creación, se pondrá a disposición del público. Este recurso proporcionará a la comunidad un banco de pruebas estandarizado para la evaluación comparativa de nuevos algoritmos de separación de fuentes, fomentando la reproducibilidad y el progreso colaborativo en el campo.

### 6.2. Limitaciones Potenciales

Es fundamental reconocer las limitaciones del enfoque propuesto para contextualizar adecuadamente los resultados:

- **Dependencia de Datos Sintéticos:** La principal limitación es la dependencia de un conjunto de datos sintético. Aunque esto es necesario para una evaluación controlada y cuantitativa, el modelo de ruido utilizado (ruido rosa) es una simplificación de los complejos y no estacionarios ruidos geofónicos y antropofónicos que se encuentran en las grabaciones del mundo real. Además, el estudio se centra en una sola especie. Por lo tanto, el rendimiento del modelo en paisajes sonoros reales, con múltiples especies vocalizando simultáneamente y con una mayor diversidad de tipos de ruido, puede diferir de los resultados obtenidos aquí.
- **Escenario de Fuente Única:** La propuesta actual aborda un escenario de separación de una única fuente de biofonía más ruido. No aborda el problema, considerablemente más complejo, de separar múltiples fuentes biofónicas que se solapan entre sí (por ejemplo, dos o más especies de aves cantando al mismo tiempo). Este es un desafío conocido como separación de fuentes infradeterminada, que a menudo requiere técnicas más avanzadas.

### 6.3. Direcciones de Investigación Futuras

Los resultados y las limitaciones de este trabajo sentarán las bases para una serie de emocionantes direcciones de investigación futuras:

- **Transición a Datos del Mundo Real:** El siguiente paso lógico será probar y ajustar el modelo AS-Net en grabaciones de campo del mundo real. Se podrían utilizar conjuntos de datos de desafíos como BirdCLEF u otros grandes archivos públicos.<sup>73</sup> Esto implicará abordar los desafíos del ruido del mundo real y podría requerir técnicas de adaptación de dominio para transferir el conocimiento aprendido de los datos sintéticos a los datos reales.
- **Separación de Múltiples Especies:** Una extensión natural del trabajo actual es adaptar la arquitectura del modelo para separar múltiples fuentes biofónicas simultáneamente. Esto transformaría el problema de una separación de una sola fuente a una de múltiples fuentes, lo que probablemente requeriría un cambio a una salida multiclase y el uso de estrategias de entrenamiento más complejas, como el aprendizaje por currículo (curriculum learning), donde el modelo se entrena primero en ejemplos más fáciles (menos fuentes superpuestas) antes de pasar a ejemplos más difíciles.<sup>78</sup>
- **Enfoques No Supervisados y Semi-Supervisados:** Dada la persistente escasez de datos bioacústicos etiquetados a gran escala, una de las fronteras más importantes de la investigación es la exploración de paradigmas de entrenamiento que puedan aprovechar las vastas cantidades de grabaciones de campo no etiquetadas. Los enfoques no supervisados, semi-supervisados o auto-supervisados podrían permitir entrenar modelos de separación potentes sin depender de la costosa anotación manual, lo que representaría un avance transformador para el campo.<sup>9</sup>
- **Implementación en Dispositivos de Campo:** La optimización continua de la arquitectura AS-Net en términos de eficiencia computacional (reducción del tamaño del modelo y del tiempo de inferencia) es crucial para su implementación final en dispositivos de campo. El objetivo a largo plazo es permitir la separación de fuentes en tiempo real en las propias ARUs de baja potencia. Esto transformaría el paradigma del monitoreo acústico pasivo de la simple "grabación" a un "análisis inteligente in situ", donde los datos se procesan y filtran en el campo, reduciendo drásticamente los requisitos de almacenamiento y transmisión de datos y permitiendo sistemas de alerta temprana para eventos ecológicos de interés.

#### Fuentes citadas

- Perspectives in ecoacoustics: A contribution to defining a discipline, acceso: octubre 6, 2025, <https://jea.jams.pub/download/article/2/2/46/pdf>
- Ecoacoustics: A Quantitative Approach to Investigate the Ecological Role of Environmental Sounds - MDPI, acceso: octubre 6, 2025, <https://www.mdpi.com/2227-7390/7/1/21>
- Editorial: Advances in ecoacoustics - Frontiers, acceso: octubre 6, 2025, <https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2022.978516/full>
- Advances in Ecoacoustics | Frontiers Research Topic, acceso: octubre 6, 2025, <https://www.frontiersin.org/research-topics/21968/advances-in-ecoacoustics/magazine>
- Advancements in preprocessing, detection and classification techniques for ecoacoustic data - ResearchOnline@JCU, acceso: octubre 6, 2025, <https://researchonline.jcu.edu.au/82825/1/82825.pdf>
- Summary for the bioacoustics monitoring using AI methods. - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/figure/Summary-for-the-bioacoustics-monitoring-using-AI-methods_tbl1_370282702>
- Monitoring soil fauna with ecoacoustics - PMC - PubMed Central, acceso: octubre 6, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC11371423/>
- Automatic detection for bioacoustic research: a practical guide from and for biologists and computer scientists, acceso: octubre 6, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC11885706/>
- (PDF) Systematic review of machine learning methods applied to ..., acceso: octubre 6, 2025, <https://www.researchgate.net/publication/374122721_Systematic_review_of_machine_learning_methods_applied_to_ecoacoustics_and_soundscape_monitoring>
- Computational bioacoustics with deep learning: a review and roadmap - PMC, acceso: octubre 6, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC8944344/>
- Listening to Nature: The Emerging Field of Bioacoustics - e360-Yale, acceso: octubre 6, 2025, <https://e360.yale.edu/features/listening-to-nature-the-emerging-field-of-bioacoustics>
- BioCPPNet: Automatic Bioacoustic Source Separation with Deep Neural Networks | bioRxiv, acceso: octubre 11, 2025, <https://www.biorxiv.org/content/10.1101/2021.06.18.449016v1.full-text>
- (PDF) BioCPPNet: automatic bioacoustic source separation with ..., acceso: octubre 11, 2025, <https://www.researchgate.net/publication/356806519_BioCPPNet_automatic_bioacoustic_source_separation_with_deep_neural_networks>
- Underdetermined Blind Source Separation of Bioacoustic Signals - Pertanika UPM, acceso: octubre 6, 2025, [http://www.pertanika.upm.edu.my/resources/files/Pertanika%20PAPERS/JST%20Vol.%2031%20(5)%20Aug.%202023/08%20JST-3720-2022.pdf](http://www.pertanika.upm.edu.my/resources/files/Pertanika%20PAPERS/JST%20Vol.%2031%20%285%29%20Aug.%202023/08%20JST-3720-2022.pdf)
- (PDF) Bioacoustic signal denoising: a review - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/publication/346516107_Bioacoustic_signal_denoising_a_review>
- (PDF) Automatic and Efficient Denoising of Bioacoustics Recordings Using MMSE STSA, acceso: octubre 6, 2025, <https://www.researchgate.net/publication/321785229_Automatic_and_Efficient_Denoising_of_Bioacoustics_Recordings_Using_MMSE_STSA>
- Semiautomated generation of species-specific training data from large, unlabeled acoustic datasets for deep supervised birdsong isolation - PubMed, acceso: octubre 6, 2025, <https://pubmed.ncbi.nlm.nih.gov/39329137/>
- A Methodological Literature Review of Acoustic Wildlife Monitoring ..., acceso: octubre 6, 2025, <https://www.mdpi.com/2071-1050/15/9/7128>
- Semiautomated generation of species-specific training data from large, unlabeled acoustic datasets for deep supervised birdsong isolation - PMC - PubMed Central, acceso: octubre 6, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC11426315/>
- An Efficient Time-Domain End-to-End Single-Channel Bird Sound Separation Network, acceso: octubre 11, 2025, <https://www.mdpi.com/2076-2615/12/22/3117>
- 05_Source_Separation_Techniq, acceso: octubre 6, 2025, <https://books.mercity.ai/books/Audio-Analysis-and-Synthesis---Introduction-to-Audio-Signal-Processing/audio_analysis_techniques/05_Source_Separation_Techniques>
- Spectral Subtractive-Type Algorithms for Enhancement of Noisy Speech: An Integrative Review - MECS Press, acceso: octubre 11, 2025, <https://www.mecs-press.org/ijigsp/ijigsp-v5-n11/IJIGSP-V5-N11-2.pdf>
- A geometric approach to spectral subtraction - PMC, acceso: octubre 11, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC2516309/>
- A Review Paper: Speech Enhancement Using Different Spectral Subtraction Algorithms - ijltemas, acceso: octubre 11, 2025, <https://www.ijltemas.in/DigitalLibrary/Vol.3Issue4/12-15.pdf>
- Noise Reduction Based on Modified Spectral Subtraction Method - IAENG, acceso: octubre 11, 2025, <https://www.iaeng.org/IJCS/issues_v38/issue_1/IJCS_38_1_10.pdf>
- Article - Journal of Emerging Investigators, acceso: octubre 6, 2025, <https://emerginginvestigators.org/articles/22-262/pdf>
- Application of Independent Component Analysis and Nelder-Mead Particle Swarm Optimization Algorithm in Non-Contact Blood Pressure Estimation - MDPI, acceso: octubre 6, 2025, <https://www.mdpi.com/1424-8220/24/11/3544>
- A Comparative Model for Blurred Text Detection in Wild Scene Using Independent Component Analysis (ICA) and Enhanced Genetic Algorithm (Using a Bird Approach) with Classifiers - MECS Press, acceso: octubre 6, 2025, <https://www.mecs-press.org/ijitcs/ijitcs-v16-n5/IJITCS-V16-N5-7.pdf>
- (PDF) Comparison of Independent Component Analysis, Principal Component Analysis, and Minimum Noise Fraction Transformation for Tree Species Classification Using APEX Hyperspectral Imagery - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/publication/329793103_Comparison_of_Independent_Component_Analysis_Principal_Component_Analysis_and_Minimum_Noise_Fraction_Transformation_for_Tree_Species_Classification_Using_APEX_Hyperspectral_Imagery>
- Penerapan Metode Fast Independent Component Analysis (FastICA) dalam Memisahkan Vokal dan Instrumen Seni Geguntangan | Jurnal Buana Informatika - Open Journal Systems, acceso: octubre 6, 2025, <https://ojs.uajy.ac.id/index.php/jbi/article/view/5693>
- A New Non-Negative Matrix Factorization Approach for Blind Source Separation of Cardiovascular and Respiratory Sound Based on the Periodicity of Heart and Lung Function - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/publication/370495055_A_New_Non-Negative_Matrix_Factorization_Approach_for_Blind_Source_Separation_of_Cardiovascular_and_Respiratory_Sound_Based_on_the_Periodicity_of_Heart_and_Lung_Function>
- A Survey: Object Feature Analysis Based on Non-negative Matrix Factorization, acceso: octubre 6, 2025, <https://www.csroc.org.tw/journal/JOC32-6/JOC3206-09.pdf>
- Deep Convex Representations: Feature Representations for Bioacoustics Classification - ISCA Archive, acceso: octubre 6, 2025, <https://www.isca-archive.org/interspeech_2018/thakur18_interspeech.pdf>
- A Survey of Artificial Intelligence Approaches in Blind Source Separation - LJMU Research Online, acceso: octubre 6, 2025, <https://researchonline.ljmu.ac.uk/id/eprint/24283/8/Final.pdf>
- Comparison of (A) blind source separation, (B) model‐based source... - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/figure/Comparison-of-A-blind-source-separation-B-model-based-source-separation-and-C-deep_fig1_337974795>
- (PDF) State-Of-The-Art Analysis of Deep Learning-Based Monaural Speech Source Separation Techniques - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/publication/366980377_State-Of-The-Art_Analysis_of_Deep_Learning-Based_Monaural_Speech_Source_Separation_Techniques>
- \[2003.10414\] Multi-channel U-Net for Music Source Separation - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/abs/2003.10414>
- \[1810.11520\] Spectrogram-channels u-net: a source separation model viewing each channel as the spectrogram of each source - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/abs/1810.11520>
- (a) Schematic overview of the BioCPPNet pipeline. Source vocalization... | Download Scientific Diagram - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/figure/a-Schematic-overview-of-the-BioCPPNet-pipeline-Source-vocalization-waveforms-are_fig1_356806519>
- ATTENTION WAVE-U-NET FOR SPEECH ENHANCEMENT Ritwik Giri, Umut Isik, and Arvindh Krishnaswamy Amazon Inc., acceso: octubre 11, 2025, <https://assets.amazon.science/ce/30/fc369da44a7284703da0dd72ea38/attention-wave-u-net-for-speech-enhancement.pdf>
- Multi-channel U-Net for Music Source Separation - Venkatesh Shenoy Kadandale, acceso: octubre 11, 2025, <https://vskadandale.github.io/pdf/MMSP_2020.pdf>
- High-Quality Visually-Guided Sound Separation from Diverse Categories - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/html/2308.00122v2>
- \[2211.08553\] Hybrid Transformers for Music Source Separation - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/abs/2211.08553>
- Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation, acceso: octubre 11, 2025, <https://pmc.ncbi.nlm.nih.gov/articles/PMC6726126/>
- \[1809.07454\] Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/abs/1809.07454>
- Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/pdf/1809.07454>
- \[2002.08688\] An empirical study of Conv-TasNet - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/abs/2002.08688>
- (PDF) LiTasNeT: A Bird Sound Separation Algorithm Based on ..., acceso: octubre 11, 2025, <https://www.researchgate.net/publication/363070271_LiTasNeT_A_Bird_Sound_Separation_Algorithm_Based_on_Deep_Learning>
- LiTasNeT: A Bird Sound Separation Algorithm Based on Deep Learning - IGI Global, acceso: octubre 11, 2025, <https://www.igi-global.com/article/litasnet/301261>
- A Review of Automated Bioacoustics and General Acoustics ... - MDPI, acceso: octubre 6, 2025, <https://www.mdpi.com/1424-8220/22/21/8361>
- ORCA-WHISPER: An Automatic Killer Whale Sound Type Generation Toolkit Using Deep Learning - ISCA Archive, acceso: octubre 6, 2025, <https://www.isca-archive.org/interspeech_2022/bergler22_interspeech.pdf>
- Benchmarking nearest neighbor retrieval of zebra finch ... - bioRxiv, acceso: octubre 6, 2025, <https://www.biorxiv.org/content/10.1101/2023.09.04.555475v1.full-text>
- Wytham Great Tit Song Dataset - nilo merino recalde, acceso: octubre 11, 2025, <https://nilomr.github.io/great-tit-hits/>
- Multiple signals for multiple messages: Great tit, Parus major, song signals age and survival | Request PDF - ResearchGate, acceso: octubre 11, 2025, <https://www.researchgate.net/publication/223257996_Multiple_signals_for_multiple_messages_Great_tit_Parus_major_song_signals_age_and_survival>
- Spectrogram of a typical great tit song. | Download Scientific Diagram - ResearchGate, acceso: octubre 11, 2025, <https://www.researchgate.net/figure/Spectrogram-of-a-typical-great-tit-song_fig1_46512664>
- A densely sampled and richly annotated acoustic dataset from a wild bird population - bioRxiv, acceso: octubre 11, 2025, <https://www.biorxiv.org/content/10.1101/2023.07.03.547484v1.full.pdf>
- Xeno-canto - Wikipedia, acceso: octubre 11, 2025, <https://en.wikipedia.org/wiki/Xeno-canto>
- Xeno-canto - Bird sounds from around the world - Research Data Australia, acceso: octubre 11, 2025, <https://researchdata.edu.au/xeno-canto-bird-sounds-world/3545720>
- Xeno-Canto Bird Recordings Dataset - Kaggle, acceso: octubre 11, 2025, <https://www.kaggle.com/datasets/imoore/xenocanto-bird-recordings-dataset>
- \[2205.13657\] An enhanced Conv-TasNet model for speech separation using a speaker distance-based loss function - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/abs/2205.13657>
- What Python/Java libraries are good for searching audio files for certain sounds - Reddit, acceso: octubre 11, 2025, <https://www.reddit.com/r/learnpython/comments/5lj6v0/what_pythonjava_libraries_are_good_for_searching/>
- Machine learning tools for acoustic bird detection - GitHub, acceso: octubre 11, 2025, <https://github.com/microsoft/acoustic-bird-detection>
- Environmental sound classification with Convolutional neural networks and the UrbanSound8K dataset. - GitHub, acceso: octubre 11, 2025, <https://github.com/mariostrbac/environmental-sound-classification>
- Neural Networks for Separation of Communication Sources - UPCommons, acceso: octubre 6, 2025, <https://upcommons.upc.edu/bitstream/handle/2117/420716/Master_Thesis_Marc_Ibanyez.pdf?sequence=3>
- a new non-negative matrix factorization approach for - arXiv, acceso: octubre 6, 2025, <https://arxiv.org/pdf/2305.01889>
- (PDF) Separation of overlapping sources in bioacoustic mixtures - ResearchGate, acceso: octubre 6, 2025, <https://www.researchgate.net/publication/339994631_Separation_of_overlapping_sources_in_bioacoustic_mixtures>
- The use of BirdNET embeddings as a fast solution to find novel sound classes in audio recordings - Frontiers, acceso: octubre 11, 2025, <https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2024.1409407/full>
- Machine Learning Detector - Quick Start Guide - Raven Sound Analysis, acceso: octubre 11, 2025, <https://www.ravensoundsoftware.com/knowledge-base/learning-detector/>
- What is BirdNET?, acceso: octubre 11, 2025, <https://birdnet.cornell.edu/home/>
- BirdNET Sound ID - The easiest way to identify birds by sound., acceso: octubre 11, 2025, <https://birdnet.cornell.edu/>
- BirdNET analyzer for scientific audio data processing. - GitHub, acceso: octubre 11, 2025, <https://github.com/birdnet-team/BirdNET-Analyzer>
- How to use BirdNET - British Ornithologists' Union, acceso: octubre 11, 2025, <https://bou.org.uk/blog-granados-birdnet/>
- Visual WetlandBirds Dataset: Bird Species Identification and Behavior Recognition in Videos - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/html/2501.08931v1>
- BirdCLEF+ 2025 - Kaggle, acceso: octubre 11, 2025, <https://www.kaggle.com/c/birdclef-2025>
- bernardocecchetto/BirdCLEF-Challenge2023-Kaggle · Datasets at Hugging Face, acceso: octubre 11, 2025, <https://huggingface.co/datasets/bernardocecchetto/BirdCLEF-Challenge2023-Kaggle>
- jfpuget/birdclef-2024: Solution to Birdclef 2024 challenge on Kaggle - GitHub, acceso: octubre 11, 2025, <https://github.com/jfpuget/birdclef-2024>
- BirdCLEF 2024 | Kaggle, acceso: octubre 11, 2025, <https://www.kaggle.com/competitions/birdclef-2024>
- Improving Singing Voice Separation Using Curriculum Learning on Recurrent Neural Networks - MDPI, acceso: octubre 11, 2025, <https://www.mdpi.com/2076-3417/10/7/2465>
- Training-Free Multi-Step Audio Source Separation - arXiv, acceso: octubre 11, 2025, <https://arxiv.org/html/2505.19534v1>
- Visually Guided Audio Source Separation With Meta Consistency Learning - CVF Open Access, acceso: octubre 11, 2025, <https://openaccess.thecvf.com/content/WACV2024/papers/Islam_Visually_Guided_Audio_Source_Separation_With_Meta_Consistency_Learning_WACV_2024_paper.pdf>
- TweetyBERT: Automated parsing of birdsong through self ... - bioRxiv, acceso: octubre 6, 2025, <https://www.biorxiv.org/content/10.1101/2025.04.09.648029v1.full-text>
- Decoding communication of non-human species - DiVA portal, acceso: octubre 6, 2025, <https://www.diva-portal.org/smash/get/diva2:1811459/FULLTEXT01.pdf>
- Identifying birdsong syllables without labelled data - arXiv, acceso: octubre 6, 2025, <https://www.arxiv.org/pdf/2509.18412>
- Identifying birdsong syllables without labelled data - arXiv, acceso: octubre 6, 2025, <https://arxiv.org/html/2509.18412v1>
