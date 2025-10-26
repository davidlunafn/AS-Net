# AS-Net Pipeline: Diagrama Completo del Proceso

## Diagrama Principal del Pipeline

```mermaid
graph TB
    subgraph "FASE 1: GENERACIÓN DE DATOS"
        A1[🎵 Datos Crudos<br/>Xeno-Canto + Wytham Dataset<br/>Vocalizaciones Limpias] --> A2[🔊 Generación de Ruido Rosa<br/>Simulación Geofonía/Antropofonía]
        A1 --> A3[🎲 Mezcla Sintética<br/>1-5 vocalizaciones por sample<br/>Duración: 20s]
        A2 --> A3
        A3 --> A4{📊 Control SNR<br/>-5, 0, 5, 10, 15 dB}
        A4 --> A5[💾 Dataset Sintético<br/>Mixed + Source + Noise<br/>+ Metadata CSV]
        A5 --> A6[📂 División de Datos<br/>80% Train | 10% Val | 10% Test<br/>Seed=42, Sin Overlap]
    end

    subgraph "FASE 2: ARQUITECTURA AS-NET"
        B1[🎙️ Input: Mixed Audio<br/>Forma de Onda 1D] --> B2[📥 Encoder<br/>Conv1D<br/>Kernel=16, Stride=8<br/>Channels: 1→128]
        B2 --> B3[🔄 Separation Module<br/>8 TCN Blocks<br/>Dilations: 1,2,4,8,16,32,64,128<br/>Dropout=0.35<br/>LayerNorm GroupNorm]
        B3 --> B4[🎭 Mask Estimation<br/>Conv1D 1x1<br/>2 Máscaras Sigmoid<br/>Biofonía + Ruido]
        B4 --> B5[✖️ Masked Features<br/>encoded * mask_source<br/>encoded * mask_noise]
        B5 --> B6[📤 Decoder<br/>ConvTranspose1D<br/>Kernel=16, Stride=8<br/>Channels: 128→1]
        B6 --> B7[🎵 Output: 2 Señales<br/>Biofonía Separada<br/>Ruido Separado]
    end

    subgraph "FASE 3: ENTRENAMIENTO"
        C1[⚙️ Configuración<br/>Batch=1, Accum=4<br/>LR=0.0001, WD=1e-4<br/>Epochs=50] --> C2[🔁 Training Loop<br/>SI-SDR Loss negativa<br/>PIT Permutation Invariant<br/>Adam Optimizer]
        C2 --> C3[📉 Gradient Accumulation<br/>4 steps efectivos<br/>Simula batch_size=4]
        C3 --> C4[✅ Validation<br/>1000 steps<br/>Sin Dropout<br/>Eval Mode]
        C4 --> C5{🛑 Early Stopping<br/>Patience=10 epochs<br/>Best Val Loss Check}
        C5 -->|Mejora| C6[💾 Save Checkpoint<br/>Model + Optimizer + Config<br/>Best Model]
        C5 -->|No Mejora| C7[⏱️ Contador Paciencia<br/>+1 epoch]
        C7 -->|< 10| C2
        C7 -->|≥ 10| C8[🏁 Training Finished]
        C6 --> C9[📊 Save History<br/>training_history.csv<br/>epoch, train_loss, val_loss]
        C9 --> C2
    end

    subgraph "FASE 4: EVALUACIÓN OBJETIVA"
        D1[📁 Test Set<br/>10% datos no vistos] --> D2[🔮 Inferencia<br/>Modelo Entrenado<br/>Forward Pass]
        D2 --> D3[📏 Métricas de Señal]
        D3 --> D4[📊 SI-SDR<br/>Scale-Invariant SDR<br/>Mejora en dB]
        D3 --> D5[📊 SDR<br/>Signal-to-Distortion<br/>Calidad General]
        D3 --> D6[📊 SIR<br/>Signal-to-Interference<br/>Supresión de Ruido]
        D4 --> D7[📈 Análisis por SNR<br/>Rendimiento en cada nivel<br/>-5dB a 15dB]
        D5 --> D7
        D6 --> D7
    end

    subgraph "FASE 5: EVALUACIÓN FUNCIONAL"
        E1[🔊 Audio Separado] --> E2[🤖 BirdNET Detector<br/>Modelo Pre-entrenado<br/>Clasificador de Especies]
        E2 --> E3[📊 Métricas de Detección]
        E3 --> E4[✓ Precision<br/>TP / TP+FP<br/>Calidad Detecciones]
        E3 --> E5[✓ Recall<br/>TP / TP+FN<br/>Cobertura]
        E3 --> E6[✓ F1-Score<br/>2·P·R / P+R<br/>Balance General]
        E4 --> E7[📊 Comparación<br/>Clean vs Mixed vs Separated]
        E5 --> E7
        E6 --> E7
    end

    subgraph "FASE 6: BENCHMARKING"
        F1[🏆 Modelos de Referencia] --> F2[📐 NMF<br/>Non-negative Matrix<br/>Método Estadístico]
        F1 --> F3[🔷 U-Net<br/>Encoder-Decoder<br/>Dominio T-F]
        F1 --> F4[⚡ Conv-TasNet<br/>Temporal Conv<br/>Dominio Tiempo]
        F1 --> F5[🌟 AS-Net<br/>Modelo Propuesto<br/>Optimizado Biofonía]
        F2 --> F6[📊 Tabla Comparativa<br/>SDR, SIR, F1-Score<br/>Por SNR Level]
        F3 --> F6
        F4 --> F6
        F5 --> F6
    end

    A6 --> B1
    A6 --> C1
    C8 --> D1
    D2 --> E1
    D7 --> F6
    E7 --> F6
    F6 --> G1[📄 Paper & Publicación]
```

## Diagrama de Flujo de Datos en Entrenamiento

```mermaid
sequenceDiagram
    participant DataLoader
    participant Model
    participant Loss
    participant Optimizer
    participant EarlyStopping
    participant Checkpoint

    loop Por cada Epoch
        DataLoader->>Model: Batch (mixture, source, noise)
        Model->>Model: Forward Pass
        Model->>Loss: Predicciones (est_source1, est_source2)
        Loss->>Loss: Calcula SI-SDR con PIT
        Loss->>Optimizer: Backward (loss / accum_steps)

        alt Cada 4 steps
            Optimizer->>Model: Update Weights
            Optimizer->>Optimizer: Zero Gradients
        end

        Note over DataLoader,Model: Repite 2000 steps (Train)

        DataLoader->>Model: Validation Batch
        Model->>Loss: Val Predictions
        Loss->>EarlyStopping: Val Loss

        alt Val Loss Improved
            EarlyStopping->>Checkpoint: Save Best Model
            EarlyStopping->>EarlyStopping: Reset Counter
        else No Improvement
            EarlyStopping->>EarlyStopping: Counter += 1
            alt Counter >= 10
                EarlyStopping-->>Model: STOP Training
            end
        end
    end
```

## Arquitectura Detallada AS-Net

```mermaid
graph LR
    subgraph Input
        I[Audio Mezclado<br/>Shape: B, T<br/>T ≈ 320,000 samples<br/>16kHz, 20s]
    end

    subgraph Encoder
        E1[Conv1D<br/>in=1, out=128<br/>kernel=16, stride=8<br/>padding=0]
        E2[ReLU]
        E1 --> E2
        E3[Encoded Features<br/>Shape: B, 128, T/8]
        E2 --> E3
    end

    subgraph "Separation Module"
        S1[TCN Block 1<br/>dilation=1]
        S2[TCN Block 2<br/>dilation=2]
        S3[TCN Block 3<br/>dilation=4]
        S4[TCN Block 4<br/>dilation=8]
        S5[TCN Block 5<br/>dilation=16]
        S6[TCN Block 6<br/>dilation=32]
        S7[TCN Block 7<br/>dilation=64]
        S8[TCN Block 8<br/>dilation=128]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8
    end

    subgraph "TCN Block Detail"
        T1[Conv1D 1x1<br/>Point-wise]
        T2[GroupNorm<br/>groups=1]
        T3[PReLU]
        T4[Dropout 0.35]
        T5[DepthWise Conv1D<br/>Dilated]
        T6[GroupNorm<br/>groups=1]
        T7[PReLU]
        T8[Dropout 0.35]
        T9[Conv1D 1x1<br/>Point-wise]
        T10[Residual Add]
        T1-->T2-->T3-->T4-->T5-->T6-->T7-->T8-->T9-->T10
    end

    subgraph "Mask Generation"
        M1[Conv1D 1x1<br/>in=128, out=256<br/>2 sources]
        M2[Sigmoid<br/>Independiente por fuente]
        M3[Mask Biofonía<br/>Shape: B, 128, T/8]
        M4[Mask Ruido<br/>Shape: B, 128, T/8]
        M1 --> M2
        M2 --> M3
        M2 --> M4
    end

    subgraph Decoder
        D1[ConvTranspose1D<br/>in=128, out=1<br/>kernel=16, stride=8]
        D2[Padding Ajuste<br/>Match Input Length]
    end

    subgraph Output
        O1[Biofonía Separada<br/>Shape: B, T]
        O2[Ruido Separado<br/>Shape: B, T]
    end

    I --> E1
    E3 --> S1
    S8 --> M1
    E3 --> M3
    E3 --> M4
    M3 --> D1
    D1 --> D2
    D2 --> O1
    M4 --> D1
    D2 --> O2
```

## Métricas de Evaluación

```mermaid
graph TD
    subgraph "Métricas Objetivas (Calidad de Señal)"
        A[🎯 SI-SDR<br/>Scale-Invariant SDR]
        A --> A1[✓ Invariante a escala<br/>✓ Mejor para audio<br/>✓ Rango: -∞ a +∞ dB<br/>✓ Mayor = Mejor]

        B[📊 SDR<br/>Signal-to-Distortion Ratio]
        B --> B1[✓ Calidad general<br/>✓ Incluye todos artefactos<br/>✓ BSS Eval estándar<br/>✓ Mayor = Mejor]

        C[🔇 SIR<br/>Signal-to-Interference Ratio]
        C --> C1[✓ Supresión de otras fuentes<br/>✓ Aislamiento<br/>✓ Mayor = Mejor<br/>✓ Complementa SDR]
    end

    subgraph "Métricas Funcionales (Utilidad Práctica)"
        D[✓ Precision<br/>TP / TP+FP]
        D --> D1[Calidad de<br/>Detecciones<br/>Positivas]

        E[✓ Recall<br/>TP / TP+FN]
        E --> E1[Cobertura de<br/>Vocalizaciones<br/>Reales]

        F[✓ F1-Score<br/>2·Precision·Recall / P+R]
        F --> F1[Balance<br/>Entre Precision<br/>y Recall]
    end

    subgraph "Análisis Comparativo"
        G[📈 Por Nivel SNR]
        G --> G1[-5dB: Muy Ruidoso<br/>0dB: SNR Crítico<br/>5dB: Moderado<br/>10dB: Bueno<br/>15dB: Excelente]

        H[🏆 Entre Modelos]
        H --> H1[NMF vs U-Net<br/>vs Conv-TasNet<br/>vs AS-Net]

        I[🔄 Mejora Relativa]
        I --> I1[Δ SDR = SDR_out - SDR_in<br/>Ganancia en dB<br/>Por procesamiento]
    end
```

## Pipeline de Evaluación Funcional

```mermaid
graph TB
    subgraph "Datos de Entrada"
        A1[🎵 Audio Limpio<br/>Ground Truth<br/>Upper Bound]
        A2[🔊 Audio Mezclado<br/>Sin Procesar<br/>Baseline]
        A3[✨ Audio Separado<br/>AS-Net Output<br/>Propuesta]
        A4[📊 Referencias<br/>NMF, U-Net, TasNet<br/>Comparación]
    end

    subgraph "Detector BirdNET"
        B1[🤖 Modelo Pre-entrenado<br/>1000+ especies<br/>Segmentos 3s]
        B1 --> B2[Confidence Score<br/>0.0 a 1.0<br/>Por especie]
        B2 --> B3[Threshold 0.5<br/>Decisión Binaria<br/>Presente / Ausente]
    end

    subgraph "Cálculo Métricas"
        C1[Confusion Matrix<br/>TP, FP, TN, FN]
        C1 --> C2[Precision = TP/TP+FP]
        C1 --> C3[Recall = TP/TP+FN]
        C2 --> C4[F1 = 2·P·R / P+R]
        C3 --> C4
    end

    subgraph "Análisis de Resultados"
        D1[📊 F1_clean<br/>Límite Superior<br/>Rendimiento Ideal]
        D2[📊 F1_mixed<br/>Baseline<br/>Sin Preprocesamiento]
        D3[📊 F1_separated<br/>Con AS-Net<br/>Mejora Propuesta]
        D4[📊 F1_reference<br/>Otros Modelos<br/>Estado del Arte]

        D1 --> D5[💡 Recuperación<br/>recovery = F1_sep - F1_mix / F1_clean - F1_mix<br/>% de pérdida recuperada]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    A4 --> B1
    B3 --> C1
    C4 --> D1
    C4 --> D2
    C4 --> D3
    C4 --> D4
```

## Estado Actual del Entrenamiento (Época 28)

```mermaid
graph LR
    subgraph "Progreso de Entrenamiento"
        A[Época 1<br/>Train: 11.95<br/>Val: 1.03<br/>Gap: 10.92]
        B[Época 10<br/>Train: -23.60<br/>Val: -24.33<br/>Gap: 0.73]
        C[Época 20<br/>Train: -27.47<br/>Val: -27.84<br/>Gap: 0.37]
        D[Época 28<br/>Train: -28.12<br/>Val: -28.77<br/>Gap: 0.65]

        A -->|Aprendizaje Rápido| B
        B -->|Convergencia| C
        C -->|Refinamiento| D
    end

    subgraph "Regularización Activa"
        E[✓ Dropout 0.35<br/>✓ Weight Decay 1e-4<br/>✓ LayerNorm<br/>✓ Gradient Accum 4<br/>✓ Early Stop P=10]
    end

    subgraph "Resultados Clave"
        F[🎯 CERO Overfitting<br/>Val Loss < Train Loss<br/>Generalización Excelente]
        G[📈 Mejora Consistente<br/>Sin Oscilaciones<br/>Curva Suave]
        H[⏱️ Paciencia: 0/10<br/>Modelo Mejorando<br/>Aún Activo]
    end

    D --> F
    D --> G
    D --> H
```

---

## 📋 Tabla Resumen de Configuración

| **Componente** | **Parámetro** | **Valor** | **Justificación** |
|----------------|---------------|-----------|-------------------|
| **Datos** | Duración samples | 20s | Captura vocalizaciones completas |
| | Vocalizaciones/sample | 1-5 | Variabilidad realista |
| | SNR levels | -5, 0, 5, 10, 15 dB | Rango condiciones reales |
| | Train/Val/Test split | 80/10/10% | Estándar ML, sin overlap |
| **Arquitectura** | Encoder channels | 128 | Balance capacidad/memoria |
| | TCN blocks | 8 | Receptive field ~2s |
| | Dilations | 1→128 | Dependencias temporales |
| | Dropout | 0.35 | Anti-overfitting |
| **Entrenamiento** | Batch size | 1 | Optimizado M1 |
| | Accumulation steps | 4 | Batch efectivo = 4 |
| | Learning rate | 1e-4 | Convergencia estable |
| | Weight decay | 1e-4 | L2 regularización |
| | Early stopping | 10 epochs | Previene overtraining |
| **Evaluación** | Train steps/epoch | 2000 | ~3.5 horas datos |
| | Val steps/epoch | 1000 | Evaluación robusta |
| | Métricas objetivo | SI-SDR, SDR, SIR | Estándar BSS Eval |
| | Métricas funcionales | P, R, F1 | Utilidad práctica |

---

## 🔄 Flujo de Trabajo Completo

1. **Generación** → Datos sintéticos con ground truth
2. **División** → 80/10/10 sin overlap (seed=42)
3. **Entrenamiento** → AS-Net con regularización fuerte
4. **Validación** → Early stopping basado en val loss
5. **Evaluación Objetiva** → SI-SDR, SDR, SIR por SNR
6. **Evaluación Funcional** → BirdNET F1-Score
7. **Benchmarking** → Comparación con SOTA
8. **Publicación** → Paper + código open source

---

## 📊 Resultados Esperados

- **SI-SDR Improvement**: >10 dB en todos los SNR
- **F1-Score**: Recuperación 60-80% pérdida por ruido
- **Generalización**: Val ≈ Train (cero overfitting)
- **Eficiencia**: Inferencia <100ms/sample en M1

