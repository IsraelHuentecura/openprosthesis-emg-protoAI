# Modelos Ligeros de Clasificación de Gestos sEMG para Prótesis de Código Abierto

Este repositorio contiene los modelos de aprendizaje profundo, el código y los resultados del proyecto de título desarrollado por Israel Huentecura en el marco del proyecto **[ProtoIA](https://github.com/P4HBionicsAcademy)**.

El objetivo principal de este trabajo es proporcionar modelos de redes neuronales de alto rendimiento y computacionalmente eficientes para la clasificación de gestos de mano a partir de señales de electromiografía de superficie (sEMG). Estos modelos están optimizados para su despliegue en hardware de bajos recursos, como microcontroladores (ej. ESP32), con el fin de potenciar una nueva generación de prótesis de mano asequibles y de código abierto.

## 🏆 Mejores Modelos por Arquitectura

A continuación se presentan los mejores modelos para cada arquitectura, basados en su precisión en el conjunto de pruebas:

| Arquitectura | Mejor Modelo | Precisión en Prueba | Tamaño |
|-------------|--------------|---------------------|--------|
| **HyT-Net** | `HyT-Net_fold1_BEST.keras` | **0.9988** | 10.4 MB |
| **EMGHandNet-2D** | `EMGHandNet-2D_fold6_BEST.keras` | 0.9984 | 4.2 MB |
| **DualStream-Original** | `DualStream-Original_fold3_BEST.keras` | 0.9986 | 36.6 MB |
| **EMGHandNet-Original** | `EMGHandNet-Original_fold9_BEST.keras` | 0.9982 | 19.8 MB |
| **DualStream-Lite** | `DualStream-Lite_fold8_BEST.keras` | 0.9946 | 3.4 MB |
| **CRNN-Attn** | `CRNN-Attn_fold1_BEST.keras` | 0.9881 | 3.6 MB |

Los modelos recomendados para despliegue en hardware de recursos limitados son **EMGHandNet-2D**, **DualStream-Lite** y **CRNN-Attn** debido a su menor tamaño y buen rendimiento.

## Características Principales

* **Alta Precisión:** Todos los modelos propuestos superaron el 98% de precisión promedio en el conjunto de datos de prueba.
* **Optimizados para el Borde (Edge):** Se incluyen tres arquitecturas (`EMGHandNet-2D`, `DualStream-Lite`, `CRNN-Attn`) con un tamaño de despliegue inferior a 1.5 MB, haciéndolas viables para microcontroladores.
* **Listos para *Fine-Tuning*:** Los pesos pre-entrenados se publican para servir como un punto de partida robusto para la personalización a nuevos usuarios, una estrategia clave para la usabilidad en el mundo real.
* **Investigación Reproducible:** Se proporciona el código y los scripts para permitir la reproducibilidad de los resultados y fomentar nuevas investigaciones sobre esta base.

## Resumen de Modelos

Los modelos fueron entrenados y evaluados en el **Estímulo 1 (12 gestos de dedos)** del dataset **NinaPro DB1**. A continuación se comparan las arquitecturas propuestas y las de referencia.

| Modelo | Accuracy en Prueba (μ ± σ) | Tamaño TFLite (MB) | Escenario de Despliegue Recomendado |
| :--- | :---: | :---: | :--- |
| **`HyT-Net`** | **0.9983 ± 0.0002** | 3.27 | **Alto Rendimiento (ej. Raspberry Pi):** Máxima precisión donde la memoria no es el factor más crítico. |
| **`EMGHandNet-2D`** | 0.9978 ± 0.0003 | 1.33 | **Microcontrolador (Balanceado):** El mejor equilibrio entre alta precisión, eficiencia y arquitectura *end-to-end*. |
| **`DualStream-Lite`** | 0.9941 ± 0.0003 | 1.07 | **Microcontrolador (Ultra-Ligero):** La opción más pequeña, ideal para hardware con restricciones de memoria extremas. |
| `CRNN-Attn` | 0.9855 ± 0.0015 | 1.14 | Experimental. Menor estabilidad en las pruebas. |
| `DualStream-Original` | 0.9982 ± 0.0003 | 11.65 | Referencia (No apto para microcontroladores). |
| `EMGHandNet-Original` | 0.9980 ± 0.0002 | 6.30 | Referencia (No apto para microcontroladores). |

## Cómo Cargar los Modelos

### Prerrequisitos
- Python 3.8+
- TensorFlow 2.10+

### Cargando los modelos Keras (.keras)

Todos los mejores modelos están disponibles en formato `.keras` en el directorio `models/`. Aquí se muestra cómo cargar cada tipo de modelo:

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Definir la ruta al modelo que deseas cargar
MODEL_PATH = "models/HyT-Net_fold1_BEST.keras"  # Cambia esto según el modelo que desees usar

# Cargar el modelo
model = load_model(MODEL_PATH)

# Verificar la arquitectura del modelo
print(f"Modelo cargado: {model.name}")
print(f"Capas del modelo:")
model.summary()

# Preparar datos de entrada para inferencia (ejemplo con datos aleatorios)
# La forma de entrada dependerá del modelo específico:

if "HyT-Net" in MODEL_PATH or "DualStream" in MODEL_PATH or "CRNN-Attn" in MODEL_PATH:
    # Estos modelos esperan dos entradas: raw y features
    t_subwin = 5  # Número de sub-ventanas por secuencia
    win_len = 20  # Longitud de la ventana
    n_channels = 10  # Número de canales EMG
    feat_dim = 100  # Dimensión de características (10 por canal × 10 canales)
    
    # Datos aleatorios para ejemplo
    raw_data = np.random.rand(1, t_subwin, win_len, n_channels, 1).astype(np.float32)
    feat_data = np.random.rand(1, t_subwin, feat_dim).astype(np.float32)
    
    # Realizar predicción
    if "DualStream" in MODEL_PATH:
        # Modelo DualStream usa nombres específicos para las entradas
        predictions = model.predict({"raw": raw_data, "feat": feat_data})
    else:
        # HyT-Net y CRNN-Attn usan nombres diferentes
        predictions = model.predict({"raw": raw_data, "feat_input": feat_data})
else:
    # EMGHandNet-2D y EMGHandNet-Original solo esperan una entrada: raw
    t_subwin = 5
    win_len = 20
    n_channels = 10
    
    # Datos aleatorios para ejemplo
    raw_data = np.random.rand(1, t_subwin, win_len, n_channels, 1).astype(np.float32)
    
    # Realizar predicción
    predictions = model.predict(raw_data)

# Obtener la clase predicha
predicted_class = np.argmax(predictions, axis=1)[0]
print(f"Clase predicha (índice 0-11): {predicted_class}")
print(f"Etiqueta de gesto (1-12): {predicted_class + 1}")
```

### Cargando modelos en formato TFLite

Para dispositivos con recursos limitados, se recomienda usar los modelos en formato TFLite:

```python
import tensorflow as tf
import numpy as np

# Cargar el modelo TFLite
interpreter = tf.lite.Interpreter(model_path="models_tflite/EMGHandNet-2D.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Mostrar información de las entradas requeridas
print("Detalles de entrada:")
for input_detail in input_details:
    print(f"  - Nombre: {input_detail['name']}, Forma: {input_detail['shape']}")

# Preparar datos de ejemplo (adaptar según el modelo específico)
t_subwin = 5
win_len = 20
n_channels = 10
input_shape = input_details[0]['shape']
input_data = np.random.rand(*input_shape).astype(np.float32)

# Establecer el tensor de entrada y ejecutar la inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtener el resultado
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_gesture_index = np.argmax(output_data)
predicted_gesture_label = predicted_gesture_index + 1

print(f"Índice de salida predicho: {predicted_gesture_index}")
print(f"Etiqueta de gesto predicha: {predicted_gesture_label}")
