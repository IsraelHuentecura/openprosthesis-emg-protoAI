# Modelos Ligeros de Clasificación de Gestos sEMG para Prótesis de Código Abierto

Este repositorio contiene los modelos de aprendizaje profundo, el código y los resultados del proyecto de título desarrollado por Israel Huentecura en el marco del proyecto **[ProtoIA](https://github.com/P4HBionicsAcademy)**.

[cite_start]El objetivo principal de este trabajo es proporcionar modelos de redes neuronales de alto rendimiento y computacionalmente eficientes para la clasificación de gestos de mano a partir de señales de electromiografía de superficie (sEMG)[cite: 6, 64]. [cite_start]Estos modelos están optimizados para su despliegue en hardware de bajos recursos, como microcontroladores (ej. ESP32), con el fin de potenciar una nueva generación de prótesis de mano asequibles y de código abierto[cite: 34, 35].

## Características Principales

* [cite_start]**Alta Precisión:** Todos los modelos propuestos superaron el 98% de precisión promedio en el conjunto de datos de prueba[cite: 8, 225].
* [cite_start]**Optimizados para el Borde (Edge):** Se incluyen tres arquitecturas (`EMGHandNet-2D`, `DualStream-Lite`, `CRNN-Attn`) con un tamaño de despliegue inferior a 1.5 MB, haciéndolas viables para microcontroladores[cite: 9, 233, 278].
* [cite_start]**Listos para *Fine-Tuning*:** Los pesos pre-entrenados se publican para servir como un punto de partida robusto para la personalización a nuevos usuarios[cite: 57, 62, 263], una estrategia clave para la usabilidad en el mundo real.
* [cite_start]**Investigación Reproducible:** Se proporciona el código y los scripts para permitir la reproducibilidad de los resultados [cite: 65, 68] y fomentar nuevas investigaciones sobre esta base.

## Resumen de Modelos

[cite_start]Los modelos fueron entrenados y evaluados en el **Estímulo 1 (12 gestos de dedos)** del dataset **NinaPro DB1**[cite: 46, 47, 48]. A continuación se comparan las arquitecturas propuestas y las de referencia.

| Modelo | Accuracy en Prueba (μ ± σ) | Tamaño TFLite (MB) | Escenario de Despliegue Recomendado |
| :--- | :---: | :---: | :--- |
| **`HyT-Net`** | [cite_start]**0.9983 ± 0.0002** [cite: 222] | [cite_start]3.27 [cite: 231] | [cite_start]**Alto Rendimiento (ej. Raspberry Pi):** Máxima precisión donde la memoria no es el factor más crítico[cite: 241, 280]. |
| **`EMGHandNet-2D`** | [cite_start]0.9978 ± 0.0003 [cite: 222] | [cite_start]1.33 [cite: 231] | [cite_start]**Microcontrolador (Balanceado):** El mejor equilibrio entre alta precisión, eficiencia y arquitectura *end-to-end*[cite: 253, 257, 281]. |
| **`DualStream-Lite`** | [cite_start]0.9941 ± 0.0003 [cite: 223] | [cite_start]1.07 [cite: 231] | [cite_start]**Microcontrolador (Ultra-Ligero):** La opción más pequeña, ideal para hardware con restricciones de memoria extremas[cite: 249, 251]. |
| `CRNN-Attn` | [cite_start]0.9855 ± 0.0015 [cite: 223] | [cite_start]1.14 [cite: 231] | Experimental. [cite_start]Menor estabilidad en las pruebas[cite: 228, 248]. |
| `DualStream-Original` | [cite_start]0.9982 ± 0.0003 [cite: 222] | [cite_start]11.65 [cite: 232] | [cite_start]Referencia (No apto para microcontroladores)[cite: 246]. |
| `EMGHandNet-Original` | [cite_start]0.9980 ± 0.0002 [cite: 222] | [cite_start]6.30 [cite: 232] | [cite_start]Referencia (No apto para microcontroladores)[cite: 246]. |

## Cómo Empezar

### Prerrequisitos
- Python 3.8+
- TensorFlow 2.10+

### Uso para Inferencia
Puedes cargar fácilmente un modelo `.tflite` para realizar predicciones.

```python
import tensorflow as tf
import numpy as np

# Cargar el modelo TFLite. Asegúrate de que la ruta sea correcta.
# Por ejemplo, para EMGHandNet-2D
interpreter = tf.lite.Interpreter(model_path="models/EMGHandNetIID.tflite")
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preparar una secuencia de datos de entrada (ejemplo con datos aleatorios)
# La forma de entrada debe coincidir con la del modelo.
# Para EMGHandNet-2D: (1, 5, 20, 10, 1) -> (batch, secuencias, ventanas, canales, 1)
input_shape = input_details[0]['shape']
input_data = np.random.rand(*input_shape).astype(np.float32)

# Establecer el tensor de entrada y ejecutar la inferencia
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Obtener el resultado
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_gesture_index = np.argmax(output_data)

# Las etiquetas van de 1 a 12, mientras que el índice de salida va de 0 a 11.
predicted_gesture_label = predicted_gesture_index + 1

print(f"Índice de salida predicho: {predicted_gesture_index}")
print(f"Etiqueta de gesto predicha: {predicted_gesture_label}")
