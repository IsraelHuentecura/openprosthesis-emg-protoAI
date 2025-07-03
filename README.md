# Clasificación de Gestos sEMG con Modelos Ligeros de Deep Learning

Este repositorio contiene los modelos `.h5`, el código y las herramientas desarrolladas para clasificar 12 gestos de la mano a partir de señales de **electromiografía de superficie (sEMG)**, como parte del proyecto de título de **Israel Huentecura** en el contexto de **[ProtoIA](https://github.com/ProtoAI-cl/ProtoAI)**.

El objetivo es proveer **modelos de alto rendimiento y bajo costo computacional**, especialmente diseñados para ser desplegados en dispositivos de bajo consumo como Raspberry Pi o microcontroladores más potentes.

---

## 🧠 ¿Qué Modelo Deberías Usar?

La elección del modelo depende del hardware donde se quiera ejecutar:

### 🔹 Para Computadores Monoplaca (Ej. Raspberry Pi, Jetson Nano)

| Modelo              | Precisión (Test)  | Tamaño `.h5` |
| ------------------- | ----------------- | ------------ |
| **HyT-Net**         | **99.83% ± 0.02** | \~10.4 MB    |
| DualStream-Original | 99.82% ± 0.03     | \~36.6 MB    |
| EMGHandNet-Original | 99.80% ± 0.02     | \~19.8 MB    |

**✔ Recomendado: HyT-Net**, por su precisión máxima con menor tamaño que los modelos de referencia.

---

### 🔹 Para Prototipos en Desarrollo y Pruebas Locales

| Modelo            | Precisión (Test)  | Tamaño `.h5` |
| ----------------- | ----------------- | ------------ |
| **EMGHandNet-2D** | **99.78% ± 0.03** | \~4.2 MB     |
| DualStream-Lite   | 99.41% ± 0.03     | \~3.4 MB     |
| CRNN-Attn         | 98.55% ± 0.15     | \~3.6 MB     |

**✔ Recomendado: EMGHandNet-2D**, el mejor balance entre precisión, arquitectura simple (*end-to-end*) y tamaño reducido.

> ⚠️ Ten en cuenta que actualmente estos modelos no están preparados para microcontroladores sin conversión a TFLite o cuantización. Este repositorio se centra en desarrollo y evaluación en entorno de entrenamiento.

---

## 🗂️ Contenido del Repositorio

```bash
📁 models_h5/               # Modelos pre-entrenados en formato .h5
📁 models_tflite/           # Modelos convertidos a TFLite
📁 saved_datasets/          # Dataset de entrenamiento y prueba (debes descargarlo)
📁 test_data_sample/        # Dataset de prueba (no debe ser descargado)
📄 script_test_h5.py        # Evalúa todos los modelos .h5 con un dataset de muestra
📄 ejemplo_finetuning.py    # Fine-tuning del modelo HyT-Net
📄 notebook_entrenamiento.ipynb # Notebook para construir y entrenar modelos desde cero
📄 requirements.txt         # Dependencias necesarias
```

---

## 🚀 Cómo Empezar

### 1. Instala dependencias

Usa un entorno virtual y ejecuta:

```bash
pip install -r requirements.txt
```

> Este proyecto está probado en **Python 3.9** y usa **TensorFlow 2.10.0**.

---

### 2. Descarga el Dataset Procesado

Debes tener una carpeta `saved_datasets/` con los siguientes datasets:

* `train_ds_h/` y `test_ds_h/`: para entrenamiento y prueba
* `sample_raw_ds/` y `sample_hybrid_ds/`: para pruebas rápidas

> 🔗 Agrega tu link de descarga aquí para que otros puedan acceder a estos datos.

---

## ⚙️ Opciones de Uso

### ✅ A. Evaluar Modelos `.h5` Rápidamente

```bash
python script_test_h5.py
```

Este script evalúa los modelos preentrenados con un dataset pequeño y muestra un resumen de precisión por modelo.

---

### ✅ B. Entrenar Desde Cero

Abre el notebook:

```bash
jupyter lab
```

Edita y ejecuta `notebook_entrenamiento.ipynb` para construir modelos desde cero.

---

### ✅ C. Fine-tuning Personalizado

Para adaptar un modelo preentrenado (ej. `HyT-Net`) a nuevos datos:

```bash
python ejemplo_finetuning.py
```

El script:

1. Carga el modelo preentrenado
2. Congela las capas base
3. Reentrena las últimas capas con una tasa de aprendizaje baja (`1e-5`) por 10 épocas
4. Evalúa el nuevo rendimiento

> ✍️ Puedes modificar `ejemplo_finetuning.py` para aplicar fine-tuning a otros modelos.

---

## 📊 Resultados Técnicos (Resumen)

| Modelo              | Test Accuracy (± σ) | Tamaño `.h5` | Entradas Requeridas       |
| ------------------- | ------------------- | ------------ | ------------------------- |
| **HyT-Net**         | 0.9983 ± 0.0002     | \~10.4 MB    | Señales crudas + features |
| DualStream-Original | 0.9982 ± 0.0003     | \~36.6 MB    | Señales crudas + features |
| EMGHandNet-Original | 0.9980 ± 0.0002     | \~19.8 MB    | Señales crudas            |
| **EMGHandNet-2D**   | 0.9978 ± 0.0003     | \~4.2 MB     | Señales crudas            |
| DualStream-Lite     | 0.9941 ± 0.0003     | \~3.4 MB     | Señales crudas + features |
| CRNN-Attn           | 0.9855 ± 0.0015     | \~3.6 MB     | Señales crudas + features |

Fuente: Evaluación sobre NinaPro DB1 - Ejercicio A, siguiendo protocolo de validación cruzada 10-fold.

---

## 📄 Referencias

* Paper asociado: [`EMG-Based Gesture Recognition Using Hybrid and Transformer-Based Deep Learning Models`](./EMG_Israel_Huentecura_IEEE.pdf)
* Dataset base: [NinaPro DB1](http://ninapro.hevs.ch/DB1)
* Proyecto madre: [ProtoIA](https://github.com/ProtoAI-cl/ProtoAI)

---

## 🧠 Autoría

Desarrollado por **Israel Huentecura Rodríguez**
---
