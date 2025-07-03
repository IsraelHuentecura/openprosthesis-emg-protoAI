#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta los MEJORES modelos de cada arquitectura a TFLite *sin* cuantización (float32).
Versión final con flags de compatibilidad para capas recurrentes.

Israel Huentecura · Junio 2025
"""

import os, sys, time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import tensorflow as tf

# ╔═════════════════════════════════════════════════════════════════╗
# 0.  CONFIGURACIÓN Y CAPA PERSONALIZADA
# ╚═════════════════════════════════════════════════════════════════╝
MODELS_DIR = Path("models")
TFLITE_DIR = Path("models_tflite"); TFLITE_DIR.mkdir(exist_ok=True)

# --- Capa Attention (necesaria para cargar modelos que la usen) ---
class Attention(tf.keras.layers.Layer):
    def build(self, inp_shape):
        u = inp_shape[-1]
        self.W = self.add_weight("W", (u, u))
        self.b = self.add_weight("b", (u,))
        self.u = self.add_weight("u", (u, 1))
    def call(self, x):
        v   = tf.tanh(tf.tensordot(x, self.W, 1) + self.b)
        vu  = tf.tensordot(v, self.u, 1)
        α   = tf.nn.softmax(tf.squeeze(vu, -1), 1)
        return tf.reduce_sum(x * tf.expand_dims(α, -1), 1)
CUSTOM_OBJECTS = {"Attention": Attention}

# --- INICIO DE LA MODIFICACIÓN ---
# Lista específica de los mejores modelos a convertir
BEST_MODEL_FILES = [
    "HyT-Net_fold1_BEST.keras",
    "DualStream-Original_fold3_BEST.keras",
    "EMGHandNet-2D_fold6_BEST.keras",
    "EMGHandNet-Original_fold1_BEST.keras",
    "DualStream-Lite_fold8_BEST.keras",
    "CRNN-Attn_fold1_BEST.keras",
]
# --- FIN DE LA MODIFICACIÓN ---

# ╔═════════════════════════════════════════════════════════════════╗
# 1.  FUNCIÓN DE CONVERSIÓN (Sin cambios)
# ╚═════════════════════════════════════════════════════════════════╝
def to_float32_tflite(model: tf.keras.Model) -> bytes:
    """Convierte un modelo Keras a TFLite float32, añadiendo flags de compatibilidad."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Flags de compatibilidad para capas RNN (GRU/LSTM)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Operadores estándar de TFLite
        tf.lite.OpsSet.SELECT_TF_OPS    # Permite usar operadores de TensorFlow
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    return converter.convert()

# ╔═════════════════════════════════════════════════════════════════╗
# 2.  BUCLE PRINCIPAL DE EXPORTACIÓN
# ╚═════════════════════════════════════════════════════════════════╝
results: List[Dict] = []

# --- INICIO DE LA MODIFICACIÓN ---
# En lugar de buscar en la carpeta, usamos la lista predefinida
checkpoints = [MODELS_DIR / fname for fname in BEST_MODEL_FILES]
# --- FIN DE LA MODIFICACIÓN ---

if not checkpoints:
    sys.exit("⚠️  La lista 'BEST_MODEL_FILES' está vacía o los archivos no se pudieron encontrar.")

print(f"🔍  Se procesarán {len(checkpoints)} modelos seleccionados.\n")

for ck_path in checkpoints:
    tag = ck_path.stem.replace("_BEST", "")
    try:
        if not ck_path.exists():
            print(f"🚧  Exportando {tag:<30s}  ❌  Archivo no encontrado. Saltando.")
            continue
            
        print(f"🚧  Exportando {tag:<30s}", end="")
        
        model = tf.keras.models.load_model(ck_path, custom_objects=CUSTOM_OBJECTS)

        # Usar la función de conversión corregida
        tflite_bytes = to_float32_tflite(model)
        
        tflite_path = TFLITE_DIR / f"{tag}_float32.tflite"
        tflite_path.write_bytes(tflite_bytes)
        size_kb = tflite_path.stat().st_size / 1024

        results.append(dict(
            Model=tag,
            Size_kB=round(size_kb, 1)
        ))
        print(f"  ✔  {size_kb:7.1f} kB")

    except Exception as e:
        print(f"\n   ❌  Error con {tag}: {e}")
    
    finally:
        tf.keras.backend.clear_session()

# ╔═════════════════════════════════════════════════════════════════╗
# 3.  REPORTE FINAL (Sin cambios)
# ╚═════════════════════════════════════════════════════════════════╝
if results:
    df = pd.DataFrame(results).sort_values("Size_kB")
    
    # Agrupar por arquitectura para el resumen
    summary_data = []
    for model_file in BEST_MODEL_FILES:
        model_name = model_file.split('_fold')[0]
        # Buscar el resultado correspondiente en la lista de resultados
        for res in results:
            if res['Model'] == model_file.replace('_BEST.keras', ''):
                summary_data.append({
                    'Arquitectura': model_name,
                    'Tamaño_kB': res['Size_kB']
                })
                break
    
    summary = pd.DataFrame(summary_data).sort_values("Tamaño_kB")

    summary_path = TFLITE_DIR / "best_models_float32_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.1f")
    
    print("\n" + "="*50)
    print("📋  Resumen de Exportación (Mejores Modelos)".center(50))
    print("="*50)
    print(summary.to_string(index=False, float_format="%.1f"))
    print(f"\n📝  Reporte CSV guardado en: {summary_path}")
else:
    print("\n😕  No se pudo exportar ningún modelo.")