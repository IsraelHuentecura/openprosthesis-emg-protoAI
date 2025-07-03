### 2. Script de Ejemplo para Fine-tuning
import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path

# ------------------------------------------------------------------------- #
# 🧠 EJEMPLO DE FINE-TUNING PARA HyT-Net
# ------------------------------------------------------------------------- #
# El fine-tuning consiste en tomar un modelo pre-entrenado, congelar la
# mayoría de sus capas (el "cuerpo" que extrae características) y re-entrenar
# únicamente las últimas capas (la "cabeza" clasificadora) con una tasa de
# aprendizaje muy baja. Es útil para adaptar un modelo a nuevos datos o
# para intentar exprimir un poco más de rendimiento.
# ------------------------------------------------------------------------- #

# --- 1. Configuración ---
SAVE_ROOT = Path("saved_datasets")
MODELS_DIR = Path("models_h5")
FINETUNE_MODEL_NAME = "HyT-Net_fold1.h5"  # Modelo que vamos a cargar y mejorar
FINETUNE_EPOCHS = 10                     # Entrenar por pocas épocas
FINETUNE_LR = 1e-5                       # Tasa de aprendizaje muy baja

# --- Capa de Atención (necesaria si el modelo la usa) ---
# Nota: Tu modelo HyT-Net usa MultiHeadAttention, que es una capa estándar
# de Keras. Si tienes otro modelo que usa una capa 'Attention' personalizada,
# deberás mantener esta clase. Para HyT-Net no es estrictamente necesaria
# al momento de cargar el modelo si no se guardó con ella.
class Attention(tf.keras.layers.Layer):
    def build(self, inp_shape):
        u = inp_shape[-1]
        self.W = tf.keras.layers.Dense(u, use_bias=True, name="att_W")
        self.u = tf.keras.layers.Dense(1, use_bias=False, name="att_u")
    def call(self, x):
        v = tf.tanh(self.W(x))
        vu = tf.squeeze(self.u(v), -1)
        al = tf.nn.softmax(vu, axis=1)
        return tf.reduce_sum(x * tf.expand_dims(al, -1), 1)

# --- 2. Carga de Datos ---
print("💾 Cargando datos de entrenamiento y prueba...")
# Asegúrate de que las rutas a tus datasets son correctas
if not (SAVE_ROOT / "train_ds_h").exists() or not (SAVE_ROOT / "test_ds_h").exists():
     raise FileNotFoundError(f"Los directorios del dataset no se encontraron en '{SAVE_ROOT}'.")
train_ds_h = tf.data.Dataset.load(str(SAVE_ROOT / "train_ds_h"))
test_ds_h = tf.data.Dataset.load(str(SAVE_ROOT / "test_ds_h"))
print("✅ Datos cargados.")

# --- 3. Cargar Modelo Pre-entrenado ---
model_path = MODELS_DIR / FINETUNE_MODEL_NAME
if not model_path.exists():
    raise FileNotFoundError(f"El modelo '{model_path}' no existe. Asegúrate de tener los modelos .h5.")

print(f"\n🧠 Cargando modelo base: {model_path}")
# Se carga sin compilar para poder modificarlo.
# Si tu modelo no usa la capa 'Attention' personalizada, puedes remover `custom_objects`.
base_model = tf.keras.models.load_model(model_path, custom_objects={'Attention': Attention}, compile=False)
base_model.summary() # Imprime un resumen para verificar los nombres de las capas

# --- 4. Congelar Capas Base ---
print("\n❄️  Congelando capas del modelo base...")
# Congelamos TODAS las capas primero
for layer in base_model.layers:
    layer.trainable = False

# Descongelamos las ÚLTIMAS capas para re-entrenarlas (el clasificador de HyT-Net)
# **CORRECCIÓN:** Se usan los nombres de las capas del clasificador final de tu modelo HyT-Net.
# Keras puede añadir un sufijo numérico si los nombres se repiten (ej. "dropout_1").
# Verifica los nombres exactos con `base_model.summary()`.
try:
    # Intenta obtener las capas con nombres por defecto
    base_model.get_layer("global_average_pooling1d").trainable = True
    base_model.get_layer("dropout").trainable = True
    base_model.get_layer("dense_2").trainable = True # La primera Dense del clasificador
    base_model.get_layer("dropout_1").trainable = True
    base_model.get_layer("output").trainable = True    # La capa de salida SIEMPRE debe ser entrenable
except ValueError as e:
    print(f"\n⚠️  Error al obtener capa: {e}")
    print("Verifica los nombres de las capas en el resumen del modelo impreso arriba y ajústalos en el script.")
    exit()


print("\nCapas que serán re-entrenadas (fine-tuning):")
for layer in base_model.layers:
    if layer.trainable:
        print(f"  -> {layer.name}")

print("\nParámetros entrenables ANTES del fine-tuning:", np.sum([np.prod(v.shape) for v in base_model.trainable_variables]))

# --- 5. Re-compilar el Modelo ---
print(f"\n⚙️  Re-compilando modelo con una tasa de aprendizaje baja (lr={FINETUNE_LR})")
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- 6. Ejecutar Fine-Tuning ---
print(f"\n🚀 Iniciando fine-tuning por {FINETUNE_EPOCHS} épocas...")
history = base_model.fit(
    train_ds_h,
    epochs=FINETUNE_EPOCHS,
    validation_data=test_ds_h, # Usamos el test set como validación en este ejemplo
    verbose=1
)

# --- 7. Evaluación Final ---
print("\n📈 Evaluando el modelo después del fine-tuning...")
loss, accuracy = base_model.evaluate(test_ds_h, verbose=0)
print(f"  -> Precisión final en Test: {accuracy:.2%}")