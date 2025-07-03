# probar_modelos.py
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras import layers

# Se necesita la definición de la capa 'Attention' para cargar algunos modelos
class Attention(layers.Layer):
    """Capa de Atención para modelos tipo CRNN."""
    def build(self, inp_shape):
        u = inp_shape[-1]
        self.W = self.add_weight("W", (u, u), initializer="glorot_uniform")
        self.b = self.add_weight("b", (u,), initializer="zeros")
        self.u = self.add_weight("u", (u, 1), initializer="glorot_uniform")
    def call(self, x):
        v = tf.tanh(tf.tensordot(x, self.W, 1) + self.b)
        vu = tf.tensordot(v, self.u, 1)
        al = tf.nn.softmax(tf.squeeze(vu, -1), 1)
        return tf.reduce_sum(x * tf.expand_dims(al, -1), 1)

# --- CONFIGURACIÓN ---
MODELS_H5_DIR = Path("models_h5")
SAMPLE_DATA_DIR = Path("test_data_sample")

# Lista de los mejores modelos por arquitectura y su tipo de dato
BEST_MODELS_INFO = [
    {"name": "HyT-Net", "file": "HyT-Net_fold1.h5", "type": "hybrid"},
    {"name": "DualStream-Original", "file": "DualStream-Original_fold3.h5", "type": "hybrid"},
    {"name": "EMGHandNet-2D", "file": "EMGHandNet-2D_fold6.h5", "type": "raw"},
    {"name": "EMGHandNet-Original", "file": "EMGHandNet-Original_fold1.h5", "type": "raw"},
    {"name": "DualStream-Lite", "file": "DualStream-Lite_fold8.h5", "type": "hybrid"},
    {"name": "CRNN-Attn", "file": "CRNN-Attn_fold1.h5", "type": "hybrid"},
]

# Nombres de los gestos para una visualización clara
CLASS_NAMES = [
    "Flexión de Índice", "Extensión de Índice", "Flexión de Dedo Medio", "Extensión de Dedo Medio",
    "Flexión de Anular", "Extensión de Anular", "Flexión de Meñique", "Extensión de Meñique",
    "Aducción de Pulgar", "Abducción de Pulgar", "Flexión de Pulgar", "Extensión de Pulgar"
]

# --- LÓGICA DE LA APLICACIÓN ---

def print_summary(all_results):
    """Imprime una tabla de resumen con los resultados de la evaluación."""
    print("\n\n" + "="*85)
    print(" " * 32 + "📊 RESUMEN DE LA EVALUACIÓN 📊")
    print("="*85)

    for result in all_results:
        model_name = result["model_name"]
        samples = result["results"]
        
        correct_predictions = sum(1 for s in samples if s["is_correct"])
        total_samples = len(samples)
        accuracy = (correct_predictions / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"\n--- Arquitectura: {model_name} " + "-"*(63-len(model_name)))
        print(f" -> Precisión en la muestra: {correct_predictions}/{total_samples} ({accuracy:.1f}%)")
        print("---" + "-"*79)
        
        # Encabezados de la tabla
        header = f"| {'Muestra':<7} | {'Gesto Real':<25} | {'Predicción':<25} | {'Resultado':<10} |"
        print(header)
        print(f"|{'-'*9}|{'-'*27}|{'-'*27}|{'-'*12}|")

        # Filas con los resultados
        for i, sample in enumerate(samples):
            true_name = CLASS_NAMES[sample["true_class"]]
            pred_name = CLASS_NAMES[sample["predicted_class"]]
            status_icon = "✅ CORRECTO" if sample["is_correct"] else "❌ INCORRECTO"
            
            row = f"| #{i+1:<6} | {true_name:<25} | {pred_name:<25} | {status_icon:<10} |"
            print(row)
        print("---" + "-"*79)

def test_best_models():
    """
    Carga los mejores modelos, realiza predicciones y almacena los resultados
    para generar un resumen final.
    """
    if not MODELS_H5_DIR.exists() or not SAMPLE_DATA_DIR.exists():
        print(f"❌ ERROR: No se encuentran las carpetas requeridas.")
        return

    print("Cargando datasets de muestra...")
    sample_datasets = {
        'hybrid': tf.data.Dataset.load(str(SAMPLE_DATA_DIR / "sample_hybrid_ds")),
        'raw': tf.data.Dataset.load(str(SAMPLE_DATA_DIR / "sample_raw_ds"))
    }
    print("✅ Muestras cargadas.")

    all_evaluation_results = []

    for model_info in BEST_MODELS_INFO:
        model_path = MODELS_H5_DIR / model_info["file"]
        print("\n" + "="*80)
        print(f"🚀 Probando: {model_info['name']} ({model_info['file']})")
        
        if not model_path.exists():
            print(f" -> ❌ Archivo no encontrado. Saltando...")
            continue

        model = tf.keras.models.load_model(model_path, custom_objects={'Attention': Attention}, compile=False)
        test_data = sample_datasets[model_info["type"]]
        
        model_sample_results = []
        for i, (features, label_data) in enumerate(test_data):
            if isinstance(features, dict):
                features_batch = {key: tf.expand_dims(value, axis=0) for key, value in features.items()}
            else:
                features_batch = tf.expand_dims(features, axis=0)

            prediction_probs = model(features_batch, training=False)
            predicted_class = np.argmax(prediction_probs[0])
            
            if isinstance(label_data, dict):
                label_tensor = list(label_data.values())[0]
            else: 
                label_tensor = label_data
            
            true_class = np.argmax(label_tensor)
            
            model_sample_results.append({
                "true_class": true_class,
                "predicted_class": predicted_class,
                "is_correct": predicted_class == true_class
            })
            print(f" -> Muestra #{i+1} procesada...", end='\r')
        
        all_evaluation_results.append({
            "model_name": model_info['name'],
            "results": model_sample_results
        })
        print(f" -> ¡Evaluación de {len(model_sample_results)} muestras completada!")
        
        del model
        tf.keras.backend.clear_session()

    # Al finalizar todas las pruebas, imprimir el resumen
    print_summary(all_evaluation_results)

if __name__ == "__main__":
    test_best_models()