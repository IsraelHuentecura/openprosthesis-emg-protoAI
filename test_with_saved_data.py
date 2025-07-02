"""
Evaluación de Modelos EMG con Conjuntos de Datos de Prueba
=========================================================

Este script permite evaluar los mejores modelos utilizando los conjuntos
de datos de prueba previamente guardados.

Autor: Israel Huentecura
"""

import os
import json
import numpy as np
import tensorflow as tf
from src.load_models import BEST_MODELS, load_best_model
from src.tflite_inference import BEST_TFLITE_MODELS, load_tflite_model, predict_with_tflite_model

# Directorio donde se encuentran los datasets guardados
SAVED_DATA_DIR = "saved_datasets"
MODELS_DIR = "models"
MODELS_TFLITE_DIR = "models_tflite"

def load_metadata():
    """Cargar metadatos de los conjuntos de datos"""
    with open(os.path.join(SAVED_DATA_DIR, "meta.json"), "r") as f:
        return json.load(f)

def load_test_data():
    """Cargar los datos de prueba"""
    print("Cargando datos de prueba...")
    
    # Cargar datos raw y features
    test_X_raw = np.load(os.path.join(SAVED_DATA_DIR, "test_X_raw.npy"))
    test_X_feats = np.load(os.path.join(SAVED_DATA_DIR, "test_X_feats.npy"))
    
    # Cargar etiquetas
    test_y_lbl = np.load(os.path.join(SAVED_DATA_DIR, "test_y_lbl.npy"))
    y_te = np.load(os.path.join(SAVED_DATA_DIR, "y_te.npy"))
    
    print(f"✅ Datos cargados: {test_X_raw.shape[0]} muestras de prueba")
    return test_X_raw, test_X_feats, test_y_lbl, y_te

def evaluate_keras_models(test_X_raw, test_X_feats, y_te):
    """Evaluar todos los modelos Keras con datos de prueba"""
    print("\n" + "=" * 80)
    print(" Evaluación de Modelos Keras (.keras) ".center(80, "="))
    print("=" * 80)
    
    results = {}
    
    for arch, model_file in BEST_MODELS.items():
        print(f"\n{'-' * 50}")
        print(f"Evaluando {arch}: {model_file}")
        
        try:
            # Cargar modelo
            model = load_best_model(arch, verbose=False)
            
            # Preparar entrada según el tipo de modelo
            if len(model.inputs) > 1:
                # Modelos duales (raw + features)
                inputs = {
                    model.inputs[0].name.split(':')[0]: test_X_raw,
                    model.inputs[1].name.split(':')[0]: test_X_feats
                }
            else:
                # Modelos con entrada única
                inputs = test_X_raw
            
            # Evaluar
            loss, accuracy = model.evaluate(inputs, y_te, verbose=0)
            results[arch] = {
                'accuracy': float(accuracy),
                'loss': float(loss),
                'model_file': model_file
            }
            
            print(f"✅ Exactitud en prueba: {accuracy:.4f} (loss: {loss:.4f})")
            
        except Exception as e:
            print(f"❌ Error al evaluar {arch}: {e}")
    
    # Mostrar resumen
    print("\n" + "=" * 80)
    print(" Resumen de Resultados ".center(80, "="))
    print("=" * 80)
    
    # Ordenar por exactitud
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\n{:<20} {:<10} {:<40}".format("Arquitectura", "Exactitud", "Modelo"))
    print("-" * 70)
    for arch, res in sorted_results:
        print("{:<20} {:<10.4f} {:<40}".format(
            arch, res['accuracy'], res['model_file']
        ))
    
    return results

def evaluate_tflite_models(test_X_raw, test_X_feats, test_y_lbl):
    """Evaluar modelos TFLite con datos de prueba"""
    print("\n" + "=" * 80)
    print(" Evaluación de Modelos TFLite para Dispositivos de Bajos Recursos ".center(80, "="))
    print("=" * 80)
    
    results = {}
    
    for arch, model_info in BEST_TFLITE_MODELS.items():
        model_file = model_info['file']
        print(f"\n{'-' * 50}")
        print(f"Evaluando {arch}: {model_file}")
        
        try:
            # Cargar modelo TFLite
            model_path = os.path.join(MODELS_TFLITE_DIR, model_file)
            interpreter = load_tflite_model(model_path)
            
            # Preparar entradas
            correct = 0
            total = min(100, test_X_raw.shape[0])  # Limitar a 100 muestras para rapidez
            
            print(f"Evaluando con {total} muestras de prueba...")
            
            for i in range(total):
                # Preparar entrada según el tipo de modelo
                if 'input_shapes' in model_info:
                    # Modelos duales (raw + features)
                    sample_raw = test_X_raw[i:i+1]
                    sample_feat = test_X_feats[i:i+1]
                    
                    # Ajustar formas si es necesario
                    if sample_raw.shape != model_info['input_shapes']['raw_input']:
                        sample_raw = np.reshape(sample_raw, model_info['input_shapes']['raw_input'])
                    
                    input_data = {
                        'raw_input': sample_raw,
                        'feat_input': sample_feat
                    }
                else:
                    # Modelos con entrada única
                    sample = test_X_raw[i:i+1]
                    
                    # Ajustar forma si es necesario
                    if sample.shape != model_info['input_shape']:
                        sample = np.reshape(sample, model_info['input_shape'])
                        
                    input_data = sample
                
                # Predecir
                pred = predict_with_tflite_model(interpreter, input_data, verbose=False)
                pred_class = np.argmax(pred[0])
                true_class = test_y_lbl[i] - 1  # Restar 1 para ajustar a índice 0-11
                
                if pred_class == true_class:
                    correct += 1
                
            accuracy = correct / total
            results[arch] = {
                'accuracy': accuracy,
                'model_file': model_file,
                'samples_tested': total
            }
            
            print(f"✅ Exactitud en muestra de prueba: {accuracy:.4f} ({correct}/{total} correctos)")
            
        except Exception as e:
            print(f"❌ Error al evaluar {arch}: {e}")
    
    # Mostrar resumen
    print("\n" + "=" * 80)
    print(" Resumen de Resultados TFLite ".center(80, "="))
    print("=" * 80)
    
    # Ordenar por exactitud
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\n{:<20} {:<10} {:<40}".format("Arquitectura", "Exactitud", "Modelo TFLite"))
    print("-" * 70)
    for arch, res in sorted_results:
        print("{:<20} {:<10.4f} {:<40}".format(
            arch, res['accuracy'], res['model_file']
        ))
    
    return results

def main():
    """Función principal"""
    print("\n📊 Evaluación de Modelos EMG con Datos de Prueba Guardados")
    print("=" * 70)
    
    # Verificar existencia de directorios
    if not os.path.exists(SAVED_DATA_DIR):
        print(f"❌ ERROR: No se encontró el directorio {SAVED_DATA_DIR}")
        print("   Ejecute primero el notebook para generar los datasets guardados.")
        return
    
    if not os.path.exists(MODELS_DIR):
        print(f"❌ ERROR: No se encontró el directorio {MODELS_DIR}")
        print("   Asegúrese de que los modelos .keras estén en la carpeta 'models/'")
        return
    
    # Cargar datos
    test_X_raw, test_X_feats, test_y_lbl, y_te = load_test_data()
    
    # Evaluar modelos Keras
    keras_results = evaluate_keras_models(test_X_raw, test_X_feats, y_te)
    
    # Evaluar modelos TFLite si existe el directorio
    if os.path.exists(MODELS_TFLITE_DIR):
        tflite_results = evaluate_tflite_models(test_X_raw, test_X_feats, test_y_lbl)
    else:
        print(f"\n⚠️ No se encontró el directorio {MODELS_TFLITE_DIR}")
        print("   Se omitió la evaluación de modelos TFLite.")
    
    print("\n✨ Evaluación completada")

if __name__ == "__main__":
    main()
