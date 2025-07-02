"""
Generador de Conjuntos de Datos Ligeros para Pruebas de Modelos EMG
==================================================================

Este script genera versiones ligeras de los conjuntos de datos de prueba para
facilitar su subida a repositorios con límites de tamaño (como GitHub).

Autor: Israel Huentecura
"""

import os
import numpy as np
import json
import shutil

# Directorios
SRC_DIR = "saved_datasets"
DEST_DIR = "dataset_test"

def create_light_dataset(sample_size=100):
    """
    Crea un conjunto de datos de prueba ligero tomando una muestra
    del conjunto de datos original.
    
    Args:
        sample_size: Número de muestras a incluir en el conjunto ligero
    """
    print(f"\n📦 Generando conjunto de datos ligero con {sample_size} muestras...")
    
    # Crear directorio destino si no existe
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"✅ Directorio '{DEST_DIR}' creado")
    
    # Cargar datos originales
    print("⏳ Cargando datos originales...")
    
    # Verificar que existan los archivos necesarios
    required_files = ["test_X_raw.npy", "test_X_feats.npy", "test_y_lbl.npy", "y_te.npy", "meta.json"]
    for file in required_files:
        file_path = os.path.join(SRC_DIR, file)
        if not os.path.exists(file_path):
            print(f"❌ Error: No se encontró el archivo {file_path}")
            return False

    # Cargar datos
    test_X_raw = np.load(os.path.join(SRC_DIR, "test_X_raw.npy"))
    test_X_feats = np.load(os.path.join(SRC_DIR, "test_X_feats.npy"))
    test_y_lbl = np.load(os.path.join(SRC_DIR, "test_y_lbl.npy"))
    y_te = np.load(os.path.join(SRC_DIR, "y_te.npy"))
    
    print(f"✅ Datos cargados: {test_X_raw.shape[0]} muestras totales")
    
    # Verificar que haya suficientes muestras
    if test_X_raw.shape[0] < sample_size:
        print(f"⚠️ Advertencia: Solo hay {test_X_raw.shape[0]} muestras disponibles")
        sample_size = test_X_raw.shape[0]
    
    # Tomar una muestra aleatoria pero asegurando que haya representación de todas las clases
    np.random.seed(42)  # Para reproducibilidad
    
    # Identificar clases únicas (ajustadas a índice 0)
    unique_classes = np.unique(test_y_lbl)
    num_classes = len(unique_classes)
    
    # Calcular muestras por clase para mantener distribución similar
    samples_per_class = sample_size // num_classes
    remaining = sample_size % num_classes
    
    # Índices de las muestras a seleccionar
    selected_indices = []
    
    # Seleccionar muestras de cada clase
    for cls in unique_classes:
        # Encontrar índices de esta clase
        cls_indices = np.where(test_y_lbl == cls)[0]
        # Tomar muestra aleatoria de esta clase
        if len(cls_indices) > samples_per_class:
            cls_selected = np.random.choice(cls_indices, samples_per_class, replace=False)
        else:
            cls_selected = cls_indices  # Tomar todas si hay menos que las requeridas
        
        selected_indices.extend(cls_selected)
    
    # Agregar muestras adicionales si hay un resto
    if remaining > 0:
        # Seleccionar de índices no seleccionados
        all_indices = set(range(len(test_y_lbl)))
        remaining_indices = list(all_indices - set(selected_indices))
        if len(remaining_indices) >= remaining:
            additional = np.random.choice(remaining_indices, remaining, replace=False)
            selected_indices.extend(additional)
    
    # Asegurarse de que no hay duplicados y convertir a array
    selected_indices = np.array(list(set(selected_indices)))
    np.random.shuffle(selected_indices)  # Mezclar índices
    
    # Recortar al tamaño deseado si hay más
    if len(selected_indices) > sample_size:
        selected_indices = selected_indices[:sample_size]
    
    # Crear subconjuntos reducidos
    light_test_X_raw = test_X_raw[selected_indices]
    light_test_X_feats = test_X_feats[selected_indices]
    light_test_y_lbl = test_y_lbl[selected_indices]
    light_y_te = y_te[selected_indices]
    
    # Guardar subconjuntos
    print(f"⏳ Guardando conjunto ligero ({len(selected_indices)} muestras)...")
    np.save(os.path.join(DEST_DIR, "test_X_raw.npy"), light_test_X_raw)
    np.save(os.path.join(DEST_DIR, "test_X_feats.npy"), light_test_X_feats)
    np.save(os.path.join(DEST_DIR, "test_y_lbl.npy"), light_test_y_lbl)
    np.save(os.path.join(DEST_DIR, "y_te.npy"), light_y_te)
    
    # Copiar metadata
    with open(os.path.join(SRC_DIR, "meta.json"), "r") as f:
        metadata = json.load(f)
    
    # Actualizar metadata para el conjunto ligero
    metadata["light_dataset"] = True
    metadata["original_samples"] = test_X_raw.shape[0]
    metadata["light_samples"] = len(selected_indices)
    metadata["reduction_factor"] = round(test_X_raw.shape[0] / len(selected_indices), 2)
    
    with open(os.path.join(DEST_DIR, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Mostrar estadísticas
    original_size_mb = (test_X_raw.nbytes + test_X_feats.nbytes + 
                       test_y_lbl.nbytes + y_te.nbytes) / (1024 * 1024)
    
    light_size_mb = (light_test_X_raw.nbytes + light_test_X_feats.nbytes + 
                    light_test_y_lbl.nbytes + light_y_te.nbytes) / (1024 * 1024)
    
    print(f"\n✅ Conjunto de datos ligero creado exitosamente en '{DEST_DIR}/'")
    print(f"   - Muestras originales: {test_X_raw.shape[0]}")
    print(f"   - Muestras en conjunto ligero: {len(selected_indices)}")
    print(f"   - Tamaño original (aprox.): {original_size_mb:.2f} MB")
    print(f"   - Tamaño reducido (aprox.): {light_size_mb:.2f} MB")
    print(f"   - Factor de reducción: {original_size_mb / light_size_mb:.2f}x")
    
    # Verificar distribución de clases
    original_class_dist = np.bincount(test_y_lbl.astype(int))
    light_class_dist = np.bincount(light_test_y_lbl.astype(int))
    
    print("\nDistribución de clases:")
    print(f"{'Clase':>5} | {'Original':>8} | {'Ligero':>8} | {'%':>5}")
    print("-" * 35)
    
    for i in range(len(original_class_dist)):
        if i in unique_classes:
            orig = original_class_dist[i]
            light = light_class_dist[i]
            percent = (light / orig) * 100
            print(f"{i:>5} | {orig:>8} | {light:>8} | {percent:>5.1f}%")
    
    return True

if __name__ == "__main__":
    if not os.path.exists(SRC_DIR):
        print(f"❌ Error: No se encontró el directorio {SRC_DIR}")
        print("   Asegúrese de tener los datos originales antes de generar la versión ligera.")
    else:
        create_light_dataset(sample_size=200)  # Crear dataset ligero con 200 muestras
