# preparar_test.py (CORREGIDO)
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import shutil

# --- CONFIGURACIÓN ---
SAVE_ROOT = Path("saved_datasets")
SAMPLE_DATA_DIR = Path("test_data_sample")
NUM_SAMPLES_TO_SAVE = 100

# --- LÓGICA PRINCIPAL ---
def preparar_datos_de_prueba():
    """
    Carga los datasets de prueba, los desagrupa para obtener muestras
    individuales y guarda una pequeña fracción en un nuevo directorio.
    """
    if not SAVE_ROOT.exists():
        print(f"❌ ERROR: La carpeta de datasets '{SAVE_ROOT}' no existe.")
        return

    if SAMPLE_DATA_DIR.exists():
        print(f"🧹 Limpiando directorio existente: '{SAMPLE_DATA_DIR}'")
        shutil.rmtree(SAMPLE_DATA_DIR)
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Creado directorio para muestras: '{SAMPLE_DATA_DIR}'")

    print("\nCargando datasets de prueba completos...")
    test_ds_h = tf.data.Dataset.load(str(SAVE_ROOT / "test_ds_h"))
    test_ds_raw = tf.data.Dataset.load(str(SAVE_ROOT / "test_ds_raw"))
    print("✅ Datasets cargados.")

    # --- INICIO DE LA CORRECCIÓN ---
    # Desagrupar los datasets para asegurar que trabajamos con muestras individuales
    print("🧬 Desagrupando lotes (unbatch) para obtener muestras individuales...")
    test_ds_h_unbatched = test_ds_h.unbatch()
    test_ds_raw_unbatched = test_ds_raw.unbatch()
    # --- FIN DE LA CORRECCIÓN ---

    print(f"\nExtrayendo {NUM_SAMPLES_TO_SAVE} muestras de cada tipo de dato...")

    # Extraer y guardar la muestra de datos 'hybrid'
    sample_hybrid = test_ds_h_unbatched.take(NUM_SAMPLES_TO_SAVE)
    sample_hybrid_path = SAMPLE_DATA_DIR / "sample_hybrid_ds"
    sample_hybrid.save(str(sample_hybrid_path))
    print(f" -> Guardado subset 'hybrid' en: '{sample_hybrid_path}'")

    # Extraer y guardar la muestra de datos 'raw'
    sample_raw = test_ds_raw_unbatched.take(NUM_SAMPLES_TO_SAVE)
    sample_raw_path = SAMPLE_DATA_DIR / "sample_raw_ds"
    sample_raw.save(str(sample_raw_path))
    print(f" -> Guardado subset 'raw' en: '{sample_raw_path}'")

    print("\n✅ ¡Preparación completada! Ya puedes usar el script 'probar_modelos.py'.")

if __name__ == "__main__":
    preparar_datos_de_prueba()