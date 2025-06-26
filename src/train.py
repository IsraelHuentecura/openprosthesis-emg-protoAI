import os
import json
import datetime as dt
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# Import from local modules
import config
import models
import data_utils

# --- Setup Reproducibility ---
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.keras.utils.set_random_seed(config.SEED)

# --- Define Model Catalog ---
# Maps model names to their builder functions and input requirements.
MODELS_CATALOG = {
    "DualStream-Lite": {"builder": models.build_dualstream_lite, "uses_features": True},
    "DualStream-Original": {"builder": models.build_dualstream_original, "uses_features": True},
    "EMGHandNet-2D": {"builder": models.build_emghandnet_2d, "uses_features": False},
    "EMGHandNet-Original": {"builder": models.build_emghandnet_original, "uses_features": False},
    "HyT-Net": {"builder": models.build_hyt_net, "uses_features": True},
    "CRNN-Attn": {"builder": models.build_crnn_attn, "uses_features": True},
}

def get_callbacks(model_name: str, fold: int) -> list:
    """Creates a list of Keras callbacks for training."""
    tag = f"{model_name}_fold_{fold}"
    log_dir = config.RUNS_DIR / f"{tag}_{dt.datetime.now():%Y%m%d-%H%M%S}"
    
    # Backup callbacks are useful for resuming long training sessions
    backup_callback = tf.keras.callbacks.BackupAndRestore(
        backup_dir=config.BACKUP_DIR / tag, delete_checkpoint=False
    )
    # Checkpoint to save the best model from the fold
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=config.MODELS_DIR / f"{tag}_best.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=0
    )
    # Early stopping to prevent overfitting
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    )
    # Reduce learning rate when a metric has stopped improving
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    # TensorBoard for logging
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    return [
        backup_callback, checkpoint_callback, early_stopping_callback,
        reduce_lr_callback, tensorboard_callback
    ]

def run_cross_validation(X_raw, y_labels, F_hand=None):
    """
    Runs the full 10-fold cross-validation and evaluation pipeline.
    """
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    stratify_labels = np.argmax(y_labels, axis=1)

    fold_results = []
    # Store the path to the best performing checkpoint for each model across all folds
    best_checkpoints = {name: {"val_acc": -1, "path": None} for name in MODELS_CATALOG}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, stratify_labels), 1):
        print(f"\n----- FOLD {fold}/{config.N_FOLDS} -----")
        
        # Create fold-specific data splits
        X_train, X_val = X_raw[train_idx], X_raw[val_idx]
        y_train, y_val = y_labels[train_idx], y_labels[val_idx]
        F_train, F_val = (F_hand[train_idx], F_hand[val_idx]) if F_hand is not None else (None, None)

        for model_name, props in MODELS_CATALOG.items():
            print(f"  Training model: {model_name}")
            
            tf.keras.backend.clear_session()
            
            # Build the model
            builder_fn = props["builder"]
            model = builder_fn(
                raw_shape=config.RAW_SHAPE,
                feat_dim=config.FEAT_DIM,
                t_subwin=config.T_SUBWIN,
                n_cls=config.NUM_CLASSES
            ) if props["uses_features"] else builder_fn(
                raw_shape=config.RAW_SHAPE,
                t_subwin=config.T_SUBWIN,
                n_cls=config.NUM_CLASSES
            )
            
            # Create tf.data.Dataset
            train_ds = data_utils.make_tf_dataset(
                X_train, y_train, F_train if props["uses_features"] else None,
                batch_size=config.BATCH_SIZE, shuffle=True
            )
            val_ds = data_utils.make_tf_dataset(
                X_val, y_val, F_val if props["uses_features"] else None,
                batch_size=config.BATCH_SIZE, shuffle=False
            )

            # Fit the model
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=config.EPOCHS,
                callbacks=get_callbacks(model_name, fold),
                verbose=2
            )

            # Record results and update best checkpoint
            best_val_acc = max(history.history["val_accuracy"])
            fold_results.append([model_name, fold, best_val_acc])

            if best_val_acc > best_checkpoints[model_name]["val_acc"]:
                best_checkpoints[model_name]["val_acc"] = best_val_acc
                best_checkpoints[model_name]["path"] = config.MODELS_DIR / f"{model_name}_fold_{fold}_best.keras"

    return pd.DataFrame(fold_results, columns=["Model", "Fold", "Val_Acc"]), best_checkpoints

def evaluate_final_models(best_checkpoints, X_test, y_test, F_test=None):
    """Evaluates the best checkpoint of each model on the hold-out test set."""
    test_results = []
    print("\n----- Evaluating Best Models on Hold-Out Test Set -----")

    for model_name, data in best_checkpoints.items():
        if data["path"] is None or not data["path"].exists():
            print(f"  ⚠️ Skipping {model_name}: No valid checkpoint found.")
            continue
        
        print(f"  🧪 Testing {model_name} from {data['path'].name}")
        model = tf.keras.models.load_model(data["path"], custom_objects={"AdditiveAttention": models.AdditiveAttention})
        
        uses_features = MODELS_CATALOG[model_name]["uses_features"]
        test_ds = data_utils.make_tf_dataset(
            X_test, y_test, F_test if uses_features else None,
            batch_size=config.BATCH_SIZE, shuffle=False
        )
        
        loss, acc = model.evaluate(test_ds, verbose=0)
        test_results.append([model_name, acc, loss])
        
    return pd.DataFrame(test_results, columns=["Model", "Test_Acc", "Test_Loss"])

def main():
    """Main execution script."""
    # Create necessary directories
    config.RUNS_DIR.mkdir(exist_ok=True)
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.BACKUP_DIR.mkdir(exist_ok=True)

    # --- 1. Load Data ---
    # Placeholder: You should load your preprocessed numpy arrays here.
    # For example:
    # X_raw_full = np.load(config.DATA_DIR / "processed" / "X_raw_full.npy")
    # y_labels_full = np.load(config.DATA_DIR / "processed" / "y_labels_full.npy")
    # F_hand_full = np.load(config.DATA_DIR / "processed" / "F_hand_full.npy")
    print("⚠️ Placeholder: Loading dummy data. Replace with your actual data loading logic.")
    num_samples = 2000
    X_raw_full = np.random.rand(num_samples, config.T_SUBWIN, *config.RAW_SHAPE)
    y_labels_full = tf.keras.utils.to_categorical(np.random.randint(0, config.NUM_CLASSES, num_samples))
    F_hand_full = np.random.rand(num_samples, config.T_SUBWIN, config.FEAT_DIM)

    # --- 2. Create Train/Test Split ---
    (X_train_val, F_train_val, y_train_val), (X_test, F_test, y_test) = \
        data_utils.create_train_test_split(
            X_raw_full, y_labels_full, F_hand_full,
            test_size=config.TEST_SPLIT_PCT, seed=config.SEED
        )

    # --- 3. Run K-Fold Cross-Validation ---
    df_folds, best_checkpoints = run_cross_validation(X_train_val, y_train_val, F_train_val)

    # --- 4. Evaluate on Test Set ---
    df_test = evaluate_final_models(best_checkpoints, X_test, y_test, F_test)

    # --- 5. Report and Save Results ---
    df_cv_summary = df_folds.groupby("Model")['Val_Acc'].agg(['mean', 'std']).round(4)
    
    print("\n--- Cross-Validation Summary ---")
    print(df_cv_summary)
    
    print("\n--- Final Test Set Performance ---")
    print(df_test.sort_values("Test_Acc", ascending=False).reset_index(drop=True))

    # Save results to CSV files
    df_folds.to_csv(config.RUNS_DIR / "cv_fold_results.csv", index=False)
    df_cv_summary.to_csv(config.RUNS_DIR / "cv_summary_results.csv")
    df_test.to_csv(config.RUNS_DIR / "test_set_results.csv", index=False)
    print(f"\n✅ Results saved to {config.RUNS_DIR}")

if __name__ == "__main__":
    main()