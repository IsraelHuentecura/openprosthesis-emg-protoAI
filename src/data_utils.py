# src/data_utils.py

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def create_train_test_split(
    X_raw: np.ndarray, y_labels: np.ndarray, F_hand: np.ndarray = None,
    test_size: float = 0.2, seed: int = 42
) -> tuple:
    """
    Splits the full dataset into training/validation and a hold-out test set.

    Args:
        X_raw (np.ndarray): Array of raw sEMG window sequences.
        y_labels (np.ndarray): Array of one-hot encoded labels.
        F_hand (np.ndarray, optional): Array of handcrafted feature sequences.
        test_size (float): Proportion of the dataset to reserve for the test set.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the split datasets:
               ((X_train_val, F_train_val, y_train_val), (X_test, F_test, y_test))
    """
    indices = np.arange(len(y_labels))
    stratify_labels = np.argmax(y_labels, axis=1)

    # Split indices to ensure arrays are aligned
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_labels,
        shuffle=True
    )

    X_train_val, X_test = X_raw[train_val_idx], X_raw[test_idx]
    y_train_val, y_test = y_labels[train_val_idx], y_labels[test_idx]

    F_train_val, F_test = (None, None)
    if F_hand is not None:
        F_train_val, F_test = F_hand[train_val_idx], F_hand[test_idx]

    print(f"✅ Data split successfully.")
    print(f"   Train/Validation set size: {len(y_train_val)}")
    print(f"   Test set size: {len(y_test)}")

    return (X_train_val, F_train_val, y_train_val), (X_test, F_test, y_test)

def make_tf_dataset(
    X_raw: np.ndarray, y_labels: np.ndarray, F_hand: np.ndarray = None,
    batch_size: int = 128, shuffle: bool = False, seed: int = 42
) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from numpy arrays.

    Args:
        X_raw (np.ndarray): Raw sEMG data.
        y_labels (np.ndarray): One-hot encoded labels.
        F_hand (np.ndarray, optional): Handcrafted features. If provided, the
            dataset will yield a dictionary of inputs {'raw': ..., 'feat': ...}.
        batch_size (int): The batch size for the dataset.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        tf.data.Dataset: The configured TensorFlow dataset.
    """
    if F_hand is not None:
        # Model requires both raw and handcrafted features
        inputs = {"raw": X_raw, "feat": F_hand}
        ds = tf.data.Dataset.from_tensor_slices((inputs, y_labels))
    else:
        # Model requires only raw data
        ds = tf.data.Dataset.from_tensor_slices((X_raw, y_labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(y_labels), seed=seed, reshuffle_each_iteration=True)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)