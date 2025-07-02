"""
Utility functions for loading the best pre-trained models for EMG gesture classification.
This module provides easy-to-use functions to load models from different architectures.

Author: Israel Huentecura
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Best model paths by architecture
BEST_MODELS = {
    'HyT-Net': 'HyT-Net_fold1_BEST.keras',              # Test Acc = 0.9988
    'DualStream-Original': 'DualStream-Original_fold3_BEST.keras', # Test Acc = 0.9986
    'EMGHandNet-2D': 'EMGHandNet-2D_fold6_BEST.keras',   # Test Acc = 0.9984
    'EMGHandNet-Original': 'EMGHandNet-Original_fold9_BEST.keras', # Test Acc = 0.9982
    'DualStream-Lite': 'DualStream-Lite_fold8_BEST.keras', # Test Acc = 0.9946
    'CRNN-Attn': 'CRNN-Attn_fold1_BEST.keras',          # Test Acc = 0.9881
}

def get_model_path(architecture, models_dir='models'):
    """
    Get the full path to the best model for a specific architecture.
    
    Parameters:
    -----------
    architecture : str
        Architecture name, must be one of the keys in BEST_MODELS
    models_dir : str, optional
        Directory where models are stored, by default 'models'
    
    Returns:
    --------
    str
        Full path to the best model file
    """
    if architecture not in BEST_MODELS:
        raise ValueError(f"Architecture must be one of: {list(BEST_MODELS.keys())}")
    
    return os.path.join(models_dir, BEST_MODELS[architecture])

def load_best_model(architecture, models_dir='models', verbose=True):
    """
    Load the best pre-trained model for the specified architecture.
    
    Parameters:
    -----------
    architecture : str
        Architecture name, must be one of the keys in BEST_MODELS
    models_dir : str, optional
        Directory where models are stored, by default 'models'
    verbose : bool, optional
        Whether to print model summary, by default True
    
    Returns:
    --------
    tf.keras.Model
        Loaded model
    """
    model_path = get_model_path(architecture, models_dir)
    
    if verbose:
        print(f"Loading best {architecture} model: {BEST_MODELS[architecture]}")
    
    model = load_model(model_path)
    
    if verbose:
        print(f"Model loaded successfully: {model.name}")
        print(f"Input shapes:")
        for input_layer in model.inputs:
            print(f"  - {input_layer.name}: {input_layer.shape}")
    
    return model

def create_dummy_input(model):
    """
    Create dummy input data for the loaded model, for testing purposes.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The loaded model
    
    Returns:
    --------
    dict or array
        Input data that matches the model's input requirements
    """
    # Check model input structure
    model_inputs = model.inputs
    
    # For models with two inputs (raw and features)
    if len(model_inputs) == 2:
        # Check which input is for raw data and which is for features
        if 'raw' in model_inputs[0].name:
            raw_input = model_inputs[0]
            feat_input = model_inputs[1]
        else:
            raw_input = model_inputs[1]
            feat_input = model_inputs[0]
        
        # Create dummy data for both inputs
        raw_shape = raw_input.shape
        feat_shape = feat_input.shape
        
        raw_data = np.random.rand(*raw_shape.as_list()).astype(np.float32)
        feat_data = np.random.rand(*feat_shape.as_list()).astype(np.float32)
        
        # Return dictionary with the appropriate input names
        input_dict = {}
        input_dict[raw_input.name.split(':')[0]] = raw_data
        input_dict[feat_input.name.split(':')[0]] = feat_data
        
        return input_dict
    
    # For models with a single input
    else:
        input_shape = model_inputs[0].shape
        return np.random.rand(*input_shape.as_list()).astype(np.float32)

def predict_with_model(model, inputs=None):
    """
    Make a prediction with the model using either provided inputs or dummy inputs.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The loaded model
    inputs : dict or array, optional
        Input data for prediction. If None, dummy inputs will be created
    
    Returns:
    --------
    tuple
        (raw predictions, predicted class index, predicted class label)
    """
    if inputs is None:
        inputs = create_dummy_input(model)
    
    # Make prediction
    predictions = model.predict(inputs)
    
    # Get predicted class (index 0-11)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # Ninapro DB1 labels are 1-12, so add 1 to the predicted index
    predicted_class_label = predicted_class_idx + 1
    
    return predictions, predicted_class_idx, predicted_class_label

def example_usage():
    """Example of how to use the functions in this module"""
    
    print("=" * 60)
    print("EMG Gesture Classification - Model Loading Example")
    print("=" * 60)
    
    # Example for a model with two inputs (HyT-Net)
    arch = "HyT-Net"
    print(f"\nLoading {arch} model (dual-input architecture):")
    model = load_best_model(arch)
    
    print("\nMaking a prediction with dummy data:")
    _, class_idx, class_label = predict_with_model(model)
    print(f"Predicted class index (0-11): {class_idx}")
    print(f"Predicted gesture label (1-12): {class_label}")
    
    # Example for a model with single input (EMGHandNet-2D)
    arch = "EMGHandNet-2D"
    print(f"\nLoading {arch} model (single-input architecture):")
    model = load_best_model(arch)
    
    print("\nMaking a prediction with dummy data:")
    _, class_idx, class_label = predict_with_model(model)
    print(f"Predicted class index (0-11): {class_idx}")
    print(f"Predicted gesture label (1-12): {class_label}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    example_usage()
