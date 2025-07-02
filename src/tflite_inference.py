"""
TFLite Inference Utilities for EMG Gesture Classification
========================================================

This module provides utility functions for loading and performing inference 
with TensorFlow Lite models for EMG gesture classification, particularly 
useful for deployment on resource-constrained devices.

Author: Israel Huentecura
"""

import os
import numpy as np
import tensorflow as tf

# Dictionary mapping architecture names to their TFLite model file names and input shapes
BEST_TFLITE_MODELS = {
    'EMGHandNet-2D': {
        'file': 'EMGHandNet-2D_fold1_float32.tflite',
        'input_shape': (1, 1, 400, 8),  # [batch, channels, time_steps, features]
        'format': 'float32',
        'size_mb': 1.33
    },
    'DualStream-Lite': {
        'file': 'DualStream-Lite_fold1_float32.tflite',
        'input_shapes': {
            'raw_input': (1, 400, 8),  # [batch, time_steps, features]
            'feat_input': (1, 9, 8)     # [batch, features, channels]
        },
        'format': 'float32',
        'size_mb': 1.07
    },
    'CRNN-Attn': {
        'file': 'CRNN-Attn_fold1_quant_int8.tflite',
        'input_shape': (1, 400, 8),  # [batch, time_steps, features]
        'format': 'int8',
        'size_mb': 0.32
    },
    'HyT-Net': {
        'file': 'HyT-Net_fold1_float32.tflite',
        'input_shapes': {
            'raw_input': (1, 400, 8),  # [batch, time_steps, features]
            'feat_input': (1, 9, 8)     # [batch, features, channels]
        },
        'format': 'float32',
        'size_mb': 3.27
    }
}

def load_tflite_model(model_path):
    """
    Load a TFLite model for inference.
    
    Parameters:
    -----------
    model_path : str
        Path to the TFLite model file
    
    Returns:
    --------
    tf.lite.Interpreter
        TFLite interpreter with allocated tensors
    """
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    
    # Load and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Print model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("TFLite Model loaded successfully")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    print(f"Input details:")
    for detail in input_details:
        print(f"  - {detail['name']}: {detail['shape']}")
    
    return interpreter

def predict_with_tflite_model(interpreter, input_data, verbose=False):
    """
    Make a prediction using a TFLite model interpreter.
    
    Parameters:
    -----------
    interpreter : tf.lite.Interpreter
        The loaded TFLite interpreter
    input_data : numpy.ndarray or dict
        Input data for the model. If the model has multiple inputs,
        provide a dictionary mapping input names to arrays.
    verbose : bool, optional
        Whether to print prediction details, by default False
    
    Returns:
    --------
    numpy.ndarray
        Output tensor (prediction result)
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Handle single or multiple inputs
    if isinstance(input_data, dict):
        # Multiple inputs
        for i, detail in enumerate(input_details):
            input_name = detail['name'].split(':')[0]
            if input_name in input_data:
                interpreter.set_tensor(detail['index'], input_data[input_name])
            else:
                raise ValueError(f"Input '{input_name}' not found in provided input data")
    else:
        # Single input
        if len(input_details) > 1:
            raise ValueError("Model has multiple inputs, but only one input tensor was provided")
        interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if verbose:
        # Print prediction results
        predicted_class = np.argmax(output_data[0])
        predicted_label = predicted_class + 1  # Add 1 to map to NinaPro DB1 labels (1-12)
        print(f"Prediction probabilities: {output_data[0]}")
        print(f"Predicted class index (0-11): {predicted_class}")
        print(f"Predicted gesture label (1-12): {predicted_label}")
    
    return output_data

def get_input_shape_for_model(model_name):
    """
    Get the expected input shape for a specific model architecture.
    
    Parameters:
    -----------
    model_name : str
        Model architecture name
        
    Returns:
    --------
    tuple or dict
        Input shape(s) expected by the model
    """
    if model_name not in BEST_TFLITE_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(BEST_TFLITE_MODELS.keys())}")
    
    model_info = BEST_TFLITE_MODELS[model_name]
    
    if 'input_shape' in model_info:
        return model_info['input_shape']
    else:
        return model_info['input_shapes']

def create_dummy_input(model_name):
    """
    Create dummy input data for the specified model architecture.
    
    Parameters:
    -----------
    model_name : str
        Model architecture name
        
    Returns:
    --------
    numpy.ndarray or dict
        Dummy input data matching the model's expected input format
    """
    model_info = BEST_TFLITE_MODELS[model_name]
    
    if 'input_shape' in model_info:
        # Single input
        return np.random.rand(*model_info['input_shape']).astype(np.float32)
    else:
        # Multiple inputs
        input_dict = {}
        for name, shape in model_info['input_shapes'].items():
            input_dict[name] = np.random.rand(*shape).astype(np.float32)
        return input_dict

def example_usage():
    """Example demonstrating how to use TFLite models for EMG gesture classification"""
    
    print("\n" + "=" * 80)
    print(" EMG Gesture Classification - TFLite Models for Low-Resource Devices ".center(80, "="))
    print("=" * 80)
    
    # Example with EMGHandNet-2D (single input model)
    model_name = "EMGHandNet-2D"
    model_file = BEST_TFLITE_MODELS[model_name]['file']
    model_path = os.path.join("models_tflite", model_file)
    
    print(f"\nTesting {model_name} (single input model):")
    print("-" * 50)
    
    try:
        # Load model
        interpreter = load_tflite_model(model_path)
        
        # Create dummy input
        dummy_input = create_dummy_input(model_name)
        print(f"Input shape: {dummy_input.shape}")
        
        # Make prediction
        print("\nMaking prediction...")
        prediction = predict_with_tflite_model(interpreter, dummy_input, verbose=True)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example with DualStream-Lite (dual input model)
    model_name = "DualStream-Lite"
    model_file = BEST_TFLITE_MODELS[model_name]['file']
    model_path = os.path.join("models_tflite", model_file)
    
    print(f"\nTesting {model_name} (dual input model):")
    print("-" * 50)
    
    try:
        # Load model
        interpreter = load_tflite_model(model_path)
        
        # Create dummy inputs
        dummy_inputs = create_dummy_input(model_name)
        for name, tensor in dummy_inputs.items():
            print(f"Input '{name}' shape: {tensor.shape}")
        
        # Make prediction
        print("\nMaking prediction...")
        prediction = predict_with_tflite_model(interpreter, dummy_inputs, verbose=True)
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Example with quantized int8 model
    model_name = "CRNN-Attn"
    model_file = BEST_TFLITE_MODELS[model_name]['file']
    model_path = os.path.join("models_tflite", model_file)
    
    print(f"\nTesting {model_name} (quantized int8 model):")
    print("-" * 50)
    
    try:
        # Load model
        interpreter = load_tflite_model(model_path)
        
        # Create dummy input
        dummy_input = create_dummy_input(model_name)
        print(f"Input shape: {dummy_input.shape}")
        
        # Make prediction
        print("\nMaking prediction...")
        prediction = predict_with_tflite_model(interpreter, dummy_input, verbose=True)
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80)
    print(" Examples Complete ".center(80, "="))
    print("=" * 80)

if __name__ == "__main__":
    example_usage()
