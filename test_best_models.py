"""
Test Best Models for EMG Gesture Classification
==============================================

This script allows you to easily test the best models from each architecture.
Run this script to see how to load and make predictions with any of the best models.

Author: Israel Huentecura
"""

import os
import sys
import numpy as np
import tensorflow as tf
from src.load_models import load_best_model, predict_with_model, BEST_MODELS

def main():
    # Print header
    print("\n" + "=" * 80)
    print(" EMG Gesture Classification - Best Models Test Suite ".center(80, "="))
    print("=" * 80)
    
    # Display available models
    print("\nAvailable models:")
    for i, (arch, model_name) in enumerate(BEST_MODELS.items()):
        print(f"{i+1}. {arch:<20} - {model_name}")
    
    # Get user selection
    try:
        while True:
            selection = input("\nSelect a model to test (1-6) or 'q' to quit: ")
            
            if selection.lower() == 'q':
                print("Exiting program...")
                sys.exit(0)
            
            try:
                idx = int(selection) - 1
                if 0 <= idx < len(BEST_MODELS):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(BEST_MODELS)}")
            except ValueError:
                print("Please enter a valid number")
    
        # Get the selected architecture
        architecture = list(BEST_MODELS.keys())[idx]
        
        print("\n" + "-" * 80)
        print(f" Testing {architecture} Model ".center(80, "-"))
        print("-" * 80)
        
        # Load the selected model
        model = load_best_model(architecture)
        
        # Make a prediction with dummy data
        print("\nMaking a prediction with random test data...")
        predictions, class_idx, class_label = predict_with_model(model)
        
        # Display results
        print(f"\nPrediction result:")
        print(f"  Class probabilities: {predictions[0][:3]}... (showing first 3 of 12)")
        print(f"  Highest probability: {np.max(predictions[0]):.4f}")
        print(f"  Predicted class index (0-11): {class_idx}")
        print(f"  Predicted gesture label (1-12): {class_label}")
        
        # Show model details
        print("\nModel details:")
        print(f"  Architecture: {architecture}")
        print(f"  Model file: {BEST_MODELS[architecture]}")
        print(f"  Model size: {os.path.getsize(os.path.join('models', BEST_MODELS[architecture])) / (1024*1024):.2f} MB")
        print(f"  Total parameters: {model.count_params():,}")
        
        # Display inference requirements
        print("\nInference requirements:")
        if len(model.inputs) > 1:
            print("  - This model requires both raw EMG data and handcrafted features")
            print("  - Suitable for systems with higher computational capabilities")
        else:
            print("  - This model only requires raw EMG data")
            print("  - Suitable for simplified deployment pipelines")
        
        if "DualStream-Lite" in architecture or "CRNN-Attn" in architecture:
            print("  - Optimized for low memory footprint (good for microcontrollers)")
        
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\n" + "=" * 80)
        print(" Test Complete ".center(80, "="))
        print("=" * 80)

if __name__ == "__main__":
    main()
