"""
This module handles the evaluation process for the trained SimpleCNN model.

It loads the saved weights of the model, runs it against the test dataset to calculate
the overall classification accuracy, and generates a confusion matrix report and 
visual heatmap to see where the model is making errors.
"""

import torch 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from data_loader import get_data_loaders
from model import SimpleCNN

def evaluate_model():
    """
    Evaluates the trained SimpleCNN model on the test dataset.

    This function:
    1. Loads the model weights from 'models/baseline_model.pth'.
    2. Runs predictions on the 10,000 test images.
    3. Calculates and prints the overall classification accuracy.
    4. Prints a text-based confusion matrix.
    5. Saves a visual confusion matrix heatmap as a PNG image in 'reports/figures/'.
    """
    # Identify the processing unit: GPU (cuda) if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Class names in the CIFAR-10 dataset (index 0 is plane, index 1 is car, etc.)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck')

    # Load only the test data loader
    _, test_loader, _ = get_data_loaders(batch_size=64)

    # Initialize the neural network structure and send it to the GPU/CPU
    model = SimpleCNN().to(device)

    try:
        # Load the saved training weights into our model structure
        # map_location=device ensures it loads correctly whether on GPU or CPU
        model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
        print("Successfully loaded 'models/baseline_model.pth'")
    except FileNotFoundError:
        print("Error: Could not find model weights. Try running train.py first.")
        return 

    # Set the model to evaluation/inference mode (turns off dropout, batchnorm training, etc.)
    model.eval()

    # Track classification performance
    correct = 0
    total = 0

    # Initialize a 10x10 grid of zeros to count correct/incorrect class predictions
    conf_matrix = torch.zeros(10, 10, dtype=torch.int32)

    print("Evaluating model on 10,000 test images...")

    # Disable backpropagation (gradient calculation) since we aren't training. Saves memory/time.
    with torch.no_grad():
        for images, labels in test_loader:
            # Move data to the active processor (GPU/CPU)
            images, labels = images.to(device), labels.to(device)

            # Get raw scores (logits) from the model
            outputs = model(images)

            # Determine the class index with the highest score (prediction)
            _, predicted = torch.max(outputs.data, 1)

            # Count total images evaluated in this batch
            total += labels.size(0)
            
            # Count how many predictions in this batch matched the actual label
            correct += (predicted == labels).sum().item()

            # Record predictions in the confusion matrix (mapped to CPU for safe indexing)
            for true_label, pred_label in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                conf_matrix[true_label, pred_label] += 1

    # Calculate overall percentage accuracy
    accuracy = 100 * correct / total

    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print("\n")
    
    # Format confusion matrix as a labeled table using a pandas DataFrame
    df_cm = pd.DataFrame(conf_matrix.numpy(), index=classes, columns=classes)
    print(df_cm)

    # Create a visual plot for the confusion matrix
    plt.figure(figsize=(10, 8))
    # Draw a heatmap where darker colors represent higher counts
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CIFAR-10 Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Define the directory where reports/charts will be saved
    directory = 'reports/figures'
    os.makedirs(directory, exist_ok=True)

    # Generate a timestamp to avoid overwriting previous evaluation runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"confusion_matrix_{timestamp}.png"
    save_path = os.path.join(directory, filename)

    # Save the generated heatmap image
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    
    print("\n")
    print("Saved confusion matrix visualization to 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()
