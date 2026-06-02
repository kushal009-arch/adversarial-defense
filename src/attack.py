"""
Adversarial Attack Utilities.

This module provides functions to extract gradients from an input image
with respect to the model's loss, which is the core step for generating
adversarial examples.
"""

import torch 
import torch.nn.functional as F 
from data_loader import get_data_loaders
from model import SimpleCNN

def extract_image_gradient():
    """
    Loads a pre-trained CNN model and computes the gradient of the loss
    with respect to a single test image.
    
    This gradient tells us how we need to perturb the image pixels to
    increase the model's classification loss (making it misclassify).
    """
    # Use GPU if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load the trained model and put it in evaluation mode
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
    model.eval()

    # 2. Get a single image and its correct label from the test dataset
    _, test_loader, _ = get_data_loaders(batch_size=1)
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    # 3. Tell PyTorch to track gradients for the input image itself
    # (By default, PyTorch only tracks gradients for model parameters)
    data.requires_grad = True

    # 4. Perform a forward pass to get the model's predictions
    output = model(data)

    # 5. Calculate the loss between predictions and the true label
    loss = F.cross_entropy(output, target)
    
    # 6. Clear any existing gradients on the model parameters
    model.zero_grad()

    # 7. Perform a backward pass to calculate gradients
    loss.backward()

    # 8. Retrieve the gradient of the loss with respect to the input image.
    # We use .detach() to safely access the gradient tensor without tracking history.
    gradient_map = data.grad.detach()

    # Log shape and sample details to verify the output
    print(f"Original Image Shape: {data.shape}")
    print(f"Gradient Map Shape: {gradient_map.shape}")
    print(f"\nSample Gradient Values (first 5 pixels of the red channel):")
    print(gradient_map[0][0][0][:5])

    return data, gradient_map, target

if __name__ == "__main__":
    extract_image_gradient()