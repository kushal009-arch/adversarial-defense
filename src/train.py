"""
This module handles the training process for the SimpleCNN neural network.

It loads the training dataset, feeds it through the model, computes the loss (how incorrect the model is), and updates the model's weights using an optimizer (CrossEntropyLoss() for this one) to make it more accurate over time.
"""

import torch 
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import SimpleCNN

def train_model(epochs=5, learning_rate=0.001):
    """
    Trains the SimpleCNN model on the loaded dataset.

    Args:
        epochs (int, optional): The number of complete times the model sees the entire 
            training dataset. Defaults to 5.
        learning_rate (float, optional): Controls how fast the model adjusts its weights 
            during training. Smaller values mean slower, more stable learning. Defaults to 0.001.

    Returns:
        torch.nn.Module: The trained neural network model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, test_loader, CIFAR10_CLASSES = get_data_loaders(batch_size=64)
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        # Loop over each batch of data in the training dataset
        for images, labels in train_loader:
            # Move images and labels to the active processor (GPU/CPU)
            images, labels = images.to(device), labels.to(device)

            # Clear previous calculations of gradients (slopes) so they don't accumulate
            optimizer.zero_grad()

            # Feed the images into the model to get predictions (outputs)
            outputs = model(images)

            # Calculate how far off the predictions were from the correct labels
            loss = criterion(outputs, labels)

            # Calculate the gradients (slopes of how we should adjust weights to reduce loss)
            loss.backward()

            # Adjust the model's weights slightly in the correct direction
            optimizer.step()

            # Add this batch's loss to our running total for the epoch
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f}")

    import os
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("Finished Training! Saved model to models/baseline_model.pth")
    return model

if __name__ == "__main__":
    trained_model = train_model()    
