"""
Baseline Training Module for SimpleCNN.

Executes standard clean training of SimpleCNN model parameters on CIFAR-10 data.
"""

from typing import Optional
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from data_loader import get_data_loaders
from model import SimpleCNN

def train_model(epochs: int = 5, learning_rate: float = 0.001) -> nn.Module:
    """
    Trains the SimpleCNN architecture on clean CIFAR-10 training images.

    Args:
        epochs (int): Total training epochs over full dataset. Defaults to 5.
        learning_rate (float): Learning rate parameter for Adam optimizer. Defaults to 0.001.

    Returns:
        nn.Module: The trained SimpleCNN model instance set to evaluation mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, _, _ = get_data_loaders(batch_size=64)
    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/baseline_model.pth")
    print("Finished Training! Saved baseline model to models/baseline_model.pth")
    return model

if __name__ == "__main__":
    trained_model = train_model()
