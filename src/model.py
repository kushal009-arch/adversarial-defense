"""
Simple Convolutional Neural Network (CNN) Architecture.

Defines the SimpleCNN PyTorch model suitable for basic 32x32 image classification
tasks on datasets like CIFAR-10.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A 2-layer Convolutional Neural Network followed by 2 Fully Connected layers.
    
    Expected input: A batch tensor of shape (batch_size, 3, 32, 32)
    Expected output: Unnormalized class logits of shape (batch_size, 10)
    """
    def __init__(self) -> None:
        """
        Initializes convolutional, max-pooling, and linear layers for SimpleCNN.
        """
        super(SimpleCNN, self).__init__()
        
        # First Convolutional Layer: 3 input channels -> 16 output filters (3x3 kernel, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Second Convolutional Layer: 16 input filters -> 32 output filters (3x3 kernel, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Max Pooling Layer: Reduces spatial dimensions by factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers: 32 channels * 16 width * 16 height = 8192 flattened features
        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass through the network layers.
        
        Args:
            x (torch.Tensor): Input batch tensor of shape (batch_size, 3, 32, 32).
            
        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch_size, 10).
        """
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
