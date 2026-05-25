"""
Defines a simple Convolutional Neural Network (CNN) architecture using PyTorch.
This model is suitable for basic image classification tasks on small images (e.g., 32x32 pixels).
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """
    A simple CNN model containing two convolutional layers followed by two fully connected layers.
    
    Expected input: A batch of images with shape (batch_size, 3, 32, 32)
    Expected output: Logits for 10 classes with shape (batch_size, 10)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # First Convolutional Layer:
        # Takes in 3-channel (RGB) images, applies 16 filters of size 3x3.
        # padding=1 ensures the spatial dimensions (width/height) stay the same.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Second Convolutional Layer:
        # Takes in the 16 feature maps from the previous layer, applies 32 filters of size 3x3.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # Max Pooling Layer:
        # Reduces the spatial dimensions (width and height) by half.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected (Linear) Layers:
        # After two convolutions and one pooling step, a 32x32 image becomes 16x16.
        # 32 channels * 16 width * 16 height = 8192 flattened features.
        self.fc1 = nn.Linear(8192, 128) # Maps the 8192 features down to 128
        self.fc2 = nn.Linear(128, 10)   # Maps the 128 features to the final 10 output classes

    def forward(self, x):
        """
        Defines the forward pass (how data flows through the network).
        
        Args:
            x (torch.Tensor): The input data (images).
            
        Returns:
            torch.Tensor: The final raw predictions (logits) for each class.
        """
        # Pass through the first conv layer and apply ReLU activation function
        x = F.relu(self.conv1(x))

        # Pass through the second conv layer, apply ReLU, then apply max pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the 3D feature maps into a 1D vector for the linear layers
        x = torch.flatten(x, 1)

        # Pass through the first linear layer and apply ReLU
        x = F.relu(self.fc1(x))

        # Pass through the final output layer (no activation here, typically used with CrossEntropyLoss)
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    # Test the model with dummy data to ensure it works
    model = SimpleCNN()
    
    # Create a dummy batch of 1 image, with 3 color channels, 32x32 pixels
    dummy_input = torch.randn(1, 3, 32, 32)
    
    # Run the dummy image through the model
    output = model(dummy_input)
    
    # Print the shape of the output to verify it's (1, 10)
    print(f"Output shape: {output.shape}")
