# Adversarial Defense

This repository is dedicated to exploring and implementing techniques for adversarial defense in deep learning.

## Current Progress

### Model Architecture & Training
- **`src/model.py`**: A basic Convolutional Neural Network (`SimpleCNN`) has been implemented using PyTorch. It is designed for small-scale image classification tasks (such as 32x32 pixel images).
  - The model features two convolutional layers followed by two fully connected layers.
  - It is fully documented with standard docstrings and inline comments.
- **`src/train.py`**: A training loop script has been implemented to train the `SimpleCNN` model on the CIFAR-10 dataset.
  - Features an optimization loop using Adam and Cross-Entropy Loss.
  - Automatically runs on GPU (`cuda`) if available, otherwise defaults to CPU.
  - Fully commented and documented for ease of understanding.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- PyTorch
- torchvision (required for loading CIFAR-10 data)

### Running the Project

#### 1. Test the Model Architecture
You can run the model script directly to perform a quick forward pass test with dummy inputs:
```bash
python src/model.py
```

#### 2. Train the Model
You can execute the training pipeline directly to train the model on the CIFAR-10 dataset:
```bash
python src/train.py
```
