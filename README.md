# Adversarial Defense

This repository is dedicated to exploring and implementing techniques for adversarial defense in deep learning.

## Current Progress

### Model Architecture, Training & Evaluation
- **`src/model.py`**: A basic Convolutional Neural Network (`SimpleCNN`) has been implemented using PyTorch. It is designed for small-scale image classification tasks (such as 32x32 pixel images).
  - The model features two convolutional layers followed by two fully connected layers.
  - It is fully documented with standard docstrings and inline comments.
- **`src/train.py`**: A training loop script has been implemented to train the `SimpleCNN` model on the CIFAR-10 dataset.
  - Features an optimization loop using Adam and Cross-Entropy Loss.
  - Automatically runs on GPU (`cuda`) if available, otherwise defaults to CPU.
  - Fully commented and documented for ease of understanding.
- **`src/evaluate.py`**: An evaluation script that loads a trained model's weights and evaluates its accuracy on the CIFAR-10 test set.
  - Computes the overall classification accuracy.
  - Builds and prints a textual confusion matrix.
  - Generates and saves a visual heatmap visualization under `reports/figures/` (ignored by git).
  - Fully annotated with docstrings and comments.
- **`src/attack.py`**: An adversarial attack utility script that implements the **Fast Gradient Sign Method (FGSM)**.
  - Demonstrates how to enable gradients on input tensors (`data.requires_grad = True`).
  - Implements gradient extraction of the loss with respect to the input image.
  - Implements the core FGSM attack logic (`fgsm_attack`) to perturb the input image in the direction of the gradient sign, scaled by `epsilon`, to maximize classification loss and trigger misclassification.
  - Fully annotated with clean, professional docstrings and comments.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- PyTorch
- torchvision (required for loading CIFAR-10 data)
- pandas, matplotlib, seaborn (required for evaluating and visualizing results)

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

#### 3. Evaluate the Model
You can evaluate the trained model and view the accuracy and confusion matrix:
```bash
python src/evaluate.py
```

#### 4. Run Adversarial Attack (FGSM)
You can run the attack script to extract the gradient map and generate a perturbed adversarial image using the FGSM attack:
```bash
python src/attack.py
```
