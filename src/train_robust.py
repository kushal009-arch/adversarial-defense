# import all necessary modules 
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from data_loader import get_data_loaders

# Hyperparameter configuration
ALPHA = 0.5
EPSILON = 8.0 / 255.0
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 15
MODEL_SAVE_PATH = "models/robust_model.pth"

# Inner maximization solver (FGSM Batch generator)
def generate_fgsm_batch(model, images, labels, epsilon, criterion):
    images_clone = images.clone().detach()
    images_clone.requires_grad = True

    outputs = model(images_clone)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    gradient_sign = images_clone.grad.data.sign()

    perturbed_images = images + epsilon * gradient_sign

    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)

    return perturbed_images.detach()

# Outer maximizatoin loop (Mixed batch training engine)
def train_robust_epoch(model, loader, optimizer, criterion, device):
    model.train()


# execution pipeline infrastructure
def main():

if __name__ == "__main__":
    main()

