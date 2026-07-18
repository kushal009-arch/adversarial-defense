# import all necessary modules 
"""
Module for robust training of a simple CNN model on CIFAR-10 using Fast Gradient Sign Method (FGSM) adversarial training.
"""
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
def generate_fgsm_batch(model, images, labels, EPSILON, criterion):
    """
    Generates adversarial images using the Fast Gradient Sign Method (FGSM).

    Args:
        model (nn.Module): The PyTorch neural network model.
        images (torch.Tensor): A batch of clean input images.
        labels (torch.Tensor): True labels associated with the images.
        EPSILON (float): Perturbation strength parameter for the attack.
        criterion (nn.modules.loss._Loss): The loss function used to calculate gradients.

    Returns:
        torch.Tensor: A batch of perturbed, adversarial images detached from the computation graph.
    """
    images_clone = images.clone().detach()
    images_clone.requires_grad = True

    outputs = model(images_clone)
    loss = criterion(outputs, labels)

    model.zero_grad()
    loss.backward()

    gradient_sign = images_clone.grad.data.sign()

    perturbed_images = images + EPSILON * gradient_sign

    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)

    return perturbed_images.detach()

# Outer maximizatoin loop (Mixed batch training engine)
def train_robust_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using a mixture of clean and adversarial examples.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): The clean data loader (unused in favor of local CIFAR10 loading).
        optimizer (optim.Optimizer): The optimizer to update model parameters.
        criterion (nn.modules.loss._Loss): The loss function.
        device (torch.device): The device (CPU or CUDA) to run training on.

    Returns:
        tuple: (epoch_loss, clean_acc, adv_acc) containing the average mixed loss,
               accuracy on clean images (%), and accuracy on adversarial images (%) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct_clean = 0
    correct_adv = 0
    total_samples = 0

    train_loader, test_loader, CIFAR10_CLASSES = get_data_loaders(batch_size=64)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        batch_size_current = images.size(0)

        images_adv = generate_fgsm_batch(model, images, labels, EPSILON, criterion)

        optimizer.zero_grad()

        output_clean = model(images)
        loss_clean = criterion(output_clean, labels)

        output_adv = model(images_adv)
        loss_adv = criterion(output_adv, labels)

        loss_mixed = (ALPHA * loss_clean) + ((1-ALPHA) * loss_adv)

        loss_mixed.backward()
        optimizer.step()

        running_loss += loss_mixed.item() * batch_size_current
        correct_clean += output_clean.argmax(dim=1).eq(labels).sum().item()
        correct_adv += output_adv.argmax(dim=1).eq(labels).sum().item()
        total_samples += batch_size_current

    epoch_loss = running_loss / total_samples
    clean_acc = (correct_clean / total_samples) * 100.0
    adv_acc = (correct_adv / total_samples) * 100.0

    return epoch_loss, clean_acc, adv_acc

# execution pipeline infrastructure
def main():
    """
    Sets up the device, initializes data loaders and the SimpleCNN model, 
    and runs the full adversarial robust training loop for the configured number of epochs.
    Finally, saves the robust model state dictionary to the designated path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    train_loader, test_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)
    model = SimpleCNN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("models", exist_ok=True)

    print("\n--- Initiating Boundary Regularization ---")
    for epoch in range(1, EPOCHS + 1):
        loss, clean_acc, adv_acc = train_robust_epoch(model, train_loader, optimizer, criterion, device)

        print(f"Epoch [{epoch:02d}/{EPOCHS}]"
              f"Loss: {loss:.4f} | "
              f"Clean Acc: {clean_acc:.2f}% | " 
              f"Robust Acc: {adv_acc:.2f}%")                       

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel checkpoint successfully compiled and saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
