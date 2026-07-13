"""
This module will train the model with adversarial examples to strengthen it against adversarial attacks

"""
import torch 
import os
import torch.nn as nn 
from data_loader import get_data_loaders
from model import SimpleCNN
from train import train_model

def train_adversarial_epoch(model, train_loader, criterion, optimizer, epsilon, device):
    """
    Executes a single epoch of adversarial training using min-max robust optimization.

    This function implements a 2-step min-max training procedure:
    1. Inner Maximizer (Attack): Generates adversarial perturbations via FGSM to maximize the classification loss.
    2. Outer Minimizer (Defense): Updates the model parameters to minimize the classification loss on those adversarial samples.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader supplying the training dataset.
        criterion (torch.nn.modules.loss._Loss): The loss function (e.g. CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimizer (e.g. SGD).
        epsilon (float): The maximum perturbation budget for the FGSM attack.
        device (torch.device): Device on which training is executed (CPU or CUDA).

    Returns:
        tuple (float, float):
            - Average clean loss over the epoch.
            - Average robust (adversarial) loss over the epoch.
    """
    # Put the model in training mode (enables dropout, updates batch norm stats, etc.)
    model.train()
    
    total_clean_loss = 0
    total_robust_loss = 0
    total_samples = 0

    for x, y in train_loader:
        # Move data to the active device (GPU or CPU)
        x, y = x.to(device), y.to(device)
        batch_size = x.size(0)
        total_samples += batch_size

        # -- Step A: The Inner Maximizer (Attack) --
        
        # Enable gradient tracking on the input images.
        # By default, PyTorch only tracks gradients for model parameters.
        # For FGSM, we need the gradient of the loss with respect to the input image (x).
        x.requires_grad = True

        # Forward pass on clean (unperturbed) images
        outputs = model(x)
        loss = criterion(outputs, y)
        total_clean_loss += loss.item() * batch_size

        # Reset model gradients before computing the input gradients
        model.zero_grad()
        
        # Backward pass to calculate gradients w.r.t. the inputs (x.grad)
        loss.backward()

        # Generate Adversarial Images using the Fast Gradient Sign Method (FGSM).
        # We perturb the image in the direction of the gradient sign to maximize loss.
        x_adv = x + epsilon * x.grad.sign()
        
        # Clamp the adversarial images to the valid pixel value range [0, 1]
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Detach the adversarial images from the current computation graph.
        # This treats them as static input data for the subsequent training step.
        x_adv = x_adv.detach()

        # -- Step B: The Outer Minimizer (Defense) --
        
        # Zero out the parameter gradients for the model optimizer
        optimizer.zero_grad()

        # Forward pass using the perturbed adversarial images
        outputs_adv = model(x_adv)
        
        # Calculate robust classification loss on adversarial samples
        loss_robust = criterion(outputs_adv, y)
        total_robust_loss += loss_robust.item() * batch_size

        # Backward pass to calculate gradients w.r.t. model parameters
        loss_robust.backward()
        
        # Update model weights based on the robust loss gradients
        optimizer.step()

    # Return the average clean and robust loss values for the epoch
    return total_clean_loss / total_samples, total_robust_loss / total_samples

def main():
    """
    Initializes training components (device, model, loader, optimizer, criterion)
    and executes the adversarial training loop, saving the robust model parameters.
    """
    # 1. Initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = SimpleCNN().to(device)
    train_loader, _, _ = get_data_loaders(batch_size=64)
    optimizer = torch.optim.SGD(model.parameters() , lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    EPSILON = 8 / 255 # Standard budget for CIFAR 10
    NUM_EPOCHS = 5

    print("[*] Launching min-max optimization training loop across adversarial batches...")
    for epoch in range(1, NUM_EPOCHS + 1):
        clean_loss, robust_loss = train_adversarial_epoch(model, train_loader, criterion, optimizer, EPSILON, device)
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] -> Clean Loss Map: {clean_loss:.4f} | Robust Adversary Loss: {robust_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = "models/robust_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[SUCCESS] Adversarially hardened weights successfully saved to disk at: {save_path}")

if __name__ == "__main__":
    main()