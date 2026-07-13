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
    Executes a single min-max robust optimization training epoch
    """
    model.train() # CHECK THIS BEFORE GIT PUSH --> model has no train module, should we use train_model() from train?
    total_clean_loss = 0
    total_robust_loss = 0
    total_samples = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device) # x, y represent images, labels
        batch_size = x.size(0)
        total_samples += batch_size

        # -- Step A: The inner maximizer (Attack) --
        x.requires_grad = True  # This is True because in standard training, requires_grad is set to True by default 
                                # for model weights (theta), but is set to False by default for input images (x). 
                                # In FGSM, we need the gradient of the Loss Function with respect to x, which we 
                                # can only obtain when we explicitly set x.requires_grad to True.

        outputs = model(x)
        loss = criterion(outputs, y)
        total_clean_loss += loss.item() * batch_size

        # Clear any existing gradient before attacking
        model.zero_grad()
        loss.backward()

        # Generate Adversarial Images
        x_adv = x +  epsilon * x.grad.sign()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        x_adv = x_adv.detach()

        # -- Step B: The outer minimizer (Defense) --
        optimizer.zero_grad() 

        outputs_adv = model(x_adv)
        loss_robust = criterion(outputs_adv, y)
        total_robust_loss += loss_robust.item() * batch_size

        loss_robust.backward()
        optimizer.step()

    return total_clean_loss / total_samples , total_robust_loss / total_samples

def main():
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