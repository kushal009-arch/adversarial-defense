"""
The goal of this script is to perform a white box vs black box audit on both 
baseline_model.pth and robust_model.pth across multiple attack intensities (epsilon)

"""

# import necessary libraries and modules 
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
import numpy as np 

from model import SimpleCNN
from attack import fgsm_attack
from data_loader import get_data_loaders
from plot_evasion_curve import plot_audit_curves

def run_audit(model, target_model_for_gradients, data_loader, epsilon, device):
    """
    Evaluates the accuracy of a model on a dataset under FGSM attack.
    If target_model_for_gradients is provided, it performs a black-box transfer attack.
    """
    model.eval()
    if target_model_for_gradients is not None:
        target_model_for_gradients.eval()

    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # Determine which model controls gradient generation map
        grad_source_model = target_model_for_gradients if target_model_for_gradients is not None else model

        # clone inputs and engage gradient extraction on inputs.
        perturbed_images = images.clone().detach()
        perturbed_images.requires_grad = True

        # forward pass on gradient source model
        outputs = grad_source_model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)

        grad_source_model.zero_grad()
        loss.backward()

        if perturbed_images.grad is not None and epsilon > 0:
            data_grad = perturbed_images.grad.data
            perturbed_images = fgsm_attack(images, epsilon, data_grad)
        else:
            perturbed_images = images

        with torch.no_grad():
            final_outputs = model(perturbed_images)
            predictions = final_outputs.argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.size(0)
            
    return (correct / total) * 100 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Security Audit on Device: {device}")
    
    # 1. Pipeline Data Setup
    _, test_loader, _ = get_data_loaders(batch_size=128)
    
    # 2. Instantiate and load weights
    baseline_net = SimpleCNN().to(device)
    robust_net = SimpleCNN().to(device)
    
    # Check if weights exist before loading
    if not os.path.exists("models/baseline_model.pth") or not os.path.exists("models/robust_model.pth"):
        print("[ERROR] Pre-trained models not found in models/ directory. Please run training scripts first.")
        return

    baseline_net.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
    robust_net.load_state_dict(torch.load("models/robust_model.pth", map_location=device))
    
    # 3. Establish the Epsilon Sweep Profile
    epsilons = [0.0, 0.01, 0.03, 0.07, 0.15]
    
    baseline_wb_list = []
    robust_wb_list = []
    robust_bb_list = []
    
    print("\nStarting Audit Metrics Logging...\n")
    print(f"{'Epsilon':<10} | {'Baseline White-Box':<20} | {'Robust White-Box':<18} | {'Robust Black-Box':<18}")
    print("-" * 75)
    
    for eps in epsilons:
        # Test A: Baseline under standard white box exploit
        base_wb = run_audit(baseline_net, None, test_loader, eps, device)
        baseline_wb_list.append(base_wb)
        
        # Test B: Robust Model under pure white box exploit
        robust_wb = run_audit(robust_net, None, test_loader, eps, device)
        robust_wb_list.append(robust_wb)
        
        # Test C: Robust Model under surrogate baseline transfer exploit (The Masking Trap Check)
        robust_bb = run_audit(robust_net, baseline_net, test_loader, eps, device)
        robust_bb_list.append(robust_bb)
        
        print(f"{eps:<10.2f} | {base_wb:<20.2f}% | {robust_wb:<18.2f}% | {robust_bb:<18.2f}%")

    # 4. Generate visual representation (Evasion Curves Comparison)
    print("\nGenerating security audit evasion curves plot...")
    plot_audit_curves(epsilons, baseline_wb_list, robust_wb_list, robust_bb_list)

if __name__ == "__main__":
    main()