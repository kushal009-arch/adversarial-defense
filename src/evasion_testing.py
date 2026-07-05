"""
Evasion Testing Module for Adversarial Robustness.

This module evaluates the robustness of the trained CNN model against adversarial
perturbations of varying budgets (epsilon values) using the Fast Gradient Sign Method (FGSM).
It calculates the model's robust classification accuracy at each epsilon level to plot
or output an evasion curve.
"""

import torch 
import torch.nn as nn
from data_loader import get_data_loaders
from model import SimpleCNN
from attack import extract_image_gradient, fgsm_attack

def evaluate_evasion_curve(model, test_loader, epsilons, device):
    """
    Evaluates the model's adversarial robustness across a list of epsilon values.

    For each epsilon level, the function loops through the entire test dataset,
    generates adversarial perturbations on the input batches using FGSM (for eps > 0),
    and computes the classification accuracy.

    Args:
        model (nn.Module): The PyTorch CNN model to evaluate.
        test_loader (DataLoader): PyTorch DataLoader for the evaluation/test dataset.
        epsilons (list of float): Adversarial perturbation magnitudes (budget).
        device (torch.device): The device (CPU or CUDA) to run evaluation on.

    Returns:
        dict: A dictionary mapping each epsilon value to the robust accuracy percentage.
    """
    # Set the model to evaluation/inference mode
    model.eval()
    results = {}

    # Loop over every attack budget level (epsilon)
    for eps in epsilons:
        print(f"Running evasion testing for epsilon: {eps}...")
        correct = 0
        total = 0

        # Process the dataset in batches
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Step 1: Extract pixel gradients if epsilon > 0
            if eps > 0:
                images, data_grad, labels = extract_image_gradient(model, images, labels)

                # Step 2: Perturb the batch of images in the direction of the gradient sign
                perturbed_images = fgsm_attack(images, eps, data_grad)
            else:
                # Epsilon of 0.0 corresponds to evaluating clean (unperturbed) images
                perturbed_images = images

            # Step 3: Run inference on the images (clean or adversarial)
            with torch.no_grad():
                outputs = model(perturbed_images)
                predictions = outputs.argmax(dim=1)

            # Step 4: Track aggregated accuracy metrics
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        # Calculate final accuracy for the current epsilon
        final_accuracy = (correct / total) * 100
        results[eps] = final_accuracy
        print(f" Accuracy at epsilon {eps}: {final_accuracy:.2f}%\n")

    return results

if __name__ == "__main__":
    # Choose processing unit: GPU (CUDA) if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Load test dataset loaders
    _, test_loader, _ = get_data_loaders(batch_size=64)
    
    # Initialize model architecture and load trained baseline weights
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))

    # Define perturbation budget values to trace the evasion curve
    eps_values = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1]

    # Run the evaluation sweep
    curve_data = evaluate_evasion_curve(model, test_loader, eps_values, device)

    # Print final summary of the evasion sweep
    print("=== FINAL SWEEP Sweeping Complete ===")
    for eps, acc in curve_data.items():
        print(f"Epsilon: {eps:<6} | Robust Accuracy: {acc:.2f}%")
