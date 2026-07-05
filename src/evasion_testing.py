# Objective: To run the FGSM attack across entire CIFAR 10 test dataset, for multiple epsilon values, and 
# to generate data values for evasion curve (epsilon vs accuracy).

import torch 
import torch.nn as nn
from data_loader import get_data_loaders
from model import SimpleCNN
from attack import extract_image_gradient, fgsm_attack

def evaluate_evasion_curve(model, test_loader, epsilons, device):
    """ 
    Evaluates a model's adversarial robustness across a list of epsilon values.
    Returns a dictionary mapping epsilon to robust accuracy.
    """

    #set the model to inference mode
    model.eval()
    results = {} # set an empty dictionary. This is what we'll return 

    # Loop over every attack budget level:
    for eps in epsilons:
        print(f"Running evasion testing for epsilon: {eps}...")
        correct = 0
        total = 0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # step 01: extract pixel gradients if eps > 0:
            if eps > 0:
                images, data_grad, targets = extract_image_gradient(model, images, labels)

                #step 2: Perturb image along the gradient sign vector
                perturbed_images = fgsm_attack(images, eps, data_grad)

            else:
                perturbed_images = images

            #step 03: run inferences on the images (clean or adversarial)
            with torch.no_grad():
                outputs = model(perturbed_images)
                predictions = outputs.argmax(dim=1)

            #step 04: track aggregated metrices
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        final_accuracy = (correct / total) * 100
        results[eps] = final_accuracy
        print(f" Accuracy at epsilon {eps}: {final_accuracy:.2f}%/n")

    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Load datasets and model weights
    _, test_loader, _ = get_data_loaders(batch_size=64)
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))

    # Define perturbation values to trace the evasion curve
    eps_values = [0.0, 0.005, 0.01, 0.03, 0.05, 0.1]

    curve_data = evaluate_evasion_curve(model, test_loader, eps_values, device)

    print("=== FINAL SWEEP Sweeping Complete ===")
    for eps, acc in curve_data.items():
        print(f"Epsilon: {eps:<6} | Robust Accuracy: {acc:.2f}%")
