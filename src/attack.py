"""
Adversarial Attack Utilities for Fast Gradient Sign Method (FGSM).

Provides gradient extraction functions and FGSM adversarial noise injection.
"""

from typing import Optional, Tuple
import torch 
import torch.nn.functional as F 
from data_loader import get_data_loaders
from model import SimpleCNN

def extract_image_gradient(
    model: Optional[torch.nn.Module] = None, 
    data: Optional[torch.Tensor] = None, 
    target: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the loss gradient with respect to an input image tensor.

    Args:
        model (Optional[torch.nn.Module]): PyTorch model instance. Defaults to baseline SimpleCNN if None.
        data (Optional[torch.Tensor]): Input image tensor batch. Defaults to next test_loader item if None.
        target (Optional[torch.Tensor]): Target class label tensor. Defaults to next test_loader item if None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A 3-tuple containing:
            - data: Original input image tensor with tracked autograd gradients.
            - gradient_map: Detached loss gradient tensor w.r.t input data.
            - target: True target class label tensor.
    """
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
        model.eval()
    else:
        device = next(model.parameters()).device

    if data is None or target is None:
        _, test_loader, _ = get_data_loaders(batch_size=1)
        data, target = next(iter(test_loader))
        data, target = data.to(device), target.to(device)

    data.requires_grad = True
    output = model(data)
    loss = F.cross_entropy(output, target)
    model.zero_grad()
    loss.backward()

    gradient_map = data.grad.detach()
    return data, gradient_map, target

def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor) -> torch.Tensor:
    """
    Applies Fast Gradient Sign Method (FGSM) perturbation to an input image.

    Formula: x_adv = clamp(x + epsilon * sign(gradient_x(loss)), -1.0, 1.0)

    Args:
        image (torch.Tensor): Original input image tensor.
        epsilon (float): Perturbation magnitude parameter (step size).
        data_grad (torch.Tensor): Loss gradient tensor w.r.t input image.

    Returns:
        torch.Tensor: Perturbed adversarial image tensor clamped to [-1.0, 1.0].
    """
    signed_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * signed_data_grad
    perturbed_image = torch.clamp(perturbed_image, -1.0, 1.0)
    return perturbed_image

if __name__ == "__main__":
    clean_image, raw_grad, true_label = extract_image_gradient()
    epsilon_val = 0.05
    adv_image = fgsm_attack(clean_image, epsilon_val, raw_grad)
    print(f"Adversarial Image generated successfully. Shape: {adv_image.shape}")