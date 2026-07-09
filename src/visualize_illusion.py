"""
Adversarial Illusion Visualization Module.

This script demonstrates and visualizes the effects of the Fast Gradient Sign Method (FGSM)
attack on individual sample images from the CIFAR-10 test dataset. It generates a 
side-by-side comparative panel containing clean images, their corresponding computed 
gradient noise maps, and the resulting perturbed adversarial images.
"""

import os 
import torch 
import matplotlib.pyplot as plt 
import numpy as np 

from data_loader import get_data_loaders
from model import SimpleCNN
from attack import fgsm_attack

# Class labels for CIFAR-10 dataset predictions mapping
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    De-normalizes a PyTorch tensor and converts it into a NumPy array ready for Matplotlib plotting.

    This function performs the inverse operation of PyTorch image normalization:
    tensor = (tensor * std) + mean

    Args:
        tensor (torch.Tensor): Normalized input image tensor.
        mean (list): The channel means used during dataset normalization.
        std (list): The channel standard deviations used during dataset normalization.

    Returns:
        numpy.ndarray: Plottable image array of shape (Height, Width, Channels) in the [0.0, 1.0] range.
    """
    # 1. Clone and detach the tensor to avoid modifying the original data in-place
    x = tensor.clone().detach().cpu()
    
    # 2. Reshape mean and std arrays to [3, 1, 1] to allow PyTorch broadcasting across (C, H, W)
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    
    # 3. Apply the inverse normalization math
    x = (x * std_tensor) + mean_tensor
    
    # 4. Clip values to prevent minor floating-point calculations from exceeding the valid [0, 1] range
    x = torch.clamp(x, 0.0, 1.0)
    
    # 5. Permute the dimensions from PyTorch (Channels, Height, Width) to Matplotlib (Height, Width, Channels)
    return x.permute(1, 2, 0).numpy()

def generate_visual_panel(model_path, save_dir="reports/figures"):
    """
    Loads the trained model, extracts a batch of test images, computes their gradients,
    creates adversarial examples using FGSM, and plots a comparative 3-column diagnostic panel.

    Columns in the panel:
    1. Clean Image (labeled with true class and model prediction)
    2. Gradient Noise Map (visualizing the sign of the loss gradients)
    3. Adversarial Image (labeled with the model's new adversarial prediction)

    Args:
        model_path (str): File path to the trained model's state dictionary (.pth).
        save_dir (str): Folder where the completed visualization panel should be saved.
    """
    # Choose processing hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Fetch a single batch of 4 images from the test dataset with shuffling enabled
    _, test_loader, _ = get_data_loaders(batch_size=4, shuffle_test=True)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # 2. Load the trained simple CNN model and set it to evaluation mode
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Enable gradient tracking on the input images
    images.requires_grad = True

    # 4. Perform forward pass and compute cross-entropy loss relative to true targets
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    
    # 5. Perform backpropagation to populate images.grad with the loss gradients
    model.zero_grad()
    loss.backward()

    # 6. Retrieve the gradient map of the loss with respect to the input pixels
    data_grad = images.grad.data 

    # 7. Apply the FGSM attack to generate perturbed adversarial images
    epsilon = 0.03
    adv_images = fgsm_attack(images, epsilon, data_grad)

    # 8. Run inference on both clean and adversarial images to compare predictions
    with torch.no_grad():
        clean_preds = model(images).argmax(dim=1)
        adv_preds = model(adv_images).argmax(dim=1)

    # 9. Initialize the Matplotlib layout grid (4 rows, 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    # Set overall title for the figure
    fig.suptitle(f"FGSM Evasion Diagnostics (Epsilon = {epsilon})", fontsize=16, weight='bold', y=0.95)

    # 10. Populate the grid panels row-by-row
    for i in range(4):
        # Column 1: Clean image display
        clean_img = denormalize(images[i])
        axes[i, 0].imshow(clean_img)
        axes[i, 0].set_title(f"Target: {classes[labels[i]]}\nPred: {classes[clean_preds[i]]}", fontsize=10)
        axes[i, 0].axis('off')

        # Column 2: Grayscale gradient sign noise map display
        noise_map = data_grad[i].sign().permute(1, 2, 0).cpu().numpy()
        noise_grayscale = np.mean(noise_map, axis=2)
        axes[i, 1].imshow(noise_grayscale, cmap='coolwarm')
        axes[i, 1].set_title("Gradient Noise Map\nsign(∇x L)", fontsize=10)
        axes[i, 1].axis('off')

        # Column 3: Adversarial image display
        adv_img = denormalize(adv_images[i])
        axes[i, 2].imshow(adv_img)
        
        # Color prediction title red if the attack successfully tricked the model (evaded)
        is_evaded = clean_preds[i] != adv_preds[i]
        title_color = 'red' if is_evaded else 'black'
        axes[i, 2].set_title(f"Pred: {classes[adv_preds[i]]}", fontsize=10, color=title_color)
        axes[i, 2].axis('off')

    # 11. Save the completed plot grid to file
    output_path = os.path.join(save_dir, "optical_illusion_panel.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Visual diagnostic panel saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_visual_panel(model_path="models/baseline_model.pth")