# import necessary modules

import os 
import torch 
import matplotlib.pyplot as plt 
import numpy as np 

from data_loader import get_data_loaders
from model import SimpleCNN
from attack import fgsm_attack

classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

def denormalize(tensor, mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):

    #clone tensor to avoid upstream inplace mutations aka to avoid directly modifying the original piece of data
    # 1. Clone and move to CPU safely
    x = tensor.clone().detach().cpu()
    
    # 2. Convert mean and std lists to PyTorch tensors
    mean_tensor = torch.tensor(mean).view(3, 1, 1)  # Reshapes to [3, 1, 1]
    std_tensor = torch.tensor(std).view(3, 1, 1)    # Reshapes to [3, 1, 1]
    
    # 3. Apply the inverse math across all channels simultaneously
    # [3, 32, 32] * [3, 1, 1] perfectly broadcasts across Height and Width!
    x = (x * std_tensor) + mean_tensor
    
    # 4. Enforce strict box constraints
    x = torch.clamp(x, 0.0, 1.0)
    
    # 5. Rearrange layout from PyTorch (C, H, W) to Matplotlib (H, W, C)
    return x.permute(1, 2, 0).numpy() # pytorch processes images in NCHW (batch, channel, height, width) format
                                  # matplotlib and other images processors understand NHWC (batch, height, width, channel)
                                  # channel is red, green blue (RGB)
                                  # this conversion is done using .permute(1, 2, 0)
                                  # This expression prepares a PyTorch image tensor to be displayed by standard plotting tools like Matplotlib:

def generate_visual_panel(model_path, save_dir="reports/figures"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Pull a single batch from the test dataloader
    _, test_loader, _ = get_data_loaders(batch_size=4, shuffle_test=True)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    images.requires_grad = True

    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    data_grad = images.grad.data # calculate the gradient of loss with respect to the input images. 

    epsilon = 0.03
    adv_images = fgsm_attack(images, epsilon, data_grad)

    with torch.no_grad(): # you do this to disable gradient cdalculation, used durign inference
        clean_preds = model(images).argmax(dim=1) # we do argmax(dim=1) to get raw model outputs to correct labels
        adv_preds = model(adv_images).argmax(dim=1)

    # Build Matplotlib layout grid (4 rows 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(12,14))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)

    fig.suptitle(f"FGSM evasion disgnostics (Epsilon = {epsilon})", fontsize=16, weight='bold', y=0.95)

    for i in range(4):
        # clean panel
        clean_img = denormalize(images[i])
        axes[i, 0].imshow(clean_img)
        axes[i, 0].set_title(f"Target: {classes[labels[i]]}\nPred: {classes[clean_preds[i]]}", fontsize=10)
        axes[i, 0].axis('off')

        # isolated noise
        noise_map = data_grad[i].sign().permute(1, 2, 0).cpu().numpy()
        noise_grayscale = np.mean(noise_map, axis=2)

        axes[i, 1].imshow(noise_grayscale, cmap='coolwarm')
        axes[i, 1].set_title("Gradient Noise map\nsign(∇x L)", fontsize=10)
        axes[i, 1].axis('off')

        # adversarial panel
        adv_img = denormalize(adv_images[i])
        axes[i, 2].imshow(adv_img)

        is_evaded = clean_preds[i] != adv_preds[i]
        title_color = 'red' if is_evaded else 'black'
        axes[i, 2].set_title(f"Pred: {classes[adv_preds[i]]}", fontsize=10, color=title_color)
        axes[i, 2].axis('off')

    output_path = os.path.join(save_dir, "optical_illusion_panel.png")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[SUCCESS] Visual diagnostic panel saved successfully to: {output_path}")

if __name__ == "__main__":
    generate_visual_panel(model_path="models/baseline_model.pth")

        
        