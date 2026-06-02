import torch 
import torch.nn.functional as F 
from data_loader import get_data_loaders
from model import SimpleCNN

def extract_image_gradient():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))

    model.eval()

    _, test_loader, _ = get_data_loaders(batch_size=1)
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)

    data.requires_grad = True

    output = model(data)

    loss = F.cross_entropy(output, target)
    
    model.zero_grad()

    loss.backward()

    gradient_map = data.grad.data

    print(f"Original Image Shape: {data.shape}")
    print(f"Gradient Map Shape: {gradient_map.shape}")
    print(f"\nSample Gradient Values (first 5 pixels of the red channel):")
    print(gradient_map[0][0][0][:5])

    return data, gradient_map, target

if __name__ == "__main__":
    extract_image_gradient()