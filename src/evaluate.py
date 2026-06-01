import torch 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from data_loader import get_data_loaders
from model import SimpleCNN

def evaluate_model():
    #setup device and class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck')

    _, test_loader, _ = get_data_loaders(batch_size=64)

    model = SimpleCNN().to(device)

    try:
        # map_location=device ensures it loads correctly whether on GPU or CPU
        model.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
        print("Successfully loaded 'models/baseline_model.pth'")
    except FileNotFoundError:
        print("Error: Could not find model weights. Try running train.py first.")
        return 

    # Inference Mode
    model.eval()

    correct = 0
    total = 0

    conf_matrix = torch.zeros(10, 10, dtype=torch.int32)

    print("Evaluating model on 10,000 test images...")

    # Turn off Autograd
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Convert to CPU list for safe and fast matrix indexing
            for true_label, pred_label in zip(labels.cpu().tolist(), predicted.cpu().tolist()):
                conf_matrix[true_label, pred_label] += 1

    accuracy = 100 * correct / total

    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print("\n")
    
    df_cm = pd.DataFrame(conf_matrix.numpy(), index=classes, columns=classes)
    print(df_cm)

    # Create a Visual Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'CIFAR-10 Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Define the directory
    directory = 'reports/figures'
    os.makedirs(directory, exist_ok=True)

    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"confusion_matrix_{timestamp}.png"
    save_path = os.path.join(directory, filename)

    # Save the plot
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    
    print("\n")
    print("Saved confusion matrix visualization to 'confusion_matrix.png'")

if __name__ == "__main__":
    evaluate_model()
