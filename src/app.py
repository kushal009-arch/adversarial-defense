"""
Streamlit Web Dashboard for CIFAR-10 Adversarial Robustness Explorer.

This module provides an interactive interface allowing users to upload images,
select between standard (baseline) and adversarial-trained (robust) CNN models,
and inspect image preprocessing and inference predictions.
"""

import streamlit as st 
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image 

# Step 1: Page configuration and custom layout setup
st.set_page_config(
    page_title="CIFAR-10 Adversarial Robustness Dashboard", 
    page_icon="🛡️",
    layout="wide"
)

st.title("CIFAR-10 Adversarial Robustness Explorer")
st.caption("A Deep Learning Security Showcase analyzing Fast Gradient Sign Method (FGSM) attacks.")
st.divider()

# Step 2: CIFAR-10 label mapping definitions
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", 
    "Frog", "Horse", "Ship", "Truck"
]

# Step 3: Model loading utility with Streamlit resource caching
@st.cache_resource
def load_model(model_path: str):
    """
    Loads and caches PyTorch SimpleCNN model weights to prevent reload overhead during UI re-runs.

    Args:
        model_path (str): Relative or absolute path to the saved .pth model state dictionary.

    Returns:
        torch.nn.Module: The loaded SimpleCNN model set to evaluation mode (`.eval()`).
    """
    from model import SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Step 4: Session state initialization for tracking user session flags
if "model_loader" not in st.session_state:
    st.session_state["model_loader"] = True

# Step 5: Image preprocessing pipeline matching CIFAR-10 training transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Step 6: UI Layout - Sidebar Controls
with st.sidebar:
    st.header("Controls and Settings")
    st.markdown("Select options to evaluate model against adversarial noise.")

    selected_model_type = st.radio(
        "Choose Model Architecture:",
        ["Baseline Model (Standard)", "Robust Model (Adversarial Trained)"]
    )

# Step 7: Main Content - Two-column interactive interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader(
        "Upload a CIFAR-10 suitable image (PNG, JPEG)", 
        type=["png", "jpeg", "jpg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess Tensor for model inference: add a batch dimension at index 0 -> shape becomes (1, 3, 32, 32)
        # PyTorch models expect input shape of (Batch_Size, Channels, Height, Width)
        input_tensor = transform(image).unsqueeze(0)
        st.success(f"Image processed into Tensor shape: '{tuple(input_tensor.shape)}'")

with col2:
    st.subheader("Model Output")
    if uploaded_file is None:
        st.info("Please upload an image on the left column to run inference.")
    else:
        st.write(f"**Active Model:** '{selected_model_type}'")
        
        # Step 8: Inference logic execution placeholder