"""
The objective is to write the base layout for app.py, define CIFAR 10 label mapping, 
set up file uploader, and leverage seteamlit's caching mechanism to efficiently load models 
without rerunning initialization on every user interaction. 

"""

import streamlit as st 
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image 

# 1. Page configuration and custom setup

st.set_page_config(
    page_title = "CIFAR-10 Adversarial Robustness Dashboard", 
    page_icon = "Adversarial-Defense Dashboard",
    layout = "wide"
)

st.title("CIFAR-10 Adversarial Robustness Explorer")
st.caption("A Deep Learning Security Showcase analyzing Fast Gradient Sign Method (FGSM) attacks.")

st.divider()

# 2. CIFAR 10 Label Mapping
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", 
    "Frog", "Horse", "Ship", "Truck"
]

# 3. Model Loading with Caching

@st.cache_resource
def load_model(model_path: str):
    """
    Loads and caches PyTorch model weights to prevent reload overhead. 
    """
    from model import SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
    model.eval()
    return model

if "model_loader" not in st.session_state:
    st.session_state["model_loader"] = True

# 4. Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((32, 32)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 5. UI Layout: Sidebar and Main Content

with st.sidebar:
    st.header("Controls and Settings")
    st.markdown("Select options to evaluate model against adversarial noise.")

    selected_model_type = st.radio(
        "Choose Model Architecture:",
        ["Baseline Model (Standard)", "Robust Model (Adversarial Trained)"]
    )

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
        
        # Inference Logic