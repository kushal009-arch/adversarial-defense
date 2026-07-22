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