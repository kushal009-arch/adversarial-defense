"""
Streamlit Web Dashboard for CIFAR-10 Adversarial Robustness Explorer.

This module provides an interactive interface allowing users to upload images,
select between standard (baseline) and adversarial-trained (robust) CNN models,
and inspect image preprocessing and real-time FGSM adversarial evasion predictions.
"""

import streamlit as st 
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np

from attack import fgsm_attack

# Step 1: Page configuration and layout setup
st.set_page_config(
    page_title="CIFAR-10 Adversarial Robustness Dashboard", 
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CIFAR-10 Adversarial Robustness Explorer")
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
    from src.model import SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Step 4: Session state initialization
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
    
    display_mode = st.radio(
        "Image Render Mode:",
        ["Smooth View (Bilinear)", "Pixel Grid View (Nearest)"]
    )

    model_file = "models/baseline_model.pth" if "Baseline" in selected_model_type else "models/robust_model.pth"
    active_model = load_model(model_file)

# Step 7: Main Content - Two-column interactive interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Input Image & Attack Setup")
    uploaded_file = st.file_uploader(
        "Upload a CIFAR-10 suitable image (PNG, JPEG)", 
        type=["png", "jpeg", "jpg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Clean Image", use_container_width=True)

        # Preprocess Tensor for model inference: add a batch dimension at index 0 -> shape (1, 3, 32, 32)
        input_tensor = transform(image).unsqueeze(0)
        st.success(f"Image processed into Tensor shape: '{tuple(input_tensor.shape)}'")

with col2:
    st.subheader("🎯 Evasion Testing & Model Output")
    if uploaded_file is None:
        st.info("Please upload an image on the left column to run inference.")
    else:
        # Isolated fragment for real-time attack evaluation
        @st.fragment
        def render_attack_interactive(input_tensor, model, orig_image, mode):
            true_label_name = st.selectbox("Ground Truth Label:", CIFAR10_CLASSES)
            true_label_idx = torch.tensor([CIFAR10_CLASSES.index(true_label_name)])

            epsilon = st.slider("Attack Strength (Epsilon - ε)", min_value=0.0, max_value=0.3, value=0.0, step=0.01)

            # Enable autograd tracking for input tensor
            input_tensor_grad = input_tensor.clone().detach().requires_grad_(True)

            # Compute Gradients w.r.t Ground Truth Label
            output = model(input_tensor_grad)
            loss = F.cross_entropy(output, true_label_idx)
            model.zero_grad()
            loss.backward()

            # Generate FGSM Perturbation
            data_grad = input_tensor_grad.grad.data
            perturbed_tensor = fgsm_attack(input_tensor, epsilon, data_grad)

            # Inference on Perturbed Image
            with torch.no_grad():
                adv_output = model(perturbed_tensor)
                probabilities = F.softmax(adv_output, dim=1)[0]
                pred_class = CIFAR10_CLASSES[probabilities.argmax().item()]
                confidence = probabilities.max().item() * 100

            # Render Metrics
            st.metric(label="Adversarial Predicted Class", value=pred_class)
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
            st.progress(float(confidence / 100))

            # De-normalize tensor back to [0, 1] range
            adv_img_np = perturbed_tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
            adv_img_np = (adv_img_np * 0.5 + 0.5).clip(0, 1)

            # Convert to PIL Image and upscale using chosen resampling mode
            adv_img_uint8 = (adv_img_np * 255).astype(np.uint8)
            resample_filter = Image.Resampling.BILINEAR if "Smooth" in mode else Image.Resampling.NEAREST
            display_img = Image.fromarray(adv_img_uint8).resize(orig_image.size, resample_filter)

            st.image(display_img, caption=f"Perturbed Image (ε = {epsilon}) - {mode}", use_container_width=True)

        render_attack_interactive(input_tensor, active_model, image, display_mode)
