"""
Streamlit Web Dashboard for CIFAR-10 Adversarial Robustness Explorer.

This module provides an interactive interface allowing users to upload images,
select ground-truth labels, evaluate baseline vs. robust model architectures in real-time,
and inspect image preprocessing, attack anatomy (clean image, magnified noise, adversarial sample),
and FGSM evasion predictions with head-to-head probability distribution charts.
"""

import gc
import streamlit as st 
import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
import pandas as pd

from attack import fgsm_attack
from src.model import SimpleCNN

def format_tensor_for_display(tensor):
    """
    Clamps and reshapes a PyTorch image tensor [1, C, H, W] or [C, H, W] into a 
    displayable NumPy array [H, W, C] for Streamlit.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    img = tensor.clone().detach().cpu()

    # De-normalize from [-1, 1] back to [0, 1] for display
    img = img * 0.5 + 0.5
    img = torch.clamp(img, 0.0, 1.0)

    img_np = img.permute(1, 2, 0).numpy()
    return img_np

def load_and_preprocess_image(uploaded_file):
    """
    Safely opens uploaded image, strips Alpha transparency, and converts to RGB.
    """
    raw_image = Image.open(uploaded_file).convert("RGB")
    return raw_image

# Page configuration and layout setup
st.set_page_config(
    page_title="CIFAR-10 Adversarial Robustness Dashboard", 
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ CIFAR-10 Adversarial Robustness Explorer")
st.caption("A Deep Learning Security Showcase analyzing Fast Gradient Sign Method (FGSM) attacks.")
st.divider()

# CIFAR-10 label mapping definitions
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", 
    "Frog", "Horse", "Ship", "Truck"
]

# Model loading utility with Streamlit resource caching
@st.cache_resource
def load_all_models():
    """
    Loads and caches both Baseline and Robust PyTorch SimpleCNN model weights to GPU/CPU.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baseline_net = SimpleCNN().to(device)
    baseline_net.load_state_dict(torch.load("models/baseline_model.pth", map_location=device))
    baseline_net.eval()
    
    robust_net = SimpleCNN().to(device)
    robust_net.load_state_dict(torch.load("models/robust_model.pth", map_location=device))
    robust_net.eval()

    return baseline_net, robust_net, device

baseline_net, robust_net, device = load_all_models()

# Define image transformation matching CIFAR-10 training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Sidebar Controls & File Upload
st.sidebar.header("Controls & Setup")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Safely load and convert image
    raw_image = load_and_preprocess_image(uploaded_file)
    clean_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Show clean baseline image in sidebar
    st.sidebar.image(raw_image, caption="Clean Uploaded Image", use_container_width=True)

    # =========================================================================
    # STEP 3: REAL-TIME INTERACTIVE FRAGMENT
    # =========================================================================
    @st.fragment
    def render_attack_fragment(img_tensor):
        st.subheader("⚡ Real-Time Evasion Diagnostics")
        
        # User Selection for Ground Truth Label
        true_label_name = st.selectbox("Select Ground Truth Class (Actual Object):", CIFAR10_CLASSES)
        true_label_idx = torch.tensor([CIFAR10_CLASSES.index(true_label_name)], device=device)

        # Interactive Epsilon Slider (0.00 to 0.30)
        epsilon = st.slider(
            "Adversarial Perturbation Budget (ε)", 
            min_value=0.00, 
            max_value=0.30, 
            value=0.00, 
            step=0.01
        )

        # 1. Prepare tensor and enable autograd to calculate image gradients
        input_tensor = img_tensor.clone().detach().requires_grad_(True)
        
        # 2. Pass clean image through baseline net to get logits
        base_logits = baseline_net(input_tensor)

        # 3. Compute loss w.r.t user-selected True Label and extract gradients
        loss = F.cross_entropy(base_logits, true_label_idx)
        baseline_net.zero_grad()
        loss.backward()

        # 4. Generate perturbed tensor using FGSM
        data_grad = input_tensor.grad.data
        perturbed_tensor = fgsm_attack(input_tensor, epsilon, data_grad)

        # 5. Inference on perturbed image for both models
        with torch.no_grad():
            adv_out_base = baseline_net(perturbed_tensor)
            adv_out_robust = robust_net(perturbed_tensor)
            
            probs_base = F.softmax(adv_out_base, dim=1)[0].cpu().numpy()
            probs_robust = F.softmax(adv_out_robust, dim=1)[0].cpu().numpy()

        # Compute absolute noise and magnify for visual inspection
        noise_tensor = (perturbed_tensor - input_tensor).abs() * 5.0

        st.divider()
        st.subheader("🖼️ Attack Anatomy")
        
        # 3-Column Layout
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                format_tensor_for_display(img_tensor), 
                caption="Clean Input (x)", 
                use_container_width=True
            )

        with col2:
            st.image(
                format_tensor_for_display(noise_tensor), 
                caption=f"Magnified Noise (5x | ε={epsilon:.2f})", 
                use_container_width=True
            )

        with col3:
            st.image(
                format_tensor_for_display(perturbed_tensor), 
                caption=f"Adversarial Sample (x_adv | ε={epsilon:.2f})", 
                use_container_width=True
            )

        # Head-to-Head Model Evaluation
        st.divider()
        st.subheader("⚔️ Head-to-Head Model Evaluation")
        
        col_base, col_robust = st.columns(2)

        # 1. Baseline Model Panel
        with col_base:
            base_pred_idx = probs_base.argmax()
            base_conf = probs_base[base_pred_idx] * 100
            
            st.error("🔴 **Standard Baseline Model**")
            st.metric("Predicted Class", CIFAR10_CLASSES[base_pred_idx])
            st.metric("Confidence", f"{base_conf:.2f}%")
            
            # Interactive Probability Bar Chart
            df_base = pd.DataFrame({
                'Class': CIFAR10_CLASSES, 
                'Probability': probs_base
            })
            st.bar_chart(df_base.set_index('Class'))

        # 2. Adversarially Trained (Robust) Model Panel
        with col_robust:
            robust_pred_idx = probs_robust.argmax()
            robust_conf = probs_robust[robust_pred_idx] * 100
            
            st.success("🟢 **Adversarially Trained Model**")
            st.metric("Predicted Class", CIFAR10_CLASSES[robust_pred_idx])
            st.metric("Confidence", f"{robust_conf:.2f}%")
            
            # Interactive Probability Bar Chart
            df_robust = pd.DataFrame({
                'Class': CIFAR10_CLASSES, 
                'Probability': probs_robust
            })
            st.bar_chart(df_robust.set_index('Class'))

        # Free autograd references
        gc.collect()

    # Call Fragment
    render_attack_fragment(clean_tensor)

else:
    st.info("👈 Upload a PNG or JPG image from the sidebar to test real-time adversarial evasion.")