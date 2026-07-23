"""
Streamlit Web Dashboard for CIFAR-10 Adversarial Robustness Explorer.

This module provides an enterprise-grade interactive interface allowing users to evaluate 
baseline vs. robust CNN model architectures in real-time, inspect multi-mode noise heatmaps, 
explore dynamic epsilon evasion curves, test preset sample images, and export diagnostic logs.
"""

import gc
import json
import pickle
import streamlit as st 
import torch 
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from streamlit_image_comparison import image_comparison

from attack import fgsm_attack
from src.model import SimpleCNN

# Step 1: Page configuration setup
st.set_page_config(
    page_title="CIFAR-10 Adversarial Robustness Dashboard", 
    layout="wide"
)

# Custom CSS Theme Injection
st.markdown("""
<style>
    /* Card Container Styling */
    div[data-testid="stVerticalBlock"] > div[data-testid="stBlock"] {
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Custom Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Section Headers */
    .sub-header {
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: #888888;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("CIFAR-10 Adversarial Robustness Explorer")
st.caption("A Deep Learning Security Showcase analyzing Fast Gradient Sign Method (FGSM) attacks.")
st.divider()

# Step 2: CIFAR-10 label mapping definitions
CIFAR10_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", 
    "Frog", "Horse", "Ship", "Truck"
]

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

def load_and_preprocess_image(image_input):
    """
    Safely opens uploaded or preset image, strips Alpha transparency, and converts to RGB.
    """
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    raw_image = Image.open(image_input).convert("RGB")
    return raw_image

@st.cache_data
def load_preset_gallery_samples():
    """
    Loads one representative sample image per class from the CIFAR-10 test batch.
    """
    samples = {}
    try:
        with open('data/cifar-10-batches-py/test_batch', 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            images = data_dict[b'data']
            labels = data_dict[b'labels']

            for i, label_idx in enumerate(labels):
                class_name = CIFAR10_CLASSES[label_idx]
                if class_name not in samples:
                    img_array = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
                    samples[class_name] = Image.fromarray(img_array)
                if len(samples) == 10:
                    break
    except Exception:
        # Fallback synthetic solid patterns if batch file is inaccessible
        for idx, class_name in enumerate(CIFAR10_CLASSES):
            color = (idx * 25 % 255, (idx * 50 + 100) % 255, (255 - idx * 20) % 255)
            samples[class_name] = Image.new('RGB', (32, 32), color=color)
    return samples

def create_styled_probability_chart(df, predicted_class, true_class):
    """
    Generates a horizontal Plotly bar chart with dynamic color-coding:
    Red for misclassifications, Green for correct predictions, Blue for other classes.
    """
    colors = [
        '#EF553B' if c == predicted_class and c != true_class 
        else '#00CC96' if c == predicted_class 
        else '#636EFA' for c in df['Class']
    ]
    
    fig = px.bar(
        df, 
        x='Probability', 
        y='Class', 
        orientation='h',
        text_auto='.1%',
        title="Class Probability Distribution"
    )
    fig.update_traces(marker_color=colors, textposition='outside')
    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_range=[0, 1],
        height=280,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig

def create_evasion_curve_plot(sweep_epsilons, base_confs, robust_confs):
    """
    Renders an interactive multi-line Plotly chart showing Epsilon vs. Confidence for both models.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep_epsilons, y=base_confs,
        mode='lines+markers',
        name='Standard Baseline CNN',
        line=dict(color='#EF553B', width=3),
        marker=dict(size=7)
    ))
    fig.add_trace(go.Scatter(
        x=sweep_epsilons, y=robust_confs,
        mode='lines+markers',
        name='Adversarially Trained CNN',
        line=dict(color='#00CC96', width=3),
        marker=dict(size=7)
    ))
    fig.update_layout(
        title="Real-Time Evasion Curve (Epsilon vs Confidence)",
        xaxis_title="Perturbation Budget (Epsilon)",
        yaxis_title="Model Confidence (%)",
        yaxis_range=[0, 105],
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def generate_heatmap_visualization(tensor, mode):
    """
    Generates multi-mode noise visualization: Grayscale 5x, Plasma Heatmap, or Sign Map.
    """
    img_np = tensor.squeeze(0).cpu().detach().permute(1, 2, 0).numpy()
    
    if mode == "Plasma Heatmap":
        # Average intensity across color channels
        intensity = np.mean(np.abs(img_np), axis=2)
        norm_intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        cmap = plt.get_cmap('plasma')
        colored = cmap(norm_intensity)[:, :, :3]
        return (colored * 255).astype(np.uint8)
    elif mode == "Gradient Sign Map":
        sign_map = np.sign(img_np)
        sign_norm = (sign_map * 0.5 + 0.5)
        return (sign_norm * 255).astype(np.uint8)
    else:
        # Default 5x Magnified Difference
        diff = np.abs(img_np) * 5.0
        diff = np.clip(diff, 0.0, 1.0)
        return (diff * 255).astype(np.uint8)

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

# Image transformation matching CIFAR-10 training
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Sidebar Controls & File Upload
st.sidebar.header("Controls and Setup")

# Preset Sample Gallery Selector
input_mode = st.sidebar.radio("Select Image Source:", ["Preset Gallery Image", "Upload Custom Image"])

selected_image = None
if input_mode == "Preset Gallery Image":
    preset_samples = load_preset_gallery_samples()
    preset_class = st.sidebar.selectbox("Choose Benchmark Preset Image:", list(preset_samples.keys()))
    selected_image = preset_samples[preset_class]
else:
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        selected_image = uploaded_file

# Model Architecture Telemetry & Layer Inspector (Sidebar Expander)
with st.sidebar.expander("Model Architecture Specifications"):
    st.markdown("""
    **SimpleCNN Specs:**
    * **Device:** CPU/CUDA Auto-detect
    * **Conv Layer 1:** 3 -> 32 channels (3x3 kernel, pad 1)
    * **Conv Layer 2:** 32 -> 64 channels (3x3 kernel, pad 1)
    * **Pooling:** MaxPool2d (2x2)
    * **FC Layer 1:** 64 * 8 * 8 -> 512 units
    * **FC Layer 2:** 512 -> 10 output classes
    * **Total Parameters:** ~1,173,770
    * **Robust Loss Formula:**  
      `Loss = 0.5 * Clean_Loss + 0.5 * FGSM_Loss`
    """)

if selected_image is not None:
    # Load and preprocess image
    raw_image = load_and_preprocess_image(selected_image)
    clean_tensor = transform(raw_image).unsqueeze(0).to(device)

    # Show clean baseline image in sidebar
    st.sidebar.image(raw_image, caption="Clean Input Image", use_container_width=True)

    # REAL-TIME INTERACTIVE FRAGMENT
    @st.fragment
    def render_attack_fragment(img_tensor):
        st.subheader("Real-Time Evasion Diagnostics")
        
        # User Selection for Ground Truth Label
        default_idx = 0
        if input_mode == "Preset Gallery Image":
            default_idx = CIFAR10_CLASSES.index(preset_class)
            
        true_label_name = st.selectbox("Select True Class (Actual Object):", CIFAR10_CLASSES, index=default_idx)
        true_label_idx = CIFAR10_CLASSES.index(true_label_name)
        true_label_tensor = torch.tensor([true_label_idx], device=device)

        # Interactive Epsilon Slider (0.00 to 0.30)
        epsilon = st.slider(
            "Adversarial Perturbation Budget (Epsilon)", 
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
        loss = F.cross_entropy(base_logits, true_label_tensor)
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

        base_pred_idx = probs_base.argmax()
        base_pred_class = CIFAR10_CLASSES[base_pred_idx]
        base_conf = probs_base[base_pred_idx] * 100

        robust_pred_idx = probs_robust.argmax()
        robust_pred_class = CIFAR10_CLASSES[robust_pred_idx]
        robust_conf = probs_robust[robust_pred_idx] * 100

        # Defense Delta & Robustness Gain Calculation
        robustness_gain = robust_conf - base_conf

        # Multi-Tab Navigation Architecture
        tab1, tab2, tab3 = st.tabs([
            "Real-Time Evasion Studio", 
            "Perturbation Anatomy", 
            "Telemetry and Benchmark Curves"
        ])

        with tab1:
            st.markdown("<div class='sub-header'>Head-to-Head Model Evaluation</div>", unsafe_allow_html=True)

            # Security Threat Banners
            if base_pred_idx != true_label_idx:
                st.error(f"Evasion Attack Successful: Baseline model forced to misclassify sample as {base_pred_class}.")
            else:
                st.success("Model Stable: Baseline prediction holds under current perturbation budget.")

            # Defense Delta Metric Card
            with st.container(border=True):
                delta_col1, delta_col2, delta_col3 = st.columns(3)
                with delta_col1:
                    st.metric("Baseline Target Confidence", f"{probs_base[true_label_idx]*100:.2f}%")
                with delta_col2:
                    st.metric("Robust Target Confidence", f"{probs_robust[true_label_idx]*100:.2f}%")
                with delta_col3:
                    st.metric("Robustness Gain (Accuracy Preserved)", f"{robustness_gain:+.2f}%", delta=f"{robustness_gain:.2f}%")

            col_base, col_robust = st.columns(2)

            # 1. Baseline Model Panel
            with col_base:
                with st.container(border=True):
                    st.markdown("<div class='sub-header'>Standard Baseline Model</div>", unsafe_allow_html=True)
                    st.metric("Predicted Class", base_pred_class)
                    st.metric("Confidence", f"{base_conf:.2f}%")
                    
                    df_base = pd.DataFrame({
                        'Class': CIFAR10_CLASSES, 
                        'Probability': probs_base
                    })
                    st.plotly_chart(
                        create_styled_probability_chart(df_base, base_pred_class, true_label_name), 
                        use_container_width=True
                    )

            # 2. Adversarially Trained (Robust) Model Panel
            with col_robust:
                with st.container(border=True):
                    st.markdown("<div class='sub-header'>Adversarially Trained Model</div>", unsafe_allow_html=True)
                    st.metric("Predicted Class", robust_pred_class)
                    st.metric("Confidence", f"{robust_conf:.2f}%")
                    
                    df_robust = pd.DataFrame({
                        'Class': CIFAR10_CLASSES, 
                        'Probability': probs_robust
                    })
                    st.plotly_chart(
                        create_styled_probability_chart(df_robust, robust_pred_class, true_label_name), 
                        use_container_width=True
                    )

        with tab2:
            st.markdown("<div class='sub-header'>Interactive Split-View Comparison</div>", unsafe_allow_html=True)
            clean_np = format_tensor_for_display(img_tensor)
            perturbed_np = format_tensor_for_display(perturbed_tensor)
            
            clean_uint8 = (clean_np * 255).astype(np.uint8)
            perturbed_uint8 = (perturbed_np * 255).astype(np.uint8)

            clean_pil = Image.fromarray(clean_uint8).resize(raw_image.size, Image.Resampling.BILINEAR)
            perturbed_pil = Image.fromarray(perturbed_uint8).resize(raw_image.size, Image.Resampling.BILINEAR)

            with st.container(border=True):
                image_comparison(
                    img1=clean_pil,
                    img2=perturbed_pil,
                    label1="Clean Image",
                    label2=f"FGSM Attack (Epsilon={epsilon:.2f})",
                    starting_position=50,
                    show_labels=True,
                    make_responsive=True,
                    in_memory=True
                )

            st.markdown("<div class='sub-header'>3-Panel Image Decomposition</div>", unsafe_allow_html=True)
            
            # Multi-Mode Noise Heatmap Selector
            heatmap_mode = st.selectbox(
                "Select Noise Heatmap Mode:", 
                ["Magnified Difference (5x)", "Plasma Heatmap", "Gradient Sign Map"]
            )
            
            raw_noise_tensor = (perturbed_tensor - input_tensor)
            noise_img_uint8 = generate_heatmap_visualization(raw_noise_tensor, heatmap_mode)
            noise_pil = Image.fromarray(noise_img_uint8).resize(raw_image.size, Image.Resampling.BILINEAR)

            with st.container(border=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.image(
                        clean_pil, 
                        caption="Clean Input (x)", 
                        use_container_width=True
                    )

                with col2:
                    st.image(
                        noise_pil, 
                        caption=f"Noise Map ({heatmap_mode} | Epsilon = {epsilon:.2f})", 
                        use_container_width=True
                    )

                with col3:
                    st.image(
                        perturbed_pil, 
                        caption=f"Adversarial Sample (x_adv | Epsilon = {epsilon:.2f})", 
                        use_container_width=True
                    )

        with tab3:
            st.markdown("<div class='sub-header'>Real-Time Evasion Curve and Telemetry</div>", unsafe_allow_html=True)
            
            # Dynamic Epsilon Sweep Calculation
            sweep_epsilons = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
            sweep_base_confs = []
            sweep_robust_confs = []

            with torch.no_grad():
                for eps in sweep_epsilons:
                    p_tensor = fgsm_attack(input_tensor, eps, data_grad)
                    out_b = baseline_net(p_tensor)
                    out_r = robust_net(p_tensor)
                    conf_b = F.softmax(out_b, dim=1)[0, true_label_idx].item() * 100
                    conf_r = F.softmax(out_r, dim=1)[0, true_label_idx].item() * 100
                    sweep_base_confs.append(conf_b)
                    sweep_robust_confs.append(conf_r)

            with st.container(border=True):
                st.plotly_chart(
                    create_evasion_curve_plot(sweep_epsilons, sweep_base_confs, sweep_robust_confs),
                    use_container_width=True
                )

            with st.container(border=True):
                st.write("**Model Evaluation Telemetry Summary**")
                telemetry_df = pd.DataFrame({
                    "Architecture": ["Standard Baseline CNN", "Adversarially Trained CNN"],
                    "Target Class": [true_label_name, true_label_name],
                    "Predicted Class": [base_pred_class, robust_pred_class],
                    "Target Confidence": [f"{probs_base[true_label_idx]*100:.2f}%", f"{probs_robust[true_label_idx]*100:.2f}%"],
                    "Top Confidence": [f"{base_conf:.2f}%", f"{robust_conf:.2f}%"],
                    "Status": [
                        "Vulnerable" if base_pred_idx != true_label_idx else "Hardened",
                        "Vulnerable" if robust_pred_idx != true_label_idx else "Hardened"
                    ]
                })
                st.dataframe(telemetry_df, use_container_width=True)

                # Export Diagnostic Log Button
                report_data = {
                    "epsilon": float(epsilon),
                    "ground_truth_class": true_label_name,
                    "baseline_model": {
                        "prediction": base_pred_class,
                        "confidence_percent": round(float(base_conf), 2),
                        "is_vulnerable": bool(base_pred_idx != true_label_idx)
                    },
                    "robust_model": {
                        "prediction": robust_pred_class,
                        "confidence_percent": round(float(robust_conf), 2),
                        "is_vulnerable": bool(robust_pred_idx != true_label_idx)
                    },
                    "robustness_gain": round(float(robustness_gain), 2)
                }
                st.divider()
                st.download_button(
                    label="Export Diagnostic Log",
                    data=json.dumps(report_data, indent=2),
                    file_name="adversarial_audit_report.json",
                    mime="application/json"
                )

        # Free autograd references
        gc.collect()

    # Call Fragment
    render_attack_fragment(clean_tensor)

else:
    st.info("Upload an image or select a Preset Gallery Image from the sidebar to test real-time adversarial evasion.")