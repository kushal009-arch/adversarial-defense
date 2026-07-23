# Hardening Neural Networks Against Fast Gradient Sign Method (FGSM) Attacks

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Code Style: PEP8](https://img.shields.io/badge/code%20style-PEP8-brightgreen.svg)](https://peps.python.org/pep-0008/)

> **Executive Summary:** An end-to-end Machine Learning Security pipeline on CIFAR-10 demonstrating how standard Convolutional Neural Networks (CNNs) are systematically derailed by imperceptible input perturbations ($L_\infty$ bounded FGSM attacks), and how to restore robust classification through Min-Max Robust Optimization (Adversarial Training). Features a reactive, real-time diagnostic dashboard built with Streamlit.

---

## Project Highlights and Architecture

* **Modular Production Pipeline:** Clean separation of data loading, model architecture, attack generation, adversarial training, and web UI.
* **Mathematically Grounded Defense:** Implemented on-the-fly FGSM adversarial training to reshape decision boundaries.
* **Interactive Diagnostic Studio:** Real-time side-by-side inference engine using `@st.fragment` isolated state re-runs to evaluate model confidence under dynamic noise injection.

```
adversarial-defense/
├── data/                       # Downloaded CIFAR-10 dataset (git-ignored)
├── models/                     # Exported PyTorch weight checkpoints (.pth)
│   ├── baseline_model.pth      # Standard ERM trained model
│   └── robust_model.pth        # Adversarially trained model
├── reports/                    # Diagnostic figures and confusion matrices
├── src/                        # Production source code
│   ├── __init__.py
│   ├── app.py                  # Interactive Streamlit dashboard
│   ├── attack.py               # FGSM mathematical perturbation engine
│   ├── data_loader.py          # Data ingestion & normalization transforms
│   ├── model.py                # 2-Conv + 2-FC CNN Architecture
│   ├── train.py                # Baseline model training script
│   ├── train_robust.py         # Robust mixed-batch training engine
│   ├── adversarial_train.py    # Min-Max robust optimization loop
│   ├── evaluate.py             # Accuracy evaluation & confusion matrix generator
│   ├── evasion_testing.py      # Multi-epsilon robust accuracy sweep engine
│   ├── plot_evasion_curve.py   # Evasion curve plotting utility
│   ├── security_audit.py       # White-box vs black-box transfer audit engine
│   └── visualize_illusion.py   # Visual optical illusion panel generator
├── requirements.txt            # Frozen environment dependencies
└── README.md                   # Technical documentation
```

---

## Mathematical Foundations

### 1. Fast Gradient Sign Method (FGSM)
Standard Empirical Risk Minimization (ERM) optimizes network parameters $\theta$ to minimize loss:

$$\min_{\theta} \mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \mathcal{L}(f_\theta(x), y) \right]$$

The FGSM evasion attack exploits high-dimensional linearity by calculating input-space gradients while freezing model parameters $\theta$. The perturbed image $x_{\text{adv}}$ is generated within an $L_\infty$ perturbation budget $\epsilon$:

$$x_{\text{adv}} = \text{clamp}\left(x + \epsilon \cdot \text{sign}\left(\nabla_x \mathcal{L}(\theta, x, y)\right), -1.0, 1.0\right)$$

### 2. Min-Max Robust Optimization
To defend against input evasion, the network is retrained under a saddle-point formulation (outer minimization of model weights, inner maximization of input noise):

$$\min_{\theta} \mathbb{E}_{(x,y)\sim \mathcal{D}} \left[ \max_{\|\delta\|_\infty \le \epsilon} \mathcal{L}(f_\theta(x + \delta), y) \right]$$

---

## Empirical Benchmarks and Telemetry

Head-to-head evaluation on 10,000 unseen CIFAR-10 test samples across perturbation budgets ($\epsilon$):

| Perturbation Budget ($\epsilon$) | Baseline Model Accuracy | Hardened (Robust) Model Accuracy | Primary System Observation |
| :--- | :---: | :---: | :--- |
| **0.00 (Clean Data)** | **68.79%** | **52.40%** | Standard Accuracy / Robustness Trade-off |
| **0.01 (Subtle Noise)** | 31.15% | **50.12%** | Baseline confidence degrades rapidly |
| **0.03 (Standard Attack)**| 12.31% | **46.85%** | Baseline completely compromised; Robust holds |
| **0.10 (Severe Noise)** | 4.80% | **28.10%** | Extreme visual artifacts emerge |

### Optimization Paradox Insights
* **The Accuracy Trade-off:** Hardening the network forces feature representations to rely on robust, low-frequency semantics rather than fragile, high-frequency edge textures. This introduces a ~16% drop in clean test accuracy while gaining a massive **+34.54% boost in robustness** under active $\epsilon=0.03$ attacks.
* **Softmax Overconfidence Mitigation:** Un-defended networks suffer from extreme overconfidence on misclassified adversarial samples ($>90\%$ confidence on wrong classes). Adversarial training smooths the loss landscape, restoring well-calibrated class probability output distributions.

---

## Interactive Diagnostic Studio (app.py)

The repository includes a Streamlit UI allowing live inspection of model behavior:

* **Real-time Epsilon Control:** Dynamic slider updating perturbation budgets instantly.
* **Performance Optimization:** Leverages Streamlit’s `@st.fragment` decorator to isolate inference reruns without reloading PyTorch models or re-rendering static layout containers.
* **Interactive Split-View Slider:** Interactive split slider (`streamlit-image-comparison`) overlaying clean and perturbed images in real time.
* **3-Panel Attack Anatomy:** Displays `[ Clean Image (x) ]` $\rightarrow$ `[ Magnified Gradient Noise ]` $\rightarrow$ `[ Perturbed Image (x_adv) ]`.
* **Side-by-Side Model Comparison:** Dynamic metric boxes and Plotly probability distribution bar charts comparing standard vs. defended model outputs.
* **Audit Report Export:** Downloadable real-time diagnostic JSON log (`adversarial_audit_report.json`).

---

## Installation and Usage Guide

### 1. Prerequisites and Environment Setup
```bash
# Clone repository
git clone https://github.com/kushal009-arch/adversarial-defense.git
cd adversarial-defense

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Models

```bash
# Train baseline model (Standard ERM)
python src/train.py

# Execute Adversarial Training (Mixed-Batch FGSM hardening)
python src/train_robust.py
```

### 3. Launching Dashboard

```bash
streamlit run src/app.py
```

---

## Code Quality and Standards

All Python modules follow strictly enforced software standards:

* **PEP 8 Compliance:** Formatted with standardized line lengths and modular imports.
* **Static Type Hinting:** Explicit return types and function argument types (`typing.Tuple`, `torch.Tensor`, etc.).
* **Google-Style Docstrings:** Complete documentation across all module functions and classes.

---

## References and Citation

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). *Explaining and Harnessing Adversarial Examples*. arXiv preprint arXiv:1412.6572.
2. Madry, A., et al. (2017). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR 2018.
