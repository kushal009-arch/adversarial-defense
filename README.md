# Adversarial Defense

This repository is dedicated to exploring and implementing techniques for adversarial defense in deep learning.

## Current Progress

### Model Architecture, Training & Evaluation
- **`src/model.py`**: A basic Convolutional Neural Network (`SimpleCNN`) has been implemented using PyTorch. It is designed for small-scale image classification tasks (such as 32x32 pixel images).
  - The model features two convolutional layers followed by two fully connected layers.
  - It is fully documented with standard docstrings and inline comments.
- **`src/train.py`**: A training loop script has been implemented to train the `SimpleCNN` model on the CIFAR-10 dataset.
  - Features an optimization loop using Adam and Cross-Entropy Loss.
  - Automatically runs on GPU (`cuda`) if available, otherwise defaults to CPU.
  - Fully commented and documented for ease of understanding.
- **`src/evaluate.py`**: An evaluation script that loads a trained model's weights and evaluates its accuracy on the CIFAR-10 test set.
  - Computes the overall classification accuracy.
  - Builds and prints a textual confusion matrix.
  - Generates and saves a visual heatmap visualization under `reports/figures/` (ignored by git).
  - Fully annotated with docstrings and comments.
- **`src/attack.py`**: An adversarial attack utility script that implements the **Fast Gradient Sign Method (FGSM)**.
  - Demonstrates how to enable gradients on input tensors (`data.requires_grad = True`).
  - Implements gradient extraction of the loss with respect to the input image(s), supporting both single-image analysis and batch processing.
  - Implements the core FGSM attack logic (`fgsm_attack`) to perturb the input image in the direction of the gradient sign, scaled by `epsilon`, to maximize classification loss and trigger misclassification.
  - Fully annotated with clean, professional docstrings and comments.
- **`src/evasion_testing.py`**: A testing script that runs the FGSM attack across the entire CIFAR-10 test dataset for multiple epsilon values.
  - Computes and outputs the evasion curve values (epsilon vs. robust accuracy) to measure the model's adversarial robustness.
  - Fully annotated with docstrings and inline comments.
- **`src/plot_evasion_curve.py`**: A plotting utility script that takes the robust accuracy metrics across different epsilon values and generates a professional line plot.
  - Saves the generated evasion curve figure to `results/figures/evasion_curve.png`.
  - Fully annotated with docstrings and comments.
- **`src/visualize_illusion.py`**: A visualization utility that generates a comparative side-by-side diagnostic panel of clean images, their extracted gradient noise maps, and the resulting adversarial images.
  - Saves the visualization grid to `reports/figures/optical_illusion_panel.png`.
- **`src/adversarial_train.py`**: An adversarial training script implementing min-max robust optimization.
  - Generates adversarial examples inline using the Fast Gradient Sign Method (FGSM).
  - Trains the model weights on these perturbed inputs using SGD with momentum.
  - Logs clean and robust loss values per epoch and saves the hardened model weights to `models/robust_model.pth`.
- **`src/train_robust.py`**: A robust mixed-batch training engine.
  - Utilizes a mixed-batch training strategy to optimize both clean accuracy and adversarial robustness.
  - Implements a loss blending technique where the total loss is a weighted sum (controlled by `ALPHA = 0.5`) of clean and FGSM-perturbed image loss.
  - Fully annotated with docstrings and clean code layout.

## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- PyTorch
- torchvision (required for loading CIFAR-10 data)
- pandas, matplotlib, seaborn (required for evaluating and visualizing results)

### Running the Project

#### 1. Test the Model Architecture
You can run the model script directly to perform a quick forward pass test with dummy inputs:
```bash
python src/model.py
```

#### 2. Train the Model (Standard)
You can execute the training pipeline directly to train the model on the CIFAR-10 dataset:
```bash
python src/train.py
```

#### 3. Train the Model (Adversarial - 100% Adversarial Batches)
You can execute the adversarial training pipeline to harden the model against FGSM attacks:
```bash
python src/adversarial_train.py
```

#### 4. Train the Model (Robust - Mixed Clean/Adversarial Batches)
You can execute the mixed robust training pipeline using a combination of clean and adversarial data:
```bash
python src/train_robust.py
```

#### 5. Evaluate the Model
You can evaluate the trained model and view the accuracy and confusion matrix:
```bash
python src/evaluate.py
```

#### 6. Run Adversarial Attack (FGSM)
You can run the attack script to extract the gradient map and generate a perturbed adversarial image using the FGSM attack:
```bash
python src/attack.py
```

#### 7. Evaluate Evasion Curve
You can run the evasion testing script to trace the model's robust accuracy across a range of epsilon perturbation budgets:
```bash
python src/evasion_testing.py
```

#### 8. Plot Evasion Curve
You can run the plotting script to generate and save a visualization of the evasion curve:
```bash
python src/plot_evasion_curve.py
```

#### 9. Visualize Adversarial Illusion Panel
You can run the visualization script to generate a side-by-side comparison of clean and adversarial images:
```bash
python src/visualize_illusion.py
```

### Adversarial Optical Illusion Diagnostics (`reports/figures/optical_illusion_panel.png`)

This diagnostic panel displays:
1. **Clean Image**: The original test image (which the model correctly classifies).
2. **Gradient Noise Map**: The direction of the loss gradient ($\text{sign}(\nabla_x L)$) visualized in grayscale.
3. **Adversarial Image**: The clean image perturbed by adding the noise map (scaled by $\epsilon=0.03$).

**Why it looks the way it does:**
- **To humans**: The clean and adversarial images look virtually identical because the added noise is extremely small.
- **To the model**: The noise is mathematically optimized to push the image's features across the model's classification decision boundaries. This triggers a misclassification (highlighted in red), creating an "optical illusion" for the CNN.

---

### Log Analysis: Why This Happened

```text
[*] Launching min-max optimization training loop across adversarial batches...
Epoch [01/05] -> Clean Loss: 2.0630 | Robust Loss: 2.0022
Epoch [02/05] -> Clean Loss: 1.9162 | Robust Loss: 1.7573
Epoch [03/05] -> Clean Loss: 2.7657 | Robust Loss: 1.6260
Epoch [04/05] -> Clean Loss: 3.9940 | Robust Loss: 1.5250
Epoch [05/05] -> Clean Loss: 6.1424 | Robust Loss: 1.4278
```

* **Robust Loss Decreased ($2.00 \to 1.43$):** The model learns to defend itself, successfully minimizing errors on worst-case FGSM inputs.
* **Clean Loss Skyrocketed ($2.06 \to 6.14$):** Over-defense degrades baseline classification accuracy.
* **Boundary Warping:** Training on 100% adversarial batches distorts decision boundaries around noise. Without clean anchors, normal boundaries shift out of alignment.
* **Feature Sacrifice:** The network ignores fragile, high-frequency details to survive attacks, breaking its ability to recognize un-attacked images.
