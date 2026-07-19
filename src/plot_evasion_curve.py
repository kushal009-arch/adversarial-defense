import os
import matplotlib.pyplot as plt

def generate_evasion_curve(sweep_data, save_path="reports/figures/evasion_curve.png"):
    """
    Plots model accuracy drops across different adversarial epsilon values.
    Saves the finalized line plot into the production directory.
    """
    # Extract keys and values from the metrics dictionary
    epsilons = list(sweep_data.keys())
    accuracies = list(sweep_data.values())
    
    # Initialize professional plot canvas
    plt.style.use('seaborn-v0_8-whitegrid')  # Clean engineering layout
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    
    # Plot the Evasion Curve with markers and line styling
    ax.plot(epsilons, accuracies, marker='o', color='#d9534f', linestyle='-', linewidth=2.5, markersize=7, label="SimpleCNN (Baseline)")
    
    # Graph annotations and titles
    ax.set_title("Evasion Curve: Model Performance Under FGSM Attack", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(r"Attack Strength / Perturbation Budget ($\epsilon$)", fontsize=11, labelpad=10)
    ax.set_ylabel("Robust Accuracy (%)", fontsize=11, labelpad=10)
    
    # Configure precise axes scaling and limit restraints
    ax.set_ylim(-5, 105)
    ax.set_xticks(epsilons)
    
    # Add numerical value callouts directly on top of data points for clear technical reporting
    for eps, acc in sweep_data.items():
        ax.annotate(f"{acc:.1f}%", 
                    xy=(eps, acc), 
                    xytext=(5, 5), 
                    textcoords='offset points', 
                    fontsize=9, 
                    fontweight='semibold')

    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    # Ensure directory existence before flushing file write
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Evasion curve figure successfully written to: {save_path}")

def plot_audit_curves(epsilons, baseline_wb, robust_wb, robust_bb, save_path="reports/figures/audit_evasion_curve.png"):
    """
    Plots a multi-line comparison of baseline and robust models under white-box and black-box attacks.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    # Plot each series
    ax.plot(epsilons, baseline_wb, marker='o', color='#d9534f', linestyle='-', linewidth=2.5, label="Baseline (White-Box)")
    ax.plot(epsilons, robust_wb, marker='s', color='#2ca02c', linestyle='-', linewidth=2.5, label="Robust (White-Box)")
    ax.plot(epsilons, robust_bb, marker='^', color='#1f77b4', linestyle='--', linewidth=2.5, label="Robust (Black-Box/Transfer)")

    # Title and labels
    ax.set_title("Security Audit: Model Robustness Sweep", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(r"Attack Strength ($\epsilon$)", fontsize=11, labelpad=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11, labelpad=10)

    # Styling limits and ticks
    ax.set_ylim(-5, 105)
    ax.set_xticks(epsilons)
    
    # Legend
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()

    # Save output
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Multi-curve audit plot successfully saved to: {save_path}")

if __name__ == "__main__":
    # live sweep data collected from the Session 7 test evaluation
    session_metrics = {
        0.0: 68.79,
        0.005: 30.62,
        0.01: 25.57,
        0.03: 12.31,
        0.05: 6.21,
        0.1: 1.76
    }
    
    generate_evasion_curve(session_metrics)