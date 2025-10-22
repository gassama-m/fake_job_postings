# ============================================================
# 📊 FedAvg vs FedProx Comparison Visualizations
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1️⃣ Accuracy & Convergence Comparison
# ============================================================

# --- Results ---
fedavg_accuracy = 95.16
fedavg_rounds = 4

fedprox_accuracy = 95.16
fedprox_rounds = 4

# --- Metrics grouped ---
metrics = ['Accuracy (%)', 'Convergence Round']
fedavg_values = [fedavg_accuracy, fedavg_rounds]
fedprox_values = [fedprox_accuracy, fedprox_rounds]

x = np.arange(len(metrics))
width = 0.35  # bar width

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(8, 5))

# Bars for FedAvg and FedProx side-by-side per metric
bars1 = ax.bar(x - width/2, fedavg_values, width, label='FedAvg', color='skyblue')
bars2 = ax.bar(x + width/2, fedprox_values, width, label='FedProx', color='orange')

# Labels and styling
ax.set_ylabel('Value')
ax.set_title('FedAvg vs FedProx — Accuracy and Convergence', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, max(fedavg_accuracy, fedprox_accuracy) + 10)

# Add value labels
for bar in bars1 + bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.5, f'{height:.2f}',
            ha='center', va='bottom', fontsize=9)

# Legend
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# Save figure
plt.savefig("fedavg_fedprox_accuracy_convergence.png", dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 2️⃣ Communication Cost Comparison
# ============================================================

# --- Data ---
methods = ['FedAvg', 'FedProx']
communication_cost = [1.02, 0.26]  # in MB

x = np.arange(len(methods))
width = 0.35

# --- Plot setup ---
fig, ax = plt.subplots(figsize=(7, 7))  # Taller plot

# Bars
bars = ax.bar(x, communication_cost, width, label='Communication Cost (MB)', color='green')

# Labels & title
ax.set_ylabel('Value (MB)')
ax.set_title('FedAvg and FedProx — Communication Costs', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)

# Annotate values
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2f}',
            ha='center', va='bottom', fontsize=9)

# Legend outside top-right
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))

# Adjust Y limit for better visibility
plt.ylim(0, max(communication_cost) * 1.2)

# Save figure
plt.savefig("fedavg_fedprox_communication_cost.png", dpi=300, bbox_inches='tight')
plt.show()
