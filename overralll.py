import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# ✅ Model Results
# -----------------------------
models = ['Centralized RF', 'FedAvg', 'FedProx']

# Metric values
accuracy = [98.76, 95.16, 95.16]            # Accuracy in %
convergence_rounds = [0, 4, 4]             # Convergence rounds
communication_cost = [0.0, 1.02, 0.26]  # Communication cost (MB)

# -----------------------------
# 📈 Prepare grouped data
# -----------------------------
metrics = ['Accuracy\n(%)', 'Convergence\nRounds', 'Communication\nCost (MB)']
values = [accuracy, convergence_rounds, communication_cost]

x = np.arange(len(metrics))  # positions for each metric
width = 0.25                 # width of each bar
fig, ax = plt.subplots(figsize=(10, 6))

# -----------------------------
# 🎨 Plot bars (grouped by metric)
# -----------------------------
colors = ['#4C72B0', '#55A868', '#C44E52']

for i, model in enumerate(models):
    bars = ax.bar(x + (i - 1) * width, [v[i] for v in values], width, 
                  label=model, color=colors[i])

    # Add model names horizontally on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, model,
                ha='center', va='bottom', fontsize=9, rotation=0)

# -----------------------------
# 🪶 Customize chart
# -----------------------------
ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Value", fontsize=12)
#ax.set_title("Overall Performance Comparison of All Approaches", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)

# Remove horizontal grid lines for a cleaner look
ax.grid(False)

# Legend in the top right
ax.legend(title="Models", loc='upper right', fontsize=10)

plt.tight_layout()
plt.show()
