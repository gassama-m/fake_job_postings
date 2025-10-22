import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


metrics = {
    "Logistic Regression": {"Accuracy": 98.56, "Precision": 89.39, "Recall": 79.91},
    "Random Forest": {"Accuracy": 98.69, "Precision": 90.87, "Recall": 81.30},
    "SVM": {"Accuracy": 98.31, "Precision": 97.10, "Recall": 67.22},
    "RF+LR Ensemble": {"Accuracy": 98.76, "Precision": 93.33, "Recall": 80.84}
}

def estimate_confusion_matrix(acc, prec, rec, pos_ratio=0.1):
    acc, prec, rec = acc / 100, prec / 100, rec / 100
    P = pos_ratio
    N = 1 - P

    TP = rec * P
    FP = TP * (1 / prec - 1)
    FN = P - TP
    TN = N - FP

    TN, FP, FN, TP = max(TN, 0), max(FP, 0), max(FN, 0), max(TP, 0)
    total = TN + FP + FN + TP
    cm = np.array([[TN, FP],
                   [FN, TP]]) / total * 100
    return cm


fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for i, (model, vals) in enumerate(metrics.items()):
    cm = estimate_confusion_matrix(vals['Accuracy'], vals['Precision'], vals['Recall'])
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=axes[i],
                annot_kws={"size": 10}, square=True)  # <-- square cells
    axes[i].set_title(f"{model} ({vals['Accuracy']:.2f}%)", fontsize=11)
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")
    axes[i].set_xticklabels(["Real", "Fake"])
    axes[i].set_yticklabels(["Real", "Fake"], rotation=0)

plt.suptitle("Estimated Confusion Matrices per Model (Values in %)", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])


plt.savefig("confusion_matricesfin.png", dpi=300, bbox_inches='tight')
plt.show()
print("Confusion matrices saved as 'confusion_matricesfin.png'")
