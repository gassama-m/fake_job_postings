import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, precision_recall_curve
)
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTENC
import joblib
from google.colab import drive

drive.mount('/content/drive')

def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    thresholds = thresholds[:-1]
    f1_scores = f1_scores[:-1]
    return thresholds[np.argmax(f1_scores)]

svm_model = SVC(max_iter= 50000, class_weight='balanced', kernel='rbf', probability=True, random_state=42)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

train_metrics = []
val_metrics = []
fold_thresholds = {}
y=df['fraudulent']

num_tfidf_cols = X_final.shape[1] - X_struct.shape[1]  
cat_indices_in_struct = list(range(preprocessor.transformers_[0][1].categories_[0].shape[0]))  
cat_feature_indices = [i + num_tfidf_cols for i in range(X_struct.shape[1]) if i in cat_indices_in_struct]

start_time_total = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y), 1):
    print(f"\n Fold {fold}")

    X_train, X_val = X_final[train_idx], X_final[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    X_train_dense = X_train.toarray()
    smote_nc = SMOTENC(categorical_features=cat_feature_indices, sampling_strategy=0.3, random_state=42)
    X_train_res, y_train_res = smote_nc.fit_resample(X_train_dense, y_train)
    print(f"After SMOTENC balancing: {np.bincount(y_train_res)}")

    start_time = time.time()
    svm_model.fit(X_train_res, y_train_res)
    train_time = time.time() - start_time

    train_probs = svm_model.predict_proba(X_train.toarray())[:, 1]
    val_probs = svm_model.predict_proba(X_val.toarray())[:, 1]

    best_thresh = find_best_threshold(y_train, train_probs)
    fold_thresholds[fold] = best_thresh

  
    y_train_pred = (train_probs >= best_thresh).astype(int)
    y_val_pred = (val_probs >= best_thresh).astype(int)

   
    train_result = {
        "accuracy": accuracy_score(y_train, y_train_pred) * 100,
        "precision": precision_score(y_train, y_train_pred, zero_division=0) * 100,
        "recall": recall_score(y_train, y_train_pred, zero_division=0) * 100,
        "f1": f1_score(y_train, y_train_pred, zero_division=0) * 100,
        "auc": roc_auc_score(y_train, train_probs) * 100,
        "train_time_s": train_time,
        "threshold": best_thresh
    }

   
    val_result = {
        "accuracy": accuracy_score(y_val, y_val_pred) * 100,
        "precision": precision_score(y_val, y_val_pred, zero_division=0) * 100,
        "recall": recall_score(y_val, y_val_pred, zero_division=0) * 100,
        "f1": f1_score(y_val, y_val_pred, zero_division=0) * 100,
        "auc": roc_auc_score(y_val, val_probs) * 100,
        "threshold": best_thresh
    }

    train_metrics.append(train_result)
    val_metrics.append(val_result)

    
    joblib.dump(svm_model, f"/content/drive/MyDrive/fake_job_postings/svm_fold{fold}_smotenc.joblib")

    print(f" Fold {fold} | Train Time: {train_time:.2f}s | Threshold: {best_thresh:.3f}")
    print("Training Metrics:", {k: f"{v:.2f}" for k, v in train_result.items() if k not in ['train_time_s', 'threshold']})
    print("Validation Metrics:", {k: f"{v:.2f}" for k, v in val_result.items() if k != 'threshold'})


def print_avg_metrics(train_metrics, val_metrics):
    print("\n Average Training Metrics Across Folds ")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{metric.capitalize()}: {np.mean([m[metric] for m in train_metrics]):.2f}%")
    time = np.sum([m['train_time_s'] for m in train_metrics])
    print(f" Train Time (s): {time:.2f}")

    print("\n Average Validation Metrics Across Folds ")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{metric.capitalize()}: {np.mean([m[metric] for m in val_metrics]):.2f}%")

print_avg_metrics(train_metrics, val_metrics)


joblib.dump(fold_thresholds, "/content/drive/MyDrive/fake_job_postings/svm_fold_thresholds_smotenc.joblib")

total_time_total = time.time() - start_time_total
print(f"\nTotal Cross-Validation Time: {total_time_total:.2f}s")

