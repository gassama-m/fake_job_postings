import time
import joblib
import numpy as np
import pandas as pd
from google.colab import drive
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import RandomOverSampler

drive.mount('/content/drive')

def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    thresholds = thresholds[:-1]
    f1_scores = f1_scores[:-1]
    best_idx = f1_scores.argmax()
    return thresholds[best_idx]

Cs = [0.001, 0.01, 0.1, 1, 10]  
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y = df['fraudulent'].copy()
train_metrics = []
test_metrics = []

start_time_total = time.time()

for fold, (train_idx, test_idx) in enumerate(skf.split(X_final, y), 1):
    print(f"\n Fold {fold} ")

    
    X_train_fold = X_final[train_idx, :]
    X_test_fold = X_final[test_idx, :]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]

    
    X_train_dense = X_train_fold.toarray()
    ros = RandomOverSampler(sampling_strategy=0.5, random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_dense, y_train_fold)
    print(f" After ROS balancing: {np.bincount(y_train_res)}")

    best_val_f1 = -1
    best_C = None
    best_model = None

    for C_val in Cs:
        lr = LogisticRegression(
            solver='liblinear',
            class_weight={0:1, 1:5},
            random_state=42,
            max_iter=1000,
            C=C_val,
        )

        start_time = time.time()
        lr.fit(X_train_res, y_train_res)
        training_time = time.time() - start_time

        val_probs = lr.predict_proba(X_test_fold.toarray())[:, 1]
        best_thresh = find_best_threshold(y_test_fold, val_probs)
        y_val_pred = (val_probs >= best_thresh).astype(int)
        val_f1 = f1_score(y_test_fold, y_val_pred)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_C = C_val
            best_model = lr
            best_thresh_final = best_thresh
            best_training_time = training_time

    train_probs = best_model.predict_proba(X_train_res)[:, 1]
    y_train_pred = (train_probs >= best_thresh_final).astype(int)

    val_probs = best_model.predict_proba(X_test_fold.toarray())[:, 1]
    y_test_pred = (val_probs >= best_thresh_final).astype(int)

    train_result = {
        "accuracy": accuracy_score(y_train_res, y_train_pred) * 100,
        "precision": precision_score(y_train_res, y_train_pred) * 100,
        "recall": recall_score(y_train_res, y_train_pred) * 100,
        "f1": f1_score(y_train_res, y_train_pred) * 100,
        "auc": roc_auc_score(y_train_res, train_probs) * 100,
        "threshold": best_thresh_final,
        "C": best_C,
        "train_time_s": best_training_time
    }

    test_result = {
        "accuracy": accuracy_score(y_test_fold, y_test_pred) * 100,
        "precision": precision_score(y_test_fold, y_test_pred) * 100,
        "recall": recall_score(y_test_fold, y_test_pred) * 100,
        "f1": f1_score(y_test_fold, y_test_pred) * 100,
        "auc": roc_auc_score(y_test_fold, val_probs) * 100,
        "threshold": best_thresh_final,
        "C": best_C,
        "train_time_s": best_training_time
    }

    train_metrics.append(train_result)
    test_metrics.append(test_result)

    joblib.dump(best_model, f"/content/drive/MyDrive/fake_job_postings/logreg_fold{fold}_C{best_C}_ros.joblib")

    print(f" Best C for Fold {fold}: {best_C}")
    print(f" Fold {fold} | Train time: {best_training_time:.2f}s")
    print(f"Train - Acc: {train_result['accuracy']:.2f}% | Prec: {train_result['precision']:.2f}% | Rec: {train_result['recall']:.2f}% | F1: {train_result['f1']:.2f}% | AUC: {train_result['auc']:.2f}%")
    print(f"Val   - Acc: {test_result['accuracy']:.2f}% | Prec: {test_result['precision']:.2f}% | Rec: {test_result['recall']:.2f}% | F1: {test_result['f1']:.2f}% | AUC: {test_result['auc']:.2f}%")

total_time_total = time.time() - start_time_total
print(f"\n Total Cross-Validation Time (s): {total_time_total:.2f}")

print("\n AVERAGE RESULTS ")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    print(f"Train {metric.capitalize()}: {np.mean([m[metric] for m in train_metrics]):.2f}%")
    print(f"Val   {metric.capitalize()}: {np.mean([m[metric] for m in test_metrics]):.2f}%")

