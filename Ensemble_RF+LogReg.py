# @title Final Logistic Regression + Random Forest Ensemble Grid Search (Colab Compatible)
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTENC
from scipy.sparse import csr_matrix
import joblib
from google.colab import drive
import time


drive.mount('/content/drive')


def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    thresholds = thresholds[:-1]
    f1_scores = f1_scores[:-1]
    return thresholds[np.argmax(f1_scores)]


lr_C_values = [0.01, 0.1, 1]
rf_n_estimators = [50, 100, 200]


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

train_metrics = []
val_metrics = []
fold_thresholds = {}
best_params_per_fold = {}
fold_training_times = []

start_time_total = time.time()

y = df['fraudulent']
num_tfidf_cols = X_final.shape[1] - X_struct.shape[1]
cat_indices_in_struct = list(range(preprocessor.transformers_[0][1].categories_[0].shape[0]))
cat_feature_indices = [i + num_tfidf_cols for i in range(X_struct.shape[1]) if i in cat_indices_in_struct]

for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y), 1):
    print(f"\n Fold {fold}")
    fold_start_time = time.time()

    X_train, X_val = X_final[train_idx], X_final[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]


    X_train_dense = X_train.toarray()
    smote_nc = SMOTENC(categorical_features=cat_feature_indices, sampling_strategy=0.8, random_state=42)
    X_train_res, y_train_res = smote_nc.fit_resample(X_train_dense, y_train)
    print(f" After SMOTENC balancing: {np.bincount(y_train_res)}")

    
    best_f1 = -1
    best_model = None
    best_lr_C = None
    best_rf_n = None
    best_thresh = None

    for lr_C in lr_C_values:
        for rf_n in rf_n_estimators:
            lr = LogisticRegression(class_weight='balanced', max_iter=2000, solver='liblinear', C=lr_C)
            rf = RandomForestClassifier(n_estimators=rf_n, class_weight='balanced', random_state=42)
            ensemble = VotingClassifier(estimators=[('lr', lr), ('rf', rf)], voting='soft')

            start_time = time.time()
            ensemble.fit(X_train_res, y_train_res)
            training_time = time.time() - start_time

            train_probs = ensemble.predict_proba(X_train_res)[:, 1]
            thresh = find_best_threshold(y_train_res, train_probs)
            val_probs = ensemble.predict_proba(X_val.toarray())[:, 1]
            y_val_pred = (val_probs >= thresh).astype(int)
            val_f1 = f1_score(y_val, y_val_pred)

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model = ensemble
                best_lr_C = lr_C
                best_rf_n = rf_n
                best_thresh = thresh
                best_training_time = training_time


    train_probs = best_model.predict_proba(X_train_res)[:, 1]
    val_probs = best_model.predict_proba(X_val.toarray())[:, 1]

    y_train_pred = (train_probs >= best_thresh).astype(int)
    y_val_pred = (val_probs >= best_thresh).astype(int)

    train_result = {
        "accuracy": accuracy_score(y_train_res, y_train_pred) * 100,
        "precision": precision_score(y_train_res, y_train_pred, zero_division=0) * 100,
        "recall": recall_score(y_train_res, y_train_pred, zero_division=0) * 100,
        "f1": f1_score(y_train_res, y_train_pred, zero_division=0) * 100,
        "auc": roc_auc_score(y_train_res, train_probs) * 100,
        "threshold": best_thresh,
        "train_time_s": best_training_time
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
    fold_thresholds[fold] = best_thresh
    best_params_per_fold[fold] = {'lr_C': best_lr_C, 'rf_n_estimators': best_rf_n}

    joblib.dump(best_model, f"/content/drive/MyDrive/fake_job_postings/ensemble_fold{fold}_grid.joblib")

    fold_time = time.time() - fold_start_time
    fold_training_times.append(fold_time)
    print(f"Fold {fold} Training Time: {fold_time:.2f} seconds")

    print(f" Fold {fold} | Best LR C: {best_lr_C}, RF n_estimators: {best_rf_n} | F1: {best_f1:.2f} | Threshold: {best_thresh:.3f}")
    print("Validation Metrics:", {k: f"{v:.2f}" for k, v in val_result.items() if k != 'threshold'})


def print_avg_metrics(train_metrics, val_metrics):
    print("\nAverage Training Metrics Across Folds ")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{metric.capitalize()}: {np.mean([m[metric] for m in train_metrics]):.2f}%")

    print("\n Average Validation Metrics Across Folds")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"{metric.capitalize()}: {np.mean([m[metric] for m in val_metrics]):.2f}%")

print_avg_metrics(train_metrics, val_metrics)


joblib.dump(fold_thresholds, "/content/drive/MyDrive/fake_job_postings/ensemble_fold_thresholds_grid.joblib")
joblib.dump(best_params_per_fold, "/content/drive/MyDrive/fake_job_postings/ensemble_fold_best_params_grid.joblib")


total_time_total = time.time() - start_time_total
print("\nTraining Time Summary")
for i, t in enumerate(fold_training_times, 1):
    print(f"Fold {i}: {t:.2f} seconds")
print(f"\n Total Cross-Validation Time: {total_time_total:.2f} seconds")
print(f"Average Training Time per Fold: {np.mean(fold_training_times):.2f} seconds")
