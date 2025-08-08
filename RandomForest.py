import numpy as np
import pandas as pd
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from google.colab import drive
drive.mount('/content/drive')
path='/content/drive/MyDrive/fake_job_postings/preprocessed_data.xlsx'

df=pd.read_excel(path)
df=df.iloc[:, [i for i in range(11, 22) if i != 17]]

feature_columns=df.drop('fraudulent', axis=1).columns
X=df[feature_columns]
y=df['fraudulent']

# Parameters
n_splits = 5
threshold = 0.364

# Model definition
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    max_features='sqrt',
    min_samples_split=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=5,
    criterion='entropy'
)

# Cross-validation setup
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_metrics = []
test_metrics = []
avg_trainingTime=0
start_time = time.time()
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

    # Encode categorical variables
    X_train_fold_enc = pd.get_dummies(X_train_fold)
    X_test_fold_enc = pd.get_dummies(X_test_fold)

    # Align test to train
    X_train_fold_enc, X_test_fold_enc = X_train_fold_enc.align(X_test_fold_enc, join='left', axis=1, fill_value=0)

    # Training
    fold_start = time.time()
    rf.fit(X_train_fold_enc, y_train_fold)
    fold_train_time = time.time() - fold_start

    # Evaluation of the training data 
    train_probs = rf.predict_proba(X_train_fold_enc)[:, 1]
    y_train_pred = (train_probs >= threshold).astype(int)

    train_result = {
        "accuracy": accuracy_score(y_train_fold, y_train_pred) * 100,
        "precision": precision_score(y_train_fold, y_train_pred, zero_division=0) * 100,
        "recall": recall_score(y_train_fold, y_train_pred, zero_division=0) * 100,
        "f1": f1_score(y_train_fold, y_train_pred, zero_division=0) * 100,
        "auc": roc_auc_score(y_train_fold, train_probs) * 100
    }

    # Testing
    test_probs = rf.predict_proba(X_test_fold_enc)[:, 1]
    y_test_pred = (test_probs >= threshold).astype(int)

    test_result = {
        "accuracy": accuracy_score(y_test_fold, y_test_pred) * 100,
        "precision": precision_score(y_test_fold, y_test_pred, zero_division=0) * 100,
        "recall": recall_score(y_test_fold, y_test_pred, zero_division=0) * 100,
        "f1": f1_score(y_test_fold, y_test_pred, zero_division=0) * 100,
        "auc": roc_auc_score(y_test_fold, test_probs) * 100
    }

    train_metrics.append(train_result)
    test_metrics.append(test_result)

    # Print per fold
    print(f"\n Fold {fold} Training :")
    print(f"Training Time : {fold_train_time:.2f} s")
    print(f"Accuracy: {train_result['accuracy']:.2f}")
    print(f"Precision:    {train_result['precision']:.2f}")
    print(f"Recall:       {train_result['recall']:.2f}")
    print(f"F1-Score:     {train_result['f1']:.2f}")
    print(f"AUC:          {train_result['auc']:.2f}")

    print(f"\n Fold {fold} Validation :")
    print(f"Accuracy: {test_result['accuracy']:.2f}")
    print(f"Precision:    {test_result['precision']:.2f}")
    print(f"Recall:       {test_result['recall']:.2f}")
    print(f"F1-Score:     {test_result['f1']:.2f}")
    print(f"AUC:          {test_result['auc']:.2f}")
    avg_trainingTime+=fold_train_time
    # Model saving
    if fold == n_splits:
        joblib.dump(rf, "random_forest_model.pkl")
        print(f"\nModel from Fold {fold} saved as 'random_forest_model.pkl'.")

total_time = time.time() - start_time
print(f"\nTotal Cross-Validation Time : {total_time:.2f} seconds")
avg_trainingTime /=n_splits
# Average results
print("\n Average Training:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    mean_score = np.mean([m[metric] for m in train_metrics])
    print(f"{metric.capitalize()}: {mean_score:.2f}")

print("\n Average Validation:")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    mean_score = np.mean([m[metric] for m in test_metrics])
    print(f"{metric.capitalize()}: {mean_score:.2f}")
#Average training time
print(f"\n Average training Time: {avg_trainingTime:.2f} seconds" )
