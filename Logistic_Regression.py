import os
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score


path='/content/drive/MyDrive/fake_job_postings/preprocessed_data.xlsx'
df0=pd.read_excel(path)
df=df0.iloc[:, [i for i in range(11, 22) if i != 17]]
feature_columns=df.drop('fraudulent', axis=1).columns

X = df[feature_columns].astype(str)
y=df['fraudulent']


preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), X.columns)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', LogisticRegression(
        class_weight='balanced', 
        max_iter=2000,
        solver='liblinear', 
        C=0.7,            
        penalty='l1'
    ))
])


# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1')


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    start = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start

    y_probs = pipeline.predict_proba(X_test)[:, 1]

    best_threshold=0.5
    y_pred = (y_probs >= best_threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)*100
    precision = precision_score(y_test, y_pred, zero_division=0)*100
    recall = recall_score(y_test, y_pred, zero_division=0)*100
    f1 = f1_score(y_test, y_pred)*100
    auc = roc_auc_score(y_test, y_probs)*100

    print(f" Fold No.{fold}")
    print(f"Training Time : {training_time:.2f} seconds")
    print(f"Threshold: {best_thresh:.3f}")
    print(f"Accuracy (%): {accuracy*100:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}\n")

    results.append({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'threshold': best_thresh,
        'time': training_time
    })

# Average results
mean_results = pd.DataFrame(results).mean()
print("Average metrics")
print(mean_results.round(4))
