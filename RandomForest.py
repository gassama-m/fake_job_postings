import time
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTENC
from google.colab import drive


drive.mount('/content/drive')

def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    thresholds = thresholds[:-1]
    f1_scores = f1_scores[:-1]
    best_idx = f1_scores.argmax()
    return thresholds[best_idx]


rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=35,
    max_features='sqrt',
    min_samples_split=10,
    min_samples_leaf=3,
    class_weight={0:1, 1:3},
    criterion='entropy',
    random_state=42,
    n_jobs=-1

)


n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

y=df['fraudulent'].copy()
train_metrics = []
test_metrics = []

start_time_total = time.time()

cat_features = [i for i, col in enumerate(preprocessor.transformers_[0][1].get_feature_names_out())
                if 'Other' in col or '_yes' in col or '_no' in col]

for fold, (train_idx, test_idx) in enumerate(skf.split(X_final, y), 1):
    X_train_fold = X_final[train_idx, :]
    X_test_fold = X_final[test_idx, :]
    y_train_fold = y.iloc[train_idx]
    y_test_fold = y.iloc[test_idx]

    X_train_dense = X_train_fold.toarray()
    smote_nc = SMOTENC(categorical_features=cat_features, sampling_strategy=0.1, random_state=42)
    X_train_res, y_train_res = smote_nc.fit_resample(X_train_dense, y_train_fold)

   
    start_time = time.time()
    rf.fit(X_train_res, y_train_res)
    training_time = time.time() - start_time

   
    train_probs = rf.predict_proba(X_train_res)[:, 1]
   
    val_probs = rf.predict_proba(X_test_fold.toarray())[:, 1]
    best_thresh = find_best_threshold(y_test_fold, val_probs)

    y_train_pred = (train_probs >= best_thresh).astype(int)
    y_test_pred = (rf.predict_proba(X_test_fold.toarray())[:, 1] >= best_thresh).astype(int)

    
    train_result = {
        "accuracy": accuracy_score(y_train_res, y_train_pred) * 100,
        "precision": precision_score(y_train_res, y_train_pred) * 100,
        "recall": recall_score(y_train_res, y_train_pred) * 100,
        "f1": f1_score(y_train_res, y_train_pred) * 100,
        "auc": roc_auc_score(y_train_res, train_probs) * 100,
        "threshold": best_thresh,
        "train_time_s": training_time
    }

    test_probs = rf.predict_proba(X_test_fold.toarray())[:, 1]
    test_result = {
        "accuracy": accuracy_score(y_test_fold, y_test_pred) * 100,
        "precision": precision_score(y_test_fold, y_test_pred) * 100,
        "recall": recall_score(y_test_fold, y_test_pred) * 100,
        "f1": f1_score(y_test_fold, y_test_pred) * 100,
        "auc": roc_auc_score(y_test_fold, test_probs) * 100,
        "threshold": best_thresh,
        "train_time_s": training_time
    }

    train_metrics.append(train_result)
    test_metrics.append(test_result)

   
    joblib.dump(rf, f"/content/drive/MyDrive/fake_job_postings/rf_fold{fold}_smotenc.joblib")

   
    print(f"\nFold {fold}:    Training Metrics ")
    for k, v in train_result.items():
        if k != "threshold":
            print(f"{k.capitalize()}: {v:.2f}")
    print(f"Best Threshold: {best_thresh:.3f}")

    print(f"\nFold {fold}     Validation Metrics ")
    for k, v in test_result.items():
        if k != "threshold":
            print(f"{k.capitalize()}: {v:.2f}")
    print(f"Best Threshold: {best_thresh:.3f}")


total_time_total = time.time() - start_time_total
print(f"\nTotal Cross-Validation Time (s): {total_time_total:.2f}")


print("\nAverage Training Metrics ")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time_s']:
    mean_score = np.mean([m[metric] for m in train_metrics])
    print(f"{metric.capitalize()}: {mean_score:.2f}")

print("\n Average Validation Metrics ")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'train_time_s']:
    mean_score = np.mean([m[metric] for m in test_metrics])
    print(f"{metric.capitalize()}: {mean_score:.2f}")
