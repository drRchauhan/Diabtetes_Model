import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

print("====================")
print("Diabetes Prediction Pipeline Starting")
print("====================")

# Step 1: Load the dataset
print("Loading data...")
data = fetch_ucirepo(id=891)
X = data.data.features
y = np.ravel(data.data.targets.values)
print("Data loaded with shape:", X.shape)

# Step 2: Split data and apply SMOTE
print("Splitting data and applying SMOTE...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 3: Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

# Step 4: Train XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42, eval_metric='logloss')
xgb_model.fit(X_train_scaled, y_train_res)
y_pred_xgb = xgb_model.predict(X_test_scaled)
y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
joblib.dump(xgb_model, os.path.join(output_dir, 'xgboost_model.pkl'))

# Step 5: Train AdaBoost
print("Training AdaBoost...")
ada_model = AdaBoostClassifier(random_state=42)
ada_model.fit(X_train_scaled, y_train_res)
y_pred_ada = ada_model.predict(X_test_scaled)
y_prob_ada = ada_model.predict_proba(X_test_scaled)[:, 1]
joblib.dump(ada_model, os.path.join(output_dir, 'adaboost_model.pkl'))

# Step 6: Evaluation and plotting
def evaluate_model(name, y_true, y_pred, y_prob, model, features):
    print(f"\n=== {name} Evaluation ===")
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_prob),
        'MSE': mean_squared_error(y_true, y_prob),
        'MAE': mean_absolute_error(y_true, y_prob),
        'Correlation': np.corrcoef(y_true, y_prob)[0, 1]
    }
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Save metrics plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(f'{name} - Performance Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name.lower()}_metrics_plot.png'))
    plt.close()

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features.columns)
        plt.title(f'{name} - Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name.lower()}_feature_importance.png'))
        plt.close()

    # Confusion Matrix heatmap
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name.lower()}_confusion_matrix.png'))
    plt.close()

# Evaluate both models
evaluate_model("XGBoost", y_test, y_pred_xgb, y_prob_xgb, xgb_model, X)
evaluate_model("AdaBoost", y_test, y_pred_ada, y_prob_ada, ada_model, X)

# Step 7: Correlation Matrix Plot
plt.figure(figsize=(12, 10))
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', square=True)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_correlation_matrix.png'))
plt.close()

# Step 8: Prediction example
print("\nRunning prediction example...")
data_point = {
    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "BMI": 30, "Smoker": 0,
    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
    "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1, "NoDocbcCost": 0,
    "GenHlth": 2, "MentHlth": 0, "PhysHlth": 0, "DiffWalk": 0, "Sex": 1,
    "Age": 8, "Education": 5, "Income": 6
}
df_input = pd.DataFrame([data_point])
features_input = scaler.transform(df_input)
prediction = xgb_model.predict(features_input)[0]
probability = xgb_model.predict_proba(features_input)[0, 1]

print(f"\nSample Prediction: {prediction} ({'Positive' if prediction == 1 else 'Negative'})")
print(f"Probability: {probability:.4f}")
print(f"Risk Level: {('High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low')}")

print("\nPipeline complete. All models and outputs saved to /output")
