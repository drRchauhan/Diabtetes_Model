# diabetes_model_training.py
"""
Rewant and Genevieve
 diabetes prediction model training with XGBoost and AdaBoost
Features:
- Direct dataset download
- 80/20 train/test split
- SMOTE for class balancing
- 5-fold cross-validation
- metrics calculation
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, f1_score, roc_auc_score,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import urllib.request
import zipfile
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Create output directory
os.makedirs('./outputs', exist_ok=True)

def download_data():
    """Load the CDC Diabetes Health Indicators dataset using ucimlrepo."""
    try:
        print("Loading CDC Diabetes Health Indicators dataset using ucimlrepo...")
        from ucimlrepo import fetch_ucirepo
        
        # Fetch dataset with ID 891 (CDC Diabetes Health Indicators)
        diabetes_dataset = fetch_ucirepo(id=891)
        
        # Extract features and targets
        features = diabetes_dataset.data.features
        target = diabetes_dataset.data.targets
        
        print(f"Successfully loaded dataset with shape: {features.shape}")
        print(f"Dataset information: {diabetes_dataset.metadata['name']}")
        
        # Ensure target is a Series
        target = pd.Series(target.values.ravel(), name='target')
        
        return features, target
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating synthetic dataset instead...")
        
        # Create a synthetic dataset with similar properties
        n_samples = 2000
        n_features = 20
        
        # Create synthetic feature data
        np.random.seed(42)
        X = np.random.normal(size=(n_samples, n_features))
        
        # Create synthetic target (imbalanced)
        y = np.zeros(n_samples)
        y[:int(n_samples * 0.3)] = 1  # 30% positive cases
        np.random.shuffle(y)
        
        # Convert to pandas DataFrame
        feature_names = [f'feature_{i}' for i in range(n_features)]
        features = pd.DataFrame(X, columns=feature_names)
        target = pd.Series(y)
        
        print("Created synthetic dataset with shape:", features.shape)
    
    return features, target

def preprocess_data(X, y):
    """Preprocess data with 80/20 split and SMOTE."""
    # Convert target to numpy array
    y = np.array(y)
    
    # Split data with 80/20 ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8, random_state=42, stratify=y
    )
    
    # Check class distribution before SMOTE
    unique_before, counts_before = np.unique(y_train, return_counts=True)
    class_dist_before = dict(zip(unique_before, counts_before))
    print("Class distribution before SMOTE:", class_dist_before)
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Check class distribution after SMOTE
    unique_after, counts_after = np.unique(y_train_resampled, return_counts=True)
    class_dist_after = dict(zip(unique_after, counts_after))
    print("Class distribution after SMOTE:", class_dist_after)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape[0]} samples")
    print(f"Test set size: {X_test_scaled.shape[0]} samples")
    
    # Save the scaler
    joblib.dump(scaler, './outputs/scaler.pkl')
    
    return X_train_scaled, X_test_scaled, y_train_resampled, y_test

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model with 5-fold CV."""
    print("\n===== Training XGBoost Model =====")
    
    # Set up hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200],
        'gamma': [0, 0.1],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Create XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    
    # Create 5-fold CV
    cv_splits = 5
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=cv_splits,
        scoring='roc_auc',
        verbose=1,
        refit=True
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, "XGBoost")
    
    # Save the model
    joblib.dump(best_model, './outputs/xgboost_model.pkl')
    
    return best_model, metrics

def train_adaboost(X_train, y_train, X_test, y_test):
    """Train AdaBoost model with 5-fold CV."""
    print("\n===== Training AdaBoost Model =====")
    
    # Set up hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    
    # Create AdaBoost classifier
    ada_model = AdaBoostClassifier(random_state=42)
    
    # Create 5-fold CV
    cv_splits = 5
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=ada_model,
        param_grid=param_grid,
        cv=cv_splits,
        scoring='roc_auc',
        verbose=1,
        refit=True
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, "AdaBoost")
    
    # Save the model
    joblib.dump(best_model, './outputs/adaboost_model.pkl')
    
    return best_model, metrics

def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """Calculate comprehensive metrics."""
    metrics = {}
    
    # Classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Regression-like metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred_proba)
    metrics['mae'] = mean_absolute_error(y_true, y_pred_proba)
    
    # Correlation metrics
    metrics['correlation'] = np.corrcoef(y_true, y_pred_proba)[0, 1]
    
    print(f"\n{model_name} Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Save metrics to file
    with open(f'./outputs/{model_name.lower()}_metrics.txt', 'w') as f:
        f.write(f"{model_name} Metrics:\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred))
    
    # Create confusion matrix figure
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix without seaborn
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", 
                    ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    
    # Save figure
    plt.savefig(f'./outputs/{model_name.lower()}_confusion_matrix.png')
    plt.close()
    
    return metrics

def compare_models(xgb_metrics, ada_metrics):
    """Compare model performance and identify the best model."""
    # Create comparison dataframe
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mse', 'mae', 'correlation']
    comparison_data = []
    
    for metric in metrics_to_compare:
        comparison_data.append({
            'Metric': metric,
            'XGBoost': xgb_metrics[metric],
            'AdaBoost': ada_metrics[metric]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(comparison_df)
    
    # Save comparison to file
    comparison_df.to_csv('./outputs/model_comparison.csv', index=False)
    
    # Create comparison figure
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Set up bar positions
    metrics = comparison_df['Metric']
    x = np.arange(len(metrics))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, comparison_df['XGBoost'], width, label='XGBoost', color='#0173B2')
    ax.bar(x + width/2, comparison_df['AdaBoost'], width, label='AdaBoost', color='#DE8F05')
    
    # Add labels and legend
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()
    
    # Add value labels on bars
    for i, v in enumerate(comparison_df['XGBoost']):
        ax.text(i - width/2, v + 0.01, f"{v:.4f}", ha='center')
    
    for i, v in enumerate(comparison_df['AdaBoost']):
        ax.text(i + width/2, v + 0.01, f"{v:.4f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('./outputs/model_comparison.png')
    plt.close()
    
    # Determine best model
    best_model_name = "XGBoost" if xgb_metrics['auc'] > ada_metrics['auc'] else "AdaBoost"
    best_auc = max(xgb_metrics['auc'], ada_metrics['auc'])
    print(f"\nBest model based on AUC: {best_model_name} (AUC: {best_auc:.4f})")
    
    # Save best model info
    with open('./outputs/best_model.txt', 'w') as f:
        f.write(f"Best model: {best_model_name}\n")
        f.write(f"AUC: {best_auc:.4f}\n")
        
        if best_model_name == "XGBoost":
            f.write("\nXGBoost Metrics:\n")
            for metric_name, metric_value in xgb_metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
        else:
            f.write("\nAdaBoost Metrics:\n")
            for metric_name, metric_value in ada_metrics.items():
                f.write(f"{metric_name}: {metric_value:.4f}\n")
    
    return comparison_df

def main():
    print("Starting diabetes prediction model training...")
    
    # Download data
    X, y = download_data()
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y.astype(int))}")
    
    # Preprocess data with 80/20 split and SMOTE
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    
    # Train XGBoost model
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    
    # Train AdaBoost model
    ada_model, ada_metrics = train_adaboost(X_train, y_train, X_test, y_test)
    
    # Compare models
    compare_models(xgb_metrics, ada_metrics)
    
    print("\nTraining completed. All outputs saved to './outputs/' directory.")

if __name__ == "__main__":
    main()