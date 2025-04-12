#!/usr/bin/env python
# coding: utf-8

"""
Azure ML Diabetes Prediction Pipeline with XGBoost and AdaBoost
Complete script for training, evaluating, registering, and deploying models.

This script performs the following steps:
1. Load the CDC Diabetes Health Indicators dataset
2. Preprocess the data with 80/20 train/test split
3. Apply SMOTE for class balancing
4. Train XGBoost and AdaBoost models with 5-fold CV
5. Calculate all metrics (MSE, MAE, correlation, F1, AUROC)
6. Compare and register the best model
7. Deploy as an API endpoint
"""

import os
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# ML and Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Azure ML and MLflow
from azureml.core import Workspace, Experiment, Run, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import mlflow
import mlflow.sklearn
import mlflow.xgboost

# Data source
from ucimlrepo import fetch_ucirepo

# Set display options for prettier output
pd.set_option('display.max_columns', None)
np.set_printoptions(precision=3, suppress=True)


def load_diabetes_data():
    """Load the CDC Diabetes Health Indicators dataset."""
    print("Loading CDC Diabetes Health Indicators dataset...")
    
    # Fetch dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    
    # Extract features and target
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets
    
    # Print dataset information
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
    print(f"Target distribution: {np.bincount(np.ravel(y.values))}")
    
    return X, y


def preprocess_data(X, y, apply_smote=True):
    """
    Preprocess the data with standardization and class balancing.
    
    Args:
        X: Features dataframe
        y: Target dataframe
        apply_smote: Whether to apply SMOTE for class balancing (default: True)
    
    Returns:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training targets (resampled if SMOTE applied)
        y_test: Test targets
        scaler: Fitted StandardScaler
        class_dist_before: Class distribution before SMOTE
        class_dist_after: Class distribution after SMOTE
    """
    # Convert target to a numpy array
    y = np.ravel(y.values)
    
    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}%)")
    
    # Check class distribution before SMOTE
    unique_before, counts_before = np.unique(y_train, return_counts=True)
    class_dist_before = dict(zip(unique_before, counts_before))
    print("\nClass distribution before SMOTE:")
    print(class_dist_before)
    
    # Apply SMOTE to balance the training data
    if apply_smote:
        print("Applying SMOTE for class balancing...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Check class distribution after SMOTE
        unique_after, counts_after = np.unique(y_train_resampled, return_counts=True)
        class_dist_after = dict(zip(unique_after, counts_after))
        print("Class distribution after SMOTE:")
        print(class_dist_after)
        
        print(f"Original training set: {X_train.shape[0]} samples")
        print(f"After SMOTE: {X_train_resampled.shape[0]} samples")
    else:
        X_train_resampled, y_train_resampled = X_train, y_train
        class_dist_after = class_dist_before
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler, class_dist_before, class_dist_after


def train_xgboost(X_train, y_train, X_test, y_test, run):
    """
    Train an XGBoost model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        run: Azure ML run for logging
    
    Returns:
        best_model: Fitted XGBoost model
        metrics: Dictionary of performance metrics
    """
    print("\n===== Training XGBoost Model =====")
    
    # Set up hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01],
        'n_estimators': [100, 200],
        'gamma': [0, 0.1],
        'colsample_bytree': [0.8, 1.0],
        'scale_pos_weight': [1, 3]  # Add weight for imbalanced problems
    }
    
    # Create XGBoost classifier
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    
    # Create cross-validation splits
    cv_splits = 5
    
    # Set up GridSearchCV with fixed CV
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
    
    # Log the best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    run.log_dict("xgboost_best_params", grid_search.best_params_)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, X_test, best_model)
    
    # Log metrics to Azure ML
    for metric_name, metric_value in metrics.items():
        run.log(f"xgboost_{metric_name}", metric_value)
    
    # Plot and log feature importance
    feature_importance_plot = plot_feature_importance(best_model, X_train)
    run.log_image("xgboost_feature_importance", feature_importance_plot)
    
    # Save the model
    model_path = "xgboost_model.pkl"
    joblib.dump(best_model, model_path)
    run.upload_file("models/" + model_path, model_path)
    
    print("XGBoost model training completed.")
    return best_model, metrics


def train_adaboost(X_train, y_train, X_test, y_test, run):
    """
    Train an AdaBoost model with hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        run: Azure ML run for logging
    
    Returns:
        best_model: Fitted AdaBoost model
        metrics: Dictionary of performance metrics
    """
    print("\n===== Training AdaBoost Model =====")
    
    # Set up hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    }
    
    # Create AdaBoost classifier
    ada_model = AdaBoostClassifier(random_state=42)
    
    # Create cross-validation splits
    cv_splits = 5
    
    # Set up GridSearchCV with fixed CV
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
    
    # Log the best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    run.log_dict("adaboost_best_params", grid_search.best_params_)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, X_test, best_model)
    
    # Log metrics to Azure ML
    for metric_name, metric_value in metrics.items():
        run.log(f"adaboost_{metric_name}", metric_value)
    
    # Save the model
    model_path = "adaboost_model.pkl"
    joblib.dump(best_model, model_path)
    run.upload_file("models/" + model_path, model_path)
    
    print("AdaBoost model training completed.")
    return best_model, metrics


def calculate_metrics(y_true, y_pred, y_pred_proba, X_test, model):
    """
    Calculate and return all required metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        X_test: Test features
        model: Trained model
    
    Returns:
        metrics: Dictionary of performance metrics
    """
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
    
    print("\nModel Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return metrics


def plot_feature_importance(model, X_train):
    """
    Plot feature importance for the model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        X_train: Training features
    
    Returns:
        plt: Matplotlib figure
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        return None
    
    # Create DataFrame for plotting
    if isinstance(X_train, np.ndarray):
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    else:
        feature_names = X_train.columns
    
    importances_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Create plot
    sns.barplot(x='Importance', y='Feature', data=importances_df[:15])
    plt.title('Feature Importance')
    plt.tight_layout()
    
    return plt


def plot_confusion_matrix(y_true, y_pred, model_name, run):
    """
    Plot and log confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for the plot title
        run: Azure ML run for logging
    
    Returns:
        plt: Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save and log the plot
    cm_path = f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)
    run.log_image(f"{model_name}_confusion_matrix", plt)
    
    return plt


def compare_models(xgb_metrics, ada_metrics, run):
    """
    Compare and visualize the performance of both models.
    
    Args:
        xgb_metrics: Dictionary of XGBoost metrics
        ada_metrics: Dictionary of AdaBoost metrics
        run: Azure ML run for logging
    
    Returns:
        plt: Matplotlib figure
    """
    # Create comparison dataframe
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'mse', 'mae']
    comparison_data = []
    
    for metric in metrics_to_compare:
        comparison_data.append({
            'Metric': metric,
            'XGBoost': xgb_metrics[metric],
            'AdaBoost': ada_metrics[metric]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Plot comparison
    plt.figure(figsize=(14, 7))
    comparison_df_melted = pd.melt(
        comparison_df, 
        id_vars=['Metric'], 
        value_vars=['XGBoost', 'AdaBoost'],
        var_name='Model', 
        value_name='Value'
    )
    
    # Create proper colormap
    colors = {"XGBoost": "#0173B2", "AdaBoost": "#DE8F05"}
    
    ax = sns.barplot(x='Metric', y='Value', hue='Model', data=comparison_df_melted, palette=colors)
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on the bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.4f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center', va = 'bottom',
                   xytext = (0, 5), 
                   textcoords = 'offset points')
    
    plt.tight_layout()
    
    # Log the comparison
    run.log_image("model_comparison", plt)
    run.log_table("model_comparison_data", comparison_df)
    
    return plt


def setup_azure_ml():
    """
    Set up Azure ML workspace and experiment.
    
    Returns:
        ws: Azure ML workspace
        experiment: Azure ML experiment
        run: Azure ML run
    """
    try:
        # Try to load configuration from config.json
        ws = Workspace.from_config()
        print("Loaded workspace configuration from config.json")
    except:
        # If that fails, use interactive authentication
        print("Could not load workspace configuration. Using interactive authentication.")
        interactive_auth = InteractiveLoginAuthentication()
        ws = Workspace.get(
            name="your-workspace-name",            # Replace with your workspace name
            subscription_id="your-subscription-id", # Replace with your subscription ID
            resource_group="your-resource-group",   # Replace with your resource group
            auth=interactive_auth
        )
    
    # Create an experiment
    experiment_name = f"diabetes-classification-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    experiment = Experiment(workspace=ws, name=experiment_name)
    
    # Start a run
    run = experiment.start_logging()
    print(f"Started experiment: {experiment_name}")
    
    return ws, experiment, run


def create_scoring_script():
    """Create scoring script for model deployment."""
    os.makedirs('./deployment', exist_ok=True)
    
    with open('./deployment/score.py', 'w') as f:
        f.write('''
import json
import numpy as np
import pandas as pd
import joblib
import os

def init():
    global model, scaler
    
    print("Initializing diabetes prediction model service...")
    
    # Get model path
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    scaler_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'scaler.pkl')
    
    # Load model and scaler
    model = joblib.load(model_path)
    
    # Load scaler if it exists
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        scaler = None
    
    print("Model loaded successfully")
    print(f"Model type: {type(model).__name__}")

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        if isinstance(data, dict):
            # Single prediction
            df = pd.DataFrame([data])
        else:
            # Multiple predictions
            df = pd.DataFrame(data)
        
        # Ensure correct features are present
        required_features = [
            'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 
            'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 
            'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 
            'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 
            'Education', 'Income'
        ]
        
        # Check for missing features
        missing_features = set(required_features) - set(df.columns)
        if missing_features:
            return json.dumps({
                "error": f"Missing required features: {list(missing_features)}"
            })
        
        # Apply preprocessing if scaler exists
        if scaler:
            features = scaler.transform(df)
        else:
            features = df.values
        
        # Generate predictions
        predictions_proba = model.predict_proba(features)[:, 1]
        predictions = model.predict(features)
        
        # Return comprehensive results
        result = {
            "predictions": predictions.tolist(),
            "probabilities": predictions_proba.tolist(),
            "prediction_details": [
                {
                    "prediction": int(pred), 
                    "probability": float(prob),
                    "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
                }
                for pred, prob in zip(predictions, predictions_proba)
            ],
            "metadata": {
                "model_type": type(model).__name__,
                "feature_count": len(df.columns),
                "sample_count": len(df)
            }
        }
        
        return json.dumps(result)
    
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
''')
    
    # Create conda environment file
    with open('./deployment/conda_env.yml', 'w') as f:
        f.write('''
name: diabetes_model_env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - numpy
  - pandas
  - scikit-learn
  - pip
  - pip:
    - azureml-defaults
    - xgboost
    - joblib
    - imbalanced-learn
''')
    
    print("Created deployment files in ./deployment/")
    

def deploy_model(ws, model_path, model_name, service_name, description):
    """
    Deploy a model to Azure ML.
    
    Args:
        ws: Azure ML workspace
        model_path: Path to the model file
        model_name: Name to register the model with
        service_name: Name for the web service
        description: Description of the model
    
    Returns:
        service: Deployed web service
    """
    print(f"\nDeploying model: {model_name} as {service_name}...")
    
    # Create scoring script and env files
    create_scoring_script()
    
    # Register the model
    model = Model.register(
        workspace=ws,
        model_path=model_path,
        model_name=model_name,
        description=description,
        tags={"type": "classification", "framework": model_name.split('_')[-1]}
    )
    
    # Create inference configuration
    inference_config = InferenceConfig(
        entry_script="./deployment/score.py",
        environment_path="./deployment/conda_env.yml"
    )
    
    # Configure the deployment
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        tags={'model': model_name, 'type': 'diabetes-classification'},
        description=f'Diabetes classification model using {model_name} with SMOTE'
    )
    
    # Deploy the model
    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config,
        overwrite=True
    )
    
    # Wait for deployment to complete
    service.wait_for_deployment(show_output=True)
    
    # Print the endpoint URL
    print(f"\nDeployment successful!")
    print(f"Service URL: {service.scoring_uri}")
    
    return service


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train and deploy ML models for diabetes prediction')
    parser.add_argument('--train', action='store_true', help='Train the models')
    parser.add_argument('--register', action='store_true', help='Register the best model in Azure ML')
    parser.add_argument('--deploy', action='store_true', help='Deploy the best model as a web service')
    parser.add_argument('--no-smote', action='store_true', help='Skip SMOTE class balancing')
    args = parser.parse_args()
    
    # Set default action if none specified
    if not (args.train or args.register or args.deploy):
        args.train = True
    
    # Set up Azure ML
    print("Setting up Azure ML...")
    ws, experiment, run = setup_azure_ml()
    
    # Train models
    if args.train:
        # Load data
        print("Loading data...")
        X, y = load_diabetes_data()
        
        # Preprocess data with SMOTE
        print("Preprocessing data...")
        apply_smote = not args.no_smote
        X_train, X_test, y_train, y_test, scaler, class_dist_before, class_dist_after = preprocess_data(
            X, y, apply_smote=apply_smote
        )
        
        # Log preprocessing parameters
        run.log("train_ratio", 0.8)
        run.log("test_ratio", 0.2)
        run.log("cross_validation_folds", 5)
        run.log("using_smote", apply_smote)
        run.log_dict("class_distribution_before", class_dist_before)
        run.log_dict("class_distribution_after", class_dist_after)
        
        # Plot class distribution before and after SMOTE
        plt.figure(figsize=(12, 5))
        
        # Before SMOTE
        plt.subplot(1, 2, 1)
        plt.bar(["Class 0", "Class 1"], [class_dist_before.get(0, 0), class_dist_before.get(1, 0)])
        plt.title("Class Distribution Before")
        plt.ylabel("Count")
        
        # After SMOTE
        plt.subplot(1, 2, 2)
        plt.bar(["Class 0", "Class 1"], [class_dist_after.get(0, 0), class_dist_after.get(1, 0)])
        plt.title("Class Distribution After" + (" SMOTE" if apply_smote else ""))
        
        plt.tight_layout()
        run.log_image("class_distribution", plt)
        
        try:
            # Enable auto-logging
            mlflow.autolog()
            
            # Train XGBoost model with 5-fold CV
            xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, run)
            
            # Train AdaBoost model with 5-fold CV
            ada_model, ada_metrics = train_adaboost(X_train, y_train, X_test, y_test, run)
            
            # Plot confusion matrices
            plot_confusion_matrix(y_test, xgb_model.predict(X_test), "XGBoost", run)
            plot_confusion_matrix(y_test, ada_model.predict(X_test), "AdaBoost", run)
            
            # Compare models
            compare_models(xgb_metrics, ada_metrics, run)
            
            # Save preprocessor
            preprocessor_path = "scaler.pkl"
            joblib.dump(scaler, preprocessor_path)
            run.upload_file("models/" + preprocessor_path, preprocessor_path)
            
            # Determine best model based on AUC
            best_auc_xgb = xgb_metrics['auc']
            best_auc_ada = ada_metrics['auc']
            
            # Register the best model if requested
            if args.register or args.deploy:
                best_model_name = "XGBoost" if best_auc_xgb > best_auc_ada else "AdaBoost"
                best_model_path = "xgboost_model.pkl" if best_model_name == "XGBoost" else "adaboost_model.pkl"
                best_model = xgb_model if best_model_name == "XGBoost" else ada_model
                best_auc = best_auc_xgb if best_model_name == "XGBoost" else best_auc_ada
                
                print(f"\nBest model: {best_model_name} with AUC: {best_auc:.4f}")
                
                registered_model_name = f"diabetes_classifier_{best_model_name.lower()}"
                description = f"Diabetes classification model using {best_model_name} with 5-fold CV"
                if apply_smote:
                    description += " and SMOTE"
                
                tags = {
                    "model_framework": best_model_name,
                    "dataset": "CDC Diabetes Health Indicators",
                    "train_test_split": "80/20",
                    "cross_validation": "5-fold",
                    "class_balancing": "SMOTE" if apply_smote else "None",
                    "auc_score": str(best_auc)
                }
                
                if args.register:
                    # Register the model in Azure ML workspace
                    registered_model = run.register_model(
                        model_name=registered_model_name,
                        model_path=f"models/{best_model_path}",
                        tags=tags,
                        description=description
                    )
                    
                    print(f"Registered model: {registered_model.name} (Version: {registered_model.version})")
                
                # Deploy the model if requested
                if args.deploy:
                    service_name = f"diabetes-predictor-{best_model_name.lower()}"
                    deploy_model(
                        ws=ws,
                        model_path=best_model_path,
                        model_name=registered_model_name,
                        service_name=service_name,
                        description=description
                    )
        
        except Exception as e:
            run.log("error", str(e))
            print(f"Error during training: {str(e)}")
            raise
        
        finally:
            # Complete the run
            run.complete()
            print("Training completed!")
    
    # Deploy existing model without training
    elif args.deploy and not args.train:
        # Query for the most recent registered model
        models = Model.list(ws, name="diabetes_classifier_xgboost")
        if not models:
            models = Model.list(ws, name="diabetes_classifier_adaboost")
        
        if models:
            latest_model = models[0]
            for model in models:
                if model.created_time > latest_model.created_time:
                    latest_model = model
            
            # Deploy the model
            service_name = f"diabetes-predictor-{latest_model.name.split('_')[-1]}"
            deploy_model(
                ws=ws,
                model_path=latest_model.id,
                model_name=latest_model.name,
                service_name=service_name,
                description=latest_model.description
            )
        else:
            print("No registered models found. Please train and register a model first.")
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()