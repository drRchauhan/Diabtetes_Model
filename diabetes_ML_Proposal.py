import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, 
    mean_squared_error, classification_report,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 1234

def load_data(file_path="D:\EDPE 579 ML\diabetes_012_health_indicators_BRFSS2015.csv"):
    """
    Load the diabetes dataset from local machine
    
    Parameters:
    file_path (str): Path to the CSV file containing diabetes dataset
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    try:
        # Load dataset from local file
        # Assuming your dataset has headers - if not, you may need to specify column names
        df = pd.read_csv(file_path)
        
        # If your dataset doesn't have column names, uncomment the following:
        # column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        #                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        # df = pd.read_csv(file_path, names=column_names, header=None)
        
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        print("Please provide the correct path to your diabetes dataset.")
        return None
    except Exception as e:
        print(f"Error loading the dataset: {e}")
        return None
    
    print(f"Dataset successfully loaded from {file_path}")
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess_data(df):
    """Preprocess the dataset for modeling"""
    # First check for null values
    print("\nNull values in dataset:")
    print(df.isnull().sum())
    
    # Handle missing values if any exist
    if df.isnull().sum().sum() > 0:
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Drop ID column if it exists (not useful for prediction)
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Check which target variable we're using
    if 'Diabetes_binary' in df.columns:
        # Binary classification (0 = no diabetes, 1 = prediabetes or diabetes)
        target_col = 'Diabetes_binary'
        is_multiclass = False
        print("\nTarget variable: Diabetes_binary (Binary classification)")
    elif 'Diabetes_012' in df.columns:
        # Multi-class classification (0 = no diabetes, 1 = prediabetes, 2 = diabetes)
        target_col = 'Diabetes_012'
        is_multiclass = True
        print("\nTarget variable: Diabetes_012 (Multi-class classification)")
    else:
        raise ValueError("No recognized target variable found in the dataset")
        
    print("\nClass distribution:")
    print(df[target_col].value_counts())
    
    # Split features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split (80-20 split as specified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test, X.columns, is_multiclass
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split (80-20 split as specified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, train_size=0.8, random_state=RANDOM_STATE
    )
    
    return X_train, X_test, y_train, y_test, X.columns

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance using various metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check if this is a multi-class or binary classification problem
    n_classes = len(np.unique(y_test))
    is_multiclass = n_classes > 2
    
    # Get prediction probabilities
    y_pred_proba = model.predict_proba(X_test)
    
    if is_multiclass:
        # For multi-class problems (3 classes: no diabetes, prediabetes, diabetes)
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # For multi-class, we use one-vs-rest ROC AUC
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    else:
        # For binary classification
        y_pred_proba_binary = y_pred_proba[:, 1]  # Probability of positive class
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba_binary)
    
    # Calculate MSE and RMSE (though these aren't typical for classification)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print results
    print(f"\n{'-'*20} {model_name} Results {'-'*20}")
    print(f"Accuracy: {accuracy:.4f}")
    if is_multiclass:
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Prepare results dictionary
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mse': mse,
        'rmse': rmse
    }
    
    # Add multi-class specific metrics if applicable
    if is_multiclass:
        results['balanced_accuracy'] = balanced_acc
        results['mcc'] = mcc
    
    # Add prediction probabilities based on classification type
    if is_multiclass:
        results['y_pred_proba_multi'] = y_pred_proba  # Full probability matrix for multi-class
    else:
        results['y_pred_proba'] = y_pred_proba[:, 1]  # Probability of positive class for binary
    
    return results

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models"""
    # Check if this is a multi-class or binary classification problem
    n_classes = len(np.unique(y_test))
    is_multiclass = n_classes > 2
    
    if is_multiclass:
        # Plot ROC curve for each class (one-vs-rest)
        plt.figure(figsize=(15, 10))
        
        # Create subplots for each class
        for class_idx in range(n_classes):
            plt.subplot(1, n_classes, class_idx + 1)
            
            for result in results:
                model_name = result['model_name']
                
                # For multi-class, we need to extract the correct column from y_pred_proba
                # This assumes y_pred_proba is stored differently in results for multi-class
                if 'y_pred_proba_multi' in result:
                    y_pred_proba_class = result['y_pred_proba_multi'][:, class_idx]
                    
                    # Convert y_test to binary one-vs-rest for this class
                    y_test_binary = (y_test == class_idx).astype(int)
                    
                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba_class)
                    # Calculate AUC for this class
                    auc_value = roc_auc_score(y_test_binary, y_pred_proba_class)
                    
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_value:.4f})')
            
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for Class {class_idx}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        # Binary classification plot
        plt.figure(figsize=(10, 8))
        
        for result in results:
            model_name = result['model_name']
            y_pred_proba = result['y_pred_proba']
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = result['auc']
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def plot_feature_importance(model, feature_names, model_name):
    """Plot feature importance for tree-based models"""
    plt.figure(figsize=(12, 6))
    
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models that don't have direct feature_importances_ attribute
        if model_name == 'XGBoost':
            importances = model.get_booster().get_score(importance_type='gain')
            # Convert to array matching feature_names order
            imp_array = np.zeros(len(feature_names))
            for key, value in importances.items():
                # XGBoost feature names are f0, f1, etc.
                idx = int(key.replace('f', ''))
                if idx < len(imp_array):
                    imp_array[idx] = value
            importances = imp_array
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Plot
    plt.title(f'Feature Importance - {model_name}')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train and tune Random Forest model"""
    print("\n" + "="*50)
    print("Training Random Forest model with hyperparameter tuning...")
    
    # Check if this is a multi-class or binary classification problem
    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2
    print(f"Number of target classes: {n_classes}")
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']  # Add class weight for potentially imbalanced data
    }
    
    # Initialize Random Forest
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='balanced_accuracy' if is_multiclass else 'roc_auc',  # Better metric for imbalanced data
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_rf = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters for Random Forest:")
    print(grid_search.best_params_)
    
    # Evaluate model
    rf_results = evaluate_model(best_rf, X_test, y_test, "Random Forest")
    
    return best_rf, rf_results

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train and tune XGBoost model"""
    print("\n" + "="*50)
    print("Training XGBoost model with hyperparameter tuning...")
    
    # Check if this is a multi-class or binary classification problem
    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2
    
    # Define parameter grid for grid search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'scale_pos_weight': [1, 5, 10] if not is_multiclass else [1]  # For imbalanced binary classification
    }
    
    # Objective function based on number of classes
    objective = 'multi:softprob' if is_multiclass else 'binary:logistic'
    
    # Initialize XGBoost
    xgb_model = xgb.XGBClassifier(
        objective=objective,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='mlogloss' if is_multiclass else 'logloss',
        num_class=n_classes if is_multiclass else None
    )
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='balanced_accuracy' if is_multiclass else 'roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_xgb = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters for XGBoost:")
    print(grid_search.best_params_)
    
    # Evaluate model
    xgb_results = evaluate_model(best_xgb, X_test, y_test, "XGBoost")
    
    return best_xgb, xgb_results

def train_adaboost(X_train, y_train, X_test, y_test):
    """Train and tune AdaBoost model"""
    print("\n" + "="*50)
    print("Training AdaBoost model with hyperparameter tuning...")
    
    # Check if this is a multi-class or binary classification problem
    n_classes = len(np.unique(y_train))
    is_multiclass = n_classes > 2
    
    # Define parameter grid for grid search
    # For multi-class, we'll need to use SAMME algorithm as SAMME.R only works for binary classification
    if is_multiclass:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'algorithm': ['SAMME']  # SAMME.R only works for binary classification
        }
    else:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }
    
    # Initialize AdaBoost
    ada = AdaBoostClassifier(random_state=RANDOM_STATE)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=ada,
        param_grid=param_grid,
        cv=5,
        scoring='balanced_accuracy' if is_multiclass else 'roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_ada = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters for AdaBoost:")
    print(grid_search.best_params_)
    
    # Evaluate model
    ada_results = evaluate_model(best_ada, X_test, y_test, "AdaBoost")
    
    return best_ada, ada_resultsSAMME', 'SAMME.R']
        }
    
    # Initialize AdaBoost
    ada = AdaBoostClassifier(random_state=RANDOM_STATE)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=ada,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_ada = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters for AdaBoost:")
    print(grid_search.best_params_)
    
    # Evaluate model
    ada_results = evaluate_model(best_ada, X_test, y_test, "AdaBoost")
    
    return best_ada, ada_results

def compare_models(results):
    """Compare all models side by side"""
    # Define metrics to compare
    basic_metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc', 'mse', 'rmse']
    
    # Filter metrics that exist in all results
    metrics = [m for m in basic_metrics if all(m in result for result in results)]
    
    # Create dataframe for comparison
    comparison = pd.DataFrame({
        result['model_name']: [result[metric] for metric in metrics]
        for result in results
    }, index=metrics)
    
    print("\n" + "="*50)
    print("Model Comparison:")
    print(comparison)
    
    # Plot comparison of performance metrics (higher is better)
    performance_metrics = [m for m in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc'] 
                          if m in metrics]
    
    plt.figure(figsize=(12, 8))
    comparison.loc[performance_metrics].T.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison (Higher is Better)')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    # Plot MSE and RMSE separately (lower is better)
    error_metrics = [m for m in ['mse', 'rmse'] if m in metrics]
    if error_metrics:
        plt.figure(figsize=(10, 6))
        comparison.loc[error_metrics].T.plot(kind='bar', figsize=(10, 6))
        plt.title('Error Metrics Comparison (Lower is Better)')
        plt.ylabel('Error')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

def main():
    """Main function to execute the workflow"""
    # Load data from local file - update the path to match your file location
    df = load_data("diabetes.csv")  # Replace with your actual file path
    
    # Exit if data loading failed
    if df is None:
        print("Exiting due to data loading failure.")
        return
    
    # Display dataset info
    print("\nData Overview:")
    print(df.head())
    
    # Feature Analysis
    if 'Diabetes_binary' in df.columns:
        target_col = 'Diabetes_binary'
    elif 'Diabetes_012' in df.columns:
        target_col = 'Diabetes_012'
    else:
        print("Error: Could not identify target column")
        return
    
    # Feature importance analysis based on correlation
    print("\nFeature Correlations with Target Variable:")
    correlations = df.corr()[target_col].sort_values(ascending=False)
    print(correlations)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names, is_multiclass = preprocess_data(df)
    
    # Train and evaluate models
    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)
    ada_model, ada_results = train_adaboost(X_train, y_train, X_test, y_test)
    
    # Collect all results
    all_results = [rf_results, xgb_results, ada_results]
    
    # Compare models
    compare_models(all_results)
    
    # Plot ROC curves
    plot_roc_curves(all_results, y_test)
    
    # Plot feature importance
    plot_feature_importance(rf_model, feature_names, "Random Forest")
    plot_feature_importance(xgb_model, feature_names, "XGBoost")
    plot_feature_importance(ada_model, feature_names, "AdaBoost")
    
    # Return the best model based on AUC
    best_model_name = max(all_results, key=lambda x: x['auc'])['model_name']
    print(f"\nBest model based on AUC: {best_model_name}")

if __name__ == "__main__":
    main()