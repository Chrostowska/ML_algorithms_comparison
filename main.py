from ml_pipeline.preprocessing import DataPreprocessor
from ml_pipeline.models import ModelFactory
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Create output directory for plots
os.makedirs('outputs', exist_ok=True)

def load_data():
    """Load and prepare the dataset."""
    print("\n1. Loading dataset...")
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class distribution:\n{pd.Series(y).value_counts().to_dict()}")
    
    return X, y

def preprocess_data(X, y):
    """Preprocess the data using our pipeline."""
    print("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        handle_outliers=True,
        feature_selection=15  # Select top 15 features
    )
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y  # Added stratification
    )
    
    # Apply preprocessing
    print("Applying preprocessing steps...")
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"Processed training set shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models."""
    print("\n3. Training and evaluating models...")
    
    # Models to evaluate
    models_to_try = [
        'logistic_regression',
        'random_forest',
        'svm',
        'gradient_boosting',
        'xgboost'
    ]
    
    results = {}
    
    for model_name in models_to_try:
        print(f"\nTraining {model_name}...")
        start_time = time.time()
        
        # Create and train model
        model = ModelFactory.create_model(model_name)
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Get probabilities for ROC curve
        try:
            test_proba = model.predict_proba(X_test)[:, 1]
        except (AttributeError, NotImplementedError):
            test_proba = None
        
        # Calculate metrics
        train_accuracy = (train_pred == y_train).mean()
        test_accuracy = (test_pred == y_test).mean()
        
        # Store results
        results[model_name] = {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_predictions': test_pred,
            'test_probabilities': test_proba,
            'training_time': time.time() - start_time
        }
        
        # Print results
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")
        print(f"Training Time: {results[model_name]['training_time']:.2f} seconds")
        print("\nClassification Report:")
        print(classification_report(y_test, test_pred))
        
    return results

def visualize_results(results, y_test):  # Added y_test parameter
    """Create visualizations of model performance."""
    print("\n4. Creating visualizations...")
    
    # Prepare data for plotting
    model_names = list(results.keys())
    train_scores = [results[name]['train_accuracy'] for name in model_names]
    test_scores = [results[name]['test_accuracy'] for name in model_names]
    training_times = [results[name]['training_time'] for name in model_names]
    
    # Plot 1: Model Performance Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, train_scores, width, label='Training Accuracy')
    ax1.bar(x + width/2, test_scores, width, label='Testing Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    
    ax2.bar(x, training_times)
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Model Training Times')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/model_comparison.png')
    print("Model comparison plot saved as 'outputs/model_comparison.png'")
    plt.close()
    
    # Plot 2: Confusion Matrices
    n_models = len(model_names)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    for idx, (name, result) in enumerate(results.items()):
        if idx < len(axes):
            cm = confusion_matrix(y_test, result['test_predictions'])
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
            axes[idx].set_title(f'Confusion Matrix - {name}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
    
    # Hide empty subplots
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrices.png')
    print("Confusion matrices saved as 'outputs/confusion_matrices.png'")
    plt.close()
    
    # Plot 3: ROC Curves
    plt.figure(figsize=(10, 6))
    for name, result in results.items():
        if result['test_probabilities'] is not None:
            fpr, tpr, _ = roc_curve(y_test, result['test_probabilities'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('outputs/roc_curves.png')
    print("ROC curves saved as 'outputs/roc_curves.png'")
    plt.close()

def main():
    """Main function to run the complete pipeline."""
    print("=== Machine Learning Pipeline ===")
    
    try:
        # Load data
        X, y = load_data()
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        # Train and evaluate models
        results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        # Visualize results
        visualize_results(results, y_test)  # Pass y_test to visualization function
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
        print(f"\nBest performing model: {best_model[0]}")
        print(f"Test accuracy: {best_model[1]['test_accuracy']:.4f}")
        print(f"Training time: {best_model[1]['training_time']:.2f} seconds")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 