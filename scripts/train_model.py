"""
Model Training Script.
End-to-end training pipeline: Data Loading -> Preprocessing -> Training (Baseline/Optuna) -> Evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.preprocess import (
        preprocess_for_training, 
        save_preprocessor
    )
    from scripts.model_utils import save_model
    from scripts.paths import (
        DATASET_PATH, MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH
    )
except ImportError:
    from preprocess import preprocess_for_training, save_preprocessor
    from model_utils import save_model
    from paths import DATASET_PATH, MODEL_PATH, PREPROCESSOR_PATH, METRICS_PATH

# Check for Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not installed. Hyperparameter optimization will be skipped.")

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_and_prepare_data(filepath: str = str(DATASET_PATH)):
    """
    Loads and splits data for training.
    
    Args:
        filepath (str): Path to CSV dataset.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor)
    """
    print(f"📦 Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"   Original shape: {df.shape}")
    
    # Preprocess
    X, y, preprocessor = preprocess_for_training(df)
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, preprocessor


def calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, model_name: str):
    """
    Calculates performance metrics.
    
    Returns:
        dict: Metrics dictionary
    """
    metrics = {
        'model_name': model_name,
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred)
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred)
        }
    }
    
    # Overfitting Check
    metrics['overfitting'] = {
        'accuracy_diff': metrics['train']['accuracy'] - metrics['test']['accuracy'],
        'f1_diff': metrics['train']['f1'] - metrics['test']['f1'],
        'is_overfitting': (metrics['train']['accuracy'] - metrics['test']['accuracy']) > 0.05
    }
    
    return metrics


def print_metrics(metrics: dict):
    """Prints metrics in a readable format."""
    print(f"\n📊 METRICS: {metrics['model_name']}")
    print("=" * 50)
    
    print("\n🔹 TRAIN:")
    for metric, value in metrics['train'].items():
        print(f"   {metric.capitalize():12}: {value:.4f}")
    
    print("\n🔹 TEST:")
    for metric, value in metrics['test'].items():
        print(f"   {metric.capitalize():12}: {value:.4f}")
    
    print("\n🔍 OVERFITTING CHECK:")
    diff_acc = metrics['overfitting']['accuracy_diff']
    status = "⚠️ OVERFITTING DETECTED" if metrics['overfitting']['is_overfitting'] else "✅ OK"
    
    print(f"   Accuracy Diff: {diff_acc:.4f} ({diff_acc*100:.2f}%)")
    print(f"   Status: {status}")


def create_optuna_objective(X_train, y_train):
    """Creates the Optuna objective function."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        
        model = RandomForestClassifier(**params)
        
        # 5-Fold Stratified Cross-Validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
        
        return scores.mean()
    
    return objective


def optimize_with_optuna(X_train, y_train, X_test, y_test, n_trials: int = 10):
    """Optimizes hyperparameters using Optuna."""
    if not OPTUNA_AVAILABLE:
        print("⚠️ Optuna unavailable. Using baseline.")
        return train_baseline_model(X_train, y_train, X_test, y_test)
    
    print(f"\n🔬 Optimizing with Optuna ({n_trials} trials)...")
    
    sampler = TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    objective = create_optuna_objective(X_train, y_train)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n✅ Best F1 (CV): {study.best_value:.4f}")
    
    # Train best model
    best_params = study.best_params.copy()
    best_params.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'class_weight': 'balanced'})
    
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, "Optuna Optimized")
    
    return {'model': model, 'metrics': metrics, 'study': study}


def train_baseline_model(X_train, y_train, X_test, y_test):
    """Trains a baseline RandomForest model."""
    print("\n🔧 Training Baseline (RandomForest)...")
    
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5, 
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    metrics = calculate_metrics(y_train, y_train_pred, y_test, y_test_pred, "Baseline")
    return {'model': model, 'metrics': metrics}


def train_model():
    """Main training execution function."""
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data()
    save_preprocessor(preprocessor, str(PREPROCESSOR_PATH))
    
    # 1. Baseline
    baseline_result = train_baseline_model(X_train, y_train, X_test, y_test)
    print_metrics(baseline_result['metrics'])
    
    # 2. Optimization
    final_model = baseline_result['model']
    final_metrics = baseline_result['metrics']
    
    if OPTUNA_AVAILABLE:
        opt_result = optimize_with_optuna(X_train, y_train, X_test, y_test, n_trials=10)
        print_metrics(opt_result['metrics'])
        
        if opt_result['metrics']['test']['f1'] > baseline_result['metrics']['test']['f1']:
            final_model = opt_result['model']
            final_metrics = opt_result['metrics']
            print("\n✅ Selected: Optimized Model")
        else:
            print("\n✅ Selected: Baseline Model")
    
    # Save Artifacts
    save_model(final_model, str(MODEL_PATH))
    joblib.dump(final_metrics, str(METRICS_PATH))
    print(f"✅ Metrics saved to: {METRICS_PATH}")
    
    return final_model


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MODEL TRAINING - AIRLINE SATISFACTION")
    print("=" * 60)
    train_model()
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED")
    print("=" * 60)