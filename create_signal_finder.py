#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Signal Finder Model
------------------
Creates a high-recall model designed to capture more potential trading opportunities.
Works as a complementary system to the main high-precision model.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN

# Import the enhanced trading system
from enhanced_adaptive_ml_trading_system import EnhancedAdaptiveMLTradingSystem

def print_banner():
    """Print a nice banner for the Signal Finder script"""
    print("\n" + "="*80)
    print("🔍 SIGNAL FINDER MODEL - HIGH RECALL TRADING OPPORTUNITIES".center(80))
    print("="*80)
    print("Creating a complementary model to capture more potential trading signals")
    print("-"*80)

def create_signal_finder():
    """Create a high-recall signal finder model"""
    
    print("🔧 Initializing Enhanced Trading System...")
    system = EnhancedAdaptiveMLTradingSystem()
    
    print("📊 Loading dataset...")
    system.load_mega_dataset()
    
    print(f"📋 Dataset loaded: {len(system.mega_dataset):,} records")
    
    # Engineer features - this will include the enhanced timeframe features
    print("🧪 Engineering features...")
    X, feature_columns, y = system.engineer_leak_free_features()
    
    # Check class imbalance
    unique_counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_counts[0], unique_counts[1]))
    print(f"\n📊 Class distribution: {class_counts}")
    
    # Calculate imbalance ratio
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
    print(f"  - Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    # Handle potential problematic columns
    print("\n🛡️ REMOVING LEAKAGE FEATURES...")
    
    # List of features that contain future information
    leakage_features = ['maxprofit', 'maxloss', 'barsheld']
    future_info_prefixes = ['max', 'final', 'total', 'end', 'result']
    
    # Find all columns that might contain future information
    potential_leakage = [col for col in X.columns if any(col.startswith(prefix) for prefix in future_info_prefixes)]
    
    # Add to our list of features to remove
    leakage_features.extend([col for col in potential_leakage if col not in leakage_features])
    
    # Remove leakage features
    X_clean = X.drop(columns=[col for col in leakage_features if col in X.columns])
    print(f"  - Removed {len(leakage_features)} leakage features")
    
    # Check for non-numeric columns
    non_numeric_cols = X_clean.select_dtypes(include=['object']).columns.tolist()
    if non_numeric_cols:
        print(f"  - Dropping {len(non_numeric_cols)} non-numeric columns: {non_numeric_cols}")
        X_clean = X_clean.drop(columns=non_numeric_cols)
    
    print(f"  - Final dataset shape: {X_clean.shape}")
    
    # Split the data
    print("\n🔪 Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.20, random_state=42, shuffle=True
    )
    
    print(f"  - Training set: {X_train.shape}")
    print(f"  - Test set: {X_test.shape}")
    
    # Scale features
    print("\n📏 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balance the dataset using SMOTE
    print("\n⚖️ Balancing dataset with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"  - Original training shape: {X_train_scaled.shape}")
    print(f"  - Balanced training shape: {X_train_balanced.shape}")
    
    unique_counts_balanced = np.unique(y_train_balanced, return_counts=True)
    class_counts_balanced = dict(zip(unique_counts_balanced[0], unique_counts_balanced[1]))
    print(f"  - Balanced class distribution: {class_counts_balanced}")
    
    # Create a high-recall model
    print("\n🧠 Training Signal Finder model (optimized for recall)...")
    
    # Option 1: Gradient Boosting with focus on recall
    signal_finder = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=50,  # Higher values reduce overfitting and improve recall
        random_state=42
    )
    
    # Train the model
    signal_finder.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = signal_finder.predict(X_test_scaled)
    y_pred_proba = signal_finder.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate default threshold performance
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n📊 INITIAL MODEL PERFORMANCE (DEFAULT THRESHOLD):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    # Find the optimal threshold for high recall
    print("\n🎯 Finding optimal threshold for high recall...")
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # Create a dataframe with thresholds and performance metrics
    thresholds_df = pd.DataFrame({
        'threshold': [0] + list(thresholds),  # Add 0 to match precision and recall lengths
        'precision': list(precision_curve),
        'recall': list(recall_curve)
    })
    
    # Calculate F1 score
    thresholds_df['f1'] = 2 * (thresholds_df['precision'] * thresholds_df['recall']) / \
                         (thresholds_df['precision'] + thresholds_df['recall']).replace(0, np.nan)
    
    # Find threshold with highest recall that maintains at least 0.30 precision
    valid_thresholds = thresholds_df[thresholds_df['precision'] >= 0.30]
    
    if len(valid_thresholds) > 0:
        best_row = valid_thresholds.sort_values('recall', ascending=False).iloc[0]
        optimal_threshold = best_row['threshold']
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Expected precision: {best_row['precision']:.4f}")
        print(f"  Expected recall: {best_row['recall']:.4f}")
        print(f"  Expected F1: {best_row['f1']:.4f}")
    else:
        # If no threshold meets our precision requirement, use default
        optimal_threshold = 0.5
        print("  No threshold meets minimum precision requirement of 0.30")
        print(f"  Using default threshold: {optimal_threshold}")
    
    # Apply the optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Evaluate optimal threshold performance
    accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
    precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)
    
    print("\n📊 PERFORMANCE WITH OPTIMAL THRESHOLD:")
    print(f"  Accuracy: {accuracy_optimal:.4f}")
    print(f"  Precision: {precision_optimal:.4f}")
    print(f"  Recall: {recall_optimal:.4f}")
    print(f"  F1: {f1_optimal:.4f}")
    
    # Display confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_optimal)
    print("\n📊 CONFUSION MATRIX:")
    print(f"  [True Neg: {conf_matrix[0,0]}, False Pos: {conf_matrix[0,1]}]")
    print(f"  [False Neg: {conf_matrix[1,0]}, True Pos: {conf_matrix[1,1]}]")
    
    # Get feature importance
    if hasattr(signal_finder, 'feature_importances_'):
        importances = signal_finder.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n🔍 TOP 10 IMPORTANT FEATURES:")
        feature_names = X_clean.columns.tolist()
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.6f}")
    
    # Save the models
    print("\n💾 SAVING SIGNAL FINDER MODEL...")
    
    signal_finder_package = {
        'model': signal_finder,
        'scaler': scaler,
        'feature_columns': X_clean.columns.tolist(),
        'training_timestamp': datetime.now(),
        'optimal_threshold': optimal_threshold,
        'performance': {
            'accuracy': accuracy_optimal,
            'precision': precision_optimal,
            'recall': recall_optimal,
            'f1': f1_optimal
        }
    }
    
    # Save the new model
    output_path = 'signal_finder_model.pkl'
    joblib.dump(signal_finder_package, output_path)
    print(f"✅ Signal Finder model saved to {output_path}")
    
    print("\n🎉 SIGNAL FINDER MODEL CREATION COMPLETE!")
    print("This model is optimized for high recall to capture more potential trading opportunities.")
    print("Use this model in conjunction with your high-precision model for optimal results.")

if __name__ == "__main__":
    print_banner()
    create_signal_finder()
