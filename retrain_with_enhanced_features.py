#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhance    # Separate numerical and categorical fe    # Create separate dataframes for numerical and categorical features
    X_train_num = X_train[numerical_columns] if numerical_columns else X_train.copy()
    X_test_num = X_test[numerical_columns] if numerical_columns else X_test.copy()
    
    X_train_cat = X_train[existing_categorical] if existing_categorical else None
    X_test_cat = X_test[existing_categorical] if existing_categorical else None
    
    # Make sure our numerical data is actually numeric
    print("\n🔍 Checking numerical data types:")
    for col in X_train_num.columns:
        if not pd.api.types.is_numeric_dtype(X_train_num[col]):
            print(f"  - Converting {col} to numeric...")
            try:
                X_train_num[col] = pd.to_numeric(X_train_num[col])
                X_test_num[col] = pd.to_numeric(X_test_num[col])
            except:
                print(f"  - Failed to convert {col}, dropping column")
                X_train_num = X_train_num.drop(columns=[col])
                X_test_num = X_test_num.drop(columns=[col])
    
    # Scale only numerical features
    print("📏 Scaling numerical features...")
    scaler = StandardScaler()
    
    try:
        X_train_scaled_num = scaler.fit_transform(X_train_num)
        X_test_scaled_num = scaler.transform(X_test_num)
    except Exception as e:
        print(f"⚠️ Scaling error: {e}")
        print("Trying alternate approach...")
        # Convert to numpy array first
        X_train_np = X_train_num.to_numpy()
        X_test_np = X_test_num.to_numpy()
        X_train_scaled_num = scaler.fit_transform(X_train_np)
        X_test_scaled_num = scaler.transform(X_test_np)print("🔄 Processing categorical features...")
    
    # Print columns to diagnose
    print("\n🔍 DEBUG: First few columns of X_train:")
    print(list(X_train.columns)[:10])
    
    # Check for non-numeric values in the dataframe
    print("\n🔍 DEBUG: Checking for non-numeric values...")
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            print(f"  - Column '{col}' has object dtype with sample: {X_train[col].iloc[0]}")
    
    # Drop any object columns as they're likely categorical and causing issues
    object_columns = X_train.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        print(f"\n⚠️ Dropping problematic object columns: {object_columns}")
        X_train = X_train.drop(columns=object_columns)
        X_test = X_test.drop(columns=object_columns)
    
    # Define potential categorical columns that should be treated as numeric
    categorical_columns = ['is_xauusdm', 'is_eurusdm', 'timeframe_m5', 'timeframe_m15', 
                          'timeframe_m30', 'timeframe_h1', 'session_asian', 'session_london', 
                          'session_ny', 'session_overlap']
    
    # Find which categorical columns actually exist in our data
    existing_categorical = [col for col in categorical_columns if col in X_train.columns]
    numerical_columns = [col for col in X_train.columns if col not in existing_categorical]
    
    print(f"  - Categorical features: {len(existing_categorical)}")
    print(f"  - Numerical features: {len(numerical_columns)}")ning Script
------------------------------
This script trains and saves updated models with the enhanced timeframe features.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import the enhanced trading system
from enhanced_adaptive_ml_trading_system import EnhancedAdaptiveMLTradingSystem

def print_banner():
    """Print a nice banner for the training script"""
    print("\n" + "="*80)
    print("🚀 ENHANCED MODEL TRAINING WITH NEW TIMEFRAME FEATURES".center(80))
    print("="*80)
    print("Training models with enhanced timeframe features and calibration")
    print("This will create new production models that leverage the new features")
    print("-"*80)

def train_models():
    """Train models with enhanced timeframe features"""
    
    print("🔧 Initializing Enhanced Trading System...")
    system = EnhancedAdaptiveMLTradingSystem()
    
    print("📊 Loading dataset...")
    system.load_mega_dataset()
    
    print(f"📋 Dataset loaded: {len(system.mega_dataset):,} records")
    
    # Engineer features - this will now include the enhanced timeframe features
    print("🧪 Engineering features with enhanced timeframe features...")
    X, feature_columns, y = system.engineer_leak_free_features()
    
    # DEBUG: Print the types of all columns in X
    print("\n🔬 DEBUG: DataFrame column types:")
    print(X.dtypes)
    
    # Handle problematic columns early
    print("\n🔧 Converting/removing problematic columns...")
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                # Try to convert to numeric
                X[col] = pd.to_numeric(X[col])
                print(f"  - Converted '{col}' to numeric")
            except:
                # If we can't convert, we need to encode or drop
                unique_values = X[col].nunique()
                if unique_values <= 10:  # Assume categorical if few unique values
                    print(f"  - One-hot encoding '{col}' with {unique_values} values")
                    dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                    X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                else:
                    print(f"  - Dropping '{col}' with {unique_values} values")
                    X = X.drop(columns=[col])
                    
    # Check that our new enhanced timeframe features are included
    timeframe_features = [col for col in feature_columns if any(tf in col for tf in ['M5', 'M15', 'M30', 'H1']) 
                          and any(suffix in col for suffix in ['_volatility_ratio', '_momentum_strength', '_session_performance'])]
    
    print(f"✅ Enhanced timeframe features: {len(timeframe_features)}")
    for feature in timeframe_features:
        if feature in X.columns:
            print(f"  - {feature} (in dataframe)")
        else:
            print(f"  - {feature} (not in dataframe)")
    
    # Split the data
    print("🔪 Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, shuffle=True
    )
    
    # Separate numerical and categorical features
    print("� Processing categorical features...")
    categorical_columns = ['is_xauusdm', 'is_eurusdm', 'timeframe_m5', 'timeframe_m15', 
                          'timeframe_m30', 'timeframe_h1', 'session_asian', 'session_london', 
                          'session_ny', 'session_overlap']
    
    # Find which categorical columns actually exist in our data
    existing_categorical = [col for col in categorical_columns if col in X_train.columns]
    numerical_columns = [col for col in X_train.columns if col not in existing_categorical]
    
    print(f"  - Categorical features: {len(existing_categorical)}")
    print(f"  - Numerical features: {len(numerical_columns)}")
    
    # Create separate dataframes for numerical and categorical features
    X_train_num = X_train[numerical_columns]
    X_test_num = X_test[numerical_columns]
    
    X_train_cat = X_train[existing_categorical] if existing_categorical else None
    X_test_cat = X_test[existing_categorical] if existing_categorical else None
    
    # Scale only numerical features
    print("�📏 Scaling numerical features...")
    scaler = StandardScaler()
    X_train_scaled_num = scaler.fit_transform(X_train_num)
    X_test_scaled_num = scaler.transform(X_test_num)
    
    # Convert scaled arrays back to dataframes
    X_train_scaled_num_df = pd.DataFrame(X_train_scaled_num, columns=numerical_columns)
    X_test_scaled_num_df = pd.DataFrame(X_test_scaled_num, columns=numerical_columns)
    
    # Combine numerical scaled features with categorical features
    if X_train_cat is not None:
        X_train_scaled = pd.concat([X_train_scaled_num_df, X_train_cat.reset_index(drop=True)], axis=1)
        X_test_scaled = pd.concat([X_test_scaled_num_df, X_test_cat.reset_index(drop=True)], axis=1)
    else:
        X_train_scaled = X_train_scaled_num_df
        X_test_scaled = X_test_scaled_num_df
    
    # Train main ensemble model
    print("🧠 Training timeframe_ensemble model...")
    timeframe_ensemble = RandomForestClassifier(
        n_estimators=150, 
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    # Convert to numpy array if it's a DataFrame
    X_train_scaled_array = X_train_scaled.values if hasattr(X_train_scaled, 'values') else X_train_scaled
    timeframe_ensemble.fit(X_train_scaled_array, y_train)
    
    # Train currency specialist models
    print("🧠 Training currency specialist models...")
    gold_specialist = RandomForestClassifier(
        n_estimators=100, 
        max_depth=4,
        random_state=43,
        n_jobs=-1
    )
    
    if 'is_xauusdm' in X_train.columns:
        gold_mask_train = (X_train['is_xauusdm'] == 1).reset_index(drop=True)
        # Align mask length with X_train_scaled
        if len(gold_mask_train) > len(X_train_scaled):
            gold_mask_train = gold_mask_train[:len(X_train_scaled)]
        elif len(gold_mask_train) < len(X_train_scaled):
            gold_mask_train = pd.Series(gold_mask_train, index=X_train_scaled.index)
        if sum(gold_mask_train) > 100:
            X_train_gold = X_train_scaled[gold_mask_train].values if hasattr(X_train_scaled, 'values') else X_train_scaled[gold_mask_train]
            y_train_gold = y_train.reset_index(drop=True)[gold_mask_train]
            gold_specialist.fit(X_train_gold, y_train_gold)
        else:
            print("⚠️ Not enough gold data to train specialist model")
            gold_specialist = None
    else:
        print("⚠️ 'is_xauusdm' column not found in dataset")
        gold_specialist = None

    eur_specialist = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=44,
        n_jobs=-1
    )
    if 'is_eurusdm' in X_train.columns:
        eur_mask_train = (X_train['is_eurusdm'] == 1).reset_index(drop=True)
        if len(eur_mask_train) > len(X_train_scaled):
            eur_mask_train = eur_mask_train[:len(X_train_scaled)]
        elif len(eur_mask_train) < len(X_train_scaled):
            eur_mask_train = pd.Series(eur_mask_train, index=X_train_scaled.index)
        if sum(eur_mask_train) > 100:
            X_train_eur = X_train_scaled[eur_mask_train].values if hasattr(X_train_scaled, 'values') else X_train_scaled[eur_mask_train]
            y_train_eur = y_train.reset_index(drop=True)[eur_mask_train]
            eur_specialist.fit(X_train_eur, y_train_eur)
        else:
            print("⚠️ Not enough EUR data to train specialist model")
            eur_specialist = None
    else:
        print("⚠️ 'is_eurusdm' column not found in dataset")
        eur_specialist = None
    
    # Initialize online learning model
    print("🧠 Initializing adaptive_learner model...")
    adaptive_learner = SGDClassifier(
        loss='log_loss',
        learning_rate='optimal',
        eta0=0.01,
        random_state=42,
        warm_start=True
    )
    
    # Initialize it with a small sample to avoid cold start
    # Convert to numpy array for partial_fit
    X_sample = X_train_scaled[:100].values if hasattr(X_train_scaled, 'values') else X_train_scaled[:100]
    adaptive_learner.partial_fit(
        X_sample,
        y_train[:100], 
        classes=np.array([0, 1])
    )
    
    # Evaluate models
    print("\n📊 EVALUATING MODELS\n" + "-"*40)
    
    # Evaluate timeframe_ensemble
    print("⚙️ Timeframe Ensemble Model:")
    # Convert to numpy array for predict
    X_test_array = X_test_scaled.values if hasattr(X_test_scaled, 'values') else X_test_scaled
    y_pred = timeframe_ensemble.predict(X_test_array)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
    
    # Get feature importance for timeframe_ensemble
    if hasattr(timeframe_ensemble, 'feature_importances_'):
        importances = timeframe_ensemble.feature_importances_
        
        # Get the feature names in the correct order
        if hasattr(X_train_scaled, 'columns'):
            # If we're using a DataFrame
            feature_names = list(X_train_scaled.columns)
        else:
            # If we're using the original feature_columns
            feature_names = feature_columns
        
        # Create a feature importance dictionary
        feature_importance_dict = {feature_names[i]: importances[i] for i in range(len(feature_names))}
        
        # Sort by importance
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🔍 TOP 10 IMPORTANT FEATURES:")
        for i, (feature, importance) in enumerate(sorted_features[:10]):
            print(f"  {i+1}. {feature}: {importance:.6f}")
        
        # Check specifically for our new enhanced timeframe features
        print("\n🔍 ENHANCED TIMEFRAME FEATURE IMPORTANCE:")
        for feature in timeframe_features:
            if feature in feature_importance_dict:
                print(f"  {feature}: {feature_importance_dict[feature]:.6f}")
    
    # Save the models
    print("\n💾 SAVING MODELS...")
    
    models = {
        'timeframe_ensemble': timeframe_ensemble,
        'adaptive_learner': adaptive_learner,
        'gold_specialist': gold_specialist,
        'EURUSDm_specialist': eur_specialist
    }
    
    model_package = {
        'models': models,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'training_timestamp': datetime.now(),
        'validation_results': {
            'timeframe_ensemble': {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0)
            }
        },
        'enhanced_timeframe_features': True,
        'calibration_enabled': True
    }
    
    # Save the new models
    output_path = 'production_models_enhanced.pkl'
    joblib.dump(model_package, output_path)
    print(f"✅ Models saved to {output_path}")
    
    # Also save as latest
    joblib.dump(model_package, 'production_models_latest.pkl')
    print(f"✅ Models saved to production_models_latest.pkl")
    
    print("\n🎉 TRAINING COMPLETE!")
    print("Your models now include enhanced timeframe features and will use calibration.")
    print("To use these models, just run your trading system as normal - it will")
    print("automatically use the enhanced features and calibration.")

if __name__ == "__main__":
    print_banner()
    train_models()
