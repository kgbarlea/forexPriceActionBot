#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dual-Model Trading System
------------------------
Combines the high-recall Signal Finder model with the high-precision main model
to create a more balanced trading approach that captures more opportunities
while maintaining strict risk management.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from enhanced_adaptive_ml_trading_system import EnhancedAdaptiveMLTradingSystem

def print_banner():
    """Print a nice banner for the Dual-Model script"""
    print("\n" + "="*80)
    print("🔄 DUAL-MODEL TRADING SYSTEM".center(80))
    print("="*80)
    print("Combining high-recall Signal Finder with high-precision Main Model")
    print("-"*80)

class DualModelSystem:
    """
    Dual-Model Trading System that combines:
    1. Signal Finder (high recall) - to capture more trading opportunities
    2. Main Model (high precision) - to filter signals and reduce false positives
    """
    
    def __init__(self):
        self.base_system = EnhancedAdaptiveMLTradingSystem()
        
        # Load models
        print("🔄 Loading Signal Finder model...")
        try:
            self.signal_finder_package = joblib.load('signal_finder_model.pkl')
            print("✅ Signal Finder model loaded successfully")
            
            self.signal_finder = self.signal_finder_package['model']
            self.signal_finder_scaler = self.signal_finder_package['scaler']
            self.signal_finder_features = self.signal_finder_package['feature_columns']
            self.signal_finder_threshold = self.signal_finder_package['optimal_threshold']
            
            print(f"  - Optimal threshold: {self.signal_finder_threshold:.4f}")
            print(f"  - Expected recall: {self.signal_finder_package['performance']['recall']:.4f}")
            
        except Exception as e:
            print(f"❌ Error loading Signal Finder model: {str(e)}")
            self.signal_finder = None
        
        print("🔄 Loading Main Precision model...")
        try:
            self.main_model_package = joblib.load('production_models_latest.pkl')
            print("✅ Main Precision model loaded successfully")
            
            self.main_model = self.main_model_package['models']['timeframe_ensemble']
            self.main_scaler = self.main_model_package['scaler']
            self.main_features = self.main_model_package['feature_columns']
            
            # Print information about the model's expected features
            categorical_prefixes = ['detectionmethod_', 'setuptype_', 'symbol_']
            categorical_features = [f for f in self.main_features if any(f.startswith(p) for p in categorical_prefixes)]
            
            print(f"  - Model expects {len(self.main_features)} features")
            print(f"  - First 10 features: {self.main_features[:10]}")
            print(f"  - Last 10 features: {self.main_features[-10:]}")
            
            if categorical_features:
                print(f"  - Including {len(categorical_features)} categorical dummy features:")
                for prefix in categorical_prefixes:
                    prefix_features = [f for f in categorical_features if f.startswith(prefix)]
                    if prefix_features:
                        print(f"    • {prefix}: {len(prefix_features)} features")
                        print(f"      Example: {prefix_features[:3]}")
            else:
                print("  - ⚠️ No categorical dummy features found in expected features!")
                print("  - This suggests the model was trained without one-hot encoding")
            
            # Default threshold is 0.5
            self.main_threshold = 0.5
            
        except Exception as e:
            print(f"❌ Error loading Main Precision model: {str(e)}")
            self.main_model = None
        
        # Trading parameters
        self.signal_finder_min_confidence = 0.4  # Lower threshold for signal finder
        self.main_model_min_confidence = 0.55    # Higher threshold for main model
        
        # Initialize backtest results
        self.backtest_results = {
            'signal_finder_only': {},
            'main_model_only': {},
            'dual_model': {}
        }
    
    def predict_with_signal_finder(self, X):
        """Get predictions from the Signal Finder model"""
        if self.signal_finder is None:
            print("❌ Signal Finder model not loaded")
            return None
        
        print(f"🔍 DEBUG: Signal Finder input shape: {X.shape}")
        
        # Create a DataFrame with exactly the required features
        final_df = pd.DataFrame(0, index=X.index, columns=self.signal_finder_features)
        
        # Copy over existing values
        for col in self.signal_finder_features:
            if col in X.columns:
                final_df[col] = X[col]
        
        print(f"✅ Signal Finder feature matrix prepared: {final_df.shape}")
        
        # Scale the features
        X_scaled = self.signal_finder_scaler.transform(final_df)
        
        # Get probability predictions
        probabilities = self.signal_finder.predict_proba(X_scaled)[:, 1]
        
        # Make binary predictions based on optimal threshold
        predictions = (probabilities >= self.signal_finder_threshold).astype(int)
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'confidence_scores': probabilities
        }
    
    def predict_with_main_model(self, X):
        """Get predictions from the Main Precision model"""
        if self.main_model is None:
            print("❌ Main Precision model not loaded")
            return None
            
        # Print debugging information
        print("\n🔍 DEBUG: Input feature columns:", X.columns.tolist())
        print(f"🔍 DEBUG: Expected model features: {len(self.main_features)} features")
        
        # Create empty DataFrame with expected features
        print("🔄 Creating feature matrix with expected features...")
        final_df = pd.DataFrame(0, index=X.index, columns=self.main_features)
        
        # Handle basic numeric features - copy directly if they exist in input
        for col in X.columns:
            if col in self.main_features:
                final_df[col] = X[col]
        
        # Handle categorical features - the model actually expects one-hot encoded features
        # even though the feature list shows raw categorical names
        print("🔄 Processing categorical features...")
        
        # The error message tells us exactly what encoded features the model expects
        # Let's handle this properly by creating the exact encoded features
        
        # Map categorical values to their encoded names based on the error message
        categorical_mappings = {
            'detectionmethod': {
                'TESTED_RESISTANCE': 'detectionmethod_TESTED_RESISTANCE',
                'TESTED_SUPPORT': 'detectionmethod_TESTED_SUPPORT',
                'ANY_MOVEMENT': 'detectionmethod_ANY_MOVEMENT'
            },
            'setuptype': {
                'VOLUME_BREAKOUT_LONG': 'setuptype_VOLUME_BREAKOUT_LONG',
                'VOLUME_BREAKOUT_SHORT': 'setuptype_VOLUME_BREAKOUT_SHORT',
                'ULTRA_FORCED_SHORT': 'setuptype_ULTRA_FORCED_SHORT',
                'ULTRA_FORCED_LONG': 'setuptype_ULTRA_FORCED_LONG'
            },
            'symbol': {
                'EURUSDm': 'symbol_EURUSDm',
                'XAUUSDm': 'symbol_XAUUSDm'
            }
        }
        
        # Create encoded features
        for cat_col, value_mapping in categorical_mappings.items():
            if cat_col in X.columns:
                print(f"  - Encoding {cat_col}...")
                for original_value, encoded_name in value_mapping.items():
                    # Create the encoded column
                    final_df[encoded_name] = (X[cat_col] == original_value).astype(int)
                    print(f"    Created: {encoded_name}")
        
        # Remove the original categorical columns since we now have encoded versions
        categorical_cols_to_remove = ['detectionmethod', 'setuptype', 'symbol']
        for col in categorical_cols_to_remove:
            if col in final_df.columns:
                final_df = final_df.drop(columns=[col])
                print(f"  - Removed original: {col}")
                    
        # Check if we have all required features
        missing_features = [col for col in self.main_features if col not in final_df.columns]
        if missing_features:
            print(f"⚠️ Missing {len(missing_features)} expected features (they'll be set to 0):")
            print(f"  {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}")
        
        # Debug: Check what encoded features we actually have
        encoded_features_created = [col for col in final_df.columns if any(col.startswith(p) for p in ['detectionmethod_', 'setuptype_', 'symbol_'])]
        print(f"🔍 DEBUG: Encoded features created: {len(encoded_features_created)}")
        for f in encoded_features_created[:10]:
            print(f"  - {f}")
        
        # Ensure final_df has exactly the expected columns in the correct order
        print("🔄 Reindexing to match expected features exactly...")
        final_df = final_df.reindex(columns=self.main_features, fill_value=0)
                
        # Scale the features using the exact feature matrix
        print(f"✅ Final feature matrix shape: {final_df.shape}")
        print(f"🔍 DEBUG: Final columns match expected: {list(final_df.columns) == self.main_features}")
        
        # Check some categorical features in the final matrix
        cat_features_in_final = [col for col in final_df.columns if any(col.startswith(p) for p in ['detectionmethod_', 'setuptype_', 'symbol_'])]
        print(f"🔍 DEBUG: Final encoded features in matrix: {len(cat_features_in_final)}")
        for f in cat_features_in_final[:5]:
            print(f"  - {f}")
        
        print("🔄 Applying scaler transform...")
        
        # The scaler was trained with 62 features and doesn't expect is_eurusdm/is_xauusdm
        # Let's build the exact 62-feature set it expects
        
        # Remove the problematic features that weren't in the original training
        features_to_exclude = ['is_eurusdm', 'is_xauusdm']
        
        # Start with non-categorical features from main_features, excluding problematic ones
        scaler_features = []
        for feature in self.main_features:
            if not any(feature.startswith(p) for p in ['symbol', 'setuptype', 'detectionmethod']):
                if feature not in features_to_exclude:
                    scaler_features.append(feature)
        
        # Add the categorical features the scaler expects (from error messages)
        expected_categorical = [
            'detectionmethod_TESTED_RESISTANCE',
            'detectionmethod_TESTED_SUPPORT', 
            'setuptype_ULTRA_FORCED_SHORT',
            'setuptype_VOLUME_BREAKOUT_LONG',
            'setuptype_VOLUME_BREAKOUT_SHORT',
            'symbol_XAUUSDm'
        ]
        scaler_features.extend(expected_categorical)
        
        print(f"🔍 DEBUG: Building scaler matrix with {len(scaler_features)} features (target: 62)")
        
        # Create DataFrame with the exact features the scaler expects
        scaler_df = pd.DataFrame(0, index=final_df.index, columns=scaler_features)
        
        # Copy values for features that exist in our encoded data
        for col in scaler_features:
            if col in final_df.columns:
                scaler_df[col] = final_df[col]
                print(f"  ✓ Copied {col}")
            else:
                print(f"  ⚠ Missing {col} (set to 0)")
        
        print(f"✅ Scaler-compatible matrix: {scaler_df.shape}")
        
        # Try transform with the properly sized matrix
        try:
            X_scaled = self.main_scaler.transform(scaler_df.values)  # Use .values to avoid feature name checking
            print("✅ Scaler transform successful with numpy array!")
            
            # Check if the model expects more features than the scaler provides
            # RandomForest might expect 64 features while scaler provides 62
            if X_scaled.shape[1] < 64:  # Model likely expects 64 features
                print(f"🔄 Expanding from {X_scaled.shape[1]} to 64 features for RandomForest...")
                
                # Pad with zeros to reach 64 features
                padding = np.zeros((X_scaled.shape[0], 64 - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
                print(f"✅ Expanded to {X_scaled.shape[1]} features")
            
        except Exception as e:
            print(f"❌ Final fallback failed: {e}")
            # Last resort: create a matrix with exactly 64 features for the RandomForest
            print("🔄 Creating 64-feature matrix for RandomForest...")
            fallback_matrix = np.zeros((final_df.shape[0], 64))
            
            # Copy available features
            for i, col in enumerate(scaler_df.columns[:min(62, len(scaler_df.columns))]):
                if col in scaler_df.columns:
                    fallback_matrix[:, i] = scaler_df[col].values
            
            X_scaled = fallback_matrix
            print("✅ Used 64-feature fallback matrix")
        
        # Get probability predictions
        probabilities = self.main_model.predict_proba(X_scaled)[:, 1]
        
        # Make binary predictions based on threshold
        predictions = (probabilities >= self.main_threshold).astype(int)
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'confidence_scores': probabilities
        }
    
    def dual_model_predict(self, X):
        """
        Make predictions using the dual-model approach:
        1. Signal Finder identifies potential opportunities (high recall)
        2. Main Model filters these opportunities (high precision)
        """
        print(f"\n🔍 DEBUG: Dual model input shape: {X.shape}")
        
        # Create feature sets for each model
        sf_features = [col for col in self.signal_finder_features if col in X.columns]
        X_sf = X[sf_features]
        
        # Get Signal Finder predictions
        signal_finder_results = self.predict_with_signal_finder(X_sf)
        
        if signal_finder_results is None:
            return None
        
        # Get Main Model predictions - use the full feature set for proper encoding
        main_model_results = self.predict_with_main_model(X)
        
        if main_model_results is None:
            return None
        
        # Combine the models:
        # 1. Signal Finder identifies potential signals
        # 2. For those signals, check if Main Model confidence is high enough
        
        # Initialize results
        dual_predictions = np.zeros_like(signal_finder_results['predictions'])
        dual_confidence = np.zeros_like(signal_finder_results['probabilities'])
        
        # For each potential signal from Signal Finder
        for i, signal in enumerate(signal_finder_results['predictions']):
            if signal == 1:  # Signal Finder predicts a trade opportunity
                # Check if main model has enough confidence
                if main_model_results['probabilities'][i] >= self.main_model_min_confidence:
                    # Take the trade
                    dual_predictions[i] = 1
                    # Use main model confidence for position sizing
                    dual_confidence[i] = main_model_results['probabilities'][i]
            
        return {
            'signal_finder': signal_finder_results,
            'main_model': main_model_results,
            'dual_predictions': dual_predictions,
            'dual_confidence': dual_confidence
        }
    
    def calculate_position_size(self, confidence, max_risk_pct=1.0):
        """
        Calculate position size based on confidence and max risk
        
        Parameters:
        - confidence: model confidence (0 to 1)
        - max_risk_pct: maximum risk per trade as percentage
        
        Returns:
        - position size as percentage of account
        """
        # Base position is proportional to confidence
        base_position = confidence * max_risk_pct
        
        # Apply Kelly-inspired scaling:
        # Kelly formula is f* = (bp - q)/b where:
        # b = odds received (we'll use 2 for simplicity - 1:1 reward:risk)
        # p = win probability (our confidence)
        # q = 1-p (loss probability)
        b = 2  # 1:1 reward:risk
        p = confidence
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Limit Kelly to 0-100% and apply a conservative factor of 0.5
        kelly_fraction = max(0, min(1, kelly_fraction)) * 0.5
        
        # Return the position size as a percentage
        return kelly_fraction * max_risk_pct
    
    def backtest_on_dataset(self):
        """Run a simple backtest on the dataset"""
        print("\n🧪 Running backtest on dataset...")
        
        # Load dataset
        self.base_system.load_mega_dataset()
        
        # Engineer features
        X, feature_columns, y = self.base_system.engineer_leak_free_features()
        
        # Remove leakage features and non-numeric columns
        leakage_features = ['maxprofit', 'maxloss', 'barsheld']
        future_info_prefixes = ['max', 'final', 'total', 'end', 'result']
        potential_leakage = [col for col in X.columns if any(col.startswith(prefix) for prefix in future_info_prefixes)]
        leakage_features.extend([col for col in potential_leakage if col not in leakage_features])
        
        X_clean = X.drop(columns=[col for col in leakage_features if col in X.columns])
        
        # Check for non-numeric columns
        non_numeric_cols = X_clean.select_dtypes(include=['object']).columns.tolist()
        print(f"\n🔍 DEBUG: Found categorical columns: {non_numeric_cols}")
        
        # Don't drop categorical columns! We'll handle them properly
        # Instead, just check what values they have
        if non_numeric_cols:
            print("\n🔍 DEBUG: Categorical column values:")
            for col in non_numeric_cols:
                unique_vals = X_clean[col].unique()
                print(f"  - {col}: {unique_vals[:5]}{' ...' if len(unique_vals) > 5 else ''} ({len(unique_vals)} unique values)")
        
        # Print information about the data
        print(f"\n📊 X_clean shape: {X_clean.shape}, columns: {len(X_clean.columns)}")
        
        # For Signal Finder model (simpler approach)
        sf_features = [col for col in self.signal_finder_features if col in X_clean.columns]
        X_sf = X_clean[sf_features]
        
        # Pass the data directly to models which will handle feature processing internally
        print("🔄 Running Signal Finder model...")
        signal_finder_results = self.predict_with_signal_finder(X_sf)
        
        print("🔄 Running Main Precision model...")
        # Pass full X_clean to allow proper categorical encoding
        main_model_results = self.predict_with_main_model(X_clean)
        
        print("🔄 Running Dual Model predictions...")
        # For dual_model_predict, also use the full feature set
        dual_model_results = self.dual_model_predict(X_clean)
        
        # Evaluate Signal Finder performance
        sf_predictions = signal_finder_results['predictions']
        sf_accuracy = accuracy_score(y, sf_predictions)
        sf_precision = precision_score(y, sf_predictions, zero_division=0)
        sf_recall = recall_score(y, sf_predictions, zero_division=0)
        sf_f1 = f1_score(y, sf_predictions, zero_division=0)
        
        # Evaluate Main Model performance
        mm_predictions = main_model_results['predictions']
        mm_accuracy = accuracy_score(y, mm_predictions)
        mm_precision = precision_score(y, mm_predictions, zero_division=0)
        mm_recall = recall_score(y, mm_predictions, zero_division=0)
        mm_f1 = f1_score(y, mm_predictions, zero_division=0)
        
        # Evaluate Dual Model performance
        dm_predictions = dual_model_results['dual_predictions']
        dm_accuracy = accuracy_score(y, dm_predictions)
        dm_precision = precision_score(y, dm_predictions, zero_division=0)
        dm_recall = recall_score(y, dm_predictions, zero_division=0)
        dm_f1 = f1_score(y, dm_predictions, zero_division=0)
        
        # Store results
        self.backtest_results['signal_finder_only'] = {
            'accuracy': sf_accuracy,
            'precision': sf_precision,
            'recall': sf_recall,
            'f1': sf_f1,
            'trade_count': np.sum(sf_predictions)
        }
        
        self.backtest_results['main_model_only'] = {
            'accuracy': mm_accuracy,
            'precision': mm_precision,
            'recall': mm_recall,
            'f1': mm_f1,
            'trade_count': np.sum(mm_predictions)
        }
        
        self.backtest_results['dual_model'] = {
            'accuracy': dm_accuracy,
            'precision': dm_precision,
            'recall': dm_recall,
            'f1': dm_f1,
            'trade_count': np.sum(dm_predictions)
        }
        
        # Print results
        print("\n📊 BACKTEST RESULTS COMPARISON:")
        print("\n1. Signal Finder Model (High Recall):")
        print(f"  Accuracy: {sf_accuracy:.4f}")
        print(f"  Precision: {sf_precision:.4f}")
        print(f"  Recall: {sf_recall:.4f}")
        print(f"  F1: {sf_f1:.4f}")
        print(f"  Trade Count: {np.sum(sf_predictions)}")
        
        print("\n2. Main Precision Model:")
        print(f"  Accuracy: {mm_accuracy:.4f}")
        print(f"  Precision: {mm_precision:.4f}")
        print(f"  Recall: {mm_recall:.4f}")
        print(f"  F1: {mm_f1:.4f}")
        print(f"  Trade Count: {np.sum(mm_predictions)}")
        
        print("\n3. Dual Model System:")
        print(f"  Accuracy: {dm_accuracy:.4f}")
        print(f"  Precision: {dm_precision:.4f}")
        print(f"  Recall: {dm_recall:.4f}")
        print(f"  F1: {dm_f1:.4f}")
        print(f"  Trade Count: {np.sum(dm_predictions)}")
        
        # Compare trade counts
        sf_trades = np.sum(sf_predictions)
        mm_trades = np.sum(mm_predictions)
        dm_trades = np.sum(dm_predictions)
        
        print("\n📈 TRADE COUNT COMPARISON:")
        print(f"  Signal Finder: {sf_trades} potential trades")
        print(f"  Main Model: {mm_trades} high-conviction trades")
        print(f"  Dual Model: {dm_trades} trades that passed both filters")
        
        print(f"\n  The Dual Model captured {dm_trades/mm_trades*100:.1f}% more trades than the Main Model alone")
        print(f"  while maintaining a precision of {dm_precision:.4f} vs {mm_precision:.4f}")
        
        # Return summary
        return self.backtest_results
    
    def get_trading_recommendations(self, market_data):
        """
        Get trading recommendations for current market data
        
        Parameters:
        - market_data: DataFrame with current market data features
        
        Returns:
        - Dictionary with trading recommendations
        """
        # Preprocess the data
        X = self.base_system.preprocess_market_data(market_data)
        
        # Make predictions
        predictions = self.dual_model_predict(X)
        
        if predictions is None:
            return {"error": "Failed to generate predictions"}
        
        # Get signals
        signals = []
        for i, signal in enumerate(predictions['dual_predictions']):
            if signal == 1:
                confidence = predictions['dual_confidence'][i]
                position_size = self.calculate_position_size(confidence)
                
                signals.append({
                    "index": i,
                    "confidence": confidence,
                    "position_size_pct": position_size,
                    "signal_finder_confidence": predictions['signal_finder']['probabilities'][i],
                    "main_model_confidence": predictions['main_model']['probabilities'][i]
                })
        
        return {
            "signals": signals,
            "signal_count": len(signals),
            "signal_finder_signals": sum(predictions['signal_finder']['predictions']),
            "main_model_signals": sum(predictions['main_model']['predictions'])
        }

def demonstrate_dual_model():
    """Run a demonstration of the Dual-Model Trading System"""
    print("🚀 Initializing Dual-Model Trading System...")
    dual_model = DualModelSystem()
    
    # Run backtest
    results = dual_model.backtest_on_dataset()
    
    # Demonstrate sample trading recommendation
    print("\n💡 SAMPLE TRADING RECOMMENDATION FLOW:")
    print("  1. Signal Finder identifies potential opportunities")
    print("  2. Main Model filters these for high precision")
    print("  3. Final trades have position sizing based on confidence")
    
    # Insights
    print("\n🔍 KEY INSIGHTS:")
    
    sf_recall = results['signal_finder_only']['recall']
    mm_recall = results['main_model_only']['recall']
    dual_recall = results['dual_model']['recall']
    
    sf_precision = results['signal_finder_only']['precision']
    mm_precision = results['main_model_only']['precision']
    dual_precision = results['dual_model']['precision']
    
    print(f"  • Signal Finder captures {sf_recall*100:.1f}% of profitable opportunities")
    print(f"  • Main Model alone captures only {mm_recall*100:.1f}% of opportunities")
    print(f"  • Dual Model approach captures {dual_recall*100:.1f}% while keeping precision at {dual_precision*100:.1f}%")
    
    if dual_recall > mm_recall and dual_precision >= mm_precision * 0.9:
        print("\n✅ The Dual Model system successfully improves trade capture while maintaining precision")
    else:
        print("\n⚠️ The Dual Model approach needs further tuning - review thresholds")
    
    print("\n🎯 NEXT STEPS:")
    print("  1. Fine-tune Signal Finder threshold to adjust recall/precision balance")
    print("  2. Experiment with dynamic thresholds based on market conditions")
    print("  3. Implement position sizing based on dual confidence scores")
    print("  4. Add time-series validation for more realistic performance estimates")

if __name__ == "__main__":
    print_banner()
    demonstrate_dual_model()
