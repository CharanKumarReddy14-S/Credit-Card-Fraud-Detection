"""
Model Training Module for Credit Card Fraud Detection
Trains multiple models and saves the best performing one
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)

# Preprocessing
from data_preprocessing import preprocess_pipeline


class ModelTrainer:
    """
    Train and evaluate multiple fraud detection models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """
        Train Logistic Regression baseline model
        """
        print("\n" + "="*60)
        print("Training Logistic Regression (Baseline)")
        print("="*60)
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_val, y_val, "Logistic Regression")
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = metrics
        
        return model, metrics
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """
        Train Random Forest model
        """
        print("\n" + "="*60)
        print("Training Random Forest")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_val, y_val, "Random Forest")
        
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        
        return model, metrics
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """
        Train XGBoost model (primary model)
        """
        print("\n" + "="*60)
        print("Training XGBoost (Primary Model)")
        print("="*60)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='auc'
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )
        
        # Evaluate
        metrics = self.evaluate_model(model, X_val, y_val, "XGBoost")
        
        self.models['xgboost'] = model
        self.results['xgboost'] = metrics
        
        return model, metrics
    
    def build_neural_network(self, input_dim):
        """
        Build deep neural network architecture
        """
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val):
        """
        Train Deep Neural Network
        """
        print("\n" + "="*60)
        print("Training Deep Neural Network")
        print("="*60)
        
        # Build model
        model = self.build_neural_network(X_train.shape[1])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=10,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=256,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        metrics = self.evaluate_neural_network(model, X_val, y_val)
        
        self.models['neural_network'] = model
        self.results['neural_network'] = metrics
        
        return model, metrics
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Comprehensive model evaluation
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # False Positive Rate (critical for fraud detection)
        fpr = fp / (fp + tn)
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn:6d}  FP: {fp:6d}")
        print(f"FN: {fn:6d}  TP: {tp:6d}")
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'false_positive_rate': float(fpr),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
        
        return metrics
    
    def evaluate_neural_network(self, model, X_test, y_test):
        """
        Evaluate neural network model
        """
        # Predictions
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
        
        print(f"\nDeep Neural Network Performance:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")
        print(f"FPR:       {fpr:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"TN: {tn:6d}  FP: {fp:6d}")
        print(f"FN: {fn:6d}  TP: {tp:6d}")
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'false_positive_rate': float(fpr),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp),
                'fn': int(fn), 'tp': int(tp)
            }
        }
        
        return metrics
    
    def compare_models(self):
        """
        Compare all trained models
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'false_positive_rate']]
        
        print(comparison_df.to_string())
        
        # Find best model based on F1-score and ROC-AUC
        best_model_name = comparison_df['f1_score'].idxmax()
        
        print(f"\n🏆 Best Model: {best_model_name.upper()}")
        print(f"   F1-Score: {self.results[best_model_name]['f1_score']:.4f}")
        print(f"   ROC-AUC:  {self.results[best_model_name]['roc_auc']:.4f}")
        
        return best_model_name
    
    def save_models(self, best_model_name):
        """
        Save all models and results
        """
        os.makedirs('models', exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            if name == 'neural_network':
                model.save(f'models/{name}.keras')
            else:
                joblib.dump(model, f'models/{name}.pkl')
            print(f"✅ Saved {name}")
        
        # Save best model separately for easy access
        if best_model_name == 'neural_network':
            self.models[best_model_name].save('models/best_model.keras')
        else:
            joblib.dump(self.models[best_model_name], 'models/best_model.pkl')
        
        # Save results
        results_data = {
            'models': self.results,
            'best_model': best_model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('models/training_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n✅ All models and results saved to models/")


def train_all_models(data_path='data/creditcard.csv'):
    """
    Complete training pipeline
    """
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - MODEL TRAINING")
    print("="*60)
    
    # Preprocess data
    print("\nStep 1: Data Preprocessing")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_pipeline(data_path)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    print("\n\nStep 2: Model Training")
    
    # 1. Logistic Regression
    trainer.train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # 2. Random Forest
    trainer.train_random_forest(X_train, y_train, X_val, y_val)
    
    # 3. XGBoost
    trainer.train_xgboost(X_train, y_train, X_val, y_val)
    
    # 4. Neural Network
    trainer.train_neural_network(X_train, y_train, X_val, y_val)
    
    # Compare models
    print("\n\nStep 3: Model Comparison")
    best_model_name = trainer.compare_models()
    
    # Final evaluation on test set
    print("\n\nStep 4: Final Evaluation on Test Set")
    best_model = trainer.models[best_model_name]
    
    if best_model_name == 'neural_network':
        final_metrics = trainer.evaluate_neural_network(best_model, X_test, y_test)
    else:
        final_metrics = trainer.evaluate_model(best_model, X_test, y_test, "Best Model (Test Set)")
    
    # Save models
    print("\n\nStep 5: Saving Models")
    trainer.save_models(best_model_name)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    return trainer


if __name__ == "__main__":
    # Check if data exists
    data_path = "data/creditcard.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        print("\n📥 Please download the Credit Card Fraud Detection dataset:")
        print("   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("\n   Place the 'creditcard.csv' file in the data/ directory")
    else:
        trainer = train_all_models(data_path)