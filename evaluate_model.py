"""
Model Evaluation and Visualization Script
Generate detailed performance reports and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_test_data():
    """Load preprocessed test data"""
    test_df = pd.read_csv('data/processed/test.csv')
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    return X_test, y_test


def load_model_and_results():
    """Load trained model and results"""
    # Load best model
    if os.path.exists('models/best_model.pkl'):
        model = joblib.load('models/best_model.pkl')
        model_type = 'sklearn'
    elif os.path.exists('models/best_model.keras'):
        from tensorflow import keras
        model = keras.models.load_model('models/best_model.keras')
        model_type = 'keras'
    else:
        model = joblib.load('models/xgboost.pkl')
        model_type = 'sklearn'
    
    # Load results
    with open('models/training_results.json', 'r') as f:
        results = json.load(f)
    
    return model, model_type, results


def plot_confusion_matrix(y_true, y_pred, save_path='reports/confusion_matrix.png'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrix saved to {save_path}")


def plot_roc_curve(y_true, y_pred_proba, save_path='reports/roc_curve.png'):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ROC curve saved to {save_path}")


def plot_precision_recall_curve(y_true, y_pred_proba, 
                                save_path='reports/precision_recall_curve.png'):
    """Plot Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Precision-Recall curve saved to {save_path}")


def plot_threshold_analysis(y_true, y_pred_proba, 
                           save_path='reports/threshold_analysis.png'):
    """Analyze performance across different thresholds"""
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    precisions = []
    recalls = []
    f1_scores = []
    fprs = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate FPR
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        fprs.append(fpr)
    
    plt.figure(figsize=(12, 8))
    plt.plot(thresholds, precisions, label='Precision', marker='o', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', marker='s', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1-Score', marker='^', linewidth=2)
    plt.plot(thresholds, fprs, label='False Positive Rate', 
             marker='d', linewidth=2, linestyle='--')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Metrics vs Classification Threshold', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Threshold analysis saved to {save_path}")


def plot_feature_importance(model, feature_names, top_n=20,
                           save_path='reports/feature_importance.png'):
    """Plot feature importance (for tree-based models)"""
    if not hasattr(model, 'feature_importances_'):
        print("⚠️ Model doesn't have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Feature importance saved to {save_path}")


def plot_model_comparison(results, save_path='reports/model_comparison.png'):
    """Compare all models side by side"""
    models_data = results.get('models', {})
    
    if not models_data:
        print("⚠️ No model results found")
        return
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    model_names = list(models_data.keys())
    
    # Prepare data
    data = {metric: [] for metric in metrics}
    
    for model_name in model_names:
        for metric in metrics:
            data[metric].append(models_data[model_name].get(metric, 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(model_names, data[metric], color=colors)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison saved to {save_path}")


def generate_classification_report(y_true, y_pred, save_path='reports/classification_report.txt'):
    """Generate and save detailed classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Legitimate', 'Fraudulent'],
                                   digits=4)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("CREDIT CARD FRAUD DETECTION - CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n" + "="*60 + "\n")
    
    print(f"✅ Classification report saved to {save_path}")
    print("\n" + report)


def create_summary_report(results, y_true, y_pred, y_pred_proba,
                         save_path='reports/summary_report.txt'):
    """Create comprehensive summary report"""
    best_model = results.get('best_model', 'Unknown')
    metrics = results['models'].get(best_model, {})
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    report = f"""
{'='*80}
CREDIT CARD FRAUD DETECTION - EVALUATION SUMMARY
{'='*80}

Date: {results.get('timestamp', 'N/A')}
Best Model: {best_model.upper()}

{'='*80}
DATASET STATISTICS
{'='*80}
Total Transactions: {len(y_true):,}
Legitimate Transactions: {(y_true == 0).sum():,} ({(y_true == 0).sum() / len(y_true) * 100:.2f}%)
Fraudulent Transactions: {(y_true == 1).sum():,} ({(y_true == 1).sum() / len(y_true) * 100:.2f}%)

{'='*80}
MODEL PERFORMANCE METRICS
{'='*80}
Accuracy:  {metrics.get('accuracy', 0):.6f}
Precision: {metrics.get('precision', 0):.6f}
Recall:    {metrics.get('recall', 0):.6f}
F1-Score:  {metrics.get('f1_score', 0):.6f}
ROC-AUC:   {metrics.get('roc_auc', 0):.6f}

False Positive Rate: {metrics.get('false_positive_rate', 0):.6f}

{'='*80}
CONFUSION MATRIX
{'='*80}
                    Predicted
                Legitimate  Fraudulent
Actual 
Legitimate      {tn:8d}    {fp:8d}
Fraudulent      {fn:8d}    {tp:8d}

True Negatives:  {tn:,}
False Positives: {fp:,}
False Negatives: {fn:,}
True Positives:  {tp:,}

{'='*80}
BUSINESS IMPACT ANALYSIS
{'='*80}
Correctly Identified Frauds: {tp:,} ({tp / (tp + fn) * 100:.2f}% of all frauds)
Missed Frauds: {fn:,} ({fn / (tp + fn) * 100:.2f}% of all frauds)
False Alarms: {fp:,} ({fp / (fp + tn) * 100:.2f}% of legitimate transactions)

Fraud Detection Rate: {tp / (tp + fn) * 100:.1f}%
Precision (Positive Predictive Value): {tp / (tp + fp) * 100:.1f}%

{'='*80}
END OF REPORT
{'='*80}
"""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"✅ Summary report saved to {save_path}")
    print(report)


def main():
    """Main evaluation pipeline"""
    print("="*80)
    print("CREDIT CARD FRAUD DETECTION - MODEL EVALUATION")
    print("="*80)
    
    # Load data
    print("\n1. Loading test data...")
    X_test, y_test = load_test_data()
    print(f"   Test set: {len(X_test)} samples")
    
    # Load model
    print("\n2. Loading trained model...")
    model, model_type, results = load_model_and_results()
    print(f"   Model type: {model_type}")
    print(f"   Best model: {results.get('best_model', 'Unknown').upper()}")
    
    # Make predictions
    print("\n3. Generating predictions...")
    if model_type == 'keras':
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Generate visualizations
    print("\n4. Creating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    plot_precision_recall_curve(y_test, y_pred_proba)
    plot_threshold_analysis(y_test, y_pred_proba)
    plot_model_comparison(results)
    
    # Feature importance (if applicable)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, X_test.columns)
    
    # Generate reports
    print("\n5. Generating reports...")
    generate_classification_report(y_test, y_pred)
    create_summary_report(results, y_test, y_pred, y_pred_proba)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nAll reports and visualizations saved to 'reports/' directory")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - threshold_analysis.png")
    print("  - model_comparison.png")
    print("  - feature_importance.png (if applicable)")
    print("  - classification_report.txt")
    print("  - summary_report.txt")


if __name__ == "__main__":
    main()