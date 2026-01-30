#!/usr/bin/env python3
"""
Model Evaluation Script for LULC Classification
Comprehensive accuracy assessment and performance metrics
"""


# --- Ensure project root is in sys.path for src imports ---
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import joblib
import yaml
from pathlib import Path
import argparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime


class LULCEvaluator:
    """Evaluate LULC classification model performance"""
    
    def __init__(self, model_path, config_path='config/config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        self.classes = self.config['classes']
        self.class_names = [self.classes[i]['name'] for i in sorted(self.classes.keys())]
        
        print(f"✓ Model loaded: {self.model.n_features_in_} features, {self.model.n_classes_} classes")
    
    def load_validation_data(self, features_path, labels_path):
        """Load validation features and labels"""
        print(f"\nLoading validation data...")
        X_val = np.load(features_path)
        y_val = np.load(labels_path)
        
        print(f"✓ Loaded {X_val.shape[0]:,} validation samples")
        return X_val, y_val
    
    def evaluate(self, X_val, y_val):
        """Perform comprehensive evaluation"""
        print("\n" + "="*70)
        print("EVALUATING MODEL PERFORMANCE")
        print("="*70)
        
        # Predictions
        print("Making predictions...")
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        
        # Calculate metrics
        results = self._calculate_metrics(y_val, y_pred, y_proba)
        
        # Print results
        self._print_results(results)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        return results, cm, y_pred, y_proba
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all evaluation metrics"""
        results = {}
        
        # Overall metrics
        results['overall_accuracy'] = accuracy_score(y_true, y_pred)
        results['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Per-class metrics
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class detailed metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        results['per_class'] = {}
        for i, class_name in enumerate(self.class_names):
            results['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'support': int(np.sum(y_true == i))
            }
        
        return results
    
    def _print_results(self, results):
        """Print evaluation results in formatted table"""
        print("\n" + "="*70)
        print("OVERALL METRICS")
        print("="*70)
        print(f"Overall Accuracy:    {results['overall_accuracy']:.4f} ({results['overall_accuracy']*100:.2f}%)")
        print(f"Cohen's Kappa:       {results['kappa']:.4f}")
        print(f"\nMacro Average:")
        print(f"  Precision:         {results['precision_macro']:.4f}")
        print(f"  Recall:            {results['recall_macro']:.4f}")
        print(f"  F1-Score:          {results['f1_macro']:.4f}")
        print(f"\nWeighted Average:")
        print(f"  Precision:         {results['precision_weighted']:.4f}")
        print(f"  Recall:            {results['recall_weighted']:.4f}")
        print(f"  F1-Score:          {results['f1_weighted']:.4f}")
        
        print("\n" + "="*70)
        print("PER-CLASS METRICS")
        print("="*70)
        print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        print("-"*70)
        
        for class_name, metrics in results['per_class'].items():
            print(f"{class_name:<15} "
                  f"{metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} "
                  f"{metrics['f1_score']:>10.4f} "
                  f"{metrics['support']:>10,}")
        
        print("="*70)
    
    def plot_confusion_matrix(self, cm, output_path='results/confusion_matrix_eval.png', normalize=False):
        """Plot confusion matrix"""
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {output_file}")
    
    def plot_per_class_metrics(self, results, output_path='results/per_class_metrics.png'):
        """Plot per-class precision, recall, and F1-score"""
        classes = list(results['per_class'].keys())
        precision = [results['per_class'][c]['precision'] for c in classes]
        recall = [results['per_class'][c]['recall'] for c in classes]
        f1 = [results['per_class'][c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width, precision, width, label='Precision', color='steelblue')
        ax.bar(x, recall, width, label='Recall', color='darkorange')
        ax.bar(x + width, f1, width, label='F1-Score', color='seagreen')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Per-class metrics plot saved to {output_file}")
    
    def plot_feature_importance(self, top_n=20, output_path='results/feature_importance_eval.png'):
        """Plot feature importance from the model"""
        if not hasattr(self.model, 'feature_importances_'):
            print("Model does not have feature_importances_ attribute")
            return
        
        importances = self.model.feature_importances_
        
        # Generate feature names
        bands = self.config['features']['bands']
        indices = self.config['features']['indices']
        feature_names = bands + indices
        
        # Get top N features
        indices_sorted = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices_sorted], color='steelblue')
        plt.yticks(range(top_n), [feature_names[i] for i in indices_sorted])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Feature importance plot saved to {output_file}")
    
    def save_results(self, results, cm, output_dir='results'):
        """Save evaluation results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON
        results_json = {
            'timestamp': datetime.now().isoformat(),
            'metrics': results,
            'confusion_matrix': cm.tolist()
        }
        
        output_file = output_path / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"✓ Results saved to {output_file}")
    
    def generate_report(self, results, cm, output_dir='results'):
        """Generate comprehensive evaluation report"""
        print("\n📊 Generating evaluation report...")
        
        # Save JSON results
        self.save_results(results, cm, output_dir)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, f'{output_dir}/confusion_matrix.png')
        self.plot_confusion_matrix(cm, f'{output_dir}/confusion_matrix_normalized.png', normalize=True)
        
        # Plot per-class metrics
        self.plot_per_class_metrics(results, f'{output_dir}/per_class_metrics.png')
        
        # Plot feature importance
        self.plot_feature_importance(output_path=f'{output_dir}/feature_importance.png')
        
        print(f"\n✅ Complete evaluation report saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LULC Random Forest model')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--features', type=str, required=True, help='Validation features (.npy)')
    parser.add_argument('--labels', type=str, required=True, help='Validation labels (.npy)')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = LULCEvaluator(args.model, args.config)
    
    # Load validation data
    X_val, y_val = evaluator.load_validation_data(args.features, args.labels)
    
    # Evaluate
    results, cm, y_pred, y_proba = evaluator.evaluate(X_val, y_val)
    
    # Generate report
    evaluator.generate_report(results, cm, args.output)
    
    print("\n" + "="*70)
    print("🎉 EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
