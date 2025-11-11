"""
Cyber-RF Anomaly Detector Challenge - Implementation Starter
============================================================

This script provides a complete baseline implementation for the Cyber-RF
Anomaly Detector Challenge, including:
- Data loading and exploration
- Preprocessing and feature engineering
- Multiple ML model training
- Comprehensive evaluation
- Model interpretation with SHAP

Author: Generated for Semantic Architectures
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    StratifiedKFold, cross_validate, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not available. Install with: pip install shap")


class CyberRFAnomalyDetector:
    """
    Complete pipeline for Cyber-RF anomaly detection.
    """
    
    def __init__(self, data_path, random_state=42):
        """
        Initialize the detector.
        
        Parameters:
        -----------
        data_path : str
            Path to the training CSV file
        random_state : int
            Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load and perform initial exploration of the dataset."""
        print("="*80)
        print("LOADING DATA")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Features: {self.df.shape[1] - 1}")
        print(f"Samples: {self.df.shape[0]}")
        
        print("\nClass Distribution:")
        print(self.df['Class'].value_counts())
        print("\nClass Proportions:")
        print(self.df['Class'].value_counts(normalize=True))
        
        print("\nMissing Values:")
        missing = self.df.isnull().sum().sum()
        print(f"Total missing values: {missing}")
        
        # Separate features and target
        self.X = self.df.drop('Class', axis=1)
        self.y = self.df['Class'].map({'normal': 0, 'anomalous': 1})
        
        print(f"\nFeature columns: {len(self.X.columns)}")
        print(f"Target encoding: normal=0, anomalous=1")
        
        return self
        
    def explore_data(self, save_plots=True):
        """Perform exploratory data analysis."""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*80)
        
        # Basic statistics
        print("\nFeature Statistics:")
        print(self.X.describe().T[['mean', 'std', 'min', 'max']])
        
        # Check for features with near-zero variance
        variances = self.X.var()
        near_zero_var = variances[variances < 0.0001]
        if len(near_zero_var) > 0:
            print(f"\n⚠️  WARNING: {len(near_zero_var)} features with near-zero variance:")
            print(near_zero_var)
        
        # Correlation analysis
        print("\nComputing correlation matrix...")
        corr_matrix = self.X.corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.9:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            print(f"\n⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.9):")
            for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
                print(f"  - {feat1} <-> {feat2}: {corr:.3f}")
        
        if save_plots:
            self._create_eda_plots()
            
        return self
    
    def _create_eda_plots(self):
        """Create and save EDA visualizations."""
        print("\nGenerating EDA plots...")
        
        # 1. Feature distribution comparison (sample of features)
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        sample_features = self.X.columns[:9]  # First 9 features
        
        for idx, feature in enumerate(sample_features):
            ax = axes[idx // 3, idx % 3]
            
            # Separate by class
            normal_data = self.X[self.y == 0][feature]
            anomalous_data = self.X[self.y == 1][feature]
            
            ax.hist(normal_data, bins=30, alpha=0.5, label='Normal', color='blue')
            ax.hist(anomalous_data, bins=30, alpha=0.5, label='Anomalous', color='red')
            ax.set_xlabel(feature, fontsize=8)
            ax.set_ylabel('Frequency', fontsize=8)
            ax.legend(fontsize=8)
            ax.tick_params(labelsize=7)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/feature_distributions.png', dpi=150)
        plt.close()
        print("  ✓ Saved: feature_distributions.png")
        
        # 2. Correlation heatmap (top features)
        fig, ax = plt.subplots(figsize=(12, 10))
        top_features = self.X.columns[:20]  # First 20 features
        sns.heatmap(self.X[top_features].corr(), annot=False, cmap='coolwarm',
                    center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap (First 20 Features)')
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/correlation_heatmap.png', dpi=150)
        plt.close()
        print("  ✓ Saved: correlation_heatmap.png")
        
    def preprocess_data(self, scaler_type='standard'):
        """
        Preprocess features with scaling.
        
        Parameters:
        -----------
        scaler_type : str
            Type of scaler: 'standard', 'robust', or 'minmax'
        """
        print("\n" + "="*80)
        print("PREPROCESSING")
        print("="*80)
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        print(f"\nApplying {scaler_type} scaling...")
        self.X_scaled = self.scaler.fit_transform(self.X)
        
        print(f"✓ Scaled data shape: {self.X_scaled.shape}")
        print(f"  Mean (should be ~0 for standard): {self.X_scaled.mean():.6f}")
        print(f"  Std (should be ~1 for standard): {self.X_scaled.std():.6f}")
        
        return self
    
    def train_baseline_models(self, cv_folds=5):
        """
        Train baseline models with cross-validation.
        
        Parameters:
        -----------
        cv_folds : int
            Number of cross-validation folds
        """
        print("\n" + "="*80)
        print("TRAINING BASELINE MODELS")
        print("="*80)
        
        # Define baseline models
        baseline_models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                            random_state=self.random_state)
        
        # Scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'recall': 'recall',
            'precision': 'precision',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        # Train and evaluate each model
        for name, model in baseline_models.items():
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print('='*60)
            
            cv_results = cross_validate(
                model, self.X_scaled, self.y,
                cv=cv, scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            # Store model and results
            model.fit(self.X_scaled, self.y)  # Fit on full data
            self.models[name] = model
            self.results[name] = cv_results
            
            # Print results
            print(f"\nCross-Validation Results ({cv_folds}-fold):")
            print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
            print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
            print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
            print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
            print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
        
        return self
    
    def train_advanced_models(self, cv_folds=5):
        """
        Train advanced models (XGBoost, LightGBM) with hyperparameter tuning.
        
        Parameters:
        -----------
        cv_folds : int
            Number of cross-validation folds
        """
        print("\n" + "="*80)
        print("TRAINING ADVANCED MODELS")
        print("="*80)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True,
                            random_state=self.random_state)
        
        # XGBoost
        if HAS_XGBOOST:
            print(f"\n{'='*60}")
            print("Training: XGBoost with Hyperparameter Tuning")
            print('='*60)
            
            xgb_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            print("\nPerforming Randomized Search (30 iterations)...")
            xgb_search = RandomizedSearchCV(
                xgb_model, xgb_params,
                n_iter=30,
                cv=cv,
                scoring='f1',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            )
            
            xgb_search.fit(self.X_scaled, self.y)
            
            print(f"\nBest Parameters:")
            for param, value in xgb_search.best_params_.items():
                print(f"  {param}: {value}")
            
            print(f"\nBest F1-Score: {xgb_search.best_score_:.4f}")
            
            # Store best model
            self.models['XGBoost'] = xgb_search.best_estimator_
            
            # Evaluate with cross-validation
            scoring = {
                'accuracy': 'accuracy',
                'recall': 'recall',
                'precision': 'precision',
                'f1': 'f1',
                'roc_auc': 'roc_auc'
            }
            
            cv_results = cross_validate(
                xgb_search.best_estimator_, self.X_scaled, self.y,
                cv=cv, scoring=scoring,
                n_jobs=-1
            )
            
            self.results['XGBoost'] = cv_results
            
            print(f"\nCross-Validation Results ({cv_folds}-fold):")
            print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
            print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
            print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
            print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
            print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
        
        # LightGBM
        if HAS_LIGHTGBM:
            print(f"\n{'='*60}")
            print("Training: LightGBM")
            print('='*60)
            
            lgb_model = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            
            lgb_model.fit(self.X_scaled, self.y)
            self.models['LightGBM'] = lgb_model
            
            # Evaluate
            cv_results = cross_validate(
                lgb_model, self.X_scaled, self.y,
                cv=cv, scoring=scoring,
                n_jobs=-1
            )
            
            self.results['LightGBM'] = cv_results
            
            print(f"\nCross-Validation Results ({cv_folds}-fold):")
            print(f"  Accuracy:  {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
            print(f"  Recall:    {cv_results['test_recall'].mean():.4f} ± {cv_results['test_recall'].std():.4f}")
            print(f"  Precision: {cv_results['test_precision'].mean():.4f} ± {cv_results['test_precision'].std():.4f}")
            print(f"  F1-Score:  {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
            print(f"  ROC-AUC:   {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
        
        return self
    
    def compare_models(self):
        """Generate comprehensive model comparison."""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Create comparison dataframe
        comparison_data = []
        
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{results['test_accuracy'].mean():.4f} ± {results['test_accuracy'].std():.4f}",
                'Recall': f"{results['test_recall'].mean():.4f} ± {results['test_recall'].std():.4f}",
                'Precision': f"{results['test_precision'].mean():.4f} ± {results['test_precision'].std():.4f}",
                'F1-Score': f"{results['test_f1'].mean():.4f} ± {results['test_f1'].std():.4f}",
                'ROC-AUC': f"{results['test_roc_auc'].mean():.4f} ± {results['test_roc_auc'].std():.4f}",
                'Accuracy_mean': results['test_accuracy'].mean(),
                'F1_mean': results['test_f1'].mean()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('F1_mean', ascending=False)
        
        print("\n" + comparison_df[['Model', 'Accuracy', 'Recall', 'Precision', 'F1-Score', 'ROC-AUC']].to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('/mnt/user-data/outputs/model_comparison.csv', index=False)
        print("\n✓ Saved: model_comparison.csv")
        
        # Identify best model
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   F1-Score: {comparison_df.iloc[0]['F1-Score']}")
        
        # Create visualization
        self._plot_model_comparison(comparison_df)
        
        return best_model_name
    
    def _plot_model_comparison(self, comparison_df):
        """Create model comparison visualization."""
        metrics = ['Accuracy_mean', 'Recall', 'Precision', 'F1_mean', 'ROC-AUC']
        
        # Extract numeric values
        plot_data = []
        for _, row in comparison_df.iterrows():
            plot_data.append({
                'Model': row['Model'],
                'Accuracy': row['Accuracy_mean'],
                'Recall': float(row['Recall'].split(' ±')[0]),
                'Precision': float(row['Precision'].split(' ±')[0]),
                'F1-Score': row['F1_mean'],
                'ROC-AUC': float(row['ROC-AUC'].split(' ±')[0])
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(plot_df))
        width = 0.15
        
        metrics_plot = ['Accuracy', 'Recall', 'Precision', 'F1-Score', 'ROC-AUC']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric, color) in enumerate(zip(metrics_plot, colors)):
            offset = width * (i - 2)
            ax.bar(x + offset, plot_df[metric], width, label=metric, color=color, alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df['Model'], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0.7, 1.0])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/model_comparison.png', dpi=150)
        plt.close()
        print("  ✓ Saved: model_comparison.png")
    
    def analyze_best_model(self, model_name):
        """
        Detailed analysis of the best model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to analyze
        """
        print("\n" + "="*80)
        print(f"ANALYZING BEST MODEL: {model_name}")
        print("="*80)
        
        model = self.models[model_name]
        
        # Generate predictions
        y_pred = model.predict(self.X_scaled)
        y_pred_proba = model.predict_proba(self.X_scaled)[:, 1]
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y, y_pred)
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(self.y, y_pred, target_names=['Normal', 'Anomalous']))
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            self._plot_feature_importance(model, model_name)
        
        # ROC and PR Curves
        self._plot_roc_pr_curves(y_pred_proba, model_name)
        
        # SHAP Analysis (if available)
        if HAS_SHAP and model_name in ['XGBoost', 'Random Forest']:
            self._shap_analysis(model, model_name)
        
        return self
    
    def _plot_feature_importance(self, model, model_name):
        """Plot feature importance."""
        print("\nGenerating feature importance plot...")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # Top 20
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([self.X.columns[i] for i in indices], fontsize=9)
        ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top 20 Feature Importances - {model_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/feature_importance_{model_name.replace(" ", "_")}.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: feature_importance_{model_name.replace(' ', '_')}.png")
    
    def _plot_roc_pr_curves(self, y_pred_proba, model_name):
        """Plot ROC and Precision-Recall curves."""
        print("\nGenerating ROC and PR curves...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y, y_pred_proba)
        roc_auc = roc_auc_score(self.y, y_pred_proba)
        
        ax1.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y, y_pred_proba)
        ap = average_precision_score(self.y, y_pred_proba)
        
        ax2.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AP = {ap:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax2.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/mnt/user-data/outputs/roc_pr_curves_{model_name.replace(" ", "_")}.png', dpi=150)
        plt.close()
        print(f"  ✓ Saved: roc_pr_curves_{model_name.replace(' ', '_')}.png")
    
    def _shap_analysis(self, model, model_name):
        """Perform SHAP analysis for model interpretation."""
        print("\nPerforming SHAP analysis (this may take a few minutes)...")
        
        try:
            # Create SHAP explainer
            if model_name == 'XGBoost':
                explainer = shap.TreeExplainer(model)
            else:  # Random Forest
                explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (use subset for speed)
            sample_size = min(200, len(self.X_scaled))
            sample_indices = np.random.choice(len(self.X_scaled), sample_size, replace=False)
            X_sample = self.X_scaled[sample_indices]
            
            shap_values = explainer.shap_values(X_sample)
            
            # If binary classification, extract the positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample,
                            feature_names=self.X.columns,
                            show=False, max_display=20)
            plt.tight_layout()
            plt.savefig(f'/mnt/user-data/outputs/shap_summary_{model_name.replace(" ", "_")}.png',
                       dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: shap_summary_{model_name.replace(' ', '_')}.png")
            
        except Exception as e:
            print(f"  ⚠️  SHAP analysis failed: {str(e)}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("CYBER-RF ANOMALY DETECTOR CHALLENGE")
    print("="*80)
    print("\nInitializing detector pipeline...\n")
    
    # Initialize detector
    detector = CyberRFAnomalyDetector(
        data_path='/mnt/user-data/uploads/Cyber-RF_Anomaly_Detector_Challenge_Dataset_TrainingSet_80.csv',
        random_state=42
    )
    
    # Run full pipeline
    detector.load_data()
    detector.explore_data(save_plots=True)
    detector.preprocess_data(scaler_type='standard')
    detector.train_baseline_models(cv_folds=5)
    detector.train_advanced_models(cv_folds=5)
    best_model = detector.compare_models()
    detector.analyze_best_model(best_model)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  - model_comparison.csv")
    print("  - model_comparison.png")
    print("  - feature_distributions.png")
    print("  - correlation_heatmap.png")
    print("  - feature_importance_*.png")
    print("  - roc_pr_curves_*.png")
    if HAS_SHAP:
        print("  - shap_summary_*.png")
    print("\n")


if __name__ == "__main__":
    main()
