"""
Visualization Module

This module handles plotting and visualization utilities for the CGMacros
CCR prediction project, including model performance, feature importance,
and data exploration visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")


def create_visualizations(features_df: pd.DataFrame, models_results: Dict[str, Any], 
                         evaluation_results: Dict[str, Any], output_dir: str) -> None:
    """
    Create all visualizations for the CGMacros project.
    
    Args:
        features_df: DataFrame with features and target
        models_results: Results from model training
        evaluation_results: Results from model evaluation
        output_dir: Directory to save visualizations
    """
    logger.info("Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Data exploration visualizations
    create_data_exploration_plots(features_df, output_path)
    
    # Model performance visualizations
    create_model_performance_plots(evaluation_results, output_path)
    
    # Feature importance visualizations
    create_feature_importance_plots(models_results, evaluation_results, output_path)
    
    # Prediction analysis visualizations
    create_prediction_analysis_plots(models_results, features_df, output_path)
    
    # Error analysis visualizations
    create_error_analysis_plots(models_results, evaluation_results, output_path)
    
    logger.info(f"All visualizations saved to: {output_path}")


def create_data_exploration_plots(df: pd.DataFrame, output_path: Path) -> None:
    """
    Create data exploration and EDA plots.
    
    Args:
        df: Features DataFrame
        output_path: Path to save plots
    """
    logger.info("Creating data exploration plots...")
    
    # Target variable distribution
    if 'ccr' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Target Variable (CCR) Analysis', fontsize=16, fontweight='bold')
        
        # Histogram
        axes[0, 0].hist(df['ccr'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('CCR Distribution')
        axes[0, 0].set_xlabel('CCR Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(df['ccr'].dropna())
        axes[0, 1].set_title('CCR Box Plot')
        axes[0, 1].set_ylabel('CCR Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(df['ccr'].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('CCR Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary statistics
        ccr_stats = df['ccr'].describe()
        axes[1, 1].text(0.1, 0.9, f"Count: {ccr_stats['count']:.0f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.8, f"Mean: {ccr_stats['mean']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f"Std: {ccr_stats['std']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"Min: {ccr_stats['min']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.5, f"25%: {ccr_stats['25%']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f"50%: {ccr_stats['50%']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.3, f"75%: {ccr_stats['75%']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.2, f"Max: {ccr_stats['max']:.4f}", transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('CCR Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Feature correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        # Select top correlated features with target for better visualization
        if 'ccr' in numeric_cols:
            correlations = df[numeric_cols].corr()['ccr'].abs().sort_values(ascending=False)
            top_features = correlations.head(20).index.tolist()
        else:
            top_features = numeric_cols[:20].tolist()
        
        plt.figure(figsize=(14, 12))
        correlation_matrix = df[top_features].corr()
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix (Top 20 Features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Missing data visualization
    if df.isnull().any().any():
        plt.figure(figsize=(12, 8))
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            plt.barh(range(len(missing_data)), missing_data.values, color='coral')
            plt.yticks(range(len(missing_data)), missing_data.index)
            plt.xlabel('Number of Missing Values')
            plt.title('Missing Data by Feature', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(output_path / 'missing_data.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_model_performance_plots(evaluation_results: Dict[str, Any], output_path: Path) -> None:
    """
    Create model performance comparison plots.
    
    Args:
        evaluation_results: Evaluation results dictionary
        output_path: Path to save plots
    """
    logger.info("Creating model performance plots...")
    
    model_metrics = evaluation_results.get('model_metrics', {})
    if not model_metrics:
        return
    
    # Performance comparison bar chart
    metrics_df = pd.DataFrame(model_metrics).T
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # NRMSE comparison
    if 'nrmse' in metrics_df.columns:
        axes[0, 0].bar(metrics_df.index, metrics_df['nrmse'], color='lightcoral', alpha=0.8)
        axes[0, 0].set_title('Normalized RMSE (Lower is Better)')
        axes[0, 0].set_ylabel('NRMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(metrics_df['nrmse']):
            axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Correlation comparison
    if 'correlation' in metrics_df.columns:
        axes[0, 1].bar(metrics_df.index, metrics_df['correlation'], color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Correlation (Higher is Better)')
        axes[0, 1].set_ylabel('Correlation')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(metrics_df['correlation']):
            axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # R² comparison
    if 'r2' in metrics_df.columns:
        axes[1, 0].bar(metrics_df.index, metrics_df['r2'], color='lightblue', alpha=0.8)
        axes[1, 0].set_title('R² Score (Higher is Better)')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(metrics_df['r2']):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MAE comparison
    if 'mae' in metrics_df.columns:
        axes[1, 1].bar(metrics_df.index, metrics_df['mae'], color='plum', alpha=0.8)
        axes[1, 1].set_title('Mean Absolute Error (Lower is Better)')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(metrics_df['mae']):
            axes[1, 1].text(i, v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model ranking visualization
    model_comparison = evaluation_results.get('model_comparison', {})
    rankings = model_comparison.get('rankings', {})
    
    if rankings:
        plt.figure(figsize=(14, 8))
        
        ranking_df = pd.DataFrame(rankings)
        
        # Create heatmap of rankings
        sns.heatmap(ranking_df.T, annot=True, fmt='.0f', cmap='RdYlGn_r', 
                   cbar_kws={'label': 'Rank (1 = Best)'})
        plt.title('Model Rankings by Metric', fontsize=14, fontweight='bold')
        plt.xlabel('Models')
        plt.ylabel('Metrics')
        plt.tight_layout()
        plt.savefig(output_path / 'model_rankings.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_feature_importance_plots(models_results: Dict[str, Any], 
                                  evaluation_results: Dict[str, Any], output_path: Path) -> None:
    """
    Create feature importance visualizations.
    
    Args:
        models_results: Results from model training
        evaluation_results: Results from model evaluation
        output_path: Path to save plots
    """
    logger.info("Creating feature importance plots...")
    
    feature_importance_data = evaluation_results.get('feature_importance', {})
    
    if not feature_importance_data:
        return
    
    # Top features importance plot
    top_features = feature_importance_data.get('top_features', {})
    
    if 'mean_importance' in top_features:
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        importance_items = list(top_features['mean_importance'].items())
        importance_items.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 15 features for better visualization
        top_15 = importance_items[:15]
        features, importances = zip(*top_15)
        
        plt.barh(range(len(features)), importances, color='steelblue', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean Feature Importance')
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add importance values
        for i, importance in enumerate(importances):
            plt.text(importance + max(importances) * 0.01, i, f'{importance:.3f}', 
                    va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path / 'top_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Category-wise importance
    category_importance = feature_importance_data.get('category_importance', {})
    
    if category_importance:
        plt.figure(figsize=(12, 6))
        
        categories = list(category_importance.keys())
        mean_importances = [category_importance[cat]['mean_importance'] for cat in categories]
        feature_counts = [category_importance[cat]['feature_count'] for cat in categories]
        
        # Create subplot with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar chart for mean importance
        bars = ax1.bar(categories, mean_importances, alpha=0.7, color='lightcoral', 
                      label='Mean Importance')
        ax1.set_ylabel('Mean Feature Importance', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add importance values on bars
        for bar, importance in zip(bars, mean_importances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_importances) * 0.01,
                    f'{importance:.3f}', ha='center', va='bottom', fontweight='bold', color='red')
        
        # Line plot for feature count
        ax2 = ax1.twinx()
        line = ax2.plot(categories, feature_counts, color='blue', marker='o', linewidth=2, 
                       markersize=8, label='Feature Count')
        ax2.set_ylabel('Number of Features', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title('Feature Importance by Category', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / 'category_importance.png', dpi=300, bbox_inches='tight')
        plt.close()


def create_prediction_analysis_plots(models_results: Dict[str, Any], 
                                   features_df: pd.DataFrame, output_path: Path) -> None:
    """
    Create prediction analysis plots.
    
    Args:
        models_results: Results from model training
        features_df: Features DataFrame
        output_path: Path to save plots
    """
    logger.info("Creating prediction analysis plots...")
    
    trainer = models_results.get('trainer')
    data_splits = models_results.get('data_splits', {})
    
    if not trainer or not data_splits:
        return
    
    X_test = data_splits.get('X_test')
    y_test = data_splits.get('y_test')
    
    if X_test is None or y_test is None:
        return
    
    # Prediction vs actual plots for top models
    models_to_plot = ['random_forest', 'xgboost', 'lightgbm', 'ridge']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Predicted vs Actual CCR Values', fontsize=16, fontweight='bold')
    
    for i, model_name in enumerate(models_to_plot):
        if model_name in trainer.models:
            row, col = i // 2, i % 2
            
            try:
                y_pred = trainer.predict(model_name, X_test)
                
                # Scatter plot
                axes[row, col].scatter(y_test, y_pred, alpha=0.6, s=50)
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                axes[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                                  label='Perfect Prediction')
                
                # Calculate and display metrics
                mse = np.mean((y_test - y_pred) ** 2)
                r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))
                corr = np.corrcoef(y_test, y_pred)[0, 1]
                
                axes[row, col].set_xlabel('Actual CCR')
                axes[row, col].set_ylabel('Predicted CCR')
                axes[row, col].set_title(f'{model_name.replace("_", " ").title()}\n'
                                       f'R²={r2:.3f}, Corr={corr:.3f}')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].legend()
                
            except Exception as e:
                logger.error(f"Failed to create prediction plot for {model_name}: {str(e)}")
                axes[row, col].text(0.5, 0.5, f'Error: {model_name}', 
                                  transform=axes[row, col].transAxes, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path / 'prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_error_analysis_plots(models_results: Dict[str, Any], 
                              evaluation_results: Dict[str, Any], output_path: Path) -> None:
    """
    Create error analysis plots.
    
    Args:
        models_results: Results from model training
        evaluation_results: Results from model evaluation
        output_path: Path to save plots
    """
    logger.info("Creating error analysis plots...")
    
    trainer = models_results.get('trainer')
    data_splits = models_results.get('data_splits', {})
    error_analysis = evaluation_results.get('error_analysis', {})
    
    if not trainer or not data_splits:
        return
    
    X_test = data_splits.get('X_test')
    y_test = data_splits.get('y_test')
    
    if X_test is None or y_test is None:
        return
    
    # Residual plots for best models
    best_models = ['random_forest', 'xgboost', 'lightgbm']
    
    fig, axes = plt.subplots(len(best_models), 2, figsize=(16, 5 * len(best_models)))
    if len(best_models) == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Residual Analysis', fontsize=16, fontweight='bold')
    
    for i, model_name in enumerate(best_models):
        if model_name in trainer.models:
            try:
                y_pred = trainer.predict(model_name, X_test)
                residuals = y_test - y_pred
                
                # Residuals vs predicted
                axes[i, 0].scatter(y_pred, residuals, alpha=0.6)
                axes[i, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
                axes[i, 0].set_xlabel('Predicted CCR')
                axes[i, 0].set_ylabel('Residuals')
                axes[i, 0].set_title(f'{model_name.replace("_", " ").title()} - Residuals vs Predicted')
                axes[i, 0].grid(True, alpha=0.3)
                
                # Residuals histogram
                axes[i, 1].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i, 1].axvline(x=0, color='red', linestyle='--', linewidth=2)
                axes[i, 1].set_xlabel('Residuals')
                axes[i, 1].set_ylabel('Frequency')
                axes[i, 1].set_title(f'{model_name.replace("_", " ").title()} - Residuals Distribution')
                axes[i, 1].grid(True, alpha=0.3)
                
                # Add statistics
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                axes[i, 1].text(0.7, 0.9, f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                               transform=axes[i, 1].transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                
            except Exception as e:
                logger.error(f"Failed to create residual plot for {model_name}: {str(e)}")
    
    plt.tight_layout()
    plt.savefig(output_path / 'residual_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model performance summary radar chart
    model_metrics = evaluation_results.get('model_metrics', {})
    if model_metrics:
        create_radar_chart(model_metrics, output_path)


def create_radar_chart(model_metrics: Dict[str, Dict[str, float]], output_path: Path) -> None:
    """
    Create a radar chart comparing model performance.
    
    Args:
        model_metrics: Dictionary of model metrics
        output_path: Path to save plots
    """
    try:
        from math import pi
        
        # Select metrics for radar chart
        metrics_to_plot = ['correlation', 'r2']  # Metrics where higher is better
        
        # Normalize metrics to 0-1 scale for radar chart
        metrics_df = pd.DataFrame(model_metrics).T
        
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        if len(available_metrics) < 2:
            return
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of metrics
        N = len(available_metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_metrics)))
        
        for i, (model_name, metrics) in enumerate(model_metrics.items()):
            values = [metrics.get(metric, 0) for metric in available_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics])
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add title and legend
        plt.title('Model Performance Comparison\n(Radar Chart)', size=16, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to create radar chart: {str(e)}")


def save_summary_plot(evaluation_results: Dict[str, Any], output_path: Path) -> None:
    """
    Create and save a summary plot with key findings.
    
    Args:
        evaluation_results: Evaluation results dictionary
        output_path: Path to save plots
    """
    logger.info("Creating summary plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract key information
    eval_summary = evaluation_results.get('evaluation_summary', {})
    model_metrics = evaluation_results.get('model_metrics', {})
    
    # Create text summary
    summary_text = "CGMacros CCR Prediction - Summary Results\n\n"
    
    if eval_summary.get('best_overall_model'):
        best_model = eval_summary['best_overall_model']
        summary_text += f"Best Model: {best_model['name']}\n"
        summary_text += f"NRMSE: {best_model['nrmse']:.4f}\n"
        if best_model.get('correlation'):
            summary_text += f"Correlation: {best_model['correlation']:.4f}\n"
    
    summary_text += f"\nTotal Models Evaluated: {eval_summary.get('total_models', 'N/A')}\n"
    
    key_findings = eval_summary.get('key_findings', [])
    if key_findings:
        summary_text += "\nKey Findings:\n"
        for finding in key_findings:
            summary_text += f"• {finding}\n"
    
    # Display summary text
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Project Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path / 'project_summary.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Example usage with sample data
    sample_features = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'ccr': np.random.uniform(0.2, 0.8, 100)
    })
    
    sample_models_results = {
        'trainer': None,  # Would be actual trainer object
        'data_splits': {
            'X_test': np.random.normal(0, 1, (20, 2)),
            'y_test': np.random.uniform(0.2, 0.8, 20)
        }
    }
    
    sample_evaluation_results = {
        'model_metrics': {
            'linear_regression': {'nrmse': 0.15, 'correlation': 0.65, 'r2': 0.42},
            'random_forest': {'nrmse': 0.12, 'correlation': 0.75, 'r2': 0.56},
            'xgboost': {'nrmse': 0.11, 'correlation': 0.78, 'r2': 0.61}
        },
        'evaluation_summary': {
            'total_models': 3,
            'best_overall_model': {'name': 'xgboost', 'nrmse': 0.11, 'correlation': 0.78},
            'key_findings': ['XGBoost achieved best performance', 'All models show good correlation']
        }
    }
    
    # Create visualizations (would need actual output directory)
    print("Visualization module ready for use!")
    print("Use create_visualizations() function with actual data and results.")