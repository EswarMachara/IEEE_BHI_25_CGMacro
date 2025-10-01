"""
Evaluation Module

This module handles model evaluation, metrics computation, and report generation
for the CGMacros CCR prediction challenge.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def evaluate_models(models_results: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive evaluation of all trained models.
    
    Args:
        models_results: Results from model training
        features_df: DataFrame with features and target
        
    Returns:
        Dictionary with comprehensive evaluation results
    """
    logger.info("Starting comprehensive model evaluation")
    
    evaluation_results = models_results.get('evaluation_results', {})
    trainer = models_results.get('trainer')
    data_splits = models_results.get('data_splits', {})
    
    # Enhanced evaluation with additional metrics
    enhanced_results = {}
    
    for model_name, base_metrics in evaluation_results.items():
        try:
            enhanced_metrics = compute_additional_metrics(
                trainer, model_name, data_splits, features_df
            )
            enhanced_results[model_name] = {**base_metrics, **enhanced_metrics}
        except Exception as e:
            logger.error(f"Failed to compute enhanced metrics for {model_name}: {str(e)}")
            enhanced_results[model_name] = base_metrics
    
    # Model comparison analysis
    comparison_results = perform_model_comparison(enhanced_results)
    
    # Feature importance analysis
    feature_importance_results = analyze_feature_importance(
        models_results.get('trainer'), 
        models_results.get('feature_names', [])
    )
    
    # Error analysis
    error_analysis_results = perform_error_analysis(
        trainer, enhanced_results, data_splits
    )
    
    final_results = {
        'model_metrics': enhanced_results,
        'model_comparison': comparison_results,
        'feature_importance': feature_importance_results,
        'error_analysis': error_analysis_results,
        'evaluation_summary': generate_evaluation_summary(enhanced_results)
    }
    
    logger.info("Model evaluation completed")
    
    return final_results


def compute_additional_metrics(trainer, model_name: str, data_splits: Dict, features_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute additional evaluation metrics beyond basic ones.
    
    Args:
        trainer: Model trainer instance
        model_name: Name of the model
        data_splits: Data splits dictionary
        features_df: Original features DataFrame
        
    Returns:
        Dictionary of additional metrics
    """
    X_test = data_splits.get('X_test')
    y_test = data_splits.get('y_test')
    
    if X_test is None or y_test is None:
        return {}
    
    # Get predictions
    y_pred = trainer.predict(model_name, X_test)
    
    additional_metrics = {}
    
    # Prediction interval metrics
    residuals = y_test - y_pred
    additional_metrics['residual_std'] = np.std(residuals)
    additional_metrics['residual_mean'] = np.mean(residuals)
    additional_metrics['residual_skewness'] = float(pd.Series(residuals).skew())
    
    # Prediction quality by CCR ranges
    low_ccr_mask = y_test < 0.33
    mid_ccr_mask = (y_test >= 0.33) & (y_test < 0.67)
    high_ccr_mask = y_test >= 0.67
    
    for range_name, mask in [('low_ccr', low_ccr_mask), ('mid_ccr', mid_ccr_mask), ('high_ccr', high_ccr_mask)]:
        if np.any(mask):
            range_rmse = np.sqrt(mean_squared_error(y_test[mask], y_pred[mask]))
            range_corr = pearsonr(y_test[mask], y_pred[mask])[0]
            additional_metrics[f'{range_name}_rmse'] = range_rmse
            additional_metrics[f'{range_name}_correlation'] = range_corr
    
    # Percentage of predictions within acceptable ranges
    abs_errors = np.abs(residuals)
    additional_metrics['pct_within_5pct'] = np.mean(abs_errors < 0.05) * 100
    additional_metrics['pct_within_10pct'] = np.mean(abs_errors < 0.10) * 100
    additional_metrics['pct_within_15pct'] = np.mean(abs_errors < 0.15) * 100
    
    # Median absolute error
    additional_metrics['median_absolute_error'] = np.median(abs_errors)
    
    # Maximum error
    additional_metrics['max_absolute_error'] = np.max(abs_errors)
    
    # Cross-validation metrics if possible
    try:
        X_train = data_splits.get('X_train')
        y_train = data_splits.get('y_train')
        if X_train is not None and y_train is not None:
            X_full = np.vstack([X_train, X_test])
            y_full = np.concatenate([y_train, y_test])
            cv_metrics = trainer.cross_validate(model_name, X_full, y_full, cv=5)
            additional_metrics.update(cv_metrics)
    except Exception as e:
        logger.warning(f"Cross-validation failed for {model_name}: {str(e)}")
    
    return additional_metrics


def perform_model_comparison(evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Perform comprehensive model comparison analysis.
    
    Args:
        evaluation_results: Dictionary of model evaluation results
        
    Returns:
        Dictionary with model comparison results
    """
    if not evaluation_results:
        return {}
    
    # Create comparison DataFrame
    metrics_df = pd.DataFrame(evaluation_results).T
    
    # Ranking by different metrics
    rankings = {}
    key_metrics = ['nrmse', 'correlation', 'rmse', 'mae', 'r2']
    
    for metric in key_metrics:
        if metric in metrics_df.columns:
            ascending = metric != 'correlation' and metric != 'r2'  # Higher is better for correlation and r2
            rankings[metric] = metrics_df[metric].rank(ascending=ascending, method='min').to_dict()
    
    # Overall ranking (weighted combination)
    if 'nrmse' in metrics_df.columns and 'correlation' in metrics_df.columns:
        # Normalize metrics to 0-1 scale
        nrmse_norm = 1 - (metrics_df['nrmse'] - metrics_df['nrmse'].min()) / (metrics_df['nrmse'].max() - metrics_df['nrmse'].min())
        corr_norm = (metrics_df['correlation'] - metrics_df['correlation'].min()) / (metrics_df['correlation'].max() - metrics_df['correlation'].min())
        
        # Weighted score (60% NRMSE, 40% correlation)
        overall_score = 0.6 * nrmse_norm + 0.4 * corr_norm
        rankings['overall'] = overall_score.rank(ascending=False, method='min').to_dict()
    
    # Best and worst models
    best_models = {}
    worst_models = {}
    
    for metric in key_metrics:
        if metric in metrics_df.columns:
            if metric in ['correlation', 'r2']:
                best_models[metric] = metrics_df[metric].idxmax()
                worst_models[metric] = metrics_df[metric].idxmin()
            else:
                best_models[metric] = metrics_df[metric].idxmin()
                worst_models[metric] = metrics_df[metric].idxmax()
    
    # Statistical significance tests (if possible)
    significance_tests = perform_significance_tests(metrics_df)
    
    comparison_results = {
        'metrics_summary': metrics_df.describe().to_dict(),
        'rankings': rankings,
        'best_models': best_models,
        'worst_models': worst_models,
        'significance_tests': significance_tests,
        'model_categories': categorize_models(metrics_df)
    }
    
    return comparison_results


def analyze_feature_importance(trainer, feature_names: List[str]) -> Dict[str, Any]:
    """
    Analyze feature importance across different models.
    
    Args:
        trainer: Model trainer instance
        feature_names: List of feature names
        
    Returns:
        Dictionary with feature importance analysis
    """
    if not trainer or not hasattr(trainer, 'feature_importance'):
        return {}
    
    feature_importance = trainer.feature_importance
    
    if not feature_importance:
        return {}
    
    # Combine feature importance from all models
    importance_df = pd.DataFrame(feature_importance)
    
    if len(feature_names) == importance_df.shape[0]:
        importance_df.index = feature_names
    
    # Overall feature ranking (average across models)
    importance_df['mean_importance'] = importance_df.mean(axis=1)
    importance_df['std_importance'] = importance_df.std(axis=1)
    importance_df['rank'] = importance_df['mean_importance'].rank(ascending=False)
    
    # Top features
    top_features = importance_df.nlargest(20, 'mean_importance')
    
    # Feature categories analysis
    feature_categories = categorize_features(feature_names)
    category_importance = {}
    
    for category, features in feature_categories.items():
        category_features = [f for f in features if f in importance_df.index]
        if category_features:
            category_importance[category] = {
                'mean_importance': importance_df.loc[category_features, 'mean_importance'].mean(),
                'total_importance': importance_df.loc[category_features, 'mean_importance'].sum(),
                'feature_count': len(category_features),
                'top_feature': importance_df.loc[category_features, 'mean_importance'].idxmax()
            }
    
    importance_results = {
        'feature_importance_matrix': importance_df.to_dict(),
        'top_features': top_features.to_dict(),
        'category_importance': category_importance,
        'feature_stability': analyze_feature_stability(importance_df)
    }
    
    return importance_results


def perform_error_analysis(trainer, evaluation_results: Dict, data_splits: Dict) -> Dict[str, Any]:
    """
    Perform detailed error analysis across models.
    
    Args:
        trainer: Model trainer instance
        evaluation_results: Model evaluation results
        data_splits: Data splits dictionary
        
    Returns:
        Dictionary with error analysis results
    """
    X_test = data_splits.get('X_test')
    y_test = data_splits.get('y_test')
    
    if X_test is None or y_test is None or not trainer:
        return {}
    
    error_analysis = {}
    
    # Analyze errors for each model
    for model_name in evaluation_results.keys():
        try:
            y_pred = trainer.predict(model_name, X_test)
            residuals = y_test - y_pred
            
            model_error_analysis = {
                'residual_distribution': {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'skewness': float(pd.Series(residuals).skew()),
                    'kurtosis': float(pd.Series(residuals).kurtosis())
                },
                'outlier_analysis': analyze_outliers(residuals, y_test, y_pred),
                'prediction_bias': analyze_prediction_bias(y_test, y_pred)
            }
            
            error_analysis[model_name] = model_error_analysis
            
        except Exception as e:
            logger.error(f"Error analysis failed for {model_name}: {str(e)}")
    
    return error_analysis


def analyze_outliers(residuals: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Analyze outliers in predictions.
    
    Args:
        residuals: Prediction residuals
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with outlier analysis
    """
    abs_residuals = np.abs(residuals)
    
    # Define outliers as points with residuals > 1.5 * IQR
    q75, q25 = np.percentile(abs_residuals, [75, 25])
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    
    outlier_mask = abs_residuals > outlier_threshold
    
    outlier_analysis = {
        'outlier_count': int(np.sum(outlier_mask)),
        'outlier_percentage': float(np.mean(outlier_mask) * 100),
        'outlier_threshold': float(outlier_threshold),
        'max_residual': float(np.max(abs_residuals)),
        'outlier_characteristics': {
            'mean_true_value': float(np.mean(y_true[outlier_mask])) if np.any(outlier_mask) else 0,
            'mean_pred_value': float(np.mean(y_pred[outlier_mask])) if np.any(outlier_mask) else 0,
            'mean_residual': float(np.mean(residuals[outlier_mask])) if np.any(outlier_mask) else 0
        }
    }
    
    return outlier_analysis


def analyze_prediction_bias(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Analyze prediction bias across different value ranges.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with bias analysis
    """
    residuals = y_true - y_pred
    
    # Bias by value ranges
    ranges = [(0, 0.33), (0.33, 0.67), (0.67, 1.0)]
    range_bias = {}
    
    for i, (low, high) in enumerate(ranges):
        mask = (y_true >= low) & (y_true < high)
        if np.any(mask):
            range_bias[f'range_{i+1}_{low}_{high}'] = {
                'mean_bias': float(np.mean(residuals[mask])),
                'rmse': float(np.sqrt(np.mean(residuals[mask]**2))),
                'sample_count': int(np.sum(mask))
            }
    
    bias_analysis = {
        'overall_bias': float(np.mean(residuals)),
        'range_bias': range_bias,
        'systematic_bias': {
            'underestimation_pct': float(np.mean(residuals > 0) * 100),
            'overestimation_pct': float(np.mean(residuals < 0) * 100)
        }
    }
    
    return bias_analysis


def categorize_features(feature_names: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features by type.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping categories to feature lists
    """
    categories = {
        'glucose': [],
        'activity': [],
        'demographic': [],
        'microbiome': [],
        'gut_health': [],
        'temporal': [],
        'interaction': [],
        'other': []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        if any(term in feature_lower for term in ['glucose', 'cgm']):
            categories['glucose'].append(feature)
        elif any(term in feature_lower for term in ['step', 'heart', 'hr', 'activity', 'calorie']):
            categories['activity'].append(feature)
        elif any(term in feature_lower for term in ['age', 'bmi', 'gender', 'weight', 'height', 'bio']):
            categories['demographic'].append(feature)
        elif 'microbiome' in feature_lower:
            categories['microbiome'].append(feature)
        elif 'gut' in feature_lower:
            categories['gut_health'].append(feature)
        elif any(term in feature_lower for term in ['hour', 'day', 'time', 'meal', 'weekend']):
            categories['temporal'].append(feature)
        elif '_x_' in feature_lower:
            categories['interaction'].append(feature)
        else:
            categories['other'].append(feature)
    
    return categories


def analyze_feature_stability(importance_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze stability of feature importance across models.
    
    Args:
        importance_df: DataFrame with feature importance
        
    Returns:
        Dictionary with stability analysis
    """
    if importance_df.shape[1] < 2:
        return {}
    
    # Calculate coefficient of variation for each feature
    cv_scores = importance_df.std(axis=1) / importance_df.mean(axis=1)
    
    stability_analysis = {
        'most_stable_features': cv_scores.nsmallest(10).to_dict(),
        'least_stable_features': cv_scores.nlargest(10).to_dict(),
        'overall_stability': {
            'mean_cv': float(cv_scores.mean()),
            'median_cv': float(cv_scores.median()),
            'stable_feature_count': int(np.sum(cv_scores < 0.5))  # CV < 0.5 considered stable
        }
    }
    
    return stability_analysis


def perform_significance_tests(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical significance tests between models.
    
    Args:
        metrics_df: DataFrame with model metrics
        
    Returns:
        Dictionary with significance test results
    """
    # This is a simplified version - in practice, you'd need access to raw predictions
    # for proper statistical tests like paired t-tests
    
    significance_results = {}
    
    if 'nrmse' in metrics_df.columns:
        nrmse_values = metrics_df['nrmse']
        
        # Basic statistical comparison
        best_model = nrmse_values.idxmin()
        worst_model = nrmse_values.idxmax()
        
        significance_results = {
            'best_vs_worst': {
                'best_model': best_model,
                'worst_model': worst_model,
                'difference': float(nrmse_values[worst_model] - nrmse_values[best_model]),
                'relative_improvement': float((nrmse_values[worst_model] - nrmse_values[best_model]) / nrmse_values[worst_model] * 100)
            }
        }
    
    return significance_results


def categorize_models(metrics_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize models by performance level.
    
    Args:
        metrics_df: DataFrame with model metrics
        
    Returns:
        Dictionary with model categories
    """
    if 'nrmse' not in metrics_df.columns:
        return {}
    
    nrmse_values = metrics_df['nrmse']
    q33, q67 = nrmse_values.quantile([0.33, 0.67])
    
    categories = {
        'high_performance': nrmse_values[nrmse_values <= q33].index.tolist(),
        'medium_performance': nrmse_values[(nrmse_values > q33) & (nrmse_values <= q67)].index.tolist(),
        'low_performance': nrmse_values[nrmse_values > q67].index.tolist()
    }
    
    return categories


def generate_evaluation_summary(evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Generate a summary of the evaluation results.
    
    Args:
        evaluation_results: Dictionary of model evaluation results
        
    Returns:
        Dictionary with evaluation summary
    """
    if not evaluation_results:
        return {}
    
    metrics_df = pd.DataFrame(evaluation_results).T
    
    summary = {
        'total_models': len(evaluation_results),
        'evaluation_metrics': list(metrics_df.columns),
        'best_overall_model': None,
        'performance_ranges': {},
        'key_findings': []
    }
    
    # Find best overall model
    if 'nrmse' in metrics_df.columns:
        best_model = metrics_df['nrmse'].idxmin()
        summary['best_overall_model'] = {
            'name': best_model,
            'nrmse': float(metrics_df.loc[best_model, 'nrmse']),
            'correlation': float(metrics_df.loc[best_model, 'correlation']) if 'correlation' in metrics_df.columns else None
        }
    
    # Performance ranges
    for metric in ['nrmse', 'correlation', 'rmse']:
        if metric in metrics_df.columns:
            summary['performance_ranges'][metric] = {
                'min': float(metrics_df[metric].min()),
                'max': float(metrics_df[metric].max()),
                'mean': float(metrics_df[metric].mean()),
                'std': float(metrics_df[metric].std())
            }
    
    # Key findings
    if 'nrmse' in metrics_df.columns:
        best_nrmse = metrics_df['nrmse'].min()
        worst_nrmse = metrics_df['nrmse'].max()
        improvement = ((worst_nrmse - best_nrmse) / worst_nrmse) * 100
        
        summary['key_findings'].append(f"Best model achieves {improvement:.1f}% improvement over worst model")
        
        if best_nrmse < 0.1:
            summary['key_findings'].append("Achieved excellent prediction accuracy (NRMSE < 0.1)")
        elif best_nrmse < 0.2:
            summary['key_findings'].append("Achieved good prediction accuracy (NRMSE < 0.2)")
    
    return summary


def generate_report(evaluation_results: Dict[str, Any], features_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        evaluation_results: Evaluation results dictionary
        features_df: Original features DataFrame
        output_dir: Directory to save the report
    """
    logger.info("Generating evaluation report...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate report content
    report_content = create_report_content(evaluation_results, features_df)
    
    # Save report
    report_file = output_path / "evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"Evaluation report saved to: {report_file}")


def create_report_content(evaluation_results: Dict[str, Any], features_df: pd.DataFrame) -> str:
    """
    Create the content for the evaluation report.
    
    Args:
        evaluation_results: Evaluation results dictionary
        features_df: Original features DataFrame
        
    Returns:
        Report content as string
    """
    content = """# CGMacros CCR Prediction - Model Evaluation Report

## Executive Summary

This report presents the evaluation results for models trained to predict Carbohydrate Caloric Ratio (CCR) from multimodal data in the CGMacros dataset.

"""
    
    # Add model performance summary
    model_metrics = evaluation_results.get('model_metrics', {})
    if model_metrics:
        content += "## Model Performance Summary\n\n"
        content += "| Model | NRMSE | Correlation | RMSE | MAE | R² |\n"
        content += "|-------|-------|-------------|------|-----|----|\n"
        
        for model_name, metrics in model_metrics.items():
            content += f"| {model_name} | {metrics.get('nrmse', 'N/A'):.4f} | {metrics.get('correlation', 'N/A'):.4f} | {metrics.get('rmse', 'N/A'):.4f} | {metrics.get('mae', 'N/A'):.4f} | {metrics.get('r2', 'N/A'):.4f} |\n"
        
        content += "\n"
    
    # Add evaluation summary
    eval_summary = evaluation_results.get('evaluation_summary', {})
    if eval_summary:
        content += "## Key Findings\n\n"
        best_model = eval_summary.get('best_overall_model')
        if best_model:
            content += f"- **Best Model**: {best_model['name']} with NRMSE of {best_model['nrmse']:.4f}\n"
        
        key_findings = eval_summary.get('key_findings', [])
        for finding in key_findings:
            content += f"- {finding}\n"
        
        content += "\n"
    
    # Add feature importance summary
    feature_importance = evaluation_results.get('feature_importance', {})
    if feature_importance and 'top_features' in feature_importance:
        content += "## Top Important Features\n\n"
        top_features = feature_importance['top_features']
        if 'mean_importance' in top_features:
            sorted_features = sorted(top_features['mean_importance'].items(), key=lambda x: x[1], reverse=True)[:10]
            for i, (feature, importance) in enumerate(sorted_features, 1):
                content += f"{i}. **{feature}**: {importance:.4f}\n"
        
        content += "\n"
    
    # Add methodology
    content += """## Methodology

### Data Preparation
- Features were engineered from multimodal data including CGM, activity, demographics, microbiome, and gut health data
- Target variable (CCR) was computed as: CCR = net_carbs / (net_carbs + protein + fat + fiber)
- Nutrient columns were removed after target computation to prevent data leakage

### Model Types
- **Baseline Models**: Linear Regression, Ridge, Lasso, Elastic Net
- **Intermediate Models**: Random Forest, XGBoost
- **Advanced Models**: LightGBM

### Evaluation Metrics
- **Primary**: Normalized Root Mean Square Error (NRMSE)
- **Secondary**: Pearson Correlation Coefficient
- **Additional**: MAE, R², residual analysis

### Cross-Validation
- Time-series aware validation when temporal data available
- Participant-aware splitting to prevent data leakage

## Conclusions

The models demonstrate varying levels of performance in predicting CCR from multimodal data. The best performing model provides actionable insights for personalized nutrition recommendations.

---

*Report generated automatically by the CGMacros evaluation pipeline*
"""
    
    return content


if __name__ == "__main__":
    # Example usage
    sample_results = {
        'model_metrics': {
            'linear_regression': {'nrmse': 0.15, 'correlation': 0.65, 'rmse': 0.12, 'mae': 0.09, 'r2': 0.42},
            'random_forest': {'nrmse': 0.12, 'correlation': 0.75, 'rmse': 0.10, 'mae': 0.07, 'r2': 0.56},
            'xgboost': {'nrmse': 0.11, 'correlation': 0.78, 'rmse': 0.09, 'mae': 0.06, 'r2': 0.61}
        }
    }
    
    sample_features = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'ccr': [0.3, 0.5, 0.7]
    })
    
    # Generate report content
    report = create_report_content(sample_results, sample_features)
    print(report)