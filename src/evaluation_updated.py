"""
Comprehensive Evaluation Framework for CGMacros CCR Prediction

This module implements participant-aware validation, extensive metrics, and model comparison
specifically designed for the CGMacros dataset challenges.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, median_absolute_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing additional libraries
try:
    from sklearn.metrics import mean_absolute_percentage_error
    MAPE_AVAILABLE = True
except ImportError:
    MAPE_AVAILABLE = False

logger = logging.getLogger(__name__)

class ParticipantAwareValidator:
    """
    Validation framework that prevents data leakage by ensuring participant separation.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def create_participant_splits(self, df: pd.DataFrame, 
                                n_splits: int = 5,
                                test_size: float = 0.2) -> Tuple[List[Tuple], Dict]:
        """
        Create participant-aware train/validation/test splits.
        
        Args:
            df: DataFrame with participant_id column
            n_splits: Number of CV splits
            test_size: Fraction of participants for test set
            
        Returns:
            Tuple of (CV splits, test split info)
        """
        participants = df['participant_id'].unique()
        np.random.seed(self.random_state)
        np.random.shuffle(participants)
        
        # Create test set
        n_test_participants = max(1, int(len(participants) * test_size))
        test_participants = participants[-n_test_participants:]
        train_val_participants = participants[:-n_test_participants]
        
        # Create cross-validation splits
        cv_splits = []
        group_kfold = GroupKFold(n_splits=n_splits)
        
        # Use participant IDs as groups for GroupKFold
        dummy_X = np.zeros(len(train_val_participants))
        dummy_y = np.zeros(len(train_val_participants))
        
        for train_idx, val_idx in group_kfold.split(dummy_X, dummy_y, groups=train_val_participants):
            train_participants = train_val_participants[train_idx]
            val_participants = train_val_participants[val_idx]
            
            train_mask = df['participant_id'].isin(train_participants)
            val_mask = df['participant_id'].isin(val_participants)
            
            cv_splits.append((
                df.index[train_mask].tolist(),
                df.index[val_mask].tolist()
            ))
        
        # Create test split
        test_mask = df['participant_id'].isin(test_participants)
        test_split = {
            'test_indices': df.index[test_mask].tolist(),
            'test_participants': test_participants.tolist(),
            'train_val_participants': train_val_participants.tolist()
        }
        
        logger.info(f"Created {n_splits} CV splits with {len(test_participants)} test participants")
        return cv_splits, test_split
    
    def time_series_splits(self, df: pd.DataFrame, 
                          n_splits: int = 5) -> List[Tuple]:
        """
        Create time-based splits for time-series validation.
        
        Args:
            df: DataFrame with Timestamp column
            n_splits: Number of time-based splits
            
        Returns:
            List of (train_indices, val_indices) tuples
        """
        # Sort by timestamp
        df_sorted = df.sort_values('Timestamp').copy()
        
        ts_splits = []
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for train_idx, val_idx in tscv.split(df_sorted):
            train_indices = df_sorted.iloc[train_idx].index.tolist()
            val_indices = df_sorted.iloc[val_idx].index.tolist()
            ts_splits.append((train_indices, val_indices))
        
        logger.info(f"Created {n_splits} time-series splits")
        return ts_splits

class MetricsCalculator:
    """
    Comprehensive metrics calculation for regression tasks.
    """
    
    def __init__(self):
        pass
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of basic metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'median_ae': median_absolute_error(y_true, y_pred)
        }
        
        # Add MAPE if available
        if MAPE_AVAILABLE:
            # Avoid division by zero
            y_true_safe = np.where(y_true == 0, 1e-8, y_true)
            metrics['mape'] = mean_absolute_percentage_error(y_true_safe, y_pred)
        else:
            # Calculate MAPE manually
            y_true_safe = np.where(y_true == 0, 1e-8, y_true)
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100
        
        return metrics
    
    def calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate advanced regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of advanced metrics
        """
        residuals = y_true - y_pred
        
        metrics = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'skewness_residual': stats.skew(residuals),
            'kurtosis_residual': stats.kurtosis(residuals),
            'max_error': np.max(np.abs(residuals)),
            'q95_error': np.percentile(np.abs(residuals), 95),
            'q90_error': np.percentile(np.abs(residuals), 90),
            'q75_error': np.percentile(np.abs(residuals), 75)
        }
        
        # Correlation between true and predicted
        if len(np.unique(y_pred)) > 1:
            metrics['pearson_correlation'] = stats.pearsonr(y_true, y_pred)[0]
            metrics['spearman_correlation'] = stats.spearmanr(y_true, y_pred)[0]
        else:
            metrics['pearson_correlation'] = 0.0
            metrics['spearman_correlation'] = 0.0
        
        return metrics
    
    def calculate_ccr_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate CCR-specific metrics considering the [0,1] range.
        
        Args:
            y_true: True CCR values
            y_pred: Predicted CCR values
            
        Returns:
            Dictionary of CCR-specific metrics
        """
        # Ensure predictions are in valid range
        y_pred_clipped = np.clip(y_pred, 0, 1)
        
        metrics = {
            'ccr_rmse': np.sqrt(mean_squared_error(y_true, y_pred_clipped)),
            'ccr_mae': mean_absolute_error(y_true, y_pred_clipped),
            'ccr_r2': r2_score(y_true, y_pred_clipped),
            'out_of_range_count': np.sum((y_pred < 0) | (y_pred > 1)),
            'out_of_range_percentage': np.mean((y_pred < 0) | (y_pred > 1)) * 100
        }
        
        # Binned accuracy (useful for CCR analysis)
        ccr_bins = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        true_bins = np.digitize(y_true, ccr_bins) - 1
        pred_bins = np.digitize(y_pred_clipped, ccr_bins) - 1
        
        metrics['bin_accuracy'] = np.mean(true_bins == pred_bins)
        metrics['bin_mae'] = mean_absolute_error(true_bins, pred_bins)
        
        return metrics

class ModelEvaluator:
    """
    Main evaluation class that orchestrates all evaluation components.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.validator = ParticipantAwareValidator(random_state)
        self.metrics_calc = MetricsCalculator()
        
        self.evaluation_results = {}
        
    def evaluate_single_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                            model_name: str, cv_splits: List[Tuple]) -> Dict[str, Any]:
        """
        Evaluate a single model using cross-validation.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target vector
            model_name: Name of the model
            cv_splits: Cross-validation splits
            
        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        
        fold_results = []
        all_predictions = []
        all_true_values = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            # Get fold data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Clone and train model for this fold
                if hasattr(model, 'fit'):
                    fold_model = self._clone_model(model)
                    fold_model.fit(X_train, y_train)
                    y_pred = fold_model.predict(X_val)
                else:
                    # For pre-trained models like neural networks
                    y_pred = model.predict(X_val)
                    if len(y_pred.shape) > 1:
                        y_pred = y_pred.flatten()
                
                # Calculate metrics for this fold
                basic_metrics = self.metrics_calc.calculate_basic_metrics(y_val, y_pred)
                advanced_metrics = self.metrics_calc.calculate_advanced_metrics(y_val, y_pred)
                ccr_metrics = self.metrics_calc.calculate_ccr_specific_metrics(y_val, y_pred)
                
                fold_result = {
                    'fold': fold_idx,
                    **basic_metrics,
                    **advanced_metrics,
                    **ccr_metrics
                }
                fold_results.append(fold_result)
                
                # Store predictions for overall analysis
                all_predictions.extend(y_pred.tolist())
                all_true_values.extend(y_val.tolist())
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name} on fold {fold_idx}: {e}")
                continue
        
        if not fold_results:
            logger.error(f"No successful evaluations for {model_name}")
            return {}
        
        # Aggregate results across folds
        aggregated_results = self._aggregate_fold_results(fold_results)
        
        # Overall metrics using all predictions
        if all_predictions:
            overall_basic = self.metrics_calc.calculate_basic_metrics(
                np.array(all_true_values), np.array(all_predictions)
            )
            overall_advanced = self.metrics_calc.calculate_advanced_metrics(
                np.array(all_true_values), np.array(all_predictions)
            )
            overall_ccr = self.metrics_calc.calculate_ccr_specific_metrics(
                np.array(all_true_values), np.array(all_predictions)
            )
            
            aggregated_results['overall'] = {
                **overall_basic,
                **overall_advanced,
                **overall_ccr
            }
        
        aggregated_results['model_name'] = model_name
        aggregated_results['n_folds'] = len(fold_results)
        
        return aggregated_results
    
    def evaluate_multiple_models(self, models: Dict[str, Any], 
                                X: np.ndarray, y: np.ndarray,
                                cv_splits: List[Tuple]) -> Dict[str, Dict]:
        """
        Evaluate multiple models.
        
        Args:
            models: Dictionary of models to evaluate
            X: Feature matrix
            y: Target vector
            cv_splits: Cross-validation splits
            
        Returns:
            Dictionary of evaluation results for all models
        """
        logger.info(f"Evaluating {len(models)} models...")
        
        results = {}
        for model_name, model in models.items():
            try:
                model_results = self.evaluate_single_model(model, X, y, model_name, cv_splits)
                if model_results:
                    results[model_name] = model_results
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                continue
        
        self.evaluation_results.update(results)
        return results
    
    def evaluate_with_participant_splits(self, models: Dict[str, Any],
                                       df: pd.DataFrame, 
                                       feature_cols: List[str],
                                       target_col: str = 'CCR') -> Dict[str, Dict]:
        """
        Evaluate models using participant-aware splits.
        
        Args:
            models: Dictionary of models to evaluate
            df: DataFrame with data
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Dictionary of evaluation results
        """
        # Prepare data
        X = df[feature_cols].fillna(0).values
        y = df[target_col].values
        
        # Create participant-aware splits
        cv_splits, test_split = self.validator.create_participant_splits(df)
        
        # Evaluate models
        results = self.evaluate_multiple_models(models, X, y, cv_splits)
        
        # Store split information
        for model_name in results:
            results[model_name]['split_info'] = {
                'n_cv_splits': len(cv_splits),
                'test_participants': test_split['test_participants'],
                'n_test_participants': len(test_split['test_participants'])
            }
        
        return results
    
    def _clone_model(self, model: Any) -> Any:
        """
        Clone a model for cross-validation.
        
        Args:
            model: Model to clone
            
        Returns:
            Cloned model
        """
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # For non-sklearn models, return the original
            return model
    
    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate results across CV folds.
        
        Args:
            fold_results: List of results from each fold
            
        Returns:
            Aggregated results with mean and std
        """
        if not fold_results:
            return {}
        
        # Get all metric names
        metric_names = set()
        for result in fold_results:
            metric_names.update(result.keys())
        metric_names.discard('fold')  # Remove fold number
        
        aggregated = {}
        for metric in metric_names:
            values = [result.get(metric, np.nan) for result in fold_results]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
        
        return aggregated

class ResultsVisualizer:
    """
    Visualization tools for evaluation results.
    """
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_model_comparison(self, results: Dict[str, Dict], 
                            metric: str = 'rmse_mean',
                            save_path: Optional[str] = None) -> None:
        """
        Plot comparison of models for a specific metric.
        
        Args:
            results: Dictionary of evaluation results
            metric: Metric to compare
            save_path: Path to save the plot
        """
        model_names = []
        metric_values = []
        metric_stds = []
        
        for model_name, model_results in results.items():
            if metric in model_results:
                model_names.append(model_name)
                metric_values.append(model_results[metric])
                # Get standard deviation if available
                std_metric = metric.replace('_mean', '_std')
                metric_stds.append(model_results.get(std_metric, 0))
        
        if not model_names:
            logger.warning(f"No results found for metric: {metric}")
            return
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values, yerr=metric_stds, capsize=5)
        plt.title(f'Model Comparison: {metric.replace("_", " ").title()}')
        plt.xlabel('Models')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_stds)*0.1,
                    f'{value:.4f}', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_heatmap(self, results: Dict[str, Dict],
                           metrics: List[str] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Plot heatmap of multiple metrics across models.
        
        Args:
            results: Dictionary of evaluation results
            metrics: List of metrics to include
            save_path: Path to save the plot
        """
        if metrics is None:
            metrics = ['rmse_mean', 'mae_mean', 'r2_mean', 'ccr_rmse_mean', 'mape_mean']
        
        # Prepare data for heatmap
        heatmap_data = []
        model_names = []
        
        for model_name, model_results in results.items():
            row = []
            for metric in metrics:
                row.append(model_results.get(metric, np.nan))
            heatmap_data.append(row)
            model_names.append(model_name)
        
        # Create DataFrame
        df_heatmap = pd.DataFrame(heatmap_data, 
                                 index=model_names, 
                                 columns=[m.replace('_mean', '').upper() for m in metrics])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_heatmap, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Metric Value'})
        plt.title('Model Performance Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_scatter(self, y_true: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Model",
                              save_path: Optional[str] = None) -> None:
        """
        Plot scatter plot of true vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('True CCR')
        plt.ylabel('Predicted CCR')
        plt.title(f'{model_name}: True vs Predicted CCR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add metrics text
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class ModelComparison:
    """
    Statistical comparison of model performance.
    """
    
    def __init__(self):
        pass
    
    def compare_models_statistical(self, results: Dict[str, Dict], 
                                 metric: str = 'rmse') -> Dict[str, Any]:
        """
        Perform statistical comparison of models.
        
        Args:
            results: Dictionary of evaluation results
            metric: Metric to compare
            
        Returns:
            Dictionary with comparison results
        """
        # Extract metric values for each model
        model_metrics = {}
        for model_name, model_results in results.items():
            # Look for fold-wise results
            fold_values = []
            for key, value in model_results.items():
                if key.startswith('fold_') and metric in key:
                    fold_values.append(value)
            
            if fold_values:
                model_metrics[model_name] = fold_values
            else:
                # Use mean value if fold-wise not available
                mean_key = f'{metric}_mean'
                if mean_key in model_results:
                    model_metrics[model_name] = [model_results[mean_key]]
        
        if len(model_metrics) < 2:
            logger.warning("Need at least 2 models for statistical comparison")
            return {}
        
        # Perform pairwise comparisons
        comparison_results = {}
        model_names = list(model_metrics.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                try:
                    # Perform t-test
                    stat, p_value = stats.ttest_ind(model_metrics[model1], model_metrics[model2])
                    
                    comparison_results[f'{model1}_vs_{model2}'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'model1_mean': np.mean(model_metrics[model1]),
                        'model2_mean': np.mean(model_metrics[model2]),
                        'better_model': model1 if np.mean(model_metrics[model1]) < np.mean(model_metrics[model2]) else model2
                    }
                except Exception as e:
                    logger.warning(f"Failed to compare {model1} and {model2}: {e}")
                    continue
        
        return comparison_results
    
    def rank_models(self, results: Dict[str, Dict], 
                   metrics: List[str] = None) -> pd.DataFrame:
        """
        Rank models based on multiple metrics.
        
        Args:
            results: Dictionary of evaluation results
            metrics: List of metrics to consider
            
        Returns:
            DataFrame with model rankings
        """
        if metrics is None:
            metrics = ['rmse_mean', 'mae_mean', 'r2_mean', 'ccr_rmse_mean']
        
        ranking_data = []
        
        for model_name, model_results in results.items():
            row = {'model': model_name}
            for metric in metrics:
                row[metric] = model_results.get(metric, np.inf)
            ranking_data.append(row)
        
        df_ranking = pd.DataFrame(ranking_data)
        
        # Rank for each metric (lower is better for error metrics, higher for R²)
        for metric in metrics:
            if 'r2' in metric:
                df_ranking[f'{metric}_rank'] = df_ranking[metric].rank(ascending=False)
            else:
                df_ranking[f'{metric}_rank'] = df_ranking[metric].rank(ascending=True)
        
        # Calculate average rank
        rank_cols = [col for col in df_ranking.columns if col.endswith('_rank')]
        df_ranking['avg_rank'] = df_ranking[rank_cols].mean(axis=1)
        df_ranking['overall_rank'] = df_ranking['avg_rank'].rank()
        
        return df_ranking.sort_values('avg_rank')

class EvaluationReport:
    """
    Generate comprehensive evaluation reports.
    """
    
    def __init__(self):
        self.visualizer = ResultsVisualizer()
        self.comparator = ModelComparison()
    
    def generate_report(self, results: Dict[str, Dict], 
                       output_dir: str = "results/evaluation") -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Dictionary of evaluation results
            output_dir: Directory to save report and plots
            
        Returns:
            Path to generated report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_path, 'w') as f:
            f.write("# CGMacros CCR Prediction - Model Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model performance summary
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | RMSE | MAE | R² | CCR RMSE | MAPE |\n")
            f.write("|-------|------|-----|----|---------|---------|\n")
            
            for model_name, model_results in results.items():
                rmse = model_results.get('rmse_mean', 'N/A')
                mae = model_results.get('mae_mean', 'N/A')
                r2 = model_results.get('r2_mean', 'N/A')
                ccr_rmse = model_results.get('ccr_rmse_mean', 'N/A')
                mape = model_results.get('mape_mean', 'N/A')
                
                f.write(f"| {model_name} | {rmse:.4f} | {mae:.4f} | {r2:.4f} | {ccr_rmse:.4f} | {mape:.4f} |\n")
            
            # Model rankings
            f.write("\n## Model Rankings\n\n")
            rankings = self.comparator.rank_models(results)
            f.write(rankings.to_markdown(index=False))
            
            # Statistical comparisons
            f.write("\n## Statistical Model Comparisons\n\n")
            comparisons = self.comparator.compare_models_statistical(results)
            for comparison, stats in comparisons.items():
                f.write(f"### {comparison}\n")
                f.write(f"- p-value: {stats['p_value']:.6f}\n")
                f.write(f"- Significant: {stats['significant']}\n")
                f.write(f"- Better model: {stats['better_model']}\n\n")
            
            # Detailed results
            f.write("\n## Detailed Results\n\n")
            for model_name, model_results in results.items():
                f.write(f"### {model_name}\n\n")
                for metric, value in model_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"- {metric}: {value:.6f}\n")
                    else:
                        f.write(f"- {metric}: {value}\n")
                f.write("\n")
        
        logger.info(f"Evaluation report saved to: {report_path}")
        return report_path