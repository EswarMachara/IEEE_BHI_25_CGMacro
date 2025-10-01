"""
Models Module

This module implements baseline, intermediate, and advanced models
for predicting Carbohydrate Caloric Ratio (CCR).
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main class for training and evaluating models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'ccr', 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            df: Input DataFrame with features and target
            target_col: Name of target column
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        # Separate features and target
        y = df[target_col].values
        X = df.drop(columns=[target_col, 'participant_id'], errors='ignore')
        
        # Remove non-numeric columns
        X = X.select_dtypes(include=[np.number])
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove rows with missing target values
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Split data
        if 'participant_id' in df.columns:
            # Use participant-aware splitting if possible
            participants = df[valid_indices]['participant_id'].values
            unique_participants = np.unique(participants)
            
            # Split participants
            train_participants, test_participants = train_test_split(
                unique_participants, test_size=test_size, random_state=self.random_state
            )
            
            train_mask = np.isin(participants, train_participants)
            test_mask = np.isin(participants, test_participants)
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            # Standard random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train baseline linear models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training baseline models...")
        
        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scalers['baseline'] = scaler
        
        baseline_models = {}
        
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        baseline_models['linear_regression'] = lr
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0, random_state=self.random_state)
        ridge.fit(X_train_scaled, y_train)
        baseline_models['ridge'] = ridge
        
        # Lasso Regression
        lasso = Lasso(alpha=0.1, random_state=self.random_state, max_iter=1000)
        lasso.fit(X_train_scaled, y_train)
        baseline_models['lasso'] = lasso
        
        # Elastic Net
        elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=1000)
        elastic.fit(X_train_scaled, y_train)
        baseline_models['elastic_net'] = elastic
        
        self.models.update(baseline_models)
        logger.info("Baseline models trained successfully")
        
        return baseline_models
    
    def train_intermediate_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train intermediate ensemble models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training intermediate models...")
        
        intermediate_models = {}
        
        # Random Forest
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        intermediate_models['random_forest'] = rf
        
        # Store feature importance
        if hasattr(rf, 'feature_importances_'):
            self.feature_importance['random_forest'] = rf.feature_importances_
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        intermediate_models['xgboost'] = xgb_model
        
        # Store feature importance
        if hasattr(xgb_model, 'feature_importances_'):
            self.feature_importance['xgboost'] = xgb_model.feature_importances_
        
        self.models.update(intermediate_models)
        logger.info("Intermediate models trained successfully")
        
        return intermediate_models
    
    def train_advanced_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train advanced models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Training advanced models...")
        
        advanced_models = {}
        
        # LightGBM
        lgb_model = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        advanced_models['lightgbm'] = lgb_model
        
        # Store feature importance
        if hasattr(lgb_model, 'feature_importances_'):
            self.feature_importance['lightgbm'] = lgb_model.feature_importances_
        
        self.models.update(advanced_models)
        logger.info("Advanced models trained successfully")
        
        return advanced_models
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Scale features if it's a baseline model
        if model_name in ['linear_regression', 'ridge', 'lasso', 'elastic_net']:
            if 'baseline' in self.scalers:
                X = self.scalers['baseline'].transform(X)
        
        return model.predict(X)
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(model_name, X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Normalized RMSE (using target range)
        y_range = y_test.max() - y_test.min()
        nrmse = rmse / y_range if y_range > 0 else np.inf
        
        # Correlation
        correlation = np.corrcoef(y_test, y_pred)[0, 1] if len(y_test) > 1 else 0
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'nrmse': nrmse,
            'correlation': correlation
        }
        
        return metrics
    
    def cross_validate(self, model_name: str, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation for a model.
        
        Args:
            model_name: Name of the model
            X: Features
            y: Targets
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        # Prepare data for cross-validation
        X_cv = X.copy()
        if model_name in ['linear_regression', 'ridge', 'lasso', 'elastic_net']:
            if 'baseline' in self.scalers:
                X_cv = self.scalers['baseline'].transform(X_cv)
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_cv, y, cv=cv, scoring='neg_mean_squared_error')
        
        cv_metrics = {
            'cv_rmse_mean': np.sqrt(-cv_scores.mean()),
            'cv_rmse_std': np.sqrt(cv_scores.std()),
            'cv_scores': cv_scores
        }
        
        return cv_metrics
    
    def save_models(self, save_dir: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            save_dir: Directory to save models
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_file = save_path / f"{model_name}.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved model: {model_file}")
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_file = save_path / f"scaler_{scaler_name}.joblib"
            joblib.dump(scaler, scaler_file)
            logger.info(f"Saved scaler: {scaler_file}")
        
        # Save feature importance
        if self.feature_importance:
            importance_file = save_path / "feature_importance.joblib"
            joblib.dump(self.feature_importance, importance_file)
            logger.info(f"Saved feature importance: {importance_file}")
    
    def load_models(self, save_dir: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            save_dir: Directory containing saved models
        """
        save_path = Path(save_dir)
        
        # Load models
        for model_file in save_path.glob("*.joblib"):
            if not model_file.name.startswith("scaler_") and model_file.name != "feature_importance.joblib":
                model_name = model_file.stem
                self.models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded model: {model_name}")
        
        # Load scalers
        for scaler_file in save_path.glob("scaler_*.joblib"):
            scaler_name = scaler_file.stem.replace("scaler_", "")
            self.scalers[scaler_name] = joblib.load(scaler_file)
            logger.info(f"Loaded scaler: {scaler_name}")
        
        # Load feature importance
        importance_file = save_path / "feature_importance.joblib"
        if importance_file.exists():
            self.feature_importance = joblib.load(importance_file)
            logger.info("Loaded feature importance")


def train_all_models(df: pd.DataFrame, save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Train all model types and return results.
    
    Args:
        df: DataFrame with features and target
        save_dir: Optional directory to save models
        
    Returns:
        Dictionary containing trained models and evaluation results
    """
    logger.info("Starting complete model training pipeline")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df)
    
    # Train all model types
    baseline_models = trainer.train_baseline_models(X_train, y_train)
    intermediate_models = trainer.train_intermediate_models(X_train, y_train)
    advanced_models = trainer.train_advanced_models(X_train, y_train)
    
    # Evaluate all models
    all_models = {**baseline_models, **intermediate_models, **advanced_models}
    evaluation_results = {}
    
    for model_name in all_models.keys():
        try:
            metrics = trainer.evaluate_model(model_name, X_test, y_test)
            evaluation_results[model_name] = metrics
            logger.info(f"{model_name} - NRMSE: {metrics['nrmse']:.4f}, Correlation: {metrics['correlation']:.4f}")
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {str(e)}")
    
    # Save models if directory provided
    if save_dir:
        trainer.save_models(save_dir)
    
    results = {
        'trainer': trainer,
        'models': all_models,
        'evaluation_results': evaluation_results,
        'data_splits': {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        },
        'feature_names': list(df.drop(columns=['ccr', 'participant_id'], errors='ignore').select_dtypes(include=[np.number]).columns)
    }
    
    logger.info("Model training pipeline completed")
    
    return results


def get_best_model(evaluation_results: Dict[str, Dict[str, float]], metric: str = 'nrmse') -> Tuple[str, float]:
    """
    Get the best performing model based on a metric.
    
    Args:
        evaluation_results: Dictionary of model evaluation results
        metric: Metric to use for comparison ('nrmse', 'correlation', 'rmse', etc.)
        
    Returns:
        Tuple of (best_model_name, best_metric_value)
    """
    if not evaluation_results:
        return None, None
    
    # For correlation, higher is better; for others, lower is better
    reverse = metric == 'correlation'
    
    best_model = None
    best_value = float('-inf') if reverse else float('inf')
    
    for model_name, metrics in evaluation_results.items():
        if metric in metrics:
            value = metrics[metric]
            if (reverse and value > best_value) or (not reverse and value < best_value):
                best_model = model_name
                best_value = value
    
    return best_model, best_value


if __name__ == "__main__":
    # Example usage with sample data
    sample_data = pd.DataFrame({
        'participant_id': [1, 1, 2, 2, 3, 3] * 20,
        'feature1': np.random.normal(0, 1, 120),
        'feature2': np.random.normal(0, 1, 120),
        'feature3': np.random.normal(0, 1, 120),
        'ccr': np.random.uniform(0.2, 0.8, 120)
    })
    
    results = train_all_models(sample_data)
    
    print("Evaluation Results:")
    for model_name, metrics in results['evaluation_results'].items():
        print(f"{model_name}: NRMSE={metrics['nrmse']:.4f}, Correlation={metrics['correlation']:.4f}")
    
    best_model, best_score = get_best_model(results['evaluation_results'], 'nrmse')
    print(f"\nBest model: {best_model} (NRMSE: {best_score:.4f})")