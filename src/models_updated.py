"""
Advanced Model Implementations for CGMacros CCR Prediction

This module implements various model architectures for predicting Carbohydrate Caloric Ratio
from multimodal data including baseline models, time-series models, and multimodal fusion.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try importing deep learning libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input, concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("TensorFlow not available. Deep learning models will be skipped.")

logger = logging.getLogger(__name__)

class BaselineModels:
    """
    Baseline model implementations for CCR prediction.
    These provide the foundation for comparison with advanced models.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        
    def create_linear_models(self) -> Dict[str, Any]:
        """
        Create linear regression models with different regularization.
        
        Returns:
            Dictionary of linear models
        """
        linear_models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state)
        }
        
        # Hyperparameter grids for optimization
        param_grids = {
            'ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'lasso': {'alpha': [0.01, 0.1, 1.0, 10.0]},
            'elastic_net': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}
        }
        
        return linear_models, param_grids
    
    def create_tree_models(self) -> Dict[str, Any]:
        """
        Create tree-based models.
        
        Returns:
            Dictionary of tree-based models
        """
        tree_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBRegressor(
                random_state=self.random_state,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        }
        
        return tree_models, param_grids

class TimeSeriesModels:
    """
    Time-series specific models for capturing temporal patterns in glucose data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        
    def prepare_time_series_data(self, df: pd.DataFrame, 
                                sequence_length: int = 24,
                                target_col: str = 'CCR') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for time-series models.
        
        Args:
            df: DataFrame with time-series data
            sequence_length: Length of input sequences
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays for time-series modeling
        """
        # Sort by participant and timestamp
        df_sorted = df.sort_values(['participant_id', 'Timestamp']).copy()
        
        # Select time-series features
        ts_features = ['Libre GL', 'Dexcom GL', 'HR', 'METs', 'Calories']
        available_features = [col for col in ts_features if col in df.columns]
        
        X_sequences = []
        y_sequences = []
        
        for participant_id in df_sorted['participant_id'].unique():
            participant_data = df_sorted[df_sorted['participant_id'] == participant_id]
            
            # Extract features and target
            features = participant_data[available_features].fillna(method='ffill').fillna(0).values
            targets = participant_data[target_col].fillna(0).values
            
            # Create sequences
            for i in range(len(features) - sequence_length + 1):
                if targets[i + sequence_length - 1] > 0:  # Only include sequences with valid target
                    X_sequences.append(features[i:i + sequence_length])
                    y_sequences.append(targets[i + sequence_length - 1])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def create_lstm_model(self, input_shape: Tuple, 
                         hidden_units: int = 64,
                         dropout_rate: float = 0.2) -> Optional[Model]:
        """
        Create LSTM model for time-series CCR prediction.
        
        Args:
            input_shape: Shape of input sequences
            hidden_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled LSTM model or None if TensorFlow not available
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot create LSTM model.")
            return None
        
        model = Sequential([
            LSTM(hidden_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(hidden_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # CCR is between 0 and 1
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def create_gru_model(self, input_shape: Tuple,
                        hidden_units: int = 64,
                        dropout_rate: float = 0.2) -> Optional[Model]:
        """
        Create GRU model for time-series CCR prediction.
        
        Args:
            input_shape: Shape of input sequences
            hidden_units: Number of GRU units
            dropout_rate: Dropout rate for regularization
            
        Returns:
            Compiled GRU model or None if TensorFlow not available
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot create GRU model.")
            return None
        
        model = Sequential([
            GRU(hidden_units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            GRU(hidden_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model

class MultimodalFusionModels:
    """
    Advanced models that combine multiple data modalities for CCR prediction.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def create_multimodal_neural_network(self, 
                                       glucose_dim: int,
                                       demographic_dim: int, 
                                       microbiome_dim: int,
                                       gut_health_dim: int) -> Optional[Model]:
        """
        Create a neural network that fuses multiple data modalities.
        
        Args:
            glucose_dim: Dimension of glucose features
            demographic_dim: Dimension of demographic features
            microbiome_dim: Dimension of microbiome features
            gut_health_dim: Dimension of gut health features
            
        Returns:
            Compiled multimodal neural network or None if TensorFlow not available
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot create multimodal neural network.")
            return None
        
        # Define input layers for each modality
        glucose_input = Input(shape=(glucose_dim,), name='glucose_input')
        demographic_input = Input(shape=(demographic_dim,), name='demographic_input')
        microbiome_input = Input(shape=(microbiome_dim,), name='microbiome_input')
        gut_health_input = Input(shape=(gut_health_dim,), name='gut_health_input')
        
        # Process each modality separately
        glucose_branch = Dense(64, activation='relu')(glucose_input)
        glucose_branch = BatchNormalization()(glucose_branch)
        glucose_branch = Dropout(0.3)(glucose_branch)
        glucose_branch = Dense(32, activation='relu')(glucose_branch)
        
        demographic_branch = Dense(32, activation='relu')(demographic_input)
        demographic_branch = BatchNormalization()(demographic_branch)
        demographic_branch = Dropout(0.2)(demographic_branch)
        demographic_branch = Dense(16, activation='relu')(demographic_branch)
        
        microbiome_branch = Dense(128, activation='relu')(microbiome_input)
        microbiome_branch = BatchNormalization()(microbiome_branch)
        microbiome_branch = Dropout(0.4)(microbiome_branch)
        microbiome_branch = Dense(64, activation='relu')(microbiome_branch)
        microbiome_branch = Dense(32, activation='relu')(microbiome_branch)
        
        gut_health_branch = Dense(32, activation='relu')(gut_health_input)
        gut_health_branch = BatchNormalization()(gut_health_branch)
        gut_health_branch = Dropout(0.2)(gut_health_branch)
        gut_health_branch = Dense(16, activation='relu')(gut_health_branch)
        
        # Fuse all modalities
        fused = concatenate([glucose_branch, demographic_branch, microbiome_branch, gut_health_branch])
        fused = Dense(128, activation='relu')(fused)
        fused = BatchNormalization()(fused)
        fused = Dropout(0.3)(fused)
        fused = Dense(64, activation='relu')(fused)
        fused = Dense(32, activation='relu')(fused)
        output = Dense(1, activation='sigmoid', name='ccr_output')(fused)
        
        model = Model(inputs=[glucose_input, demographic_input, microbiome_input, gut_health_input],
                     outputs=output)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_multimodal_data(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Prepare data for multimodal fusion models.
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Dictionary with separate arrays for each modality
        """
        # Define feature categories
        glucose_features = [col for col in df.columns if 'gl' in col.lower() or 'glucose' in col.lower()]
        demographic_features = [col for col in df.columns if col in ['Age', 'Gender', 'BMI', 'A1c', 'gender_encoded']]
        microbiome_features = [col for col in df.columns if col.startswith('microbiome_') or 
                             any(bacteria in col for bacteria in ['Bacteroides', 'Bifidobacterium', 'Lactobacillus'])]
        gut_health_features = [col for col in df.columns if any(term in col for term in 
                             ['Gut', 'LPS', 'Biofilm', 'Production', 'Metabolism', 'gut_health'])]
        
        # Add activity and temporal features to glucose category
        glucose_features.extend([col for col in df.columns if col in ['HR', 'METs', 'Calories', 'hour', 'day_of_week']])
        
        # Remove duplicates and ensure columns exist
        glucose_features = list(set([col for col in glucose_features if col in df.columns]))
        demographic_features = list(set([col for col in demographic_features if col in df.columns]))
        microbiome_features = list(set([col for col in microbiome_features if col in df.columns]))
        gut_health_features = list(set([col for col in gut_health_features if col in df.columns]))
        
        # Prepare data arrays
        data = {
            'glucose': df[glucose_features].fillna(0).values if glucose_features else np.zeros((len(df), 1)),
            'demographic': df[demographic_features].fillna(0).values if demographic_features else np.zeros((len(df), 1)),
            'microbiome': df[microbiome_features].fillna(0).values if microbiome_features else np.zeros((len(df), 1)),
            'gut_health': df[gut_health_features].fillna(0).values if gut_health_features else np.zeros((len(df), 1))
        }
        
        return data

class EnsembleModels:
    """
    Ensemble methods that combine predictions from multiple models.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.base_models = {}
        self.meta_model = None
        
    def create_stacking_ensemble(self, base_models: Dict[str, Any]) -> Any:
        """
        Create a stacking ensemble of base models.
        
        Args:
            base_models: Dictionary of trained base models
            
        Returns:
            Meta-model for stacking ensemble
        """
        from sklearn.ensemble import StackingRegressor
        
        # Convert models to list of (name, estimator) tuples
        estimators = [(name, model) for name, model in base_models.items()]
        
        # Use Ridge regression as meta-learner
        meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
        
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        return stacking_model
    
    def create_voting_ensemble(self, base_models: Dict[str, Any]) -> Any:
        """
        Create a voting ensemble of base models.
        
        Args:
            base_models: Dictionary of trained base models
            
        Returns:
            Voting ensemble model
        """
        from sklearn.ensemble import VotingRegressor
        
        # Convert models to list of (name, estimator) tuples
        estimators = [(name, model) for name, model in base_models.items()]
        
        voting_model = VotingRegressor(
            estimators=estimators,
            n_jobs=-1
        )
        
        return voting_model

class FeatureSelection:
    """
    Feature selection methods for high-dimensional data.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
    def select_features_univariate(self, X: np.ndarray, y: np.ndarray, k: int = 100) -> Any:
        """
        Select features using univariate statistical tests.
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            Fitted feature selector
        """
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))
        selector.fit(X, y)
        return selector
    
    def select_features_rfe(self, estimator: Any, X: np.ndarray, y: np.ndarray, 
                           n_features: int = 100) -> Any:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            estimator: Base estimator for RFE
            X: Feature matrix
            y: Target vector
            n_features: Number of features to select
            
        Returns:
            Fitted RFE selector
        """
        rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
        rfe.fit(X, y)
        return rfe
    
    def select_features_correlation(self, X: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            X: Feature DataFrame
            threshold: Correlation threshold
            
        Returns:
            List of selected feature names
        """
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find pairs of highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Select features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        # Return features to keep
        selected_features = [col for col in X.columns if col not in to_drop]
        return selected_features

class ModelTrainer:
    """
    Main class for training and managing all model types.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.baseline_models = BaselineModels(random_state)
        self.ts_models = TimeSeriesModels(random_state)
        self.multimodal_models = MultimodalFusionModels(random_state)
        self.ensemble_models = EnsembleModels(random_state)
        self.feature_selector = FeatureSelection(random_state)
        
        self.trained_models = {}
        self.feature_selectors = {}
        self.scalers = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'CCR',
                    scale_features: bool = True,
                    select_features: bool = True,
                    max_features: int = 200) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            scale_features: Whether to scale features
            select_features: Whether to perform feature selection
            max_features: Maximum number of features to select
            
        Returns:
            Tuple of (prepared data dict, target array)
        """
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, 'participant_id', 'Timestamp']]
        X = df[feature_cols].copy()
        y = df[target_col].values
        
        # Handle missing values
        X = X.fillna(0)
        
        # Feature selection for high-dimensional data
        if select_features and X.shape[1] > max_features:
            logger.info(f"Performing feature selection: {X.shape[1]} -> {max_features} features")
            
            # Use correlation-based selection first
            selected_features = self.feature_selector.select_features_correlation(X, threshold=0.95)
            X = X[selected_features]
            
            # If still too many features, use univariate selection
            if X.shape[1] > max_features:
                selector = self.feature_selector.select_features_univariate(X.values, y, k=max_features)
                X = pd.DataFrame(selector.transform(X.values), 
                               columns=X.columns[selector.get_support()],
                               index=X.index)
                self.feature_selectors['univariate'] = selector
        
        # Scale features
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            self.scalers['standard'] = scaler
        else:
            X_scaled = X.values
        
        # Prepare multimodal data
        multimodal_data = self.multimodal_models.prepare_multimodal_data(
            pd.DataFrame(X_scaled, columns=X.columns)
        )
        
        # Add scaled features for baseline models
        multimodal_data['all_features'] = X_scaled
        
        logger.info(f"Data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return multimodal_data, y
    
    def train_baseline_models(self, X: np.ndarray, y: np.ndarray,
                            optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train baseline models.
        
        Args:
            X: Feature matrix
            y: Target vector
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Dictionary of trained baseline models
        """
        logger.info("Training baseline models...")
        
        # Get models and parameter grids
        linear_models, linear_params = self.baseline_models.create_linear_models()
        tree_models, tree_params = self.baseline_models.create_tree_models()
        
        all_models = {**linear_models, **tree_models}
        all_params = {**linear_params, **tree_params}
        
        trained_models = {}
        
        for name, model in all_models.items():
            try:
                if optimize_hyperparameters and name in all_params:
                    # Hyperparameter optimization
                    grid_search = GridSearchCV(
                        model, all_params[name],
                        cv=5, scoring='neg_mean_squared_error',
                        n_jobs=-1
                    )
                    grid_search.fit(X, y)
                    trained_models[name] = grid_search.best_estimator_
                    logger.info(f"Trained {name} with best params: {grid_search.best_params_}")
                else:
                    # Train with default parameters
                    model.fit(X, y)
                    trained_models[name] = model
                    logger.info(f"Trained {name} with default parameters")
                    
            except Exception as e:
                logger.warning(f"Failed to train {name}: {e}")
                continue
        
        self.trained_models.update(trained_models)
        return trained_models
    
    def train_time_series_models(self, df: pd.DataFrame, target_col: str = 'CCR') -> Dict[str, Any]:
        """
        Train time-series models.
        
        Args:
            df: DataFrame with time-series data
            target_col: Target column name
            
        Returns:
            Dictionary of trained time-series models
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("Deep learning not available. Skipping time-series models.")
            return {}
        
        logger.info("Training time-series models...")
        
        # Prepare time-series data
        X_ts, y_ts = self.ts_models.prepare_time_series_data(df, target_col=target_col)
        
        if len(X_ts) == 0:
            logger.warning("No time-series sequences created. Skipping time-series models.")
            return {}
        
        trained_models = {}
        
        # Split data for training
        split_idx = int(0.8 * len(X_ts))
        X_train, X_val = X_ts[:split_idx], X_ts[split_idx:]
        y_train, y_val = y_ts[:split_idx], y_ts[split_idx:]
        
        # Train LSTM
        try:
            lstm_model = self.ts_models.create_lstm_model(X_train.shape[1:])
            if lstm_model is not None:
                history = lstm_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50, batch_size=32,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)
                    ],
                    verbose=0
                )
                trained_models['lstm'] = lstm_model
                logger.info("Trained LSTM model")
        except Exception as e:
            logger.warning(f"Failed to train LSTM: {e}")
        
        # Train GRU
        try:
            gru_model = self.ts_models.create_gru_model(X_train.shape[1:])
            if gru_model is not None:
                history = gru_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50, batch_size=32,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)
                    ],
                    verbose=0
                )
                trained_models['gru'] = gru_model
                logger.info("Trained GRU model")
        except Exception as e:
            logger.warning(f"Failed to train GRU: {e}")
        
        self.trained_models.update(trained_models)
        return trained_models
    
    def train_multimodal_models(self, multimodal_data: Dict[str, np.ndarray], 
                              y: np.ndarray) -> Dict[str, Any]:
        """
        Train multimodal fusion models.
        
        Args:
            multimodal_data: Dictionary with different modality data
            y: Target vector
            
        Returns:
            Dictionary of trained multimodal models
        """
        if not DEEP_LEARNING_AVAILABLE:
            logger.warning("Deep learning not available. Skipping multimodal models.")
            return {}
        
        logger.info("Training multimodal models...")
        
        trained_models = {}
        
        # Split data
        split_idx = int(0.8 * len(y))
        train_data = {key: data[:split_idx] for key, data in multimodal_data.items()}
        val_data = {key: data[split_idx:] for key, data in multimodal_data.items()}
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train multimodal neural network
        try:
            mm_model = self.multimodal_models.create_multimodal_neural_network(
                glucose_dim=train_data['glucose'].shape[1],
                demographic_dim=train_data['demographic'].shape[1],
                microbiome_dim=train_data['microbiome'].shape[1],
                gut_health_dim=train_data['gut_health'].shape[1]
            )
            
            if mm_model is not None:
                history = mm_model.fit(
                    [train_data['glucose'], train_data['demographic'], 
                     train_data['microbiome'], train_data['gut_health']],
                    y_train,
                    validation_data=(
                        [val_data['glucose'], val_data['demographic'],
                         val_data['microbiome'], val_data['gut_health']],
                        y_val
                    ),
                    epochs=100, batch_size=32,
                    callbacks=[
                        EarlyStopping(patience=15, restore_best_weights=True),
                        ReduceLROnPlateau(patience=7, factor=0.5)
                    ],
                    verbose=0
                )
                trained_models['multimodal_nn'] = mm_model
                logger.info("Trained multimodal neural network")
        except Exception as e:
            logger.warning(f"Failed to train multimodal model: {e}")
        
        self.trained_models.update(trained_models)
        return trained_models
    
    def train_ensemble_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble models using trained base models.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Dictionary of trained ensemble models
        """
        logger.info("Training ensemble models...")
        
        # Get sklearn-compatible base models for ensembling
        base_models = {name: model for name, model in self.trained_models.items() 
                      if hasattr(model, 'predict') and not hasattr(model, 'predict_proba')}
        
        if len(base_models) < 2:
            logger.warning("Need at least 2 base models for ensembling. Skipping ensemble models.")
            return {}
        
        trained_ensembles = {}
        
        # Train stacking ensemble
        try:
            stacking_model = self.ensemble_models.create_stacking_ensemble(base_models)
            stacking_model.fit(X, y)
            trained_ensembles['stacking'] = stacking_model
            logger.info("Trained stacking ensemble")
        except Exception as e:
            logger.warning(f"Failed to train stacking ensemble: {e}")
        
        # Train voting ensemble
        try:
            voting_model = self.ensemble_models.create_voting_ensemble(base_models)
            voting_model.fit(X, y)
            trained_ensembles['voting'] = voting_model
            logger.info("Trained voting ensemble")
        except Exception as e:
            logger.warning(f"Failed to train voting ensemble: {e}")
        
        self.trained_models.update(trained_ensembles)
        return trained_ensembles
    
    def train_all_models(self, df: pd.DataFrame, target_col: str = 'CCR',
                        include_time_series: bool = True,
                        include_multimodal: bool = True,
                        include_ensemble: bool = True) -> Dict[str, Any]:
        """
        Train all model types.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            include_time_series: Whether to train time-series models
            include_multimodal: Whether to train multimodal models
            include_ensemble: Whether to train ensemble models
            
        Returns:
            Dictionary of all trained models
        """
        logger.info("Starting comprehensive model training...")
        
        # Prepare data
        multimodal_data, y = self.prepare_data(df, target_col)
        X = multimodal_data['all_features']
        
        # Train baseline models
        baseline_models = self.train_baseline_models(X, y)
        
        # Train time-series models
        if include_time_series:
            ts_models = self.train_time_series_models(df, target_col)
        
        # Train multimodal models
        if include_multimodal:
            mm_models = self.train_multimodal_models(multimodal_data, y)
        
        # Train ensemble models
        if include_ensemble:
            ensemble_models = self.train_ensemble_models(X, y)
        
        logger.info(f"Training completed. Total models trained: {len(self.trained_models)}")
        return self.trained_models