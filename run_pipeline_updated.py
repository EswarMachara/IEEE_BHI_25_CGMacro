"""
Complete End-to-End Pipeline for CGMacros CCR Prediction

This is the main execution script that orchestrates the entire machine learning pipeline
from data loading to model evaluation and results reporting.
"""

import os
import sys
import logging
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
from src.data_loader_updated import DataLoader
from src.feature_engineering_updated import FeatureEngineer
from src.target_updated import compute_ccr, remove_nutrient_columns
from src.models_updated import ModelTrainer
from src.evaluation_updated import ModelEvaluator, EvaluationReport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cgmacros_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CGMacrosPipeline:
    """
    Complete pipeline for CGMacros CCR prediction.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.random_state = self.config.get('random_state', 42)
        
        # Initialize components with memory optimization
        self.data_loader = DataLoader()
        self.feature_engineer = FeatureEngineer(memory_efficient=True)
        self.model_trainer = ModelTrainer(random_state=self.random_state)
        self.evaluator = ModelEvaluator(random_state=self.random_state)
        self.report_generator = EvaluationReport()
        
        # Data storage
        self.raw_data = {}
        self.processed_data = None
        self.feature_data = None
        self.target_data = None
        self.trained_models = {}
        self.evaluation_results = {}
        
        logger.info("CGMacros Pipeline initialized")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """
        Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'data': {
                'raw_data_dir': 'data/raw',
                'processed_data_dir': 'data/processed',
                'cgmacros_dir': 'data/raw/CGMacros_CSVs'
            },
            'features': {
                'glucose_window_hours': [1, 2, 4, 6, 12],
                'activity_window_hours': [1, 2, 4],
                'include_microbiome': True,
                'include_gut_health': True,
                'include_temporal': True,
                'max_features': 2000  # Increased to accommodate ALL 1979 microbiome features + others
            },
            'models': {
                'include_baseline': True,
                'include_time_series': True,
                'include_multimodal': True,
                'include_ensemble': True,
                'optimize_hyperparameters': True
            },
            'evaluation': {
                'cv_splits': 5,
                'test_size': 0.2,
                'metrics': ['rmse', 'mae', 'r2', 'ccr_rmse', 'mape']
            },
            'output': {
                'results_dir': 'results',
                'models_dir': 'models',
                'plots_dir': 'results/plots'
            },
            'random_state': 42
        }
    
    def run_data_loading(self) -> None:
        """
        Execute data loading phase.
        """
        logger.info("=== Phase 1: Data Loading ===")
        
        try:
            # Load CGMacros data
            cgmacros_data = self.data_loader.load_cgmacros_data(
                self.config['data']['cgmacros_dir']
            )
            logger.info(f"Loaded CGMacros data: {cgmacros_data.shape}")
            
            # Load auxiliary data
            bio_data = self.data_loader.load_bio_data(
                os.path.join(self.config['data']['raw_data_dir'], 'bio.csv')
            )
            logger.info(f"Loaded bio data: {bio_data.shape}")
            
            microbes_data = self.data_loader.load_microbes_data(
                os.path.join(self.config['data']['raw_data_dir'], 'microbes.csv')
            )
            logger.info(f"Loaded microbes data: {microbes_data.shape}")
            
            gut_health_data = self.data_loader.load_gut_health_data(
                os.path.join(self.config['data']['raw_data_dir'], 'gut_health_test.csv')
            )
            logger.info(f"Loaded gut health data: {gut_health_data.shape}")
            
            # Store raw data
            self.raw_data = {
                'cgmacros': cgmacros_data,
                'bio': bio_data,
                'microbes': microbes_data,
                'gut_health': gut_health_data
            }
            
            # Merge all data sources
            self.processed_data = self.data_loader.merge_data_sources(
                cgmacros_data, bio_data, microbes_data, gut_health_data
            )
            logger.info(f"Merged data shape: {self.processed_data.shape}")
            
            # Save processed data
            os.makedirs(self.config['data']['processed_data_dir'], exist_ok=True)
            processed_path = os.path.join(
                self.config['data']['processed_data_dir'], 
                'merged_data.csv'
            )
            self.processed_data.to_csv(processed_path, index=False)
            logger.info(f"Processed data saved to {processed_path}")
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def run_feature_engineering(self) -> None:
        """
        Execute feature engineering phase.
        """
        logger.info("=== Phase 2: Feature Engineering ===")
        
        try:
            if self.processed_data is None:
                raise ValueError("No processed data available. Run data loading first.")
            
            # Initialize feature engineer with config
            self.feature_engineer = FeatureEngineer(
                glucose_window_hours=self.config['features']['glucose_window_hours'],
                activity_window_hours=self.config['features']['activity_window_hours']
            )
            
            # Start with processed data
            feature_data = self.processed_data.copy()
            
            # Add glucose features
            if any(col in feature_data.columns for col in ['Libre GL', 'Dexcom GL']):
                feature_data = self.feature_engineer.add_glucose_features(feature_data)
                logger.info("Added glucose features")
            
            # Add activity features
            if any(col in feature_data.columns for col in ['HR', 'METs', 'Calories']):
                feature_data = self.feature_engineer.add_activity_features(feature_data)
                logger.info("Added activity features")
            
            # Add meal timing features
            if 'Timestamp' in feature_data.columns:
                feature_data = self.feature_engineer.add_meal_timing_features(feature_data)
                logger.info("Added meal timing features")
            
            # Add demographic features
            if any(col in feature_data.columns for col in ['Age', 'Gender', 'BMI']):
                feature_data = self.feature_engineer.add_demographic_features(feature_data)
                logger.info("Added demographic features")
            
            # Add microbiome features
            if self.config['features']['include_microbiome']:
                microbiome_cols = [col for col in feature_data.columns if 
                                 any(bacteria in col for bacteria in ['Bacteroides', 'Bifidobacterium', 'Lactobacillus'])]
                if microbiome_cols:
                    feature_data = self.feature_engineer.add_microbiome_features(feature_data)
                    logger.info("Added microbiome features")
            
            # Add gut health features
            if self.config['features']['include_gut_health']:
                gut_health_cols = [col for col in feature_data.columns if 
                                 any(term in col for term in ['Gut', 'LPS', 'Biofilm'])]
                if gut_health_cols:
                    feature_data = self.feature_engineer.add_gut_health_features(feature_data)
                    logger.info("Added gut health features")
            
            # Add temporal features
            if self.config['features']['include_temporal'] and 'Timestamp' in feature_data.columns:
                feature_data = self.feature_engineer.add_temporal_features(feature_data)
                logger.info("Added temporal features")
            
            self.feature_data = feature_data
            logger.info(f"Feature engineering completed. Final shape: {feature_data.shape}")
            
            # Save feature data
            feature_path = os.path.join(
                self.config['data']['processed_data_dir'], 
                'feature_data.csv'
            )
            feature_data.to_csv(feature_path, index=False)
            logger.info(f"Feature data saved to {feature_path}")
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            raise
    
    def run_target_engineering(self) -> None:
        """
        Execute target engineering phase.
        """
        logger.info("=== Phase 3: Target Engineering ===")
        
        try:
            if self.feature_data is None:
                raise ValueError("No feature data available. Run feature engineering first.")
            
            # Compute CCR target
            target_data = compute_ccr(self.feature_data)
            logger.info(f"Computed CCR for {len(target_data)} samples")
            
            # Remove nutrient columns to prevent leakage
            target_data = remove_nutrient_columns(target_data)
            logger.info("Removed nutrient columns to prevent data leakage")
            
            self.target_data = target_data
            
            # Analyze target distribution
            ccr_stats = target_data['CCR'].describe()
            logger.info(f"CCR statistics:\n{ccr_stats}")
            
            # Save target data
            target_path = os.path.join(
                self.config['data']['processed_data_dir'], 
                'target_data.csv'
            )
            target_data.to_csv(target_path, index=False)
            logger.info(f"Target data saved to {target_path}")
            
        except Exception as e:
            logger.error(f"Target engineering failed: {e}")
            raise
    
    def run_model_training(self) -> None:
        """
        Execute model training phase.
        """
        logger.info("=== Phase 4: Model Training ===")
        
        try:
            if self.target_data is None:
                raise ValueError("No target data available. Run target engineering first.")
            
            # Train all models
            self.trained_models = self.model_trainer.train_all_models(
                df=self.target_data,
                target_col='CCR',
                include_time_series=self.config['models']['include_time_series'],
                include_multimodal=self.config['models']['include_multimodal'],
                include_ensemble=self.config['models']['include_ensemble']
            )
            
            logger.info(f"Trained {len(self.trained_models)} models: {list(self.trained_models.keys())}")
            
            # Save models
            models_dir = self.config['output']['models_dir']
            os.makedirs(models_dir, exist_ok=True)
            
            for model_name, model in self.trained_models.items():
                try:
                    model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
                    import pickle
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    logger.info(f"Saved {model_name} to {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to save {model_name}: {e}")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise
    
    def run_model_evaluation(self) -> None:
        """
        Execute model evaluation phase.
        """
        logger.info("=== Phase 5: Model Evaluation ===")
        
        try:
            if not self.trained_models:
                raise ValueError("No trained models available. Run model training first.")
            
            if self.target_data is None:
                raise ValueError("No target data available.")
            
            # Get feature columns (exclude metadata and target)
            exclude_cols = ['participant_id', 'Timestamp', 'CCR', 'Carbs', 'Protein', 'Fat', 'Fiber']
            feature_cols = [col for col in self.target_data.columns if col not in exclude_cols]
            
            # Evaluate models with participant-aware validation
            self.evaluation_results = self.evaluator.evaluate_with_participant_splits(
                models=self.trained_models,
                df=self.target_data,
                feature_cols=feature_cols,
                target_col='CCR'
            )
            
            logger.info(f"Evaluated {len(self.evaluation_results)} models")
            
            # Print quick summary
            print("\n=== MODEL EVALUATION SUMMARY ===")
            print("Model\t\t\tRMSE\t\tMAE\t\tR²")
            print("-" * 60)
            for model_name, results in self.evaluation_results.items():
                rmse = results.get('rmse_mean', 'N/A')
                mae = results.get('mae_mean', 'N/A')
                r2 = results.get('r2_mean', 'N/A')
                print(f"{model_name:<20}\t{rmse:.4f}\t\t{mae:.4f}\t\t{r2:.4f}")
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            raise
    
    def run_results_reporting(self) -> None:
        """
        Execute results reporting phase.
        """
        logger.info("=== Phase 6: Results Reporting ===")
        
        try:
            if not self.evaluation_results:
                raise ValueError("No evaluation results available. Run model evaluation first.")
            
            # Create output directories
            results_dir = self.config['output']['results_dir']
            plots_dir = self.config['output']['plots_dir']
            os.makedirs(results_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate comprehensive report
            report_path = self.report_generator.generate_report(
                results=self.evaluation_results,
                output_dir=results_dir
            )
            logger.info(f"Generated evaluation report: {report_path}")
            
            # Generate visualizations
            try:
                # Model comparison plot
                self.report_generator.visualizer.plot_model_comparison(
                    results=self.evaluation_results,
                    metric='rmse_mean',
                    save_path=os.path.join(plots_dir, 'model_comparison_rmse.png')
                )
                
                # Metrics heatmap
                self.report_generator.visualizer.plot_metrics_heatmap(
                    results=self.evaluation_results,
                    save_path=os.path.join(plots_dir, 'metrics_heatmap.png')
                )
                
                logger.info(f"Visualizations saved to {plots_dir}")
                
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
            
            # Save evaluation results
            results_path = os.path.join(results_dir, 'evaluation_results.pkl')
            import pickle
            with open(results_path, 'wb') as f:
                pickle.dump(self.evaluation_results, f)
            logger.info(f"Evaluation results saved to {results_path}")
            
            # Create summary CSV
            summary_data = []
            for model_name, results in self.evaluation_results.items():
                summary_data.append({
                    'model': model_name,
                    'rmse_mean': results.get('rmse_mean', np.nan),
                    'rmse_std': results.get('rmse_std', np.nan),
                    'mae_mean': results.get('mae_mean', np.nan),
                    'mae_std': results.get('mae_std', np.nan),
                    'r2_mean': results.get('r2_mean', np.nan),
                    'r2_std': results.get('r2_std', np.nan),
                    'ccr_rmse_mean': results.get('ccr_rmse_mean', np.nan),
                    'mape_mean': results.get('mape_mean', np.nan)
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(results_dir, 'model_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Model summary saved to {summary_path}")
            
        except Exception as e:
            logger.error(f"Results reporting failed: {e}")
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        logger.info("=== STARTING CGMACROS CCR PREDICTION PIPELINE ===")
        
        try:
            # Execute all phases
            self.run_data_loading()
            self.run_feature_engineering()
            self.run_target_engineering()
            self.run_model_training()
            self.run_model_evaluation()
            self.run_results_reporting()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            # Pipeline summary
            pipeline_results = {
                'status': 'completed',
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'data_shape': self.target_data.shape if self.target_data is not None else None,
                'models_trained': len(self.trained_models),
                'models_evaluated': len(self.evaluation_results),
                'best_model': self._get_best_model(),
                'config': self.config
            }
            
            logger.info(f"=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Duration: {duration}")
            logger.info(f"Best model: {pipeline_results['best_model']}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'start_time': start_time,
                'end_time': datetime.now()
            }
    
    def _get_best_model(self) -> str:
        """
        Identify the best performing model.
        
        Returns:
            Name of the best model
        """
        if not self.evaluation_results:
            return "No models evaluated"
        
        best_model = None
        best_rmse = float('inf')
        
        for model_name, results in self.evaluation_results.items():
            rmse = results.get('rmse_mean', float('inf'))
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model_name
        
        return best_model or "Unknown"

def main():
    """
    Main function to run the pipeline.
    """
    try:
        # Initialize and run pipeline
        pipeline = CGMacrosPipeline()
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "="*60)
        print("CGMACROS CCR PREDICTION PIPELINE RESULTS")
        print("="*60)
        print(f"Status: {results['status']}")
        if results['status'] == 'completed':
            print(f"Duration: {results['duration']}")
            print(f"Data shape: {results['data_shape']}")
            print(f"Models trained: {results['models_trained']}")
            print(f"Models evaluated: {results['models_evaluated']}")
            print(f"Best model: {results['best_model']}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"PIPELINE FAILED: {e}")

if __name__ == "__main__":
    main()