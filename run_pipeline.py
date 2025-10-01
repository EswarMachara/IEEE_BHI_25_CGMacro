#!/usr/bin/env python3
"""
Main pipeline script for IEEE BHI 2025 Track 2 Challenge
Predicting Carbohydrate Caloric Ratio (CCR) from multimodal data

Usage:
    python run_pipeline.py [--config config.yaml] [--output results/]
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from data_loader import load_cgmacros_data
from target import compute_ccr
from feature_engineering import engineer_features
from models import train_all_models
from evaluation import evaluate_models, generate_report
from visualization import create_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='Run CGMacros CCR prediction pipeline')
    parser.add_argument('--data-dir', default='data/raw', help='Path to raw data directory')
    parser.add_argument('--output-dir', default='results', help='Path to output directory')
    parser.add_argument('--models-dir', default='models', help='Path to save trained models')
    parser.add_argument('--config', help='Path to configuration file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    logger.info("Starting CGMacros CCR Prediction Pipeline")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Load raw data
        logger.info("Step 1: Loading raw data...")
        data = load_cgmacros_data(args.data_dir)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Step 2: Compute target variable (CCR)
        logger.info("Step 2: Computing target variable (CCR)...")
        data_with_target = compute_ccr(data)
        logger.info(f"Target variable computed. Data shape: {data_with_target.shape}")
        
        # Step 3: Feature engineering
        logger.info("Step 3: Performing feature engineering...")
        features_df = engineer_features(data_with_target)
        logger.info(f"Feature engineering completed. Features shape: {features_df.shape}")
        
        # Save processed data
        processed_data_path = Path('data/processed/features_with_target.csv')
        features_df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to: {processed_data_path}")
        
        # Step 4: Train models
        logger.info("Step 4: Training models...")
        models_results = train_all_models(features_df, save_dir=args.models_dir)
        logger.info("Model training completed")
        
        # Step 5: Evaluate models
        logger.info("Step 5: Evaluating models...")
        evaluation_results = evaluate_models(models_results, features_df)
        logger.info("Model evaluation completed")
        
        # Step 6: Generate visualizations
        logger.info("Step 6: Creating visualizations...")
        create_visualizations(
            features_df, 
            models_results, 
            evaluation_results, 
            output_dir=args.output_dir
        )
        logger.info("Visualizations created")
        
        # Step 7: Generate final report
        logger.info("Step 7: Generating final report...")
        generate_report(
            evaluation_results, 
            features_df, 
            output_dir=args.output_dir
        )
        logger.info("Final report generated")
        
        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()