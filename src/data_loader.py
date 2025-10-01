"""
Data Loading Module for CGMacros Dataset

This module handles loading and initial preprocessing of the CGMacros dataset
including participant CGM data, biomarker data, microbiome data, and gut health scores.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_cgmacros_data(data_dir: str) -> pd.DataFrame:
    """
    Load all CGMacros data files and merge them into a single DataFrame.
    
    Args:
        data_dir: Path to directory containing raw data files
        
    Returns:
        Combined DataFrame with all participant data and supplementary information
    """
    data_dir = Path(data_dir)
    
    logger.info(f"Loading CGMacros data from: {data_dir}")
    
    # Load participant CGMacros files
    cgmacros_data = load_participant_files(data_dir)
    
    # Load supplementary data
    bio_data = load_bio_data(data_dir)
    microbes_data = load_microbes_data(data_dir)
    gut_health_data = load_gut_health_data(data_dir)
    
    # Merge all data sources
    combined_data = merge_data_sources(
        cgmacros_data, bio_data, microbes_data, gut_health_data
    )
    
    logger.info(f"Successfully loaded and merged data. Final shape: {combined_data.shape}")
    
    return combined_data


def load_participant_files(data_dir: Path) -> pd.DataFrame:
    """
    Load all participant CGMacros CSV files.
    
    Args:
        data_dir: Path to directory containing CGMacros files
        
    Returns:
        Combined DataFrame with all participant data
    """
    cgmacros_dir = data_dir / "CGMacros_CSVs"
    
    if not cgmacros_dir.exists():
        # Try loading from main data directory
        cgmacros_files = list(data_dir.glob("CGMacros-*.csv"))
    else:
        cgmacros_files = list(cgmacros_dir.glob("CGMacros-*.csv"))
    
    if not cgmacros_files:
        raise FileNotFoundError(f"No CGMacros files found in {data_dir}")
    
    logger.info(f"Found {len(cgmacros_files)} CGMacros participant files")
    
    all_data = []
    
    for file_path in sorted(cgmacros_files):
        try:
            # Extract participant ID from filename
            participant_id = file_path.stem.split('-')[1]
            
            # Load participant data
            df = pd.read_csv(file_path)
            df['participant_id'] = participant_id
            
            # Basic data quality checks
            df = validate_participant_data(df, participant_id)
            
            all_data.append(df)
            logger.debug(f"Loaded {file_path.name}: {df.shape}")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No valid participant data files could be loaded")
    
    # Combine all participant data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    logger.info(f"Combined participant data shape: {combined_df.shape}")
    
    return combined_df


def load_bio_data(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load biomarker and demographic data.
    
    Args:
        data_dir: Path to directory containing bio.csv
        
    Returns:
        DataFrame with biomarker data or None if file not found
    """
    bio_file = data_dir / "bio.csv"
    
    if not bio_file.exists():
        logger.warning("bio.csv not found - skipping biomarker data")
        return None
    
    try:
        bio_df = pd.read_csv(bio_file)
        logger.info(f"Loaded bio data: {bio_df.shape}")
        return bio_df
    except Exception as e:
        logger.error(f"Failed to load bio.csv: {str(e)}")
        return None


def load_microbes_data(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load microbiome composition data.
    
    Args:
        data_dir: Path to directory containing microbes.csv
        
    Returns:
        DataFrame with microbiome data or None if file not found
    """
    microbes_file = data_dir / "microbes.csv"
    
    if not microbes_file.exists():
        logger.warning("microbes.csv not found - skipping microbiome data")
        return None
    
    try:
        microbes_df = pd.read_csv(microbes_file)
        logger.info(f"Loaded microbiome data: {microbes_df.shape}")
        return microbes_df
    except Exception as e:
        logger.error(f"Failed to load microbes.csv: {str(e)}")
        return None


def load_gut_health_data(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load gut health test scores.
    
    Args:
        data_dir: Path to directory containing gut_health_test.csv
        
    Returns:
        DataFrame with gut health data or None if file not found
    """
    gut_file = data_dir / "gut_health_test.csv"
    
    if not gut_file.exists():
        logger.warning("gut_health_test.csv not found - skipping gut health data")
        return None
    
    try:
        gut_df = pd.read_csv(gut_file)
        logger.info(f"Loaded gut health data: {gut_df.shape}")
        return gut_df
    except Exception as e:
        logger.error(f"Failed to load gut_health_test.csv: {str(e)}")
        return None


def validate_participant_data(df: pd.DataFrame, participant_id: str) -> pd.DataFrame:
    """
    Perform basic validation and cleaning of participant data.
    
    Args:
        df: Participant DataFrame
        participant_id: Participant identifier
        
    Returns:
        Validated DataFrame
    """
    initial_shape = df.shape
    
    # Remove completely empty rows
    df = df.dropna(how='all')
    
    # Convert datetime columns if present
    datetime_columns = ['timestamp', 'datetime', 'time']
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Log data quality issues
    missing_data_pct = (df.isnull().sum() / len(df) * 100).round(2)
    high_missing_cols = missing_data_pct[missing_data_pct > 50]
    
    if len(high_missing_cols) > 0:
        logger.warning(f"Participant {participant_id} has high missing data in columns: {high_missing_cols.to_dict()}")
    
    final_shape = df.shape
    if initial_shape != final_shape:
        logger.debug(f"Participant {participant_id} data shape changed from {initial_shape} to {final_shape}")
    
    return df


def merge_data_sources(
    cgmacros_data: pd.DataFrame,
    bio_data: Optional[pd.DataFrame],
    microbes_data: Optional[pd.DataFrame],
    gut_health_data: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """
    Merge all data sources into a single DataFrame.
    
    Args:
        cgmacros_data: Main participant data
        bio_data: Biomarker data (optional)
        microbes_data: Microbiome data (optional)
        gut_health_data: Gut health data (optional)
        
    Returns:
        Combined DataFrame
    """
    combined_df = cgmacros_data.copy()
    
    # Merge biomarker data
    if bio_data is not None:
        if 'participant_id' in bio_data.columns:
            combined_df = combined_df.merge(bio_data, on='participant_id', how='left', suffixes=('', '_bio'))
            logger.info("Merged biomarker data")
        else:
            logger.warning("Cannot merge bio data - no participant_id column")
    
    # Merge microbiome data
    if microbes_data is not None:
        if 'participant_id' in microbes_data.columns:
            combined_df = combined_df.merge(microbes_data, on='participant_id', how='left', suffixes=('', '_microbes'))
            logger.info("Merged microbiome data")
        else:
            logger.warning("Cannot merge microbiome data - no participant_id column")
    
    # Merge gut health data
    if gut_health_data is not None:
        if 'participant_id' in gut_health_data.columns:
            combined_df = combined_df.merge(gut_health_data, on='participant_id', how='left', suffixes=('', '_gut'))
            logger.info("Merged gut health data")
        else:
            logger.warning("Cannot merge gut health data - no participant_id column")
    
    return combined_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of the loaded data.
    
    Args:
        df: Combined DataFrame
        
    Returns:
        Dictionary with data summary statistics
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'participants': df['participant_id'].nunique() if 'participant_id' in df.columns else 'Unknown',
        'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100).round(2),
        'column_types': df.dtypes.value_counts().to_dict(),
        'memory_usage_mb': (df.memory_usage(deep=True).sum() / 1024 / 1024).round(2)
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    data_dir = "data/raw"
    df = load_cgmacros_data(data_dir)
    summary = get_data_summary(df)
    print("Data Summary:", summary)