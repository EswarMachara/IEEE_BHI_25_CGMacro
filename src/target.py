"""
Target Variable Computation Module

This module handles the computation of the Carbohydrate Caloric Ratio (CCR)
from nutritional data and removes nutrient columns to prevent data leakage.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_ccr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Carbohydrate Caloric Ratio (CCR) and remove nutrient columns.
    
    CCR = net_carbs / (net_carbs + protein + fat + fiber)
    
    Args:
        df: DataFrame containing nutritional information
        
    Returns:
        DataFrame with CCR target variable and nutrient columns removed
    """
    logger.info("Computing Carbohydrate Caloric Ratio (CCR)")
    
    # Make a copy to avoid modifying original data
    df_with_target = df.copy()
    
    # Identify nutrient columns
    nutrient_columns = identify_nutrient_columns(df_with_target)
    
    if not nutrient_columns:
        logger.warning("No nutrient columns found - cannot compute CCR")
        return df_with_target
    
    logger.info(f"Found nutrient columns: {nutrient_columns}")
    
    # Compute CCR
    df_with_target = calculate_ccr_values(df_with_target, nutrient_columns)
    
    # Remove nutrient columns to prevent data leakage
    df_with_target = remove_nutrient_columns(df_with_target, nutrient_columns)
    
    # Log CCR statistics
    log_ccr_statistics(df_with_target)
    
    return df_with_target


def identify_nutrient_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Identify columns containing nutritional macronutrient information.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping nutrient types to column names
    """
    nutrient_columns = {}
    
    # Common nutrient column name patterns
    nutrient_patterns = {
        'net_carbs': ['net_carbs', 'net_carbohydrates', 'carbs_net', 'netcarbs'],
        'protein': ['protein', 'proteins', 'protein_g', 'protein_content'],
        'fat': ['fat', 'fats', 'total_fat', 'fat_g', 'fat_content', 'lipids'],
        'fiber': ['fiber', 'fibre', 'dietary_fiber', 'fiber_g', 'total_fiber']
    }
    
    # Search for nutrient columns (case-insensitive)
    for nutrient_type, patterns in nutrient_patterns.items():
        for pattern in patterns:
            matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            if matching_cols:
                # Take the first match
                nutrient_columns[nutrient_type] = matching_cols[0]
                break
    
    return nutrient_columns


def calculate_ccr_values(df: pd.DataFrame, nutrient_columns: Dict[str, str]) -> pd.DataFrame:
    """
    Calculate CCR values for each row.
    
    Args:
        df: DataFrame with nutrient columns
        nutrient_columns: Dictionary mapping nutrient types to column names
        
    Returns:
        DataFrame with CCR column added
    """
    required_nutrients = ['net_carbs', 'protein', 'fat', 'fiber']
    missing_nutrients = [n for n in required_nutrients if n not in nutrient_columns]
    
    if missing_nutrients:
        logger.warning(f"Missing required nutrients for CCR calculation: {missing_nutrients}")
        # Try to proceed with available nutrients
        available_nutrients = [n for n in required_nutrients if n in nutrient_columns]
        if len(available_nutrients) < 2:
            logger.error("Insufficient nutrient columns for CCR calculation")
            df['ccr'] = np.nan
            return df
    
    # Extract nutrient values
    nutrient_values = {}
    for nutrient_type, column_name in nutrient_columns.items():
        if nutrient_type in required_nutrients:
            values = pd.to_numeric(df[column_name], errors='coerce')
            # Replace negative values with 0 (shouldn't have negative nutrients)
            values = values.clip(lower=0)
            nutrient_values[nutrient_type] = values
    
    # Calculate CCR
    if 'net_carbs' in nutrient_values:
        net_carbs = nutrient_values['net_carbs']
        
        # Sum all available macronutrients for denominator
        total_macros = net_carbs.copy()
        for nutrient in ['protein', 'fat', 'fiber']:
            if nutrient in nutrient_values:
                total_macros += nutrient_values[nutrient]
        
        # Calculate CCR with zero-division handling
        ccr = np.where(total_macros > 0, net_carbs / total_macros, np.nan)
        
        # Clip CCR to valid range [0, 1]
        ccr = np.clip(ccr, 0, 1)
        
    else:
        logger.error("Cannot calculate CCR without net_carbs column")
        ccr = np.full(len(df), np.nan)
    
    df['ccr'] = ccr
    
    return df


def remove_nutrient_columns(df: pd.DataFrame, nutrient_columns: Dict[str, str]) -> pd.DataFrame:
    """
    Remove nutrient columns to prevent data leakage.
    
    Args:
        df: DataFrame with nutrient columns
        nutrient_columns: Dictionary of nutrient column names
        
    Returns:
        DataFrame with nutrient columns removed
    """
    columns_to_remove = list(nutrient_columns.values())
    
    # Also remove any other potential nutrient-related columns
    additional_patterns = [
        'calorie', 'kcal', 'energy', 'sugar', 'sodium', 'cholesterol',
        'saturated_fat', 'trans_fat', 'carbohydrate', 'total_carbs'
    ]
    
    for pattern in additional_patterns:
        additional_cols = [col for col in df.columns if pattern.lower() in col.lower()]
        columns_to_remove.extend(additional_cols)
    
    # Remove duplicates
    columns_to_remove = list(set(columns_to_remove))
    
    # Only remove columns that actually exist
    columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if columns_to_remove:
        logger.info(f"Removing nutrient columns to prevent data leakage: {columns_to_remove}")
        df = df.drop(columns=columns_to_remove)
    
    return df


def log_ccr_statistics(df: pd.DataFrame) -> None:
    """
    Log statistics about the computed CCR values.
    
    Args:
        df: DataFrame with CCR column
    """
    if 'ccr' not in df.columns:
        logger.warning("No CCR column found for statistics")
        return
    
    ccr_values = df['ccr'].dropna()
    
    if len(ccr_values) == 0:
        logger.warning("No valid CCR values computed")
        return
    
    stats = {
        'total_samples': len(df),
        'valid_ccr_samples': len(ccr_values),
        'missing_ccr_percentage': ((len(df) - len(ccr_values)) / len(df) * 100).round(2),
        'ccr_mean': ccr_values.mean().round(4),
        'ccr_std': ccr_values.std().round(4),
        'ccr_min': ccr_values.min().round(4),
        'ccr_max': ccr_values.max().round(4),
        'ccr_median': ccr_values.median().round(4)
    }
    
    logger.info(f"CCR Statistics: {stats}")
    
    # Check for potential data quality issues
    if stats['ccr_min'] < 0 or stats['ccr_max'] > 1:
        logger.warning("CCR values outside expected range [0, 1] detected")
    
    if stats['missing_ccr_percentage'] > 10:
        logger.warning(f"High percentage of missing CCR values: {stats['missing_ccr_percentage']}%")


def validate_ccr_computation(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the CCR computation results.
    
    Args:
        df: DataFrame with computed CCR
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if 'ccr' not in df.columns:
        issues.append("CCR column not found")
        return False, issues
    
    ccr_values = df['ccr'].dropna()
    
    if len(ccr_values) == 0:
        issues.append("No valid CCR values computed")
        return False, issues
    
    # Check value range
    if (ccr_values < 0).any():
        issues.append("Negative CCR values found")
    
    if (ccr_values > 1).any():
        issues.append("CCR values greater than 1 found")
    
    # Check for missing values
    missing_pct = (df['ccr'].isnull().sum() / len(df)) * 100
    if missing_pct > 50:
        issues.append(f"High percentage of missing CCR values: {missing_pct:.2f}%")
    
    # Check for constant values (potential data issue)
    if ccr_values.nunique() == 1:
        issues.append("All CCR values are identical - potential data issue")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues


def get_ccr_distribution_info(df: pd.DataFrame) -> Dict:
    """
    Get detailed information about CCR distribution.
    
    Args:
        df: DataFrame with CCR column
        
    Returns:
        Dictionary with distribution information
    """
    if 'ccr' not in df.columns:
        return {}
    
    ccr_values = df['ccr'].dropna()
    
    if len(ccr_values) == 0:
        return {}
    
    # Compute percentiles
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(ccr_values, percentiles)
    
    distribution_info = {
        'count': len(ccr_values),
        'missing_count': df['ccr'].isnull().sum(),
        'mean': ccr_values.mean(),
        'std': ccr_values.std(),
        'min': ccr_values.min(),
        'max': ccr_values.max(),
        'skewness': ccr_values.skew(),
        'kurtosis': ccr_values.kurtosis(),
        'percentiles': dict(zip(percentiles, percentile_values))
    }
    
    return distribution_info


if __name__ == "__main__":
    # Example usage
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'participant_id': [1, 2, 3, 4, 5],
        'net_carbs': [30, 45, 60, 25, 55],
        'protein': [20, 15, 25, 30, 18],
        'fat': [10, 20, 15, 8, 22],
        'fiber': [5, 8, 10, 6, 7],
        'glucose': [120, 140, 110, 130, 125]
    })
    
    print("Sample data:")
    print(sample_data)
    
    result = compute_ccr(sample_data)
    print("\nData with CCR computed:")
    print(result)
    
    distribution_info = get_ccr_distribution_info(result)
    print("\nCCR Distribution Info:")
    print(distribution_info)