"""
Target Variable Computation Module

This module handles computation of the Carbohydrate Caloric Ratio (CCR) target variable
from nutrient data and ensures proper data leakage prevention.
Updated to handle actual dataset structure.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

def compute_ccr(df: pd.DataFrame, 
               carbs_col: str = 'Carbs',
               protein_col: str = 'Protein', 
               fat_col: str = 'Fat',
               fiber_col: str = 'Fiber') -> pd.DataFrame:
    """
    Compute Carbohydrate Caloric Ratio (CCR) from nutrient columns.
    
    CCR = net_carbs / (net_carbs + protein + fat + fiber)
    where net_carbs = carbs (fiber already separate in this dataset)
    
    Args:
        df: DataFrame with nutrient information
        carbs_col: Name of carbohydrates column
        protein_col: Name of protein column
        fat_col: Name of fat column
        fiber_col: Name of fiber column
        
    Returns:
        DataFrame with CCR column added
    """
    logger.info("Computing CCR (Carbohydrate Caloric Ratio)...")
    
    df = df.copy()
    
    # Check if required nutrient columns exist
    required_cols = [carbs_col, protein_col, fat_col, fiber_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required nutrient columns: {missing_cols}")
        raise ValueError(f"Missing required columns for CCR computation: {missing_cols}")
    
    # Handle missing values by filling with 0 (assuming no nutrients if missing)
    nutrient_data = df[required_cols].fillna(0)
    
    # Compute net carbs (in this dataset, carbs and fiber are separate)
    net_carbs = nutrient_data[carbs_col]
    
    # Compute total macronutrients
    total_macros = (net_carbs + 
                   nutrient_data[protein_col] + 
                   nutrient_data[fat_col] + 
                   nutrient_data[fiber_col])
    
    # Compute CCR, handle division by zero
    df['CCR'] = np.where(total_macros > 0, 
                        net_carbs / total_macros, 
                        0)
    
    # Log some statistics
    valid_ccr = df['CCR'][df['CCR'] > 0]
    if len(valid_ccr) > 0:
        logger.info(f"CCR computed for {len(valid_ccr)} records")
        logger.info(f"CCR statistics - Mean: {valid_ccr.mean():.3f}, "
                   f"Std: {valid_ccr.std():.3f}, "
                   f"Min: {valid_ccr.min():.3f}, "
                   f"Max: {valid_ccr.max():.3f}")
    else:
        logger.warning("No valid CCR values computed")
    
    return df

def remove_nutrient_columns(df: pd.DataFrame, 
                          columns_to_remove: List[str] = None) -> pd.DataFrame:
    """
    Remove nutrient columns to prevent data leakage after CCR computation.
    
    Args:
        df: DataFrame with nutrient columns
        columns_to_remove: List of column names to remove. If None, removes default nutrient columns.
        
    Returns:
        DataFrame with nutrient columns removed
    """
    if columns_to_remove is None:
        # Default nutrient columns based on actual dataset structure
        columns_to_remove = ['Carbs', 'Protein', 'Fat', 'Fiber']
    
    df = df.copy()
    
    # Remove columns that exist in the dataframe
    existing_cols_to_remove = [col for col in columns_to_remove if col in df.columns]
    
    if existing_cols_to_remove:
        df = df.drop(columns=existing_cols_to_remove)
        logger.info(f"Removed nutrient columns to prevent data leakage: {existing_cols_to_remove}")
    else:
        logger.warning("No nutrient columns found to remove")
    
    return df

def identify_nutrient_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify columns that contain nutrient information and should be removed
    after CCR computation to prevent data leakage.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of column names that appear to contain nutrient information
    """
    # Standard nutrient column names
    standard_nutrients = ['Carbs', 'Protein', 'Fat', 'Fiber', 'Calories']
    
    # Additional patterns that might indicate nutrient data
    nutrient_patterns = ['carb', 'protein', 'fat', 'fiber', 'calor', 'sugar', 'sodium']
    
    identified_cols = []
    
    # Check for exact matches
    for col in df.columns:
        if col in standard_nutrients:
            identified_cols.append(col)
        else:
            # Check for partial matches (case insensitive)
            col_lower = col.lower()
            for pattern in nutrient_patterns:
                if pattern in col_lower:
                    identified_cols.append(col)
                    break
    
    # Remove duplicates while preserving order
    identified_cols = list(dict.fromkeys(identified_cols))
    
    logger.info(f"Identified potential nutrient columns: {identified_cols}")
    return identified_cols

def validate_ccr_computation(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate that CCR computation was successful and values are reasonable.
    
    Args:
        df: DataFrame with CCR column
        
    Returns:
        Tuple of (is_valid, validation_message)
    """
    if 'CCR' not in df.columns:
        return False, "CCR column not found in DataFrame"
    
    ccr_values = df['CCR']
    
    # Check for valid range (0 to 1)
    if ccr_values.min() < 0 or ccr_values.max() > 1:
        return False, f"CCR values outside valid range [0,1]: min={ccr_values.min():.3f}, max={ccr_values.max():.3f}"
    
    # Check for sufficient non-zero values
    non_zero_count = (ccr_values > 0).sum()
    total_count = len(ccr_values)
    
    if non_zero_count == 0:
        return False, "No non-zero CCR values found"
    
    if non_zero_count / total_count < 0.1:
        return False, f"Too few non-zero CCR values: {non_zero_count}/{total_count} ({non_zero_count/total_count:.1%})"
    
    # Check for reasonable distribution
    mean_ccr = ccr_values[ccr_values > 0].mean()
    if mean_ccr < 0.1 or mean_ccr > 0.9:
        logger.warning(f"CCR mean seems unusual: {mean_ccr:.3f}")
    
    return True, f"CCR validation passed: {non_zero_count} valid values, mean={mean_ccr:.3f}"

def process_target_variable(df: pd.DataFrame, 
                          remove_nutrients: bool = True,
                          validate: bool = True) -> pd.DataFrame:
    """
    Complete target variable processing pipeline.
    
    Args:
        df: Input DataFrame with nutrient data
        remove_nutrients: Whether to remove nutrient columns after CCR computation
        validate: Whether to validate CCR computation
        
    Returns:
        DataFrame with CCR computed and nutrient columns optionally removed
    """
    logger.info("Starting target variable processing...")
    
    # Compute CCR
    df = compute_ccr(df)
    
    # Validate computation if requested
    if validate:
        is_valid, message = validate_ccr_computation(df)
        if not is_valid:
            logger.error(f"CCR validation failed: {message}")
            raise ValueError(f"CCR validation failed: {message}")
        else:
            logger.info(f"CCR validation: {message}")
    
    # Remove nutrient columns if requested
    if remove_nutrients:
        nutrient_cols = identify_nutrient_columns(df)
        df = remove_nutrient_columns(df, nutrient_cols)
    
    logger.info("Target variable processing completed")
    return df

def create_meal_level_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create meal-level CCR targets by aggregating time-series data.
    
    This function handles the case where we have time-series data but want
    to predict meal-level CCR values.
    
    Args:
        df: DataFrame with time-series data and CCR values
        
    Returns:
        DataFrame with meal-level aggregated data
    """
    logger.info("Creating meal-level targets...")
    
    if 'Meal Type' not in df.columns:
        logger.warning("No Meal Type column found, cannot create meal-level targets")
        return df
    
    # Filter to actual meals (not "No Meal" entries)
    meal_data = df[df['Meal Type'] != 'No Meal'].copy()
    
    if len(meal_data) == 0:
        logger.warning("No meal data found")
        return df
    
    # Aggregate by participant and timestamp (assuming one meal per timestamp)
    groupby_cols = ['participant_id', 'Timestamp', 'Meal Type'] if 'Timestamp' in df.columns else ['participant_id', 'Meal Type']
    
    # Aggregate features and target
    agg_dict = {'CCR': 'first'}  # Take first CCR value for each meal
    
    # Add other columns to aggregate
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['participant_id', 'CCR']:
            agg_dict[col] = 'mean'
    
    meal_level_df = meal_data.groupby(groupby_cols).agg(agg_dict).reset_index()
    
    logger.info(f"Created meal-level data: {len(meal_level_df)} meals from {len(meal_data)} time-series records")
    return meal_level_df