"""
Feature Engineering Module

This module handles feature extraction and engineering from multimodal data
including glucose, activity, demographics, microbiome, and meal metadata.
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main feature engineering pipeline for CGMacros data.
    
    Args:
        df: Input DataFrame with raw data
        
    Returns:
        DataFrame with engineered features
    """
    logger.info("Starting feature engineering pipeline")
    
    features_df = df.copy()
    
    # 1. Glucose response features
    logger.info("Engineering glucose response features...")
    features_df = add_glucose_features(features_df)
    
    # 2. Activity features
    logger.info("Engineering activity features...")
    features_df = add_activity_features(features_df)
    
    # 3. Demographics and lab features
    logger.info("Engineering demographics and lab features...")
    features_df = add_demographic_features(features_df)
    
    # 4. Microbiome features
    logger.info("Engineering microbiome features...")
    features_df = add_microbiome_features(features_df)
    
    # 5. Gut health features
    logger.info("Engineering gut health features...")
    features_df = add_gut_health_features(features_df)
    
    # 6. Temporal and meal metadata features
    logger.info("Engineering temporal and meal features...")
    features_df = add_temporal_features(features_df)
    
    # 7. Interaction features
    logger.info("Creating interaction features...")
    features_df = add_interaction_features(features_df)
    
    # 8. Clean and validate features
    logger.info("Cleaning and validating features...")
    features_df = clean_and_validate_features(features_df)
    
    logger.info(f"Feature engineering completed. Final shape: {features_df.shape}")
    
    return features_df


def add_glucose_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract glucose response features from CGM data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with glucose features added
    """
    # Identify glucose columns
    glucose_cols = [col for col in df.columns if 'glucose' in col.lower() or 'cgm' in col.lower()]
    
    if not glucose_cols:
        logger.warning("No glucose columns found")
        return df
    
    # Use the first glucose column as primary
    glucose_col = glucose_cols[0]
    glucose_values = pd.to_numeric(df[glucose_col], errors='coerce')
    
    # Basic glucose statistics
    df['glucose_mean'] = glucose_values
    df['glucose_median'] = glucose_values
    df['glucose_std'] = df.groupby('participant_id')[glucose_col].transform('std') if 'participant_id' in df.columns else glucose_values.std()
    df['glucose_min'] = df.groupby('participant_id')[glucose_col].transform('min') if 'participant_id' in df.columns else glucose_values.min()
    df['glucose_max'] = df.groupby('participant_id')[glucose_col].transform('max') if 'participant_id' in df.columns else glucose_values.max()
    df['glucose_range'] = df['glucose_max'] - df['glucose_min']
    
    # Glucose variability metrics
    df['glucose_cv'] = df['glucose_std'] / df['glucose_mean']  # Coefficient of variation
    
    # Glucose response patterns (if temporal data available)
    if 'participant_id' in df.columns:
        df = add_glucose_response_patterns(df, glucose_col)
    
    # Glucose percentiles
    df['glucose_p25'] = df.groupby('participant_id')[glucose_col].transform(lambda x: x.quantile(0.25)) if 'participant_id' in df.columns else glucose_values.quantile(0.25)
    df['glucose_p75'] = df.groupby('participant_id')[glucose_col].transform(lambda x: x.quantile(0.75)) if 'participant_id' in df.columns else glucose_values.quantile(0.75)
    df['glucose_iqr'] = df['glucose_p75'] - df['glucose_p25']
    
    return df


def add_glucose_response_patterns(df: pd.DataFrame, glucose_col: str) -> pd.DataFrame:
    """
    Add glucose response pattern features.
    
    Args:
        df: Input DataFrame
        glucose_col: Name of glucose column
        
    Returns:
        DataFrame with glucose response features
    """
    # Sort by participant and time if available
    time_cols = [col for col in df.columns if any(t in col.lower() for t in ['time', 'timestamp', 'datetime'])]
    
    if time_cols and 'participant_id' in df.columns:
        df = df.sort_values(['participant_id', time_cols[0]])
        
        # Calculate glucose changes
        df['glucose_diff'] = df.groupby('participant_id')[glucose_col].diff()
        df['glucose_diff_abs'] = df['glucose_diff'].abs()
        
        # Rolling statistics (simulate post-meal response)
        window_size = min(5, len(df) // 10)  # Adaptive window size
        df['glucose_rolling_mean'] = df.groupby('participant_id')[glucose_col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).mean()
        )
        df['glucose_rolling_std'] = df.groupby('participant_id')[glucose_col].transform(
            lambda x: x.rolling(window=window_size, min_periods=1).std()
        )
        
        # Peak detection approximation
        df['glucose_local_max'] = df.groupby('participant_id')[glucose_col].transform(
            lambda x: (x == x.rolling(window=3, center=True, min_periods=1).max()).astype(int)
        )
        
        # Time to peak (simplified)
        df['time_since_peak'] = df.groupby('participant_id')['glucose_local_max'].transform(
            lambda x: (x == 1).cumsum()
        )
    
    return df


def add_activity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract activity features from Fitbit data.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with activity features added
    """
    # Identify activity columns
    activity_patterns = ['step', 'heart', 'hr', 'activity', 'calorie', 'distance', 'sleep']
    activity_cols = []
    
    for pattern in activity_patterns:
        cols = [col for col in df.columns if pattern in col.lower()]
        activity_cols.extend(cols)
    
    if not activity_cols:
        logger.warning("No activity columns found")
        return df
    
    logger.info(f"Found activity columns: {activity_cols}")
    
    # Process each activity metric
    for col in activity_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            
            # Basic statistics
            if 'participant_id' in df.columns:
                df[f'{col}_mean'] = df.groupby('participant_id')[col].transform('mean')
                df[f'{col}_std'] = df.groupby('participant_id')[col].transform('std')
                df[f'{col}_max'] = df.groupby('participant_id')[col].transform('max')
                df[f'{col}_min'] = df.groupby('participant_id')[col].transform('min')
            else:
                df[f'{col}_mean'] = values.mean()
                df[f'{col}_std'] = values.std()
                df[f'{col}_max'] = values.max()
                df[f'{col}_min'] = values.min()
    
    # Activity intensity features
    heart_rate_cols = [col for col in activity_cols if 'heart' in col.lower() or 'hr' in col.lower()]
    if heart_rate_cols:
        hr_col = heart_rate_cols[0]
        hr_values = pd.to_numeric(df[hr_col], errors='coerce')
        
        # Heart rate zones (approximate)
        df['hr_rest_zone'] = (hr_values < 100).astype(int)
        df['hr_moderate_zone'] = ((hr_values >= 100) & (hr_values < 140)).astype(int)
        df['hr_vigorous_zone'] = (hr_values >= 140).astype(int)
        
        # Heart rate variability approximation
        if 'participant_id' in df.columns:
            df['hr_variability'] = df.groupby('participant_id')[hr_col].transform(
                lambda x: x.diff().abs().mean()
            )
    
    return df


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and engineer demographic and laboratory features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with demographic features added
    """
    # Identify demographic and lab columns
    demo_patterns = ['age', 'bmi', 'gender', 'sex', 'weight', 'height', 'lab', 'blood', 'biomarker']
    demo_cols = []
    
    for pattern in demo_patterns:
        cols = [col for col in df.columns if pattern in col.lower()]
        demo_cols.extend(cols)
    
    if not demo_cols:
        logger.warning("No demographic/lab columns found")
        return df
    
    logger.info(f"Found demographic columns: {demo_cols}")
    
    # Process demographic features
    for col in demo_cols:
        if col in df.columns:
            # Handle categorical variables
            if df[col].dtype == 'object':
                # Simple encoding for categorical variables
                unique_values = df[col].nunique()
                if unique_values <= 10:  # Reasonable number of categories
                    # One-hot encode
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df, dummies], axis=1)
            else:
                # Numerical features - create derived features
                values = pd.to_numeric(df[col], errors='coerce')
                
                # Standardized features
                df[f'{col}_zscore'] = (values - values.mean()) / values.std()
                
                # Binned features for some metrics
                if 'bmi' in col.lower():
                    df['bmi_underweight'] = (values < 18.5).astype(int)
                    df['bmi_normal'] = ((values >= 18.5) & (values < 25)).astype(int)
                    df['bmi_overweight'] = ((values >= 25) & (values < 30)).astype(int)
                    df['bmi_obese'] = (values >= 30).astype(int)
                
                if 'age' in col.lower():
                    df['age_young'] = (values < 30).astype(int)
                    df['age_middle'] = ((values >= 30) & (values < 50)).astype(int)
                    df['age_older'] = (values >= 50).astype(int)
    
    return df


def add_microbiome_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract microbiome features using PCA and diversity metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with microbiome features added
    """
    # Identify microbiome columns
    microbiome_patterns = ['microbe', 'bacteria', 'otu', 'genus', 'species', 'phylum']
    microbiome_cols = []
    
    for pattern in microbiome_patterns:
        cols = [col for col in df.columns if pattern in col.lower()]
        microbiome_cols.extend(cols)
    
    if not microbiome_cols:
        logger.warning("No microbiome columns found")
        return df
    
    logger.info(f"Found {len(microbiome_cols)} microbiome columns")
    
    # Extract microbiome data
    microbiome_data = df[microbiome_cols].select_dtypes(include=[np.number])
    
    if microbiome_data.empty:
        logger.warning("No numerical microbiome data found")
        return df
    
    # Fill missing values with zeros (assuming missing = not detected)
    microbiome_data = microbiome_data.fillna(0)
    
    # Alpha diversity metrics (Shannon diversity approximation)
    # Assuming relative abundance data
    microbiome_data_rel = microbiome_data.div(microbiome_data.sum(axis=1), axis=0)
    microbiome_data_rel = microbiome_data_rel.fillna(0)
    
    # Shannon diversity
    shannon_div = -(microbiome_data_rel * np.log(microbiome_data_rel + 1e-10)).sum(axis=1)
    df['microbiome_shannon_diversity'] = shannon_div
    
    # Species richness (number of non-zero taxa)
    df['microbiome_richness'] = (microbiome_data > 0).sum(axis=1)
    
    # Simpson diversity
    simpson_div = 1 - (microbiome_data_rel ** 2).sum(axis=1)
    df['microbiome_simpson_diversity'] = simpson_div
    
    # Evenness
    df['microbiome_evenness'] = shannon_div / np.log(df['microbiome_richness'] + 1)
    
    # PCA for dimensionality reduction
    try:
        # Standardize microbiome data
        scaler = StandardScaler()
        microbiome_scaled = scaler.fit_transform(microbiome_data)
        
        # Apply PCA
        n_components = min(10, microbiome_data.shape[1], microbiome_data.shape[0] - 1)
        pca = PCA(n_components=n_components)
        microbiome_pca = pca.fit_transform(microbiome_scaled)
        
        # Add PCA components as features
        for i in range(n_components):
            df[f'microbiome_pc{i+1}'] = microbiome_pca[:, i]
        
        # Add explained variance information
        df['microbiome_pca_explained_var'] = pca.explained_variance_ratio_.sum()
        
        logger.info(f"Added {n_components} microbiome PCA components")
        
    except Exception as e:
        logger.warning(f"PCA failed for microbiome data: {str(e)}")
    
    return df


def add_gut_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract gut health features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with gut health features added
    """
    # Identify gut health columns
    gut_patterns = ['gut', 'gi', 'gastrointestinal', 'digestive', 'bloat', 'ibs']
    gut_cols = []
    
    for pattern in gut_patterns:
        cols = [col for col in df.columns if pattern in col.lower()]
        gut_cols.extend(cols)
    
    if not gut_cols:
        logger.warning("No gut health columns found")
        return df
    
    logger.info(f"Found gut health columns: {gut_cols}")
    
    # Process gut health scores
    gut_scores = []
    for col in gut_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors='coerce')
            gut_scores.append(values)
            
            # Individual score features
            df[f'{col}_normalized'] = (values - values.mean()) / values.std()
    
    # Composite gut health score
    if gut_scores:
        gut_matrix = np.column_stack(gut_scores)
        df['gut_health_composite'] = np.nanmean(gut_matrix, axis=1)
        df['gut_health_std'] = np.nanstd(gut_matrix, axis=1)
    
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal and meal metadata features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with temporal features added
    """
    # Identify temporal columns
    time_cols = [col for col in df.columns if any(t in col.lower() for t in ['time', 'date', 'timestamp'])]
    
    if time_cols:
        time_col = time_cols[0]
        
        try:
            # Convert to datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
            
            # Extract temporal features
            df['hour_of_day'] = df[time_col].dt.hour
            df['day_of_week'] = df[time_col].dt.dayofweek
            df['month'] = df[time_col].dt.month
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Meal timing features
            df['is_breakfast'] = ((df['hour_of_day'] >= 6) & (df['hour_of_day'] < 10)).astype(int)
            df['is_lunch'] = ((df['hour_of_day'] >= 11) & (df['hour_of_day'] < 15)).astype(int)
            df['is_dinner'] = ((df['hour_of_day'] >= 17) & (df['hour_of_day'] < 21)).astype(int)
            df['is_snack'] = (1 - df['is_breakfast'] - df['is_lunch'] - df['is_dinner']).clip(0, 1)
            
            # Circadian features
            df['sin_hour'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['cos_hour'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['sin_day'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['cos_day'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
        except Exception as e:
            logger.warning(f"Failed to process temporal features: {str(e)}")
    
    # Meal frequency features (if participant data available)
    if 'participant_id' in df.columns:
        df['meals_per_day'] = df.groupby(['participant_id', df[time_col].dt.date if time_cols else 'participant_id']).transform('count').iloc[:, 0]
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between different modalities.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with interaction features added
    """
    # Glucose-Activity interactions
    glucose_cols = [col for col in df.columns if 'glucose' in col.lower() and '_mean' in col]
    activity_cols = [col for col in df.columns if any(a in col.lower() for a in ['step', 'heart']) and '_mean' in col]
    
    for g_col in glucose_cols[:2]:  # Limit to avoid too many features
        for a_col in activity_cols[:2]:
            if g_col in df.columns and a_col in df.columns:
                df[f'{g_col}_x_{a_col}'] = df[g_col] * df[a_col]
    
    # BMI-Glucose interactions
    bmi_cols = [col for col in df.columns if 'bmi' in col.lower()]
    for bmi_col in bmi_cols[:1]:
        for g_col in glucose_cols[:1]:
            if bmi_col in df.columns and g_col in df.columns:
                df[f'{bmi_col}_x_{g_col}'] = df[bmi_col] * df[g_col]
    
    # Microbiome-Glucose interactions
    microbiome_cols = [col for col in df.columns if 'microbiome' in col.lower() and 'diversity' in col]
    for m_col in microbiome_cols[:2]:
        for g_col in glucose_cols[:1]:
            if m_col in df.columns and g_col in df.columns:
                df[f'{m_col}_x_{g_col}'] = df[m_col] * df[g_col]
    
    return df


def clean_and_validate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the engineered features.
    
    Args:
        df: DataFrame with engineered features
        
    Returns:
        Cleaned DataFrame
    """
    initial_shape = df.shape
    
    # Remove columns with too many missing values
    missing_threshold = 0.8
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index
    
    if len(cols_to_drop) > 0:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >80% missing values")
        df = df.drop(columns=cols_to_drop)
    
    # Remove columns with zero variance
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_var_cols = []
    
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            zero_var_cols.append(col)
    
    if zero_var_cols:
        logger.info(f"Dropping {len(zero_var_cols)} zero-variance columns")
        df = df.drop(columns=zero_var_cols)
    
    # Handle infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining missing values with median for numeric columns
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    
    final_shape = df.shape
    logger.info(f"Feature cleaning completed. Shape changed from {initial_shape} to {final_shape}")
    
    return df


def get_feature_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of engineered features.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Dictionary with feature summary
    """
    feature_groups = {
        'glucose': [col for col in df.columns if 'glucose' in col.lower()],
        'activity': [col for col in df.columns if any(a in col.lower() for a in ['step', 'heart', 'hr', 'activity'])],
        'demographic': [col for col in df.columns if any(d in col.lower() for d in ['age', 'bmi', 'gender', 'weight'])],
        'microbiome': [col for col in df.columns if 'microbiome' in col.lower()],
        'gut_health': [col for col in df.columns if 'gut' in col.lower()],
        'temporal': [col for col in df.columns if any(t in col.lower() for t in ['hour', 'day', 'time', 'meal'])],
        'interaction': [col for col in df.columns if '_x_' in col],
        'target': [col for col in df.columns if col.lower() == 'ccr']
    }
    
    summary = {
        'total_features': len(df.columns),
        'feature_groups': {group: len(cols) for group, cols in feature_groups.items()},
        'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100).round(2),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object']).columns)
    }
    
    return summary


if __name__ == "__main__":
    # Example usage with sample data
    sample_data = pd.DataFrame({
        'participant_id': [1, 1, 2, 2, 3, 3],
        'glucose': [120, 140, 110, 130, 125, 135],
        'steps': [8000, 12000, 6000, 10000, 9000, 11000],
        'heart_rate': [70, 85, 65, 80, 75, 90],
        'age': [30, 30, 45, 45, 25, 25],
        'bmi': [24.5, 24.5, 28.2, 28.2, 22.1, 22.1],
        'ccr': [0.4, 0.5, 0.3, 0.4, 0.6, 0.5]
    })
    
    result = engineer_features(sample_data)
    summary = get_feature_summary(result)
    
    print("Feature Engineering Summary:")
    print(summary)
    print(f"\nResult shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")