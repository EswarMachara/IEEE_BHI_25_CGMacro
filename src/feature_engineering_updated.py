"""
Feature Engineering Module for CGMacros Dataset

This module handles extraction and engineering of features from multimodal data
including glucose patterns, activity metrics, microbiome features, and derived metrics.
Updated to handle actual dataset structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for creating features from multimodal CGMacros data.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        
    def add_glucose_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract glucose-related features from CGM data.
        
        Expected glucose columns: 'Libre GL', 'Dexcom GL'
        
        Args:
            df: DataFrame with glucose measurements
            
        Returns:
            DataFrame with additional glucose features
        """
        logger.info("Adding glucose features...")
        
        glucose_cols = ['Libre GL', 'Dexcom GL']
        available_glucose_cols = [col for col in glucose_cols if col in df.columns]
        
        if not available_glucose_cols:
            logger.warning("No glucose columns found")
            return df
        
        df = df.copy()
        
        for glucose_col in available_glucose_cols:
            # Skip if column is all NaN
            if df[glucose_col].isna().all():
                continue
                
            prefix = glucose_col.replace(' ', '_').lower()
            
            # Basic statistics
            df[f'{prefix}_mean'] = df.groupby('participant_id')[glucose_col].transform('mean')
            df[f'{prefix}_std'] = df.groupby('participant_id')[glucose_col].transform('std')
            df[f'{prefix}_min'] = df.groupby('participant_id')[glucose_col].transform('min')
            df[f'{prefix}_max'] = df.groupby('participant_id')[glucose_col].transform('max')
            df[f'{prefix}_median'] = df.groupby('participant_id')[glucose_col].transform('median')
            
            # Variability metrics
            df[f'{prefix}_cv'] = df[f'{prefix}_std'] / df[f'{prefix}_mean']
            df[f'{prefix}_range'] = df[f'{prefix}_max'] - df[f'{prefix}_min']
            
            # Time-based features (if timestamp available)
            if 'Timestamp' in df.columns:
                df_sorted = df.sort_values(['participant_id', 'Timestamp'])
                
                # Glucose rate of change
                df_sorted[f'{prefix}_diff'] = df_sorted.groupby('participant_id')[glucose_col].diff()
                df_sorted[f'{prefix}_rate_change'] = df_sorted.groupby('participant_id')[f'{prefix}_diff'].transform('mean')
                
                # Time in range features (normal glucose: 70-180 mg/dL)
                glucose_values = df_sorted[glucose_col].fillna(0)
                df_sorted[f'{prefix}_time_in_range'] = df_sorted.groupby('participant_id').apply(
                    lambda x: ((x[glucose_col] >= 70) & (x[glucose_col] <= 180)).mean()
                ).reset_index(level=0, drop=True)
                
                df_sorted[f'{prefix}_time_above_range'] = df_sorted.groupby('participant_id').apply(
                    lambda x: (x[glucose_col] > 180).mean()
                ).reset_index(level=0, drop=True)
                
                df_sorted[f'{prefix}_time_below_range'] = df_sorted.groupby('participant_id').apply(
                    lambda x: (x[glucose_col] < 70).mean()
                ).reset_index(level=0, drop=True)
                
                # Update original dataframe
                df = df_sorted
        
        logger.info(f"Added glucose features for columns: {available_glucose_cols}")
        return df
    
    def add_activity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract activity-related features.
        
        Expected activity columns: 'HR', 'METs', 'Calories'
        
        Args:
            df: DataFrame with activity measurements
            
        Returns:
            DataFrame with additional activity features
        """
        logger.info("Adding activity features...")
        
        activity_cols = ['HR', 'METs', 'Calories']
        available_activity_cols = [col for col in activity_cols if col in df.columns]
        
        if not available_activity_cols:
            logger.warning("No activity columns found")
            return df
        
        df = df.copy()
        
        for activity_col in available_activity_cols:
            if df[activity_col].isna().all():
                continue
                
            prefix = activity_col.lower()
            
            # Basic statistics per participant
            df[f'{prefix}_mean'] = df.groupby('participant_id')[activity_col].transform('mean')
            df[f'{prefix}_std'] = df.groupby('participant_id')[activity_col].transform('std')
            df[f'{prefix}_min'] = df.groupby('participant_id')[activity_col].transform('min')
            df[f'{prefix}_max'] = df.groupby('participant_id')[activity_col].transform('max')
            df[f'{prefix}_median'] = df.groupby('participant_id')[activity_col].transform('median')
            
            # Activity intensity categories (for HR)
            if activity_col == 'HR':
                df['hr_zone_1'] = (df[activity_col] < 100).astype(int)  # Rest
                df['hr_zone_2'] = ((df[activity_col] >= 100) & (df[activity_col] < 120)).astype(int)  # Light
                df['hr_zone_3'] = ((df[activity_col] >= 120) & (df[activity_col] < 140)).astype(int)  # Moderate
                df['hr_zone_4'] = ((df[activity_col] >= 140) & (df[activity_col] < 160)).astype(int)  # High
                df['hr_zone_5'] = (df[activity_col] >= 160).astype(int)  # Maximum
        
        logger.info(f"Added activity features for columns: {available_activity_cols}")
        return df
    
    def add_meal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract meal-related features.
        
        Expected meal columns: 'Meal Type', 'Amount Consumed'
        Expected nutrient columns: 'Carbs', 'Protein', 'Fat', 'Fiber'
        
        Args:
            df: DataFrame with meal information
            
        Returns:
            DataFrame with additional meal features
        """
        logger.info("Adding meal features...")
        
        df = df.copy()
        
        # Encode meal type if present
        if 'Meal Type' in df.columns:
            # Fill missing meal types
            df['Meal Type'] = df['Meal Type'].fillna('No Meal')
            
            # Create meal type dummies
            meal_dummies = pd.get_dummies(df['Meal Type'], prefix='meal_type')
            df = pd.concat([df, meal_dummies], axis=1)
            
            # Count meals per participant
            df['meals_per_day'] = df.groupby('participant_id')['Meal Type'].transform(
                lambda x: (x != 'No Meal').sum()
            )
        
        # Amount consumed features
        if 'Amount Consumed' in df.columns:
            df['amount_consumed_mean'] = df.groupby('participant_id')['Amount Consumed'].transform('mean')
            df['amount_consumed_std'] = df.groupby('participant_id')['Amount Consumed'].transform('std')
        
        # Macronutrient features (before removing for CCR calculation)
        nutrient_cols = ['Carbs', 'Protein', 'Fat', 'Fiber']
        available_nutrients = [col for col in nutrient_cols if col in df.columns]
        
        if available_nutrients:
            for nutrient in available_nutrients:
                if not df[nutrient].isna().all():
                    df[f'{nutrient.lower()}_mean'] = df.groupby('participant_id')[nutrient].transform('mean')
                    df[f'{nutrient.lower()}_total'] = df.groupby('participant_id')[nutrient].transform('sum')
        
        logger.info("Added meal features")
        return df
    
    def add_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process demographic and laboratory features.
        
        Expected columns: Age, Gender, BMI, A1c, etc.
        
        Args:
            df: DataFrame with demographic data
            
        Returns:
            DataFrame with processed demographic features
        """
        logger.info("Adding demographic features...")
        
        df = df.copy()
        
        # Gender encoding
        if 'Gender' in df.columns:
            if 'gender' not in self.encoders:
                self.encoders['gender'] = LabelEncoder()
                df['gender_encoded'] = self.encoders['gender'].fit_transform(df['Gender'].fillna('Unknown'))
            else:
                df['gender_encoded'] = self.encoders['gender'].transform(df['Gender'].fillna('Unknown'))
        
        # Age categories
        if 'Age' in df.columns:
            df['age_category'] = pd.cut(df['Age'], 
                                      bins=[0, 30, 50, 65, 100], 
                                      labels=['Young', 'Middle', 'Senior', 'Elderly'])
            age_dummies = pd.get_dummies(df['age_category'], prefix='age')
            df = pd.concat([df, age_dummies], axis=1)
        
        # BMI categories
        if 'BMI' in df.columns:
            df['bmi_category'] = pd.cut(df['BMI'], 
                                      bins=[0, 18.5, 25, 30, 50], 
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
            bmi_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi')
            df = pd.concat([df, bmi_dummies], axis=1)
        
        # Laboratory values - create derived features
        lab_cols = ['A1c', 'Fasting GLU', 'Insulin', 'Triglycerides', 'Cholesterol']
        available_labs = [col for col in lab_cols if col in df.columns]
        
        for lab_col in available_labs:
            if not df[lab_col].isna().all():
                # Create categories for key lab values
                if lab_col == 'A1c':
                    df['a1c_category'] = pd.cut(df[lab_col], 
                                              bins=[0, 5.7, 6.5, 15], 
                                              labels=['Normal', 'Prediabetic', 'Diabetic'])
                elif lab_col == 'Fasting GLU':
                    df['fasting_glu_category'] = pd.cut(df[lab_col], 
                                                       bins=[0, 100, 126, 500], 
                                                       labels=['Normal', 'Prediabetic', 'Diabetic'])
        
        logger.info("Added demographic features")
        return df
    
    def add_microbiome_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process microbiome features.
        
        Args:
            df: DataFrame with microbiome data
            
        Returns:
            DataFrame with processed microbiome features
        """
        logger.info("Adding microbiome features...")
        
        # Identify microbiome columns (exclude participant_id and other non-microbiome columns)
        exclude_cols = ['participant_id', 'Timestamp', 'Libre GL', 'Dexcom GL', 'HR', 'Calories', 
                       'METs', 'Meal Type', 'Carbs', 'Protein', 'Fat', 'Fiber', 'Amount Consumed',
                       'Image path', 'Age', 'Gender', 'BMI', 'A1c', 'Fasting GLU', 'Insulin']
        
        microbiome_cols = [col for col in df.columns if col not in exclude_cols and 
                          not col.endswith(('_mean', '_std', '_min', '_max', '_median', '_encoded', '_category'))]
        
        if not microbiome_cols:
            logger.warning("No microbiome columns identified")
            return df
        
        df = df.copy()
        
        # Calculate microbiome diversity metrics
        microbiome_data = df[microbiome_cols].fillna(0)
        
        # Alpha diversity (Shannon diversity index)
        def shannon_diversity(row):
            proportions = row / row.sum() if row.sum() > 0 else row
            proportions = proportions[proportions > 0]
            return -np.sum(proportions * np.log(proportions)) if len(proportions) > 0 else 0
        
        df['microbiome_shannon_diversity'] = microbiome_data.apply(shannon_diversity, axis=1)
        
        # Species richness (number of non-zero species)
        df['microbiome_richness'] = (microbiome_data > 0).sum(axis=1)
        
        # Total abundance
        df['microbiome_total_abundance'] = microbiome_data.sum(axis=1)
        
        # Dominant species features
        if len(microbiome_cols) > 10:
            # Select top 10 most prevalent species across all participants
            species_prevalence = (microbiome_data > 0).sum().sort_values(ascending=False)
            top_species = species_prevalence.head(10).index.tolist()
            
            for species in top_species:
                df[f'microbiome_{species}_present'] = (df[species] > 0).astype(int)
        
        logger.info(f"Added microbiome features from {len(microbiome_cols)} species")
        return df
    
    def add_gut_health_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process gut health features.
        
        Expected columns from gut_health_test.csv
        
        Args:
            df: DataFrame with gut health data
            
        Returns:
            DataFrame with processed gut health features
        """
        logger.info("Adding gut health features...")
        
        gut_health_cols = [
            'Gut Lining Health', 'LPS Biosynthesis Pathways',
            'Biofilm, Chemotaxis, and Virulence Pathways', 'TMA Production Pathways',
            'Ammonia Production Pathways', 'Metabolic Fitness', 'Active Microbial Diversity',
            'Butyrate Production Pathways', 'Digestive Efficiency', 'Protein Fermentation'
        ]
        
        available_gut_health = [col for col in gut_health_cols if col in df.columns]
        
        if not available_gut_health:
            logger.warning("No gut health columns found")
            return df
        
        df = df.copy()
        
        # Create composite gut health scores
        if len(available_gut_health) > 3:
            # Overall gut health score (mean of available metrics)
            df['gut_health_composite'] = df[available_gut_health].mean(axis=1)
            
            # Gut health categories
            df['gut_health_category'] = pd.cut(df['gut_health_composite'], 
                                             bins=[0, 1.5, 2.5, 4], 
                                             labels=['Poor', 'Fair', 'Good'])
        
        logger.info(f"Added gut health features from {len(available_gut_health)} metrics")
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from timestamp data.
        
        Args:
            df: DataFrame with Timestamp column
            
        Returns:
            DataFrame with temporal features
        """
        if 'Timestamp' not in df.columns:
            logger.warning("No Timestamp column found")
            return df
        
        logger.info("Adding temporal features...")
        
        df = df.copy()
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Extract time components
        df['hour'] = df['Timestamp'].dt.hour
        df['day_of_week'] = df['Timestamp'].dt.dayofweek
        df['month'] = df['Timestamp'].dt.month
        
        # Create time periods
        df['time_of_day'] = pd.cut(df['hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Create dummy variables for time periods
        time_dummies = pd.get_dummies(df['time_of_day'], prefix='time')
        df = pd.concat([df, time_dummies], axis=1)
        
        logger.info("Added temporal features")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering steps.
        
        Args:
            df: Raw merged DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Apply all feature engineering steps
        df = self.add_temporal_features(df)
        df = self.add_glucose_features(df)
        df = self.add_activity_features(df)
        df = self.add_meal_features(df)
        df = self.add_demographic_features(df)
        df = self.add_microbiome_features(df)
        df = self.add_gut_health_features(df)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: DataFrame with features
            method: Scaling method ('standard' or 'minmax')
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {method} scaler...")
        
        df = df.copy()
        
        # Identify numerical columns to scale
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = ['participant_id', 'Timestamp']
        numerical_cols = [col for col in numerical_cols if col not in categorical_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        
        # Fit and transform numerical features
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(0))
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        logger.info(f"Scaled {len(numerical_cols)} numerical features")
        return df