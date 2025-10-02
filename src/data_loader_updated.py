"""
Data Loading Module for CGMacros Dataset

This module handles loading and initial preprocessing of the CGMacros dataset
including participant CGM data, biomarker data, microbiome data, and gut health scores.
Updated to handle actual dataset structure discovered.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader class for loading and merging CGMacros dataset files.
    
    Handles:
    - Loading individual CGMacros participant files (time-series data)
    - Loading supplementary data (demographics, microbiome, gut health)
    - Merging all data sources on participant ID
    """
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Path to directory containing raw data files
        """
        self.data_dir = Path(data_dir)
        self.cgmacros_dir = self.data_dir / "CGMacros_CSVs"
        self.cgmacros_pattern = "CGMacros-*.csv"
        
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame dtypes to reduce memory usage.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            DataFrame with optimized dtypes
        """
        df_optimized = df.copy()
        
        # Optimize numeric columns
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            if col in ['participant_id']:
                # Keep participant_id as int for consistency
                df_optimized[col] = df_optimized[col].astype('int16')
            elif df_optimized[col].dtype == 'int64':
                # Try to downcast integers
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df_optimized[col] = df_optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df_optimized[col] = df_optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df_optimized[col] = df_optimized[col].astype(np.int32)
            elif df_optimized[col].dtype == 'float64':
                # Try to downcast floats
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Optimize object columns (strings)
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if col not in ['Timestamp']:  # Skip timestamp column
                num_unique_values = len(df_optimized[col].unique())
                num_total_values = len(df_optimized[col])
                if num_unique_values / num_total_values < 0.5:  # If less than 50% unique values
                    df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized
        
    def load_cgmacros_data(self, chunk_size: int = 5) -> pd.DataFrame:
        """
        Load all CGMacros participant files and combine them using memory-efficient chunked processing.
        Each file contains time-series data for one participant.
        
        Expected columns in CGMacros files:
        - Timestamp: Time of measurement
        - Libre GL: Libre glucose level
        - Dexcom GL: Dexcom glucose level  
        - HR: Heart rate
        - Calories: Calorie measurement
        - METs: Metabolic equivalent
        - Meal Type: Type of meal
        - Carbs, Protein, Fat, Fiber: Macronutrients
        - Amount Consumed: Amount of food consumed
        - Image path: Path to meal image
        
        Args:
            chunk_size: Number of participant files to process at once (default: 5)
            
        Returns:
            Combined DataFrame with all participants' CGMacros data
        """
        logger.info("Loading CGMacros participant files with memory optimization...")
        
        cgmacros_files = list(self.cgmacros_dir.glob(self.cgmacros_pattern))
        if not cgmacros_files:
            raise FileNotFoundError(f"No CGMacros files found in {self.cgmacros_dir}")
        
        logger.info(f"Found {len(cgmacros_files)} participant files. Processing in chunks of {chunk_size}...")
        
        # Initialize empty list to store chunk results
        all_chunks = []
        total_records = 0
        
        # Process files in chunks to manage memory
        for i in range(0, len(cgmacros_files), chunk_size):
            chunk_files = cgmacros_files[i:i + chunk_size]
            chunk_data = []
            
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(cgmacros_files)-1)//chunk_size + 1} "
                       f"(files {i+1}-{min(i+chunk_size, len(cgmacros_files))})")
            
            for file_path in chunk_files:
                # Extract participant ID from filename (e.g., CGMacros-001.csv -> 1)
                participant_id = int(file_path.stem.split('-')[1])
                
                try:
                    # Use efficient dtypes and optimize memory usage
                    df = pd.read_csv(file_path, low_memory=False)
                    df['participant_id'] = participant_id
                    
                    # Convert timestamp efficiently
                    if 'Timestamp' in df.columns:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                    
                    # Optimize numeric dtypes to reduce memory
                    df = self._optimize_dtypes(df)
                    
                    chunk_data.append(df)
                    total_records += len(df)
                    logger.info(f"Loaded {len(df)} records for participant {participant_id}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    continue
            
            if chunk_data:
                # Combine chunk and store result
                chunk_combined = pd.concat(chunk_data, ignore_index=True)
                all_chunks.append(chunk_combined)
                logger.info(f"Chunk {i//chunk_size + 1} combined: {len(chunk_combined)} records")
                
                # Clear chunk_data to free memory
                del chunk_data
                import gc
                gc.collect()
        
        if not all_chunks:
            raise ValueError("No CGMacros data could be loaded")
        
        # Final combination of all chunks
        logger.info("Combining all chunks into final dataset...")
        combined_df = pd.concat(all_chunks, ignore_index=True)
        
        # Final memory optimization
        combined_df = self._optimize_dtypes(combined_df)
        
        logger.info(f"Successfully loaded complete CGMacros data: {len(combined_df)} total records "
                   f"from {len(cgmacros_files)} participants")
        
        return combined_df
    
    def load_demographics(self) -> pd.DataFrame:
        """
        Load demographic and laboratory data from bio.csv with memory optimization.
        
        Expected columns in bio.csv:
        - subject: Participant ID
        - Age, Gender, BMI, Body weight, Height
        - A1c, Fasting GLU, Insulin
        - Triglycerides, Cholesterol, etc.
        
        Returns:
            DataFrame with participant demographics and lab results
        """
        demo_file = self.data_dir / "bio.csv"
        if not demo_file.exists():
            logger.warning("Demographics file (bio.csv) not found")
            return pd.DataFrame()
        
        df = pd.read_csv(demo_file, low_memory=False)
        
        # Rename 'subject' to 'participant_id' for consistency
        if 'subject' in df.columns:
            df = df.rename(columns={'subject': 'participant_id'})
        
        # Optimize dtypes
        df = self._optimize_dtypes(df)
            
        logger.info(f"Loaded demographics for {len(df)} participants")
        return df
    
    def load_microbiome(self, max_features: int = None) -> pd.DataFrame:
        """
        Load microbiome composition data from microbes.csv with memory optimization for Colab.
        Contains abundance/presence data for thousands of microbial species.
        
        Expected structure:
        - subject: Participant ID
        - Thousands of columns for different bacterial species (binary or abundance values)
        
        Args:
            max_features: Maximum number of most prevalent microbial features to keep 
                         (None = use ALL features for maximum biological diversity)
        
        Returns:
            DataFrame with microbiome data for each participant
        """
        microbiome_file = self.data_dir / "microbes.csv"
        if not microbiome_file.exists():
            logger.warning("Microbiome file (microbes.csv) not found")
            return pd.DataFrame()
        
        # Read file with optimized memory usage
        df = pd.read_csv(microbiome_file, low_memory=False)
        
        # Rename 'subject' to 'participant_id' for consistency
        if 'subject' in df.columns:
            df = df.rename(columns={'subject': 'participant_id'})
        
        # Feature selection for microbiome data (only if max_features is specified)
        microbiome_cols = [col for col in df.columns if col != 'participant_id']
        
        if max_features and len(microbiome_cols) > max_features:
            logger.info(f"Reducing microbiome features from {len(microbiome_cols)} to {max_features} most prevalent")
            
            # Calculate prevalence (non-zero values) for each microbial feature
            prevalence = (df[microbiome_cols] > 0).sum().sort_values(ascending=False)
            top_features = prevalence.head(max_features).index.tolist()
            
            # Keep only top features plus participant_id
            df = df[['participant_id'] + top_features]
        else:
            logger.info(f"Using ALL {len(microbiome_cols)} microbiome features for maximum biological diversity")
            
        # Optimize dtypes
        df = self._optimize_dtypes(df)
            
        logger.info(f"Loaded microbiome data for {len(df)} participants with {df.shape[1]-1} microbial features")
        return df
    
    def load_gut_health(self) -> pd.DataFrame:
        """
        Load gut health assessment scores from gut_health_test.csv with memory optimization.
        
        Expected columns:
        - subject: Participant ID
        - Various gut health metrics (Gut Lining Health, LPS Biosynthesis, etc.)
        
        Returns:
            DataFrame with gut health scores for each participant
        """
        gut_health_file = self.data_dir / "gut_health_test.csv"
        if not gut_health_file.exists():
            logger.warning("Gut health file (gut_health_test.csv) not found")
            return pd.DataFrame()
        
        df = pd.read_csv(gut_health_file, low_memory=False)
        
        # Rename 'subject' to 'participant_id' for consistency
        if 'subject' in df.columns:
            df = df.rename(columns={'subject': 'participant_id'})
        
        # Optimize dtypes
        df = self._optimize_dtypes(df)
            
        logger.info(f"Loaded gut health data for {len(df)} participants with {df.shape[1]-1} health metrics")
        return df
    
    def merge_data_sources(self, cgmacros_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge CGMacros time-series data with participant-level supplementary data using memory optimization.
        
        Args:
            cgmacros_df: Main CGMacros DataFrame (time-series)
            
        Returns:
            Merged DataFrame with all data sources
        """
        logger.info("Merging data sources with memory optimization...")
        
        # Start with CGMacros data (already optimized)
        merged_df = cgmacros_df.copy()
        initial_memory = merged_df.memory_usage(deep=True).sum() / 1024**2
        logger.info(f"Initial CGMacros data memory usage: {initial_memory:.1f} MB")
        
        # Load and merge demographics (participant-level data)
        demographics_df = self.load_demographics()
        if not demographics_df.empty:
            demographics_df = self._optimize_dtypes(demographics_df)
            merged_df = merged_df.merge(demographics_df, on='participant_id', how='left')
            logger.info("Merged demographics data")
            
            # Clear demographics_df to free memory
            del demographics_df
            import gc
            gc.collect()
        
        # Load and merge microbiome data (participant-level data with ALL 1979 features)
        microbiome_df = self.load_microbiome()  # Use ALL microbiome features for maximum biological diversity
        if not microbiome_df.empty:
            merged_df = merged_df.merge(microbiome_df, on='participant_id', how='left')
            logger.info("Merged microbiome data")
            
            # Clear microbiome_df to free memory
            del microbiome_df
            gc.collect()
        
        # Load and merge gut health data (participant-level data)
        gut_health_df = self.load_gut_health()
        if not gut_health_df.empty:
            gut_health_df = self._optimize_dtypes(gut_health_df)
            merged_df = merged_df.merge(gut_health_df, on='participant_id', how='left')
            logger.info("Merged gut health data")
            
            # Clear gut_health_df to free memory
            del gut_health_df
            gc.collect()
        
        # Final optimization of merged data
        merged_df = self._optimize_dtypes(merged_df)
        final_memory = merged_df.memory_usage(deep=True).sum() / 1024**2
        
        logger.info(f"Final merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        logger.info(f"Final memory usage: {final_memory:.1f} MB")
        
        return merged_df
    
    def load_all_data(self) -> pd.DataFrame:
        """
        Load and merge all data sources.
        
        Returns:
            Complete merged DataFrame ready for feature engineering
        """
        # Load main CGMacros data
        cgmacros_df = self.load_cgmacros_data()
        
        # Merge with supplementary data
        merged_df = self.merge_data_sources(cgmacros_df)
        
        return merged_df

def load_participant_files(data_dir: str = "data/raw") -> Dict[int, pd.DataFrame]:
    """
    Load individual participant files as separate DataFrames.
    
    Args:
        data_dir: Path to directory containing CGMacros files
        
    Returns:
        Dictionary mapping participant ID to their DataFrame
    """
    data_path = Path(data_dir) / "CGMacros_CSVs"
    participant_files = list(data_path.glob("CGMacros-*.csv"))
    
    participant_data = {}
    
    for file_path in participant_files:
        participant_id = int(file_path.stem.split('-')[1])
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp if present
            if 'Timestamp' in df.columns:
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                
            participant_data[participant_id] = df
            logger.info(f"Loaded {len(df)} records for participant {participant_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {e}")
            continue
    
    return participant_data

# Legacy function for backward compatibility
def load_cgmacros_data(data_dir: str) -> pd.DataFrame:
    """
    Legacy function to load CGMacros data.
    Uses the new DataLoader class internally.
    
    Args:
        data_dir: Path to directory containing raw data files
        
    Returns:
        Combined DataFrame with all data sources
    """
    loader = DataLoader(data_dir)
    return loader.load_all_data()