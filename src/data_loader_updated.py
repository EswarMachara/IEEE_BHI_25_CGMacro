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
        
    def load_cgmacros_data(self) -> pd.DataFrame:
        """
        Load all CGMacros participant files and combine them.
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
        
        Returns:
            Combined DataFrame with all participants' CGMacros data
        """
        logger.info("Loading CGMacros participant files...")
        
        cgmacros_files = list(self.cgmacros_dir.glob(self.cgmacros_pattern))
        if not cgmacros_files:
            raise FileNotFoundError(f"No CGMacros files found in {self.cgmacros_dir}")
        
        all_data = []
        
        for file_path in cgmacros_files:
            # Extract participant ID from filename (e.g., CGMacros-001.csv -> 1)
            participant_id = int(file_path.stem.split('-')[1])
            
            try:
                df = pd.read_csv(file_path)
                df['participant_id'] = participant_id
                
                # Convert timestamp if present
                if 'Timestamp' in df.columns:
                    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                
                all_data.append(df)
                logger.info(f"Loaded {len(df)} records for participant {participant_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No CGMacros data could be loaded")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined CGMacros data: {len(combined_df)} total records from {len(all_data)} participants")
        
        return combined_df
    
    def load_demographics(self) -> pd.DataFrame:
        """
        Load demographic and laboratory data from bio.csv.
        
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
        
        df = pd.read_csv(demo_file)
        
        # Rename 'subject' to 'participant_id' for consistency
        if 'subject' in df.columns:
            df = df.rename(columns={'subject': 'participant_id'})
            
        logger.info(f"Loaded demographics for {len(df)} participants")
        return df
    
    def load_microbiome(self) -> pd.DataFrame:
        """
        Load microbiome composition data from microbes.csv.
        Contains abundance/presence data for thousands of microbial species.
        
        Expected structure:
        - subject: Participant ID
        - Thousands of columns for different bacterial species (binary or abundance values)
        
        Returns:
            DataFrame with microbiome data for each participant
        """
        microbiome_file = self.data_dir / "microbes.csv"
        if not microbiome_file.exists():
            logger.warning("Microbiome file (microbes.csv) not found")
            return pd.DataFrame()
        
        df = pd.read_csv(microbiome_file)
        
        # Rename 'subject' to 'participant_id' for consistency
        if 'subject' in df.columns:
            df = df.rename(columns={'subject': 'participant_id'})
            
        logger.info(f"Loaded microbiome data for {len(df)} participants with {df.shape[1]-1} microbial features")
        return df
    
    def load_gut_health(self) -> pd.DataFrame:
        """
        Load gut health assessment scores from gut_health_test.csv.
        
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
        
        df = pd.read_csv(gut_health_file)
        
        # Rename 'subject' to 'participant_id' for consistency
        if 'subject' in df.columns:
            df = df.rename(columns={'subject': 'participant_id'})
            
        logger.info(f"Loaded gut health data for {len(df)} participants with {df.shape[1]-1} health metrics")
        return df
    
    def merge_data_sources(self, cgmacros_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge CGMacros time-series data with participant-level supplementary data.
        
        Args:
            cgmacros_df: Main CGMacros DataFrame (time-series)
            
        Returns:
            Merged DataFrame with all data sources
        """
        logger.info("Merging data sources...")
        
        # Load supplementary data
        demographics_df = self.load_demographics()
        microbiome_df = self.load_microbiome()
        gut_health_df = self.load_gut_health()
        
        # Start with CGMacros data
        merged_df = cgmacros_df.copy()
        
        # Merge demographics (participant-level data)
        if not demographics_df.empty:
            merged_df = merged_df.merge(demographics_df, on='participant_id', how='left')
            logger.info("Merged demographics data")
        
        # Merge microbiome data (participant-level data)
        if not microbiome_df.empty:
            merged_df = merged_df.merge(microbiome_df, on='participant_id', how='left')
            logger.info("Merged microbiome data")
        
        # Merge gut health data (participant-level data)
        if not gut_health_df.empty:
            merged_df = merged_df.merge(gut_health_df, on='participant_id', how='left')
            logger.info("Merged gut health data")
        
        logger.info(f"Final merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
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