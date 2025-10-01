# IEEE BHI 2025 Track 2: CGMacros Challenge

## Challenge Objective

This repository contains our solution for the **IEEE BHI 2025 Track 2 Challenge**: Predicting the **Carbohydrate Caloric Ratio (CCR)** of meals from multimodal data using the CGMacros dataset.

**Target Variable**: CCR = net_carbs / (net_carbs + protein + fat + fiber)

# IEEE BHI 2025 Track 2: CGMacros Challenge

## Challenge Objective

This repository contains our solution for the **IEEE BHI 2025 Track 2 Challenge**: Predicting the **Carbohydrate Caloric Ratio (CCR)** of meals from multimodal data using the CGMacros dataset.

**Target Variable**: CCR = net_carbs / (net_carbs + protein + fat + fiber)

## Actual Dataset Structure (After Analysis)

After examining the actual CSV files, we discovered the following dataset structure:

### Data Sources

1. **CGMacros Files (`CGMacros_CSVs/`)**:
   - **44 participant files** (CGMacros-001.csv through CGMacros-049.csv)
   - **Time-series data** with columns:
     - `Timestamp`: Time of measurement
     - `Libre GL`: Libre glucose level
     - `Dexcom GL`: Dexcom glucose level
     - `HR`: Heart rate
     - `Calories`: Calorie measurement
     - `METs`: Metabolic equivalent
     - `Meal Type`: Type of meal
     - `Carbs`, `Protein`, `Fat`, `Fiber`: Macronutrients
     - `Amount Consumed`: Amount of food consumed
     - `Image path`: Path to meal image

2. **Demographics & Lab Data (`bio.csv`)**:
   - **Participant-level data**:
     - Demographics: Age, Gender, BMI, Body weight, Height
     - Lab values: A1c, Fasting GLU, Insulin, Triglycerides, Cholesterol
     - Other metabolic markers

3. **Microbiome Data (`microbes.csv`)**:
   - **Thousands of bacterial species columns**
   - Binary/abundance values for each microbial species
   - 4 participants with complete microbiome profiles

4. **Gut Health Scores (`gut_health_test.csv`)**:
   - **22 gut health metrics** including:
     - Gut Lining Health, LPS Biosynthesis Pathways
     - Biofilm/Chemotaxis/Virulence Pathways
     - Metabolic Fitness, Active Microbial Diversity
     - Butyrate Production, Digestive Efficiency, etc.

### Important Data Characteristics
- **Time-series nature**: CGMacros files contain timestamped measurements
- **Participant-level supplementary data**: Demographics, microbiome, and gut health are per-participant
- **Multimodal fusion**: Need to merge time-series with participant-level features
- **Data sparsity**: Not all time points have meal information or complete sensor data

## Updated Repository Structure

```
├── data/
│   ├── raw/                    # Raw CSV files
│   │   ├── CGMacros_CSVs/     # 44 participant time-series files
│   │   │   ├── CGMacros-001.csv
│   │   │   ├── CGMacros-002.csv
│   │   │   └── ... (through CGMacros-049.csv)
│   │   ├── bio.csv            # Demographics and laboratory data
│   │   ├── microbes.csv       # Microbiome composition data
│   │   └── gut_health_test.csv # Gut health assessment scores
│   └── processed/             # Cleaned and feature-engineered datasets
├── notebooks/                 # Jupyter notebooks for exploration and experiments
│   ├── 01_data_exploration_updated.ipynb  # Updated exploration notebook
│   └── 02_model_training.ipynb
├── src/                       # Modular Python source code
│   ├── data_loader_updated.py         # Updated loader for actual data structure
│   ├── feature_engineering_updated.py # Updated feature engineering
│   ├── target_updated.py              # Updated CCR computation
│   ├── models.py              # Model implementations
│   ├── evaluation.py          # Metrics and evaluation utilities
│   └── visualization.py       # Plotting and visualization utilities
├── models/                    # Saved trained models
├── results/                   # Metrics, plots, feature importance outputs
├── requirements.txt           # Project dependencies
└── run_pipeline.py           # Main entry point to run the full pipeline
```

## Updated Pipeline Flow

### 1. Data Loading and Integration
```python
from src.data_loader_updated import DataLoader

# Load all data sources
loader = DataLoader(data_dir='data/raw')
merged_df = loader.load_all_data()
```

### 2. Target Variable Computation
```python
from src.target_updated import process_target_variable

# Compute CCR and remove nutrient columns to prevent leakage
df_with_target = process_target_variable(merged_df, remove_nutrients=True)
```

### 3. Feature Engineering
```python
from src.feature_engineering_updated import FeatureEngineer

# Engineer comprehensive multimodal features
feature_engineer = FeatureEngineer()
feature_df = feature_engineer.engineer_features(df_with_target)
```

### 4. Model Training and Evaluation
```python
from src.models import ModelTrainer
from src.evaluation import evaluate_models

# Train multiple model types
trainer = ModelTrainer()
models = trainer.train_all_models(feature_df, target_col='CCR')

# Evaluate performance
results = evaluate_models(models, test_data)
```

## Key Data Insights Discovered

1. **Time-Series Nature**: Each participant has multiple timestamped measurements
2. **Data Sparsity**: Not all time points contain meal information
3. **Multimodal Fusion**: Requires combining time-series data with participant-level demographics
4. **Microbiome Complexity**: Thousands of bacterial species with sparse representation
5. **Participant Variability**: Significant differences in data availability per participant

## Quick Start

1. **Explore the Data**:
   ```bash
   jupyter notebook notebooks/01_data_exploration_updated.ipynb
   ```

2. **Run the Complete Pipeline**:
   ```bash
   python run_pipeline.py
   ```

3. **Use Updated Modules**:
   ```python
   # Load data with actual structure
   from src.data_loader_updated import DataLoader
   loader = DataLoader()
   data = loader.load_all_data()
   
   # Engineer features for actual data
   from src.feature_engineering_updated import FeatureEngineer
   fe = FeatureEngineer()
   features = fe.engineer_features(data)
   ```

## Data Processing Recommendations

Based on our analysis of the actual dataset:

1. **Meal-Level Aggregation**: Consider aggregating time-series data to meal-level predictions
2. **Participant-Aware CV**: Use participant-aware cross-validation to prevent data leakage
3. **Feature Selection**: Focus on most important microbiome and gut health features
4. **Missing Data**: Handle glucose sensor gaps with interpolation
5. **Temporal Features**: Extract time-of-day and meal timing patterns

## Important Notes

- **Updated Modules**: Use `*_updated.py` files which handle the actual dataset structure
- **Data Leakage Prevention**: Nutrient columns are removed after CCR computation
- **Participant IDs**: Range from 1-49 with some gaps (44 total participants)
- **Microbiome Sparsity**: Many bacterial species have zero values for most participants
- **Time-Series Alignment**: Timestamps may not be perfectly aligned across modalities
- **Data Quality Assessment**: Missing data patterns, outlier detection
- **Distributional Analysis**: Target variable and feature distributions
- **Correlation Analysis**: Inter-feature relationships and target correlations
- **Temporal Patterns**: Time-series characteristics in CGM and activity data

### 2. Feature Engineering
- **Glucose Response Features**: 
  - Peak glucose levels, time to peak, area under curve (AUC)
  - Glucose variability metrics, incremental AUC
  - Pre-meal and post-meal glucose trends
- **Activity Features**:
  - Heart rate variability, step patterns, sleep quality metrics
  - Activity intensity distributions, sedentary behavior patterns
- **Demographics & Labs**:
  - Age, BMI, metabolic markers, clinical indicators
  - Derived health status indicators
- **Microbiome Features**:
  - Principal Component Analysis (PCA) of microbiome composition
  - Alpha and beta diversity metrics
  - Key bacterial taxa abundance
- **Gut Health Integration**:
  - Gut health scores and derived indicators
- **Meal Metadata**:
  - Meal timing, frequency, size indicators
  - Temporal meal patterns

### 3. Model Development

#### Baseline Models
- **Linear Regression**: Simple linear relationship modeling
- **Ridge Regression**: L2 regularization for feature selection
- **Lasso Regression**: L1 regularization for sparse feature selection

#### Intermediate Models
- **Random Forest**: Ensemble method for non-linear relationships
- **XGBoost**: Gradient boosting for complex feature interactions

#### Advanced Models
- **LightGBM**: Efficient gradient boosting implementation
- **Temporal Models**: Time-series aware architectures
- **Neural Networks**: Deep learning for complex pattern recognition (optional)

### 4. Evaluation Methodology
- **Primary Metric**: Normalized Root Mean Square Error (NRMSE)
- **Secondary Metric**: Pearson correlation coefficient
- **Cross-Validation**: Time-series aware validation splits
- **Feature Importance Analysis**: Model-specific feature ranking

## Requirements

- Python 3.10+
- See `requirements.txt` for complete dependency list

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/EswarMachara/IEEE_BHI_25_CGMacro.git
cd IEEE_BHI_25_CGMacro
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your data files in the `data/raw/` directory

### Running the Pipeline

Execute the complete machine learning pipeline:

```bash
python run_pipeline.py
```

This will:
1. Load and preprocess the raw data
2. Perform feature engineering
3. Train all model variants
4. Generate evaluation metrics and visualizations
5. Save results to the `results/` directory

### Using Individual Components

You can also run individual pipeline components:

```python
# Load data
from src.data_loader import load_cgmacros_data
data = load_cgmacros_data("data/raw/")

# Compute target variable
from src.target import compute_ccr
data_with_target = compute_ccr(data)

# Feature engineering
from src.feature_engineering import engineer_features
features = engineer_features(data_with_target)

# Train models
from src.models import train_all_models
models = train_all_models(features)
```

## Deliverables

1. **Complete Codebase**: Modular, well-documented Python implementation
2. **Results and Visualizations**: Model performance metrics, feature importance plots, error analysis
3. **8-Page Technical Report**: Comprehensive methodology and results documentation
4. **Reproducible Pipeline**: Automated workflow for result reproduction

## Evaluation Metrics

- **Normalized RMSE**: Primary evaluation metric
- **Pearson Correlation**: Secondary evaluation metric
- **Feature Importance Rankings**: Model interpretability analysis
- **Cross-Validation Scores**: Robust performance assessment

## Contributing

1. Follow PEP 8 style guidelines
2. Include comprehensive docstrings for all functions
3. Add unit tests for new functionality
4. Update documentation for significant changes

## License

This project is developed for the IEEE BHI 2025 Challenge and follows academic use guidelines.

## Contact

For questions or collaboration, please contact [Your Contact Information].

---

**Note**: This repository is part of the IEEE BHI 2025 Track 2 Challenge submission. All code and methodologies are developed in compliance with challenge guidelines and ethical AI practices.