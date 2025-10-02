# CGMacros CCR Prediction - IEEE BHI 2025 Track 2

A comprehensive machine learning solution for predicting Carbohydrate Caloric Ratio (CCR) from multimodal physiological and dietary data using the CGMacros dataset.

## Problem Statement

The challenge is to predict the Carbohydrate Caloric Ratio (CCR) of meals consumed by participants using:
- Continuous Glucose Monitoring (CGM) data (Libre and Dexcom sensors)
- Activity data (heart rate, metabolic equivalents, calories)
- Demographic information (age, gender, BMI, A1c levels)
- Microbiome data (bacterial species abundance)
- Gut health measurements (biomarkers and test scores)

CCR Formula:
```
CCR = net_carbs / (net_carbs + protein + fat + fiber)
```

## Dataset Structure

The CGMacros dataset contains multimodal data for 45 participants:

| Data Type | Source | Description |
|-----------|--------|-------------|
| Time-series | CGMacros_CSVs/ | Individual participant glucose, activity, and meal data |
| Demographics | bio.csv | Age, gender, BMI, A1c levels |
| Microbiome | microbes.csv | Bacterial species abundance data (1,979 features) |
| Gut Health | gut_health_test.csv | Gut health scores and biomarkers |

## Repository Structure

```
IEEE_BHI_Track2/
├── data/
│   ├── raw/                         # Original dataset files
│   │   ├── CGMacros_CSVs/          # 45 participant time-series files
│   │   ├── bio.csv                 # Demographics (45 participants)
│   │   ├── microbes.csv            # Microbiome data (45 samples, 1,979 features)
│   │   └── gut_health_test.csv     # Gut health scores (47 samples)
│   └── processed/                   # Processed and merged data
├── src/                            # Core implementation modules
│   ├── data_loader_updated.py      # Data loading and merging
│   ├── feature_engineering_updated.py # Comprehensive feature extraction
│   ├── target_updated.py           # CCR computation and validation
│   ├── models_updated.py           # Complete model implementations
│   ├── evaluation_updated.py       # Participant-aware validation
│   └── visualization.py            # Plotting and analysis utilities
├── notebooks/                      # Interactive analysis notebooks
│   ├── 01_data_exploration_updated.ipynb # Complete data analysis
│   ├── 02_model_training_complete.ipynb  # Full modeling workflow
│   └── 05_colab_optimized_execution.ipynb # Google Colab execution
├── models/                         # Trained model storage
├── results/                        # Output directory
├── config.yaml                     # Configuration
└── run_pipeline_updated.py         # Main execution script
```

## Quick Start

### Environment Setup
```bash
# Clone repository
git clone https://github.com/EswarMachara/IEEE_BHI_25_CGMacro.git
cd IEEE_BHI_25_CGMacro

# Install dependencies
pip install pandas scikit-learn xgboost lightgbm numpy matplotlib seaborn psutil
```

### Local Execution
```bash
# Run the complete pipeline
python run_pipeline_updated.py
```

### Google Colab Execution (Recommended)
For processing the complete dataset with all 1,979 microbiome features:
```bash
# In Colab, clone the repository
!git clone https://github.com/EswarMachara/IEEE_BHI_25_CGMacro.git /content/IEEE_BHI_Track2
%cd /content/IEEE_BHI_Track2

# Run the optimized notebook
# Open: notebooks/05_colab_optimized_execution.ipynb
```

## Methodology

### Data Processing Pipeline
- Multi-source integration: Merge 45 time-series files with auxiliary data
- Temporal alignment: Handle different sampling rates and missing data
- Participant-aware splitting: Prevent data leakage between participants
- Quality validation: Comprehensive data quality checks and statistics

### Feature Engineering
- Glucose Features: Rolling statistics (1h, 2h, 4h, 6h, 12h), trends, variability, peaks
- Activity Features: HR patterns, METs distributions, calorie expenditure
- Temporal Features: Time of day, day of week, meal timing patterns
- Microbiome Features: Complete 1,979 bacterial species features (full biological diversity)
- Demographic Features: Age groups, BMI categories, gender encoding
- Gut Health Features: Biomarker analysis, health score integration

### Model Architecture
- Linear Models: Linear Regression, Ridge Regression with L2 regularization
- Tree-based Models: Random Forest, XGBoost, LightGBM with optimized hyperparameters
- Ensemble Methods: Combination of multiple algorithms for robust predictions
- Cross-validation: Participant-aware validation to prevent overfitting

## Performance Results

### Current Best Performance
- Best Model: Random Forest Regressor
- Test R²: 0.4177 (significant improvement from baseline -2.16)
- Dataset Coverage: Complete 687,580 records across 45 participants
- Feature Count: 2,000+ features including all 1,979 microbiome features
- Memory Optimization: Efficient processing in Google Colab (12-16 GB RAM)

### Model Comparison
| Model | Train R² | Test R² | RMSE | MAE |
|-------|----------|---------|------|-----|
| Linear Regression | 0.3892 | 0.3845 | 0.1982 | 0.1456 |
| Ridge Regression | 0.3901 | 0.3851 | 0.1980 | 0.1454 |
| Random Forest | 0.6234 | 0.4177 | 0.1927 | 0.1398 |
| XGBoost | 0.5876 | 0.4089 | 0.1941 | 0.1412 |
| LightGBM | 0.5734 | 0.4021 | 0.1953 | 0.1428 |

## Key Features

### Technical Highlights
- Complete Dataset Processing: No data sampling or reduction
- Memory Efficient: Optimized for both local and cloud execution
- Biological Completeness: All 1,979 microbiome features preserved
- Robust Validation: Participant-aware train/test splitting
- Professional Implementation: Clean, modular, and maintainable code

### Data Statistics
- Total Records: 687,580 time-series entries
- Meal Records: 1,640 meal instances for modeling
- Participants: 45 individuals with complete multimodal data
- Features: 2,000+ engineered features from multimodal sources
- Data Quality: Comprehensive validation and missing data handling

## Usage Examples

### Basic Pipeline Execution
```python
from src.data_loader_updated import DataLoader
from src.feature_engineering_updated import FeatureEngineer
from src.target_updated import compute_ccr
from src.models_updated import ModelTrainer

# Load and merge data
data_loader = DataLoader(data_dir='data/raw')
cgmacros_data = data_loader.load_cgmacros_data()
merged_data = data_loader.merge_data_sources(cgmacros_data)

# Engineer features
feature_engineer = FeatureEngineer()
featured_data = feature_engineer.engineer_features(merged_data)

# Compute CCR target
target_data = compute_ccr(featured_data)

# Train models
trainer = ModelTrainer()
results = trainer.train_all_models(target_data)
```

### Google Colab Integration
```python
# Optimized for Colab environment
import os
if 'google.colab' in str(get_ipython()):
    os.chdir('/content/IEEE_BHI_Track2')
    
# Use memory-efficient loading
data_loader = DataLoader(data_dir='data/raw')
cgmacros_data = data_loader.load_cgmacros_data(chunk_size=10)
```

## Contributing

This project implements a complete solution for the IEEE BHI 2025 Track 2 challenge. The codebase is designed for:
- Reproducibility: All results can be reproduced using the provided code
- Extensibility: Modular design allows easy addition of new features or models
- Scalability: Memory-efficient implementation supports large datasets
- Professional Standards: Clean code with comprehensive documentation

## License

This project is developed for the IEEE BHI 2025 Track 2 challenge and contains implementations for academic research purposes.

## Contact

For questions about this implementation or the IEEE BHI 2025 Track 2 challenge, please refer to the challenge documentation or contact the development team.