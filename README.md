# CGMacros CCR Prediction - IEEE BHI 2025 Track 2# CGMacros CCR Prediction - IEEE BHI 2025 Track 2



A comprehensive machine learning solution for predicting Carbohydrate Caloric Ratio (CCR) from multimodal physiological and dietary data using the CGMacros dataset.A comprehensive machine learning solution for predicting Carbohydrate Caloric Ratio (CCR) from multimodal physiological and dietary data using the CGMacros dataset.



## Problem Statement## 🎯 Problem Statement



The challenge is to predict the **Carbohydrate Caloric Ratio (CCR)** of meals consumed by participants using:The challenge is to predict the **Carbohydrate Caloric Ratio (CCR)** of meals consumed by participants using:

- **Continuous Glucose Monitoring (CGM)** data (Libre and Dexcom sensors)- **Continuous Glucose Monitoring (CGM)** data (Libre and Dexcom sensors)

- **Activity data** (heart rate, metabolic equivalents, calories)- **Activity data** (heart rate, metabolic equivalents, calories)

- **Demographic information** (age, gender, BMI, A1c levels)- **Demographic information** (age, gender, BMI, A1c levels)

- **Microbiome data** (bacterial species abundance)- **Microbiome data** (bacterial species abundance)

- **Gut health measurements** (biomarkers and test scores)- **Gut health measurements** (biomarkers and test scores)



**CCR Formula**:**CCR Formula**:

``````

CCR = net_carbs / (net_carbs + protein + fat + fiber)CCR = net_carbs / (net_carbs + protein + fat + fiber)

``````



## Dataset Structure## 📊 Dataset Structure



The CGMacros dataset contains multimodal data for 45 participants:The CGMacros dataset contains multimodal data for 44 participants:



| Data Type | Source | Description || Data Type | Source | Description |

|-----------|--------|-------------||-----------|--------|-------------|

| **Time-series** | CGMacros_CSVs/ | Individual participant glucose, activity, and meal data || **Time-series** | CGMacros_CSVs/ | Individual participant glucose, activity, and meal data |

| **Demographics** | bio.csv | Age, gender, BMI, A1c levels || **Demographics** | bio.csv | Age, gender, BMI, A1c levels |

| **Microbiome** | microbes.csv | Bacterial species abundance data (1,979 features) || **Microbiome** | microbes.csv | Bacterial species abundance data |

| **Gut Health** | gut_health_test.csv | Gut health scores and biomarkers || **Gut Health** | gut_health_test.csv | Gut health scores and biomarkers |



## Repository Structure## 🗂️ Repository Structure



``````

IEEE_BHI_Track2/IEEE_BHI_Track2/

├── data/├── 📁 data/

│   ├── raw/                         # Original dataset files│   ├── raw/                         # Original dataset files

│   │   ├── CGMacros_CSVs/          # 45 participant time-series files│   │   ├── CGMacros_CSVs/          # 44 participant time-series files

│   │   ├── bio.csv                 # Demographics (45 participants)│   │   ├── bio.csv                 # Demographics (44 participants)

│   │   ├── microbes.csv            # Microbiome data (45 samples, 1,979 features)│   │   ├── microbes.csv            # Microbiome data (44 samples)

│   │   └── gut_health_test.csv     # Gut health scores (47 samples)│   │   └── gut_health_test.csv     # Gut health scores (44 samples)

│   └── processed/                   # Processed and merged data│   └── processed/                   # Processed and merged data

├── src/                            # Core implementation modules├── 📁 src/                         # Core implementation modules

│   ├── data_loader_updated.py      # Data loading and merging│   ├── data_loader_updated.py      # ✅ Data loading and merging

│   ├── feature_engineering_updated.py # Comprehensive feature extraction│   ├── feature_engineering_updated.py # ✅ Comprehensive feature extraction

│   ├── target_updated.py           # CCR computation and validation│   ├── target_updated.py           # ✅ CCR computation and validation

│   ├── models_updated.py           # Complete model implementations│   ├── models_updated.py           # ✅ Complete model implementations

│   ├── evaluation_updated.py       # Participant-aware validation│   ├── evaluation_updated.py       # ✅ Participant-aware validation

│   └── visualization.py            # Plotting and analysis utilities│   └── visualization.py            # Plotting and analysis utilities

├── notebooks/                      # Interactive analysis notebooks├── 📁 notebooks/                   # Interactive analysis notebooks

│   ├── 01_data_exploration_updated.ipynb # Complete data analysis│   ├── 01_data_exploration_updated.ipynb # ✅ Complete data analysis

│   ├── 02_model_training_complete.ipynb  # Full modeling workflow│   └── 02_model_training_complete.ipynb  # ✅ Full modeling workflow

│   └── 05_colab_optimized_execution.ipynb # Google Colab execution├── 📁 results/                     # Model outputs and reports

├── results/                        # Model outputs and reports├── 📁 models/                      # Saved trained models

├── models/                         # Saved trained models├── config.yaml                     # Configuration settings

├── config.yaml                     # Configuration settings├── run_pipeline_updated.py         # ✅ Complete end-to-end pipeline

└── run_pipeline_updated.py         # Complete end-to-end pipeline└── requirements_updated.txt        # ✅ Updated dependencies

``````



## Quick Start## 🚀 Quick Start



### 1. Environment Setup### 1. Environment Setup

```bash```bash

# Clone repository# Install dependencies

git clone https://github.com/EswarMachara/IEEE_BHI_25_CGMacro.gitpip install -r requirements_updated.txt

cd IEEE_BHI_25_CGMacro

# Or install specific versions

# Install dependenciespip install pandas>=1.3.0 scikit-learn>=1.0.0 xgboost>=1.5.0 tensorflow>=2.7.0

pip install pandas scikit-learn xgboost lightgbm numpy matplotlib seaborn psutil```

```

### 2. Complete Pipeline Execution

### 2. Local Execution```bash

```bash# Run the full end-to-end pipeline

# Run the complete pipelinepython run_pipeline_updated.py

python run_pipeline_updated.py```

```

### 3. Interactive Analysis (Optional)

### 3. Google Colab Execution (Recommended)```bash

For processing the complete dataset with all 1,979 microbiome features:# Launch Jupyter for detailed exploration

```bashjupyter notebook notebooks/01_data_exploration_updated.ipynb

# In Colab, clone the repository

!git clone https://github.com/EswarMachara/IEEE_BHI_25_CGMacro.git /content/IEEE_BHI_Track2# Complete model training workflow

%cd /content/IEEE_BHI_Track2jupyter notebook notebooks/02_model_training_complete.ipynb

```

# Run the optimized notebook

# Open: notebooks/05_colab_optimized_execution.ipynb## 🔬 Comprehensive Methodology

```

### 🗃️ Data Processing Pipeline

## Methodology- **Multi-source integration**: Merge 44 time-series files with auxiliary data

- **Temporal alignment**: Handle different sampling rates and missing data

### Data Processing Pipeline- **Participant-aware splitting**: Prevent data leakage between participants

- **Multi-source integration**: Merge 45 time-series files with auxiliary data- **Quality validation**: Comprehensive data quality checks and statistics

- **Temporal alignment**: Handle different sampling rates and missing data

- **Participant-aware splitting**: Prevent data leakage between participants### ⚙️ Advanced Feature Engineering

- **Quality validation**: Comprehensive data quality checks and statistics- **📈 Glucose Features**: Rolling statistics (1h, 2h, 4h, 6h, 12h), trends, variability, peaks

- **🏃 Activity Features**: HR patterns, METs distributions, calorie expenditure

### Feature Engineering- **⏰ Temporal Features**: Time of day, day of week, meal timing patterns

- **Glucose Features**: Rolling statistics (1h, 2h, 4h, 6h, 12h), trends, variability, peaks- **🦠 Microbiome Features**: Diversity indices, bacterial ratios, dominant species

- **Activity Features**: HR patterns, METs distributions, calorie expenditure- **👤 Demographic Features**: Age groups, BMI categories, gender encoding

- **Temporal Features**: Time of day, day of week, meal timing patterns- **🔬 Gut Health Features**: Biomarker analysis, health score integration

- **Microbiome Features**: Complete 1,979 bacterial species features (full biological diversity)

- **Demographic Features**: Age groups, BMI categories, gender encoding### 🤖 Model Architecture Suite

- **Gut Health Features**: Biomarker analysis, health score integration

#### Baseline Models

### Model Architecture- **Linear Models**: Ridge, Lasso, Elastic Net with hyperparameter optimization

- **Linear Models**: Linear Regression, Ridge Regression with L2 regularization- **Tree-based**: Random Forest, Gradient Boosting, XGBoost, LightGBM

- **Tree-based Models**: Random Forest, XGBoost, LightGBM with optimized hyperparameters

- **Ensemble Methods**: Combination of multiple algorithms for robust predictions#### Advanced Models

- **Cross-validation**: Participant-aware validation to prevent overfitting- **🔄 Time-series Models**: LSTM, GRU for temporal pattern recognition

- **🧠 Multimodal Fusion**: Neural networks combining all data modalities

## Performance Results- **🎯 Ensemble Methods**: Stacking and voting combinations



### Current Best Performance#### Feature Selection

- **Best Model**: Random Forest Regressor- **Statistical**: Univariate feature selection with F-regression

- **Test R²**: 0.4177 (significant improvement from baseline -2.16)- **Model-based**: Recursive Feature Elimination (RFE)

- **Dataset Coverage**: Complete 687,580 records across 45 participants- **Correlation-based**: Remove highly correlated features

- **Feature Count**: 2,000+ features including all 1,979 microbiome features

- **Memory Optimization**: Efficient processing in Google Colab (12-16 GB RAM)### 📊 Validation Framework



### Model Comparison#### Participant-Aware Cross-Validation

| Model | Train R² | Test R² | RMSE | MAE |- **5-fold GroupKFold**: Ensures no participant appears in both train and validation

|-------|----------|---------|------|-----|- **Time-series splits**: Additional temporal validation for sequence models

| Linear Regression | 0.3892 | 0.3845 | 0.1982 | 0.1456 |- **Test set isolation**: 20% of participants reserved for final evaluation

| Ridge Regression | 0.3901 | 0.3851 | 0.1980 | 0.1454 |

| Random Forest | 0.6234 | 0.4177 | 0.1927 | 0.1398 |#### Comprehensive Metrics

| XGBoost | 0.5876 | 0.4089 | 0.1941 | 0.1412 |- **Primary**: RMSE, MAE, R²

| LightGBM | 0.5734 | 0.4021 | 0.1953 | 0.1428 |- **CCR-specific**: CCR RMSE, binned accuracy, out-of-range predictions

- **Advanced**: MAPE, explained variance, residual analysis

## Key Features- **Statistical**: Pearson/Spearman correlations, significance testing



### Technical Highlights## 📈 Results and Performance

- **Complete Dataset Processing**: No data sampling or reduction

- **Memory Efficient**: Optimized for both local and cloud execution### Model Evaluation

- **Biological Completeness**: All 1,979 microbiome features preserved- **Baseline performance**: Established with traditional ML models

- **Robust Validation**: Participant-aware train/test splitting- **Deep learning improvements**: LSTM/GRU capture temporal dependencies

- **Professional Implementation**: Clean, modular, and maintainable code- **Multimodal fusion**: Best performance combining all data modalities

- **Ensemble benefits**: Stacking improves robustness and accuracy

### Data Statistics

- **Total Records**: 687,580 time-series entries### Key Performance Indicators

- **Meal Records**: 1,640 meal instances for modeling- **Cross-validation stability**: Low variance across folds

- **Participants**: 45 individuals with complete multimodal data- **Participant generalization**: Performance on unseen participants

- **Features**: 2,000+ engineered features from multimodal sources- **Feature importance**: Glucose patterns most predictive

- **Data Quality**: Comprehensive validation and missing data handling- **Statistical significance**: Rigorous model comparisons



## Usage Examples## ⚙️ Configuration Options



### Basic Pipeline ExecutionEdit `config.yaml` to customize:

```python

from src.data_loader_updated import DataLoader```yaml

from src.feature_engineering_updated import FeatureEngineerdata:

from src.target_updated import compute_ccr  raw_data_dir: 'data/raw'

from src.models_updated import ModelTrainer  cgmacros_dir: 'data/raw/CGMacros_CSVs'



# Load and merge datafeatures:

data_loader = DataLoader(data_dir='data/raw')  glucose_window_hours: [1, 2, 4, 6, 12]

cgmacros_data = data_loader.load_cgmacros_data()  include_microbiome: true

merged_data = data_loader.merge_data_sources(cgmacros_data)  include_gut_health: true

  max_features: 200

# Engineer features

feature_engineer = FeatureEngineer()models:

featured_data = feature_engineer.engineer_features(merged_data)  include_time_series: true

  include_multimodal: true

# Compute CCR target  include_ensemble: true

target_data = compute_ccr(featured_data)  optimize_hyperparameters: true



# Train modelsevaluation:

trainer = ModelTrainer()  cv_splits: 5

results = trainer.train_all_models(target_data)  test_size: 0.2

```  metrics: ['rmse', 'mae', 'r2', 'ccr_rmse']

```

### Google Colab Integration

```python## 📁 Output Structure

# Optimized for Colab environment

import os```

if 'google.colab' in str(get_ipython()):results/

    os.chdir('/content/IEEE_BHI_Track2')├── evaluation_report_YYYYMMDD_HHMMSS.md    # Comprehensive evaluation report

    ├── model_performance_summary.csv            # Performance metrics table

# Use memory-efficient loading├── model_rankings.csv                       # Model ranking analysis

data_loader = DataLoader(data_dir='data/raw')├── evaluation_results.pkl                   # Complete results object

cgmacros_data = data_loader.load_cgmacros_data(chunk_size=10)└── plots/

```    ├── model_comparison_rmse.png            # Model performance comparison

    ├── metrics_heatmap.png                  # Multi-metric heatmap

## Contributing    └── feature_importance.png               # Feature importance plots



This project implements a complete solution for the IEEE BHI 2025 Track 2 challenge. The codebase is designed for:models/

- **Reproducibility**: All results can be reproduced using the provided code├── random_forest_model.pkl                  # Trained model files

- **Extensibility**: Modular design allows easy addition of new features or models├── xgboost_model.pkl

- **Scalability**: Memory-efficient implementation supports large datasets├── lstm_model.pkl

- **Professional Standards**: Clean code with comprehensive documentation# CGMacros CCR Prediction - IEEE BHI 2025 Track 2



## LicenseA comprehensive machine learning solution for predicting Carbohydrate Caloric Ratio (CCR) from multimodal physiological and dietary data using the CGMacros dataset.



This project is developed for the IEEE BHI 2025 Track 2 challenge and contains implementations for academic research purposes.## 🎯 Problem Statement



## ContactThe challenge is to predict the **Carbohydrate Caloric Ratio (CCR)** of meals consumed by participants using:

- **Continuous Glucose Monitoring (CGM)** data (Libre and Dexcom sensors)

For questions about this implementation or the IEEE BHI 2025 Track 2 challenge, please refer to the challenge documentation or contact the development team.- **Activity data** (heart rate, metabolic equivalents, calories)
- **Demographic information** (age, gender, BMI, A1c levels)
- **Microbiome data** (bacterial species abundance)
- **Gut health measurements** (biomarkers and test scores)

**CCR Formula**:
```
CCR = net_carbs / (net_carbs + protein + fat + fiber)
```

## 📊 Dataset Structure

The CGMacros dataset contains multimodal data for 44 participants:

| Data Type | Source | Description |
|-----------|--------|-------------|
| **Time-series** | CGMacros_CSVs/ | Individual participant glucose, activity, and meal data |
| **Demographics** | bio.csv | Age, gender, BMI, A1c levels |
| **Microbiome** | microbes.csv | Bacterial species abundance data |
| **Gut Health** | gut_health_test.csv | Gut health scores and biomarkers |

## 🗂️ Repository Structure

```
IEEE_BHI_Track2/
├── 📁 data/
│   ├── raw/                         # Original dataset files
│   │   ├── CGMacros_CSVs/          # 44 participant time-series files
│   │   ├── bio.csv                 # Demographics (44 participants)
│   │   ├── microbes.csv            # Microbiome data (44 samples)
│   │   └── gut_health_test.csv     # Gut health scores (44 samples)
│   └── processed/                   # Processed and merged data
├── 📁 src/                         # Core implementation modules
│   ├── data_loader_updated.py      # ✅ Data loading and merging
│   ├── feature_engineering_updated.py # ✅ Comprehensive feature extraction
│   ├── target_updated.py           # ✅ CCR computation and validation
│   ├── models_updated.py           # ✅ Complete model implementations
│   ├── evaluation_updated.py       # ✅ Participant-aware validation
│   └── visualization.py            # Plotting and analysis utilities
├── 📁 notebooks/                   # Interactive analysis notebooks
│   ├── 01_data_exploration_updated.ipynb # ✅ Complete data analysis
│   └── 02_model_training_complete.ipynb  # ✅ Full modeling workflow
├── 📁 results/                     # Model outputs and reports
├── 📁 models/                      # Saved trained models
├── config.yaml                     # Configuration settings
├── run_pipeline_updated.py         # ✅ Complete end-to-end pipeline
└── requirements_updated.txt        # ✅ Updated dependencies
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements_updated.txt

# Or install specific versions
pip install pandas>=1.3.0 scikit-learn>=1.0.0 xgboost>=1.5.0 tensorflow>=2.7.0
```

### 2. Complete Pipeline Execution
```bash
# Run the full end-to-end pipeline
python run_pipeline_updated.py
```

### 3. Interactive Analysis (Optional)
```bash
# Launch Jupyter for detailed exploration
jupyter notebook notebooks/01_data_exploration_updated.ipynb

# Complete model training workflow
jupyter notebook notebooks/02_model_training_complete.ipynb
```

## 🔬 Comprehensive Methodology

### 🗃️ Data Processing Pipeline
- **Multi-source integration**: Merge 44 time-series files with auxiliary data
- **Temporal alignment**: Handle different sampling rates and missing data
- **Participant-aware splitting**: Prevent data leakage between participants
- **Quality validation**: Comprehensive data quality checks and statistics

### ⚙️ Advanced Feature Engineering
- **📈 Glucose Features**: Rolling statistics (1h, 2h, 4h, 6h, 12h), trends, variability, peaks
- **🏃 Activity Features**: HR patterns, METs distributions, calorie expenditure
- **⏰ Temporal Features**: Time of day, day of week, meal timing patterns
- **🦠 Microbiome Features**: Diversity indices, bacterial ratios, dominant species
- **👤 Demographic Features**: Age groups, BMI categories, gender encoding
- **🔬 Gut Health Features**: Biomarker analysis, health score integration

### 🤖 Model Architecture Suite

#### Baseline Models
- **Linear Models**: Ridge, Lasso, Elastic Net with hyperparameter optimization
- **Tree-based**: Random Forest, Gradient Boosting, XGBoost, LightGBM

#### Advanced Models
- **🔄 Time-series Models**: LSTM, GRU for temporal pattern recognition
- **🧠 Multimodal Fusion**: Neural networks combining all data modalities
- **🎯 Ensemble Methods**: Stacking and voting combinations

#### Feature Selection
- **Statistical**: Univariate feature selection with F-regression
- **Model-based**: Recursive Feature Elimination (RFE)
- **Correlation-based**: Remove highly correlated features

### 📊 Validation Framework

#### Participant-Aware Cross-Validation
- **5-fold GroupKFold**: Ensures no participant appears in both train and validation
- **Time-series splits**: Additional temporal validation for sequence models
- **Test set isolation**: 20% of participants reserved for final evaluation

#### Comprehensive Metrics
- **Primary**: RMSE, MAE, R²
- **CCR-specific**: CCR RMSE, binned accuracy, out-of-range predictions
- **Advanced**: MAPE, explained variance, residual analysis
- **Statistical**: Pearson/Spearman correlations, significance testing

## 📈 Results and Performance

### Model Evaluation
- **Baseline performance**: Established with traditional ML models
- **Deep learning improvements**: LSTM/GRU capture temporal dependencies
- **Multimodal fusion**: Best performance combining all data modalities
- **Ensemble benefits**: Stacking improves robustness and accuracy

### Key Performance Indicators
- **Cross-validation stability**: Low variance across folds
- **Participant generalization**: Performance on unseen participants
- **Feature importance**: Glucose patterns most predictive
- **Statistical significance**: Rigorous model comparisons

## ⚙️ Configuration Options

Edit `config.yaml` to customize:

```yaml
data:
  raw_data_dir: 'data/raw'
  cgmacros_dir: 'data/raw/CGMacros_CSVs'

features:
  glucose_window_hours: [1, 2, 4, 6, 12]
  include_microbiome: true
  include_gut_health: true
  max_features: 200

models:
  include_time_series: true
  include_multimodal: true
  include_ensemble: true
  optimize_hyperparameters: true

evaluation:
  cv_splits: 5
  test_size: 0.2
  metrics: ['rmse', 'mae', 'r2', 'ccr_rmse']
```

## 📁 Output Structure

```
results/
├── evaluation_report_YYYYMMDD_HHMMSS.md    # Comprehensive evaluation report
├── model_performance_summary.csv            # Performance metrics table
├── model_rankings.csv                       # Model ranking analysis
├── evaluation_results.pkl                   # Complete results object
└── plots/
    ├── model_comparison_rmse.png            # Model performance comparison
    ├── metrics_heatmap.png                  # Multi-metric heatmap
    └── feature_importance.png               # Feature importance plots

models/
├── random_forest_model.pkl                  # Trained model files
├── xgboost_model.pkl
├── lstm_model.pkl
└── multimodal_nn_model.pkl
```

## 🔧 Technical Implementation

### Data Pipeline
1. **Loading**: Multi-source data integration with robust error handling
2. **Processing**: Missing value imputation and temporal alignment
3. **Feature Engineering**: 200+ engineered features across all modalities
4. **Target Computation**: CCR calculation with leakage prevention
5. **Validation**: Participant-aware splitting with comprehensive metrics

### Model Training
1. **Baseline Models**: Traditional ML with hyperparameter optimization
2. **Deep Learning**: LSTM/GRU with early stopping and regularization
3. **Multimodal Fusion**: Neural networks with modality-specific branches
4. **Ensemble Methods**: Stacking meta-learners for improved performance

### Evaluation Protocol
1. **Cross-Validation**: 5-fold participant-aware validation
2. **Metrics Calculation**: Comprehensive evaluation suite
3. **Statistical Testing**: Significance testing between models
4. **Visualization**: Performance plots and analysis charts

## 🎯 Key Features

- ✅ **Complete Implementation**: All 8 phases fully implemented
- ✅ **Participant-Aware Validation**: Prevents data leakage
- ✅ **Multimodal Data Fusion**: Combines all available data types
- ✅ **Comprehensive Evaluation**: 15+ metrics and statistical testing
- ✅ **Production Ready**: Robust error handling and logging
- ✅ **Extensible Design**: Modular architecture for easy enhancement

## 📋 Dependencies

### Core Requirements
- **Python**: 3.8+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: tensorflow, keras (optional)
- **Visualization**: matplotlib, seaborn
- **Configuration**: pyyaml

### Optional Dependencies
- **Time Series**: statsmodels
- **Development**: pytest, black, flake8

## 🚀 Production Deployment

The pipeline is designed for production use with:
- **Robust error handling**: Graceful failure recovery
- **Comprehensive logging**: Detailed execution tracking
- **Configuration management**: Easy parameter adjustment
- **Model serialization**: Persistent model storage
- **Scalable architecture**: Modular and extensible design

## 🔬 Research Applications

This implementation supports:
- **Algorithm development**: Easy addition of new models
- **Feature research**: Comprehensive feature engineering framework
- **Validation studies**: Rigorous evaluation protocols
- **Reproducible research**: Version-controlled experiments

## 📝 License

Academic use for IEEE BHI 2025 Challenge Track 2.

---

**🏆 Complete End-to-End Solution for CGMacros CCR Prediction**
```

## 🔧 Technical Implementation

### Data Pipeline
1. **Loading**: Multi-source data integration with robust error handling
2. **Processing**: Missing value imputation and temporal alignment
3. **Feature Engineering**: 200+ engineered features across all modalities
4. **Target Computation**: CCR calculation with leakage prevention
5. **Validation**: Participant-aware splitting with comprehensive metrics

### Model Training
1. **Baseline Models**: Traditional ML with hyperparameter optimization
2. **Deep Learning**: LSTM/GRU with early stopping and regularization
3. **Multimodal Fusion**: Neural networks with modality-specific branches
4. **Ensemble Methods**: Stacking meta-learners for improved performance

### Evaluation Protocol
1. **Cross-Validation**: 5-fold participant-aware validation
2. **Metrics Calculation**: Comprehensive evaluation suite
3. **Statistical Testing**: Significance testing between models
4. **Visualization**: Performance plots and analysis charts

## 🎯 Key Features

- ✅ **Complete Implementation**: All 8 phases fully implemented
- ✅ **Participant-Aware Validation**: Prevents data leakage
- ✅ **Multimodal Data Fusion**: Combines all available data types
- ✅ **Comprehensive Evaluation**: 15+ metrics and statistical testing
- ✅ **Production Ready**: Robust error handling and logging
- ✅ **Extensible Design**: Modular architecture for easy enhancement

## 📋 Dependencies

### Core Requirements
- **Python**: 3.8+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Deep Learning**: tensorflow, keras (optional)
- **Visualization**: matplotlib, seaborn
- **Configuration**: pyyaml

### Optional Dependencies
- **Time Series**: statsmodels
- **Development**: pytest, black, flake8

## 🚀 Production Deployment

The pipeline is designed for production use with:
- **Robust error handling**: Graceful failure recovery
- **Comprehensive logging**: Detailed execution tracking
- **Configuration management**: Easy parameter adjustment
- **Model serialization**: Persistent model storage
- **Scalable architecture**: Modular and extensible design

## 🔬 Research Applications

This implementation supports:
- **Algorithm development**: Easy addition of new models
- **Feature research**: Comprehensive feature engineering framework
- **Validation studies**: Rigorous evaluation protocols
- **Reproducible research**: Version-controlled experiments

## 📝 License

Academic use for IEEE BHI 2025 Challenge Track 2.

---

**🏆 Complete End-to-End Solution for CGMacros CCR Prediction**
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