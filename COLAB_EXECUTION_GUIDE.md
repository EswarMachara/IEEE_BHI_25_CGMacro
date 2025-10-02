# 🚀 Google Colab Execution Guide for CGMacros Pipeline

## 📋 Prerequisites
- Google Colab account
- Repository already cloned to Colab

## 🔗 Step 1: Clone Repository to Colab

If you haven't cloned the repository yet, run this in a Colab cell:

```python
!git clone https://github.com/EswarMachara/IEEE_BHI_25_CGMacro.git /content/IEEE_BHI_Track2
%cd /content/IEEE_BHI_Track2
!ls -la
```

## 📁 Step 2: Upload the Colab-Optimized Notebook

1. Upload `notebooks/05_colab_optimized_execution.ipynb` to your Colab
2. Or copy the notebook content directly into a new Colab notebook

## ⚡ Step 3: Verify Colab Environment

Run this to check your Colab specs:

```python
!cat /proc/meminfo | grep MemTotal
!cat /proc/cpuinfo | grep "model name" | head -1
!nvidia-smi  # Check if GPU is available (optional)
```

Expected Colab resources:
- **RAM**: 12-16 GB (vs. your local 1.3 GB)
- **CPU**: 2-4 cores
- **Storage**: 25-100 GB temporary

## 🎯 Step 4: Execute the Pipeline

### **Optimizations for Colab:**

1. **Microbiome Features**: 1000 features (vs. 20 on local)
   - Preserves biological diversity
   - Uses Colab's high memory efficiently

2. **Chunk Size**: 10 files per chunk (vs. 2 on local)
   - Faster processing with more memory

3. **Complete Dataset**: All 687,580 records
   - No sampling or data reduction
   - Full meal records for training

4. **Advanced Models**: XGBoost + LightGBM
   - Leverage Colab's computational power

### **Execution Order:**
1. **Setup & Dependencies** (installs packages automatically)
2. **Data Loading** (optimal chunking for Colab)
3. **Data Merging** (1000 microbiome features)
4. **Feature Engineering** (complete pipeline)
5. **Target Computation** (CCR calculation)
6. **Model Training** (all advanced models)
7. **Results Analysis** (comprehensive evaluation)

## 🔧 Step 5: Monitor Execution

### **Memory Monitoring:**
Each phase shows memory usage to ensure stability:
```
Memory before loading: XXX MB
Memory after loading: XXX MB
```

### **Progress Indicators:**
- ✅ Successful completion
- ⚠️ Warnings (handled automatically)
- 📊 Statistics and results

## 🏆 Expected Results

### **Performance Improvements:**
- **Dataset**: 687,580 records (100% vs. 1% sample)
- **Features**: 1000+ microbiome features
- **Models**: R² > 0.5 (vs. -2.16 previously)
- **Training**: 1000+ meal records (vs. 15)

### **Memory Usage:**
- **Peak Usage**: ~2-4 GB (well within Colab limits)
- **Final Usage**: ~1-2 GB
- **Efficiency**: 5-10x better than aggressive local optimization

## 🚨 Troubleshooting

### **If Memory Issues Occur:**
1. Restart runtime: `Runtime → Restart Runtime`
2. Ensure using `05_colab_optimized_execution.ipynb`
3. Clear variables manually if needed:
   ```python
   import gc
   gc.collect()
   ```

### **If Repository Not Found:**
```python
import os
if not os.path.exists('/content/IEEE_BHI_Track2'):
    !git clone https://github.com/EswarMachara/IEEE_BHI_25_CGMacro.git /content/IEEE_BHI_Track2
os.chdir('/content/IEEE_BHI_Track2')
```

### **If Package Installation Fails:**
```python
!pip install --upgrade pip
!pip install xgboost lightgbm psutil scikit-learn pandas numpy
```

## 🎯 Execution Summary

**Total Runtime**: ~15-30 minutes for complete pipeline
**Memory Required**: 12+ GB (Colab provides this)
**Output**: High-quality CCR prediction models with full dataset

## 📊 What You'll Get

1. **Complete Data Processing**: All 687,580 records processed
2. **Rich Feature Set**: 1000 microbiome + engineered features
3. **Multiple Models**: Linear, RF, XGBoost, LightGBM results
4. **Performance Metrics**: R², RMSE, MAE for all models
5. **Memory Efficiency**: Optimized for Colab environment

## 🚀 Ready to Execute!

Simply run all cells in `05_colab_optimized_execution.ipynb` sequentially in Google Colab. The notebook is designed to be completely self-contained and will guide you through each phase with clear progress indicators and results.