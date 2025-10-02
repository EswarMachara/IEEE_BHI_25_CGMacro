"""
Comprehensive Visualization Generator for CGMacros Pipeline

This script generates and saves all crucial figures from each pipeline component:
- EDA visualizations
- Feature engineering analysis
- Data quality assessment
- Target variable analysis
- Model preparation insights

All figures are organized into structured directories for easy access.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
from src.data_loader_updated import DataLoader
from src.feature_engineering_updated import FeatureEngineer
from src.target_updated import compute_ccr, remove_nutrient_columns

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

class ComprehensiveVisualizer:
    """
    Generate all crucial visualizations for the CGMacros pipeline.
    """
    
    def __init__(self, save_dir: str = "figures"):
        """
        Initialize the visualizer.
        
        Args:
            save_dir: Base directory to save all figures
        """
        self.save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create organized directory structure
        self.dirs = {
            'eda': os.path.join(save_dir, '01_EDA'),
            'data_quality': os.path.join(save_dir, '02_Data_Quality'), 
            'feature_engineering': os.path.join(save_dir, '03_Feature_Engineering'),
            'target_analysis': os.path.join(save_dir, '04_Target_Analysis'),
            'correlation': os.path.join(save_dir, '05_Correlation_Analysis'),
            'distribution': os.path.join(save_dir, '06_Distribution_Analysis'),
            'temporal': os.path.join(save_dir, '07_Temporal_Analysis'),
            'multimodal': os.path.join(save_dir, '08_Multimodal_Analysis')
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"📁 Figure directories created in: {save_dir}")
        print(f"🕒 Timestamp: {self.timestamp}")
    
    def save_figure(self, fig, filename: str, category: str, dpi: int = 300):
        """
        Save figure with organized naming and directory structure.
        
        Args:
            fig: Matplotlib figure object
            filename: Base filename
            category: Category directory
            dpi: Resolution for saving
        """
        filepath = os.path.join(self.dirs[category], f"{filename}_{self.timestamp}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"💾 Saved: {filepath}")
        plt.close(fig)
    
    def generate_eda_figures(self, cgmacros_data, bio_data, microbes_data, gut_health_data):
        """
        Generate comprehensive EDA figures.
        """
        print("\n📊 Generating EDA Figures...")
        
        # 1. Dataset Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CGMacros Dataset Overview', fontsize=16, fontweight='bold')
        
        # Participant distribution
        participant_counts = cgmacros_data['participant_id'].value_counts().sort_index()
        axes[0, 0].bar(range(len(participant_counts)), participant_counts.values)
        axes[0, 0].set_title('Data Points per Participant')
        axes[0, 0].set_xlabel('Participant ID')
        axes[0, 0].set_ylabel('Number of Records')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Data completeness
        completeness = cgmacros_data.isnull().mean() * 100
        completeness = completeness.sort_values(ascending=True)
        axes[0, 1].barh(range(len(completeness)), completeness.values)
        axes[0, 1].set_yticks(range(len(completeness)))
        axes[0, 1].set_yticklabels(completeness.index, fontsize=8)
        axes[0, 1].set_title('Missing Data Percentage by Column')
        axes[0, 1].set_xlabel('Missing %')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bio data distribution
        if 'Age' in bio_data.columns:
            axes[1, 0].hist(bio_data['Age'].dropna(), bins=20, alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Age Distribution')
            axes[1, 0].set_xlabel('Age')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Gender distribution
        if 'Gender' in bio_data.columns:
            gender_counts = bio_data['Gender'].value_counts()
            axes[1, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Gender Distribution')
        
        plt.tight_layout()
        self.save_figure(fig, 'dataset_overview', 'eda')
        
        # 2. Glucose Data Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Glucose Data Analysis', fontsize=16, fontweight='bold')
        
        # Glucose distributions
        if 'Libre GL' in cgmacros_data.columns:
            glucose_data = cgmacros_data['Libre GL'].dropna()
            axes[0, 0].hist(glucose_data, bins=50, alpha=0.7, label='Libre GL', edgecolor='black')
            axes[0, 0].set_title('Libre Glucose Distribution')
            axes[0, 0].set_xlabel('Glucose Level')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        if 'Dexcom GL' in cgmacros_data.columns:
            glucose_data = cgmacros_data['Dexcom GL'].dropna()
            axes[0, 1].hist(glucose_data, bins=50, alpha=0.7, label='Dexcom GL', color='orange', edgecolor='black')
            axes[0, 1].set_title('Dexcom Glucose Distribution')
            axes[0, 1].set_xlabel('Glucose Level')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Glucose time series sample
        sample_participant = cgmacros_data[cgmacros_data['participant_id'] == cgmacros_data['participant_id'].iloc[0]]
        if 'Timestamp' in sample_participant.columns and 'Libre GL' in sample_participant.columns:
            sample_participant = sample_participant.sort_values('Timestamp')
            axes[1, 0].plot(sample_participant['Timestamp'], sample_participant['Libre GL'], 'b-', alpha=0.7)
            axes[1, 0].set_title(f'Sample Glucose Time Series (Participant {sample_participant["participant_id"].iloc[0]})')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Glucose Level')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Activity data
        if 'HR' in cgmacros_data.columns:
            hr_data = cgmacros_data['HR'].dropna()
            axes[1, 1].hist(hr_data, bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[1, 1].set_title('Heart Rate Distribution')
            axes[1, 1].set_xlabel('Heart Rate (bpm)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'glucose_analysis', 'eda')
        
        # 3. Microbiome Analysis
        if not microbes_data.empty:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Microbiome Data Analysis', fontsize=16, fontweight='bold')
            
            # Microbiome data sparsity
            numeric_cols = microbes_data.select_dtypes(include=[np.number]).columns
            sparsity = (microbes_data[numeric_cols] == 0).mean() * 100
            sparsity_sorted = sparsity.sort_values(ascending=False)
            
            axes[0, 0].hist(sparsity_sorted.values, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Microbiome Feature Sparsity Distribution')
            axes[0, 0].set_xlabel('Sparsity (%)')
            axes[0, 0].set_ylabel('Number of Features')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Top non-sparse features
            top_features = sparsity_sorted.head(20)
            axes[0, 1].barh(range(len(top_features)), top_features.values)
            axes[0, 1].set_yticks(range(len(top_features)))
            axes[0, 1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                                       for name in top_features.index], fontsize=8)
            axes[0, 1].set_title('Top 20 Most Sparse Microbiome Features')
            axes[0, 1].set_xlabel('Sparsity (%)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Diversity analysis
            if len(numeric_cols) > 10:
                diversity_per_sample = (microbes_data[numeric_cols] > 0).sum(axis=1)
                axes[1, 0].hist(diversity_per_sample, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Microbiome Diversity per Sample')
                axes[1, 0].set_xlabel('Number of Present Species')
                axes[1, 0].set_ylabel('Number of Samples')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Feature distribution
            if len(numeric_cols) > 0:
                sample_feature = microbes_data[numeric_cols[0]].dropna()
                if len(sample_feature) > 0:
                    axes[1, 1].hist(sample_feature, bins=30, alpha=0.7, edgecolor='black')
                    axes[1, 1].set_title(f'Sample Feature Distribution: {numeric_cols[0][:30]}')
                    axes[1, 1].set_xlabel('Feature Value')
                    axes[1, 1].set_ylabel('Frequency')
                    axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.save_figure(fig, 'microbiome_analysis', 'eda')
    
    def generate_data_quality_figures(self, merged_data):
        """
        Generate data quality assessment figures.
        """
        print("\n🔍 Generating Data Quality Figures...")
        
        # 1. Missing Data Heatmap
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # Missing data pattern
        missing_data = merged_data.isnull()
        if missing_data.sum().sum() > 0:
            # Sample for visualization if too many columns
            if missing_data.shape[1] > 50:
                sample_cols = missing_data.columns[:50]
                missing_sample = missing_data[sample_cols]
            else:
                missing_sample = missing_data
            
            sns.heatmap(missing_sample.head(100).T, cbar=True, cmap='viridis', 
                       ax=axes[0], yticklabels=True)
            axes[0].set_title('Missing Data Pattern (First 100 rows, First 50 columns)')
            axes[0].set_xlabel('Sample Index')
            axes[0].set_ylabel('Features')
        
        # Missing data summary
        missing_summary = merged_data.isnull().sum().sort_values(ascending=False)
        missing_summary = missing_summary[missing_summary > 0]
        
        if len(missing_summary) > 0:
            top_missing = missing_summary.head(20)
            axes[1].barh(range(len(top_missing)), top_missing.values)
            axes[1].set_yticks(range(len(top_missing)))
            axes[1].set_yticklabels(top_missing.index, fontsize=8)
            axes[1].set_title('Top 20 Features with Missing Data')
            axes[1].set_xlabel('Number of Missing Values')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'data_quality_overview', 'data_quality')
        
        # 2. Data Distribution Analysis
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Numeric Data Distributions', fontsize=16, fontweight='bold')
            
            # Sample 6 important numeric columns
            important_cols = [col for col in ['Libre GL', 'Dexcom GL', 'HR', 'METs', 'Calories', 'Age'] 
                            if col in numeric_cols]
            if len(important_cols) < 6:
                important_cols.extend([col for col in numeric_cols if col not in important_cols][:6-len(important_cols)])
            
            for i, col in enumerate(important_cols[:6]):
                row, col_idx = i // 3, i % 3
                data = merged_data[col].dropna()
                if len(data) > 0:
                    axes[row, col_idx].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[row, col_idx].set_title(f'{col} Distribution')
                    axes[row, col_idx].set_xlabel(col)
                    axes[row, col_idx].set_ylabel('Frequency')
                    axes[row, col_idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            self.save_figure(fig, 'numeric_distributions', 'data_quality')
    
    def generate_feature_engineering_figures(self, feature_data):
        """
        Generate feature engineering analysis figures.
        """
        print("\n⚙️ Generating Feature Engineering Figures...")
        
        # 1. Feature Categories Overview
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Feature Engineering Overview', fontsize=16, fontweight='bold')
        
        # Categorize features
        feature_categories = {
            'Glucose': [col for col in feature_data.columns if 'gl' in col.lower() or 'glucose' in col.lower()],
            'Activity': [col for col in feature_data.columns if any(term in col.lower() for term in ['hr', 'met', 'calorie', 'activity'])],
            'Temporal': [col for col in feature_data.columns if any(term in col.lower() for term in ['hour', 'day', 'time', 'temporal'])],
            'Microbiome': [col for col in feature_data.columns if 'microbiome' in col.lower() or any(bacteria in col for bacteria in ['Bacteroides', 'Bifidobacterium'])],
            'Gut Health': [col for col in feature_data.columns if any(term in col.lower() for term in ['gut', 'lps', 'biofilm'])],
            'Demographics': [col for col in feature_data.columns if any(term in col.lower() for term in ['age', 'gender', 'bmi', 'a1c'])],
            'Other': []
        }
        
        # Calculate category sizes
        category_sizes = {}
        total_features = set(feature_data.columns)
        categorized_features = set()
        
        for category, features in feature_categories.items():
            if category != 'Other':
                category_features = set(features) & total_features
                category_sizes[category] = len(category_features)
                categorized_features.update(category_features)
        
        # Add uncategorized features to 'Other'
        other_features = total_features - categorized_features - {'participant_id', 'Timestamp', 'CCR'}
        category_sizes['Other'] = len(other_features)
        
        # Feature category pie chart
        axes[0, 0].pie(category_sizes.values(), labels=category_sizes.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Feature Distribution by Category')
        
        # Feature count by category
        axes[0, 1].bar(category_sizes.keys(), category_sizes.values())
        axes[0, 1].set_title('Number of Features by Category')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Feature sparsity analysis
        numeric_features = feature_data.select_dtypes(include=[np.number]).columns
        sparsity = (feature_data[numeric_features] == 0).mean() * 100
        
        axes[1, 0].hist(sparsity.values, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Feature Sparsity Distribution')
        axes[1, 0].set_xlabel('Sparsity (%)')
        axes[1, 0].set_ylabel('Number of Features')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature variance analysis
        feature_vars = feature_data[numeric_features].var().sort_values(ascending=False)
        top_var_features = feature_vars.head(20)
        
        axes[1, 1].barh(range(len(top_var_features)), top_var_features.values)
        axes[1, 1].set_yticks(range(len(top_var_features)))
        axes[1, 1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                                   for name in top_var_features.index], fontsize=8)
        axes[1, 1].set_title('Top 20 Features by Variance')
        axes[1, 1].set_xlabel('Variance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'feature_engineering_overview', 'feature_engineering')
        
        # 2. Sample Feature Distributions by Category
        for category, features in feature_categories.items():
            if len(features) > 0 and category != 'Other':
                available_features = [f for f in features if f in feature_data.columns][:6]
                if len(available_features) > 0:
                    n_features = len(available_features)
                    n_rows = (n_features + 2) // 3
                    n_cols = min(3, n_features)
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                    fig.suptitle(f'{category} Features Distribution', fontsize=16, fontweight='bold')
                    
                    if n_features == 1:
                        axes = [axes]
                    elif n_rows == 1:
                        axes = [axes] if n_cols == 1 else axes
                    else:
                        axes = axes.flatten()
                    
                    for i, feature in enumerate(available_features):
                        if i < len(axes):
                            data = feature_data[feature].dropna()
                            if len(data) > 0:
                                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                                axes[i].set_title(f'{feature}')
                                axes[i].set_xlabel('Value')
                                axes[i].set_ylabel('Frequency')
                                axes[i].grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for i in range(len(available_features), len(axes)):
                        axes[i].set_visible(False)
                    
                    plt.tight_layout()
                    self.save_figure(fig, f'{category.lower()}_features', 'feature_engineering')
    
    def generate_target_analysis_figures(self, target_data):
        """
        Generate comprehensive target variable analysis figures.
        """
        print("\n🎯 Generating Target Analysis Figures...")
        
        if 'CCR' not in target_data.columns:
            print("❌ CCR column not found in target data")
            return
        
        # 1. CCR Distribution Analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CCR Target Variable Analysis', fontsize=16, fontweight='bold')
        
        ccr_data = target_data['CCR'].dropna()
        
        # Basic distribution
        axes[0, 0].hist(ccr_data, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('CCR Distribution')
        axes[0, 0].set_xlabel('CCR Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(ccr_data)
        axes[0, 1].set_title('CCR Box Plot')
        axes[0, 1].set_ylabel('CCR Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(ccr_data, dist="norm", plot=axes[0, 2])
        axes[0, 2].set_title('CCR Q-Q Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # CCR by participant
        if 'participant_id' in target_data.columns:
            participant_ccr = target_data.groupby('participant_id')['CCR'].agg(['mean', 'std', 'count']).reset_index()
            
            axes[1, 0].bar(range(len(participant_ccr)), participant_ccr['mean'])
            axes[1, 0].set_title('Average CCR by Participant')
            axes[1, 0].set_xlabel('Participant Index')
            axes[1, 0].set_ylabel('Average CCR')
            axes[1, 0].grid(True, alpha=0.3)
            
            # CCR variability by participant
            axes[1, 1].bar(range(len(participant_ccr)), participant_ccr['std'])
            axes[1, 1].set_title('CCR Std Deviation by Participant')
            axes[1, 1].set_xlabel('Participant Index')
            axes[1, 1].set_ylabel('CCR Std Dev')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Sample count by participant
            axes[1, 2].bar(range(len(participant_ccr)), participant_ccr['count'])
            axes[1, 2].set_title('CCR Sample Count by Participant')
            axes[1, 2].set_xlabel('Participant Index')
            axes[1, 2].set_ylabel('Number of Samples')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'ccr_distribution_analysis', 'target_analysis')
        
        # 2. CCR vs Key Features
        key_features = ['Libre GL', 'Dexcom GL', 'HR', 'METs', 'Calories', 'Age']
        available_features = [f for f in key_features if f in target_data.columns]
        
        if len(available_features) > 0:
            n_features = len(available_features)
            n_rows = (n_features + 2) // 3
            n_cols = min(3, n_features)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            fig.suptitle('CCR vs Key Features', fontsize=16, fontweight='bold')
            
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(available_features):
                if i < len(axes):
                    valid_data = target_data[[feature, 'CCR']].dropna()
                    if len(valid_data) > 0:
                        axes[i].scatter(valid_data[feature], valid_data['CCR'], alpha=0.6)
                        axes[i].set_xlabel(feature)
                        axes[i].set_ylabel('CCR')
                        axes[i].set_title(f'CCR vs {feature}')
                        axes[i].grid(True, alpha=0.3)
                        
                        # Add correlation coefficient
                        corr = valid_data[feature].corr(valid_data['CCR'])
                        axes[i].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[i].transAxes, 
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Hide unused subplots
            for i in range(len(available_features), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            self.save_figure(fig, 'ccr_vs_features', 'target_analysis')
    
    def generate_correlation_figures(self, feature_data):
        """
        Generate correlation analysis figures.
        """
        print("\n🔗 Generating Correlation Analysis Figures...")
        
        numeric_features = feature_data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 1:
            # Sample features for correlation analysis (too many features make heatmap unreadable)
            if len(numeric_features) > 50:
                # Select most important features
                important_features = [col for col in numeric_features if any(term in col.lower() 
                                    for term in ['ccr', 'gl', 'glucose', 'hr', 'met', 'age', 'bmi'])]
                
                # Add some random features if we don't have enough important ones
                remaining_features = [col for col in numeric_features if col not in important_features]
                np.random.seed(42)
                additional_features = np.random.choice(remaining_features, 
                                                     min(50 - len(important_features), len(remaining_features)), 
                                                     replace=False)
                sample_features = important_features + list(additional_features)
            else:
                sample_features = list(numeric_features)
            
            # 1. Correlation heatmap
            correlation_matrix = feature_data[sample_features].corr()
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')
            
            # Full correlation heatmap
            sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
                       square=True, ax=axes[0], cbar=True)
            axes[0].set_title(f'Feature Correlation Matrix ({len(sample_features)} features)')
            
            # High correlation pairs
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:  # High correlation threshold
                        corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j], 
                            'correlation': corr_val
                        })
            
            if corr_pairs:
                corr_df = pd.DataFrame(corr_pairs)
                corr_df = corr_df.sort_values('correlation', key=abs, ascending=False).head(20)
                
                y_pos = range(len(corr_df))
                bars = axes[1].barh(y_pos, corr_df['correlation'])
                axes[1].set_yticks(y_pos)
                axes[1].set_yticklabels([f"{row['feature1'][:15]}...vs\n{row['feature2'][:15]}..." 
                                       for _, row in corr_df.iterrows()], fontsize=8)
                axes[1].set_title('Top 20 Feature Correlations (|r| > 0.7)')
                axes[1].set_xlabel('Correlation Coefficient')
                axes[1].grid(True, alpha=0.3)
                
                # Color bars by correlation strength
                for i, bar in enumerate(bars):
                    if corr_df.iloc[i]['correlation'] > 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('blue')
            
            plt.tight_layout()
            self.save_figure(fig, 'correlation_analysis', 'correlation')
    
    def generate_temporal_figures(self, merged_data):
        """
        Generate temporal analysis figures.
        """
        print("\n⏰ Generating Temporal Analysis Figures...")
        
        if 'Timestamp' not in merged_data.columns:
            print("❌ Timestamp column not found")
            return
        
        # Convert timestamp
        merged_data = merged_data.copy()
        merged_data['Timestamp'] = pd.to_datetime(merged_data['Timestamp'])
        merged_data['hour'] = merged_data['Timestamp'].dt.hour
        merged_data['day_of_week'] = merged_data['Timestamp'].dt.dayofweek
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Temporal Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Data points by hour
        hourly_counts = merged_data.groupby('hour').size()
        axes[0, 0].bar(hourly_counts.index, hourly_counts.values)
        axes[0, 0].set_title('Data Points by Hour of Day')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Records')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Data points by day of week
        daily_counts = merged_data.groupby('day_of_week').size()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0, 1].bar(range(len(daily_counts)), daily_counts.values)
        axes[0, 1].set_xticks(range(len(daily_counts)))
        axes[0, 1].set_xticklabels([day_names[i] for i in daily_counts.index])
        axes[0, 1].set_title('Data Points by Day of Week')
        axes[0, 1].set_ylabel('Number of Records')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Glucose patterns by hour
        if 'Libre GL' in merged_data.columns:
            hourly_glucose = merged_data.groupby('hour')['Libre GL'].mean()
            axes[0, 2].plot(hourly_glucose.index, hourly_glucose.values, 'b-o')
            axes[0, 2].set_title('Average Glucose by Hour')
            axes[0, 2].set_xlabel('Hour')
            axes[0, 2].set_ylabel('Average Glucose')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Activity patterns by hour
        if 'HR' in merged_data.columns:
            hourly_hr = merged_data.groupby('hour')['HR'].mean()
            axes[1, 0].plot(hourly_hr.index, hourly_hr.values, 'r-o')
            axes[1, 0].set_title('Average Heart Rate by Hour')
            axes[1, 0].set_xlabel('Hour')
            axes[1, 0].set_ylabel('Average Heart Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Sample participant timeline
        if 'participant_id' in merged_data.columns:
            sample_participant = merged_data[merged_data['participant_id'] == merged_data['participant_id'].iloc[0]]
            sample_participant = sample_participant.sort_values('Timestamp')
            
            if 'Libre GL' in sample_participant.columns and len(sample_participant) > 1:
                axes[1, 1].plot(sample_participant['Timestamp'], sample_participant['Libre GL'], 'g-')
                axes[1, 1].set_title(f'Sample Timeline (Participant {sample_participant["participant_id"].iloc[0]})')
                axes[1, 1].set_xlabel('Time')
                axes[1, 1].set_ylabel('Glucose Level')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True, alpha=0.3)
        
        # CCR patterns by hour (if available)
        if 'CCR' in merged_data.columns:
            hourly_ccr = merged_data.groupby('hour')['CCR'].mean()
            axes[1, 2].plot(hourly_ccr.index, hourly_ccr.values, 'm-o')
            axes[1, 2].set_title('Average CCR by Hour')
            axes[1, 2].set_xlabel('Hour')
            axes[1, 2].set_ylabel('Average CCR')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'temporal_patterns', 'temporal')
    
    def generate_multimodal_figures(self, merged_data):
        """
        Generate multimodal data integration figures.
        """
        print("\n🔬 Generating Multimodal Analysis Figures...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multimodal Data Integration Analysis', fontsize=16, fontweight='bold')
        
        # Data availability by participant
        if 'participant_id' in merged_data.columns:
            participant_data_counts = merged_data.groupby('participant_id').size().sort_values(ascending=False)
            
            axes[0, 0].bar(range(len(participant_data_counts)), participant_data_counts.values)
            axes[0, 0].set_title('Data Availability by Participant')
            axes[0, 0].set_xlabel('Participant (sorted by data volume)')
            axes[0, 0].set_ylabel('Number of Records')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Feature completeness across modalities
        modality_features = {
            'Glucose': [col for col in merged_data.columns if 'gl' in col.lower()],
            'Activity': [col for col in merged_data.columns if any(term in col.lower() for term in ['hr', 'met', 'calorie'])],
            'Demographics': [col for col in merged_data.columns if any(term in col.lower() for term in ['age', 'gender', 'bmi'])],
            'Microbiome': [col for col in merged_data.columns if 'microbiome' in col.lower()],
            'Gut Health': [col for col in merged_data.columns if 'gut' in col.lower()]
        }
        
        completeness_data = []
        for modality, features in modality_features.items():
            available_features = [f for f in features if f in merged_data.columns]
            if available_features:
                completeness = (1 - merged_data[available_features].isnull().mean().mean()) * 100
                completeness_data.append({'modality': modality, 'completeness': completeness})
        
        if completeness_data:
            comp_df = pd.DataFrame(completeness_data)
            axes[0, 1].bar(comp_df['modality'], comp_df['completeness'])
            axes[0, 1].set_title('Data Completeness by Modality')
            axes[0, 1].set_ylabel('Completeness (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Cross-modality correlations
        if 'CCR' in merged_data.columns:
            key_features = ['Libre GL', 'Dexcom GL', 'HR', 'METs', 'Age', 'BMI']
            available_key_features = [f for f in key_features if f in merged_data.columns]
            
            if len(available_key_features) > 1:
                corr_with_ccr = merged_data[available_key_features + ['CCR']].corr()['CCR'].drop('CCR')
                
                axes[1, 0].barh(range(len(corr_with_ccr)), corr_with_ccr.values)
                axes[1, 0].set_yticks(range(len(corr_with_ccr)))
                axes[1, 0].set_yticklabels(corr_with_ccr.index)
                axes[1, 0].set_title('Feature Correlation with CCR')
                axes[1, 0].set_xlabel('Correlation Coefficient')
                axes[1, 0].grid(True, alpha=0.3)
        
        # Sample size distribution
        if 'participant_id' in merged_data.columns:
            sample_sizes = merged_data.groupby('participant_id').size()
            axes[1, 1].hist(sample_sizes.values, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Sample Size Distribution Across Participants')
            axes[1, 1].set_xlabel('Number of Records per Participant')
            axes[1, 1].set_ylabel('Number of Participants')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, 'multimodal_integration', 'multimodal')
    
    def generate_all_figures(self):
        """
        Generate all visualization figures for the pipeline.
        """
        print("🎨 Starting Comprehensive Figure Generation...")
        print(f"📁 Saving all figures to: {self.save_dir}")
        
        try:
            # Load data
            print("\n📥 Loading data...")
            data_loader = DataLoader()
            
            # Load individual data sources
            cgmacros_data = data_loader.load_cgmacros_data('data/raw/CGMacros_CSVs')
            bio_data = data_loader.load_bio_data('data/raw/bio.csv')
            microbes_data = data_loader.load_microbes_data('data/raw/microbes.csv')
            gut_health_data = data_loader.load_gut_health_data('data/raw/gut_health_test.csv')
            
            # Merge all data
            merged_data = data_loader.merge_data_sources(cgmacros_data, bio_data, microbes_data, gut_health_data)
            print(f"✅ Data loaded successfully. Merged shape: {merged_data.shape}")
            
            # Generate EDA figures
            self.generate_eda_figures(cgmacros_data, bio_data, microbes_data, gut_health_data)
            
            # Generate data quality figures
            self.generate_data_quality_figures(merged_data)
            
            # Feature engineering
            print("\n⚙️ Performing feature engineering...")
            feature_engineer = FeatureEngineer()
            feature_data = merged_data.copy()
            
            # Add features step by step
            feature_data = feature_engineer.add_glucose_features(feature_data)
            feature_data = feature_engineer.add_activity_features(feature_data)
            feature_data = feature_engineer.add_meal_timing_features(feature_data)
            feature_data = feature_engineer.add_demographic_features(feature_data)
            feature_data = feature_engineer.add_microbiome_features(feature_data)
            feature_data = feature_engineer.add_gut_health_features(feature_data)
            feature_data = feature_engineer.add_temporal_features(feature_data)
            
            print(f"✅ Feature engineering completed. Shape: {feature_data.shape}")
            
            # Generate feature engineering figures
            self.generate_feature_engineering_figures(feature_data)
            
            # Target engineering
            print("\n🎯 Computing target variable...")
            target_data = compute_ccr(feature_data)
            target_data = remove_nutrient_columns(target_data)
            print(f"✅ Target computation completed. Shape: {target_data.shape}")
            
            # Generate target analysis figures
            self.generate_target_analysis_figures(target_data)
            
            # Generate correlation figures
            self.generate_correlation_figures(target_data)
            
            # Generate temporal figures
            self.generate_temporal_figures(target_data)
            
            # Generate multimodal figures
            self.generate_multimodal_figures(target_data)
            
            print(f"\n🎉 All figures generated successfully!")
            print(f"📁 Figures saved in: {self.save_dir}")
            print(f"🕒 Timestamp: {self.timestamp}")
            
            # Generate summary report
            self.generate_summary_report()
            
        except Exception as e:
            print(f"❌ Error generating figures: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_summary_report(self):
        """
        Generate a summary report of all generated figures.
        """
        report_path = os.path.join(self.save_dir, f"figure_summary_{self.timestamp}.md")
        
        with open(report_path, 'w') as f:
            f.write(f"# CGMacros Pipeline - Figure Generation Summary\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Timestamp:** {self.timestamp}\n\n")
            
            f.write("## 📁 Directory Structure\n\n")
            for category, path in self.dirs.items():
                f.write(f"- **{category.replace('_', ' ').title()}**: `{path}`\n")
            
            f.write("\n## 📊 Generated Figure Categories\n\n")
            f.write("### 1. EDA (Exploratory Data Analysis)\n")
            f.write("- Dataset overview and participant distribution\n")
            f.write("- Glucose data analysis and distributions\n") 
            f.write("- Microbiome data sparsity and diversity analysis\n\n")
            
            f.write("### 2. Data Quality Assessment\n")
            f.write("- Missing data patterns and heatmaps\n")
            f.write("- Numeric data distributions\n")
            f.write("- Data completeness analysis\n\n")
            
            f.write("### 3. Feature Engineering\n")
            f.write("- Feature category distribution\n")
            f.write("- Feature sparsity and variance analysis\n")
            f.write("- Sample distributions by feature category\n\n")
            
            f.write("### 4. Target Analysis\n")
            f.write("- CCR distribution and statistical analysis\n")
            f.write("- CCR patterns by participant\n")
            f.write("- CCR vs key features relationships\n\n")
            
            f.write("### 5. Correlation Analysis\n")
            f.write("- Feature correlation matrices\n")
            f.write("- High correlation pairs identification\n\n")
            
            f.write("### 6. Temporal Analysis\n")
            f.write("- Time-based patterns in data collection\n")
            f.write("- Glucose and activity patterns by hour\n")
            f.write("- Sample participant timelines\n\n")
            
            f.write("### 7. Multimodal Analysis\n")
            f.write("- Cross-modality data integration\n")
            f.write("- Data completeness across modalities\n")
            f.write("- Sample size distributions\n\n")
            
            f.write("## 🚀 Usage\n\n")
            f.write("All figures are saved with timestamps for version control.\n")
            f.write("Use these visualizations for:\n")
            f.write("- Research presentations\n")
            f.write("- Technical reports\n")
            f.write("- Pipeline validation\n")
            f.write("- Feature selection insights\n\n")
            
            f.write("## 📋 Next Steps\n\n")
            f.write("1. Review all generated figures\n")
            f.write("2. Use insights for model selection\n")
            f.write("3. Run the complete pipeline: `python run_pipeline_updated.py`\n")
            f.write("4. Generate model performance figures\n\n")
        
        print(f"📄 Summary report saved: {report_path}")

def main():
    """
    Main function to generate all figures.
    """
    print("🎨 CGMacros Pipeline - Comprehensive Figure Generation")
    print("=" * 60)
    
    # Create visualizer
    visualizer = ComprehensiveVisualizer(save_dir="figures")
    
    # Generate all figures
    visualizer.generate_all_figures()
    
    print("\n" + "=" * 60)
    print("✅ Figure generation completed successfully!")
    print(f"📁 All figures saved in: {visualizer.save_dir}")
    print("🚀 Ready to run the pipeline!")

if __name__ == "__main__":
    main()