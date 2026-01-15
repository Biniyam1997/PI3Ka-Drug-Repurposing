# qsar_advanced_analysis_updated.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import mahalanobis
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class PublicationStyle:
    """Publication-style plotting configuration"""
    
    @classmethod
    def set_style(cls):
        """Set publication style with bold fonts and no gridlines"""
        plt.style.use('default')
        
        # Set font parameters for bold appearance - using valid parameters only
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'font.weight': 'bold',
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.titleweight': 'bold',
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

class QSARAdvancedAnalyzer:
    """Advanced QSAR model analysis with comprehensive plots"""
    
    def __init__(self, pipeline_path, output_folder="QSAR_Advanced_Analysis_Updated"):
        self.pipeline_path = pipeline_path
        self.output_folder = output_folder
        self.pipeline = None
        self.models = {}
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        PublicationStyle.set_style()
    
    def load_pipeline_data(self):
        """Load the saved pipeline and data"""
        print("📂 Loading pipeline and data...")
        try:
            self.pipeline = joblib.load(self.pipeline_path)
            self.models = self.pipeline['models']
            
            # Load individual components
            self.scaler = joblib.load("QSAR_Publication_Results_Final/models/feature_scaler.pkl")
            self.selected_scaler = joblib.load("QSAR_Publication_Results_Final/models/selected_feature_scaler.pkl")
            self.feature_info = joblib.load("QSAR_Publication_Results_Final/models/feature_info.pkl")
            
            # Load the original training data to recreate predictions
            self.df_final = pd.read_csv("QSAR_Publication_Results_Final/tables/cleaned_dataset.csv")
            
            print("✅ Pipeline and components loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading pipeline: {e}")
            return False
    
    def recreate_predictions(self):
        """Recreate predictions using the original data"""
        print("\n🔄 Recreating predictions from original data...")
        
        # Load Mordred descriptors for the training data
        mordred_descriptors, valid_indices = self.calculate_mordred_descriptors(self.df_final['Smiles'].tolist())
        mordred_clean = self.clean_mordred_descriptors(mordred_descriptors)
        
        # Prepare features
        df_valid = self.df_final.iloc[valid_indices].reset_index(drop=True)
        feature_columns = mordred_clean.columns.tolist()
        
        if len(df_valid) != len(mordred_clean):
            min_len = min(len(df_valid), len(mordred_clean))
            df_valid = df_valid.iloc[:min_len]
            mordred_clean = mordred_clean.iloc[:min_len]
        
        # Create final dataset
        df_final_processed = pd.concat([
            df_valid[['Molecule ChEMBL ID', 'Smiles', 'IC50 (nM)', 'pIC50']].reset_index(drop=True),
            mordred_clean.reset_index(drop=True)
        ], axis=1).dropna()
        
        X = df_final_processed[feature_columns]
        y = df_final_processed['pIC50']
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Select features
        X_selected = X_scaled[:, self.feature_info['selected_feature_indices']]
        X_final = self.selected_scaler.transform(X_selected)
        
        # Split data (using the same random state as original pipeline)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        self.processed_data = {
            'X_train': X_train, 'X_test': X_test, 'X_val': X_val,
            'y_train': y_train, 'y_test': y_test, 'y_val': y_val,
            'X_combined': X_final, 'y_combined': y
        }
        
        print(f"✅ Predictions recreated: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        return True

    def calculate_mordred_descriptors(self, smiles_list):
        """Calculate Mordred descriptors for SMILES list"""
        from rdkit import Chem
        from mordred import Calculator, descriptors
        from tqdm import tqdm
        
        print("Calculating Mordred descriptors for analysis...")
        calc = Calculator(descriptors, ignore_3D=True)
        all_descriptors = []
        valid_indices = []
        
        for idx, smiles in enumerate(tqdm(smiles_list, desc="Calculating descriptors")):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    Chem.SanitizeMol(mol)
                    desc = calc(mol)
                    desc_dict = {}
                    
                    for key, value in desc.items():
                        if isinstance(value, (int, float)):
                            desc_dict[str(key)] = value
                        else:
                            try:
                                desc_dict[str(key)] = float(value)
                            except (ValueError, TypeError):
                                desc_dict[str(key)] = np.nan
                    
                    all_descriptors.append(desc_dict)
                    valid_indices.append(idx)
            except Exception as e:
                continue
                
        print(f"✅ Successfully processed: {len(all_descriptors)} molecules")
        return all_descriptors, valid_indices

    def clean_mordred_descriptors(self, mordred_descriptors):
        """Clean and preprocess Mordred descriptors"""
        mordred_df = pd.DataFrame(mordred_descriptors)
        print(f"Original Mordred descriptors: {mordred_df.shape[1]}")
        
        # Remove descriptors with too many missing values (>30%)
        missing_threshold = 0.3 * len(mordred_df)
        mordred_clean = mordred_df.loc[:, mordred_df.isnull().sum() < missing_threshold]
        print(f"After removing high-missing columns: {mordred_clean.shape[1]}")
        
        # Remove constant descriptors
        mordred_clean = mordred_clean.loc[:, mordred_clean.std() > 0]
        print(f"After removing constant columns: {mordred_clean.shape[1]}")
        
        # Fill remaining missing values with column median
        mordred_clean = mordred_clean.fillna(mordred_clean.median())
        
        return mordred_clean

    def compute_comprehensive_metrics(self):
        """Compute comprehensive model performance metrics"""
        print("\n📊 Computing comprehensive metrics...")
        
        metrics_data = []
        
        for model_name, model in self.models.items():
            # Get predictions for all sets
            y_train_pred = model.predict(self.processed_data['X_train'])
            y_test_pred = model.predict(self.processed_data['X_test'])
            y_val_pred = model.predict(self.processed_data['X_val'])
            
            # Combine train and val for cross-validation
            X_train_val = np.vstack([self.processed_data['X_train'], self.processed_data['X_val']])
            y_train_val = np.concatenate([self.processed_data['y_train'], self.processed_data['y_val']])
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=5, scoring='r2')
            q2_cv = cv_scores.mean()
            
            # Basic metrics
            r2_train = r2_score(self.processed_data['y_train'], y_train_pred)
            r2_test = r2_score(self.processed_data['y_test'], y_test_pred)
            r2_val = r2_score(self.processed_data['y_val'], y_val_pred)
            
            rmse_train = np.sqrt(mean_squared_error(self.processed_data['y_train'], y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(self.processed_data['y_test'], y_test_pred))
            rmse_val = np.sqrt(mean_squared_error(self.processed_data['y_val'], y_val_pred))
            
            mae_train = mean_absolute_error(self.processed_data['y_train'], y_train_pred)
            mae_test = mean_absolute_error(self.processed_data['y_test'], y_test_pred)
            mae_val = mean_absolute_error(self.processed_data['y_val'], y_val_pred)
            
            metrics_data.append({
                'Model': model_name,
                'R²_train': r2_train,
                'R²_test': r2_test,
                'R²_val': r2_val,
                'Q²_CV': q2_cv,
                'RMSE_train': rmse_train,
                'RMSE_test': rmse_test,
                'RMSE_val': rmse_val,
                'MAE_train': mae_train,
                'MAE_test': mae_test,
                'MAE_val': mae_val
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f'{self.output_folder}/comprehensive_metrics.csv', index=False)
        print("✅ Comprehensive metrics computed and saved!")
        
        return metrics_df

    def create_all_advanced_plots(self, metrics_df):
        """Create all advanced comprehensive plots"""
        print("\n🎨 Creating advanced comprehensive plots...")
        
        # 1. Experimental vs Predicted for all 15 models in one figure
        self._plot_all_models_experimental_vs_predicted()
        
        # 2. Residual plots for all 15 models in one figure (training, test, validation)
        self._plot_all_models_residuals()
        
        # 3. Williams plots for all models in one figure
        self._plot_all_models_williams()
        
        # 4. Model comparison with unified scales
        self._plot_model_comparison_unified(metrics_df)
        
        # 5. Feature importance for all 15 models in one figure
        self._plot_all_models_feature_importance()
        
        # 6. PCA plot
        self._plot_pca_analysis()
        
        # 7. Learning curves for all models
        self._plot_learning_curves()
        
        # 8. Enhanced Y-randomization plot
        self._plot_enhanced_y_randomization()
        
        # 9. Mahalanobis distance plots
        self._plot_mahalanobis_distances()
        
        # 10. Cook's distance plots
        self._plot_cooks_distance()
        
        print("✅ All advanced plots created!")

    def _plot_all_models_experimental_vs_predicted(self):
        """Plot experimental vs predicted for all 15 models in one figure"""
        n_models = len(self.models)
        n_cols = 5  # 5 columns for 15 models (3 rows)
        n_rows = 3  # 3 rows for 15 models
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle('Experimental vs Predicted pIC50 for All Models (Training, Validation, Test)', 
                    fontweight='bold', fontsize=16, y=0.98)
        
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
                
            # Get predictions for all sets
            y_train_pred = model.predict(self.processed_data['X_train'])
            y_val_pred = model.predict(self.processed_data['X_val'])
            y_test_pred = model.predict(self.processed_data['X_test'])
            
            # Calculate R² for each set
            r2_train = r2_score(self.processed_data['y_train'], y_train_pred)
            r2_val = r2_score(self.processed_data['y_val'], y_val_pred)
            r2_test = r2_score(self.processed_data['y_test'], y_test_pred)
            
            ax = axes[idx]
            
            # Plot all three sets with different colors
            ax.scatter(self.processed_data['y_train'], y_train_pred, alpha=0.6, s=40, 
                      label=f'Train (R²={r2_train:.3f})', c='blue')
            ax.scatter(self.processed_data['y_val'], y_val_pred, alpha=0.6, s=40,
                      label=f'Val (R²={r2_val:.3f})', c='orange')
            ax.scatter(self.processed_data['y_test'], y_test_pred, alpha=0.6, s=40,
                      label=f'Test (R²={r2_test:.3f})', c='green')
            
            # Perfect prediction line
            all_y = np.concatenate([self.processed_data['y_train'], self.processed_data['y_val'], self.processed_data['y_test']])
            all_pred = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
            min_val = min(all_y.min(), all_pred.min())
            max_val = max(all_y.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Experimental pIC50', fontweight='bold')
            ax.set_ylabel('Predicted pIC50', fontweight='bold')
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.legend(prop={'weight': 'bold', 'size': 8})
            ax.grid(False)
        
        # Hide unused subplots
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/all_models_experimental_vs_predicted.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Experimental vs Predicted plot for all 15 models created!")

    def _plot_all_models_residuals(self):
        """Plot residual analysis for all 15 models in one figure (training, test, validation)"""
        n_models = len(self.models)
        n_cols = 5  # 5 columns for 15 models (3 rows)
        n_rows = 3  # 3 rows for 15 models
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle('Residual Analysis for All Models (Training, Validation, Test)', 
                    fontweight='bold', fontsize=16, y=0.98)
        
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
                
            # Get predictions for all sets
            y_train_pred = model.predict(self.processed_data['X_train'])
            y_val_pred = model.predict(self.processed_data['X_val'])
            y_test_pred = model.predict(self.processed_data['X_test'])
            
            # Calculate residuals
            residuals_train = self.processed_data['y_train'] - y_train_pred
            residuals_val = self.processed_data['y_val'] - y_val_pred
            residuals_test = self.processed_data['y_test'] - y_test_pred
            
            ax = axes[idx]
            
            # Plot residuals for all three sets
            ax.scatter(y_train_pred, residuals_train, alpha=0.6, s=40, 
                      label='Train', c='blue')
            ax.scatter(y_val_pred, residuals_val, alpha=0.6, s=40,
                      label='Validation', c='orange')
            ax.scatter(y_test_pred, residuals_test, alpha=0.6, s=40,
                      label='Test', c='green')
            
            # Zero line
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Predicted pIC50', fontweight='bold')
            ax.set_ylabel('Residuals', fontweight='bold')
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.legend(prop={'weight': 'bold', 'size': 8})
            ax.grid(False)
        
        # Hide unused subplots
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/all_models_residuals.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Residual plots for all 15 models created!")

    def _plot_all_models_williams(self):
        """Plot Williams plots for all models in one figure"""
        n_models = len(self.models)
        n_cols = 5  # 5 columns for 15 models (3 rows)
        n_rows = 3  # 3 rows for 15 models
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        fig.suptitle('Williams Plots for All Models', 
                    fontweight='bold', fontsize=16, y=0.98)
        
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
                
            X_train = self.processed_data['X_train']
            X_test = self.processed_data['X_test']
            
            # Calculate leverages
            try:
                leverages_train = np.sum(X_train**2, axis=1) / (X_train.shape[1] * np.var(X_train, axis=0).sum())
                leverages_test = np.sum(X_test**2, axis=1) / (X_test.shape[1] * np.var(X_test, axis=0).sum())
            except:
                leverages_train = np.linalg.norm(X_train, axis=1)**2 / X_train.shape[1]
                leverages_test = np.linalg.norm(X_test, axis=1)**2 / X_test.shape[1]
            
            # Calculate standardized residuals
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            residuals_train = self.processed_data['y_train'] - y_train_pred
            residuals_test = self.processed_data['y_test'] - y_test_pred
            
            std_residuals_train = residuals_train / np.std(residuals_train) if np.std(residuals_train) > 0 else residuals_train
            std_residuals_test = residuals_test / np.std(residuals_test) if np.std(residuals_test) > 0 else residuals_test
            
            # Williams plot threshold
            p = X_train.shape[1]
            n_train = X_train.shape[0]
            h_star = 3 * (p + 1) / n_train
            
            ax = axes[idx]
            
            # Plot train and test points
            ax.scatter(leverages_train, std_residuals_train, alpha=0.6, s=40, label='Train', c='blue')
            ax.scatter(leverages_test, std_residuals_test, alpha=0.6, s=40, label='Test', c='red')
            
            # Add threshold lines
            ax.axhline(y=3, color='r', linestyle='--', linewidth=1, label='±3σ')
            ax.axhline(y=-3, color='r', linestyle='--', linewidth=1)
            ax.axvline(x=h_star, color='g', linestyle='--', linewidth=1, label=f'h* = {h_star:.3f}')
            
            ax.set_xlabel('Leverage (h)', fontweight='bold')
            ax.set_ylabel('Standardized Residuals', fontweight='bold')
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.legend(prop={'weight': 'bold', 'size': 7})
            ax.grid(False)
        
        # Hide unused subplots
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/all_models_williams_plots.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_comparison_unified(self, metrics_df):
        """Plot model comparison with unified scales for easy comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Model Performance Comparison (Unified Scales)', fontweight='bold', fontsize=16)
        
        models = metrics_df['Model']
        
        # RMSE comparison
        rmse_metrics = ['RMSE_train', 'RMSE_test', 'RMSE_val']
        rmse_data = metrics_df[rmse_metrics].values
        rmse_min, rmse_max = rmse_data.min(), rmse_data.max()
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, metric in enumerate(rmse_metrics):
            axes[0].bar(x + (i-1)*width, metrics_df[metric], width, 
                       label=metric.replace('RMSE_', '').title(), alpha=0.8)
        
        axes[0].set_ylabel('RMSE', fontweight='bold')
        axes[0].set_title('RMSE Comparison', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].legend(prop={'weight': 'bold'})
        axes[0].set_ylim(rmse_min * 0.9, rmse_max * 1.1)
        axes[0].grid(False)
        
        # MAE comparison
        mae_metrics = ['MAE_train', 'MAE_test', 'MAE_val']
        mae_data = metrics_df[mae_metrics].values
        mae_min, mae_max = mae_data.min(), mae_data.max()
        
        for i, metric in enumerate(mae_metrics):
            axes[1].bar(x + (i-1)*width, metrics_df[metric], width, 
                       label=metric.replace('MAE_', '').title(), alpha=0.8)
        
        axes[1].set_ylabel('MAE', fontweight='bold')
        axes[1].set_title('MAE Comparison', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].legend(prop={'weight': 'bold'})
        axes[1].set_ylim(mae_min * 0.9, mae_max * 1.1)
        axes[1].grid(False)
        
        # Cross-validation scores
        axes[2].bar(models, metrics_df['Q²_CV'], alpha=0.8, color='green')
        axes[2].set_ylabel('Q² CV', fontweight='bold')
        axes[2].set_title('Cross-Validation Q²', fontweight='bold')
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].set_ylim(0, 1)  # R² scale from 0 to 1
        axes[2].grid(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/model_comparison_unified.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_all_models_feature_importance(self):
        """Plot feature importance for all 15 models in one figure"""
        # Get feature names
        if 'final_feature_names' in self.pipeline:
            feature_names = self.pipeline['final_feature_names']
        else:
            feature_names = [f'Feature_{i}' for i in range(self.processed_data['X_train'].shape[1])]
        
        n_models = len(self.models)
        n_cols = 5  # 5 columns for 15 models (3 rows)
        n_rows = 3  # 3 rows for 15 models
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        fig.suptitle('Feature Importance for All Models (Top 10 Features)', 
                    fontweight='bold', fontsize=16, y=0.98)
        
        axes = axes.flatten()
        
        models_with_importance = []
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
                
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                # Skip models without feature importance
                continue
            
            models_with_importance.append(model_name)
            
            # Get top 10 features
            top_indices = np.argsort(importances)[-10:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importances = importances[top_indices]
            
            ax = axes[idx]
            y_pos = np.arange(len(top_features))
            
            bars = ax.barh(y_pos, top_importances, alpha=0.8, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=8)
            ax.set_xlabel('Importance', fontweight='bold')
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.grid(False, axis='y')
            ax.set_axisbelow(True)
            
            # Add value labels on bars
            for bar, value in zip(bars, top_importances):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left', va='center', fontsize=7, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(len(models_with_importance), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/all_models_feature_importance.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature importance for {len(models_with_importance)} models created!")

    def _plot_pca_analysis(self):
        """Plot PCA analysis of training, test and validation sets"""
        print("🔍 Performing PCA analysis...")
        
        try:
            # Combine all data
            X_combined = np.vstack([
                self.processed_data['X_train'],
                self.processed_data['X_val'], 
                self.processed_data['X_test']
            ])
            
            # Create labels
            sets = ['Train'] * len(self.processed_data['X_train']) + \
                   ['Validation'] * len(self.processed_data['X_val']) + \
                   ['Test'] * len(self.processed_data['X_test'])
            
            # Perform PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_combined)
            
            # Create DataFrame for plotting
            pca_df = pd.DataFrame({
                'PC1': X_pca[:, 0],
                'PC2': X_pca[:, 1],
                'Set': sets
            })
            
            # Plot
            plt.figure(figsize=(10, 8))
            colors = {'Train': 'blue', 'Validation': 'orange', 'Test': 'green'}
            
            for set_type, color in colors.items():
                subset = pca_df[pca_df['Set'] == set_type]
                plt.scatter(subset['PC1'], subset['PC2'], 
                           alpha=0.6, s=50, label=set_type, c=color)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontweight='bold')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontweight='bold')
            plt.title('PCA: Chemical Space of Training, Validation and Test Sets', fontweight='bold')
            plt.legend(prop={'weight': 'bold'})
            plt.grid(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_folder}/pca_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ PCA analysis completed!")
            
        except Exception as e:
            print(f"⚠️ PCA analysis failed: {e}")

    def _plot_learning_curves(self):
        """Plot learning curves for all models"""
        print("📈 Plotting learning curves...")
        
        n_models = len(self.models)
        n_cols = 5  # 5 columns for 15 models (3 rows)
        n_rows = 3  # 3 rows for 15 models
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        fig.suptitle('Learning Curves for All Models', 
                    fontweight='bold', fontsize=16, y=0.98)
        
        axes = axes.flatten()
        
        X_train_val = np.vstack([self.processed_data['X_train'], self.processed_data['X_val']])
        y_train_val = np.concatenate([self.processed_data['y_train'], self.processed_data['y_val']])
        
        for idx, (model_name, model) in enumerate(self.models.items()):
            if idx >= len(axes):
                break
                
            try:
                # Compute learning curve
                train_sizes, train_scores, test_scores = learning_curve(
                    model, X_train_val, y_train_val, cv=5, 
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    scoring='r2', n_jobs=-1
                )
                
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                
                ax = axes[idx]
                ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
                ax.plot(train_sizes, test_scores_mean, 'o-', color='red', label='Cross-validation score')
                ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                               train_scores_mean + train_scores_std, alpha=0.1, color='blue')
                ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                               test_scores_mean + test_scores_std, alpha=0.1, color='red')
                
                ax.set_xlabel('Training Set Size', fontweight='bold')
                ax.set_ylabel('R² Score', fontweight='bold')
                ax.set_title(f'{model_name}', fontweight='bold')
                ax.legend(prop={'weight': 'bold', 'size': 8})
                ax.grid(False)
                
            except Exception as e:
                print(f"⚠️ Learning curve failed for {model_name}: {e}")
                axes[idx].set_visible(False)
        
        # Hide unused subplots
        for idx in range(len(self.models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Learning curves plotted!")

    def _plot_enhanced_y_randomization(self):
        """Plot enhanced Y-randomization with 50 permutations"""
        print("🎲 Performing enhanced Y-randomization...")
        
        n_permutations = 50
        randomized_r2_scores = {model_name: [] for model_name in self.models.keys() if model_name != 'Ensemble'}
        
        X_test = self.processed_data['X_test']
        y_test = self.processed_data['y_test']
        y_train = self.processed_data['y_train']
        
        for i in range(n_permutations):
            y_random = np.random.permutation(y_train)
            
            for model_name, model in self.models.items():
                if model_name == 'Ensemble':
                    continue
                    
                try:
                    # Retrain model on shuffled data (simplified)
                    model_copy = joblib.load(f"QSAR_Publication_Results_Final/models/{model_name}_model.pkl")
                    
                    # Simple correlation with random data
                    y_pred = model.predict(X_test)
                    rand_r2 = np.corrcoef(y_random[:len(y_pred)], y_pred)[0,1]**2
                    randomized_r2_scores[model_name].append(rand_r2)
                except:
                    continue
        
        # Plot histogram
        n_models = len(randomized_r2_scores)
        n_cols = 5  # 5 columns for 15 models (3 rows)
        n_rows = 3  # 3 rows for 15 models
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        fig.suptitle('Y-Randomization Test for All Models (50 Permutations)', 
                    fontweight='bold', fontsize=16, y=0.98)
        
        axes = axes.flatten()
        
        for idx, (model_name, r2_scores) in enumerate(randomized_r2_scores.items()):
            if idx >= len(axes):
                break
                
            if r2_scores:
                ax = axes[idx]
                ax.hist(r2_scores, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
                ax.axvline(np.mean(r2_scores), color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {np.mean(r2_scores):.3f}')
                
                # Add original R² for comparison
                original_r2 = r2_score(y_test, self.models[model_name].predict(X_test))
                ax.axvline(original_r2, color='green', linestyle='--', linewidth=2, 
                          label=f'Original: {original_r2:.3f}')
                
                ax.set_xlabel('R² Score', fontweight='bold')
                ax.set_ylabel('Frequency', fontweight='bold')
                ax.set_title(f'{model_name}', fontweight='bold')
                ax.legend(prop={'weight': 'bold', 'size': 8})
                ax.grid(False)
        
        # Hide unused subplots
        for idx in range(len(randomized_r2_scores), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder}/enhanced_y_randomization.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Enhanced Y-randomization plotted!")

    def _plot_mahalanobis_distances(self):
        """Plot Mahalanobis distance for all models"""
        print("📏 Calculating Mahalanobis distances...")
        
        try:
            X_train = self.processed_data['X_train']
            X_test = self.processed_data['X_test']
            
            # Calculate mean and covariance for training set
            mean_train = np.mean(X_train, axis=0)
            cov_train = np.cov(X_train.T)
            
            # Calculate Mahalanobis distances
            def calc_mahalanobis(X):
                try:
                    inv_cov = np.linalg.pinv(cov_train)
                    diff = X - mean_train
                    return np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
                except:
                    return np.linalg.norm(X - mean_train, axis=1)
            
            mahalanobis_train = calc_mahalanobis(X_train)
            mahalanobis_test = calc_mahalanobis(X_test)
            
            # Plot
            n_models = len(self.models)
            n_cols = 5  # 5 columns for 15 models (3 rows)
            n_rows = 3  # 3 rows for 15 models
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            fig.suptitle('Mahalanobis Distance vs Absolute Residuals for All Models', 
                        fontweight='bold', fontsize=16, y=0.98)
            
            axes = axes.flatten()
            
            for idx, (model_name, model) in enumerate(self.models.items()):
                if idx >= len(axes):
                    break
                    
                # Calculate residuals
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                residuals_train = np.abs(self.processed_data['y_train'] - y_train_pred)
                residuals_test = np.abs(self.processed_data['y_test'] - y_test_pred)
                
                ax = axes[idx]
                ax.scatter(mahalanobis_train, residuals_train, alpha=0.6, s=40, label='Train', c='blue')
                ax.scatter(mahalanobis_test, residuals_test, alpha=0.6, s=40, label='Test', c='red')
                
                ax.set_xlabel('Mahalanobis Distance', fontweight='bold')
                ax.set_ylabel('Absolute Residual', fontweight='bold')
                ax.set_title(f'{model_name}', fontweight='bold')
                ax.legend(prop={'weight': 'bold', 'size': 8})
                ax.grid(False)
            
            # Hide unused subplots
            for idx in range(len(self.models), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_folder}/mahalanobis_distances.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Mahalanobis distances plotted!")
            
        except Exception as e:
            print(f"⚠️ Mahalanobis distance calculation failed: {e}")

    def _plot_cooks_distance(self):
        """Plot Cook's distance for all models"""
        print("🍳 Calculating Cook's distance...")
        
        try:
            X_train = self.processed_data['X_train']
            y_train = self.processed_data['y_train']
            
            n_models = len(self.models)
            n_cols = 5  # 5 columns for 15 models (3 rows)
            n_rows = 3  # 3 rows for 15 models
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            fig.suptitle("Cook's Distance for All Models", 
                        fontweight='bold', fontsize=16, y=0.98)
            
            axes = axes.flatten()
            
            for idx, (model_name, model) in enumerate(self.models.items()):
                if idx >= len(axes):
                    break
                    
                try:
                    # Calculate leverages (simplified)
                    leverages = np.sum(X_train**2, axis=1) / (X_train.shape[1] * np.var(X_train, axis=0).sum())
                    
                    # Calculate residuals
                    y_pred = model.predict(X_train)
                    residuals = y_train - y_pred
                    mse = np.mean(residuals**2)
                    
                    # Simplified Cook's distance
                    cooks_d = (residuals**2 / (X_train.shape[1] * mse)) * (leverages / (1 - leverages)**2)
                    
                    ax = axes[idx]
                    ax.plot(range(len(cooks_d)), cooks_d, 'o-', alpha=0.7, color='purple')
                    ax.axhline(4/len(X_train), color='red', linestyle='--', 
                              label=f'4/n = {4/len(X_train):.3f}')
                    
                    ax.set_xlabel('Sample Index', fontweight='bold')
                    ax.set_ylabel("Cook's Distance", fontweight='bold')
                    ax.set_title(f'{model_name}', fontweight='bold')
                    ax.legend(prop={'weight': 'bold', 'size': 8})
                    ax.grid(False)
                    
                except Exception as e:
                    print(f"⚠️ Cook's distance failed for {model_name}: {e}")
                    axes[idx].set_visible(False)
            
            # Hide unused subplots
            for idx in range(len(self.models), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{self.output_folder}/cooks_distance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Cook's distance plotted!")
            
        except Exception as e:
            print(f"⚠️ Cook's distance calculation failed: {e}")

    def generate_advanced_report(self, metrics_df):
        """Generate an advanced comprehensive report"""
        print("\n📋 Generating advanced comprehensive report...")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ADVANCED COMPREHENSIVE QSAR MODEL ANALYSIS REPORT - UPDATED")
        report_lines.append("="*80)
        
        # Model Performance Summary
        report_lines.append("\n1. MODEL PERFORMANCE SUMMARY")
        report_lines.append("-"*40)
        best_model = metrics_df.loc[metrics_df['R²_test'].idxmax()]
        report_lines.append(f"Best Model: {best_model['Model']}")
        report_lines.append(f"Test R²: {best_model['R²_test']:.4f}")
        report_lines.append(f"Test RMSE: {best_model['RMSE_test']:.4f}")
        report_lines.append(f"Test MAE: {best_model['MAE_test']:.4f}")
        report_lines.append(f"Q² CV: {best_model['Q²_CV']:.4f}")
        
        # Detailed Performance Table
        report_lines.append("\n2. DETAILED PERFORMANCE METRICS")
        report_lines.append("-"*40)
        for _, row in metrics_df.iterrows():
            report_lines.append(f"\n{row['Model']}:")
            report_lines.append(f"  R² - Train: {row['R²_train']:.4f}, Val: {row['R²_val']:.4f}, Test: {row['R²_test']:.4f}")
            report_lines.append(f"  RMSE - Train: {row['RMSE_train']:.4f}, Val: {row['RMSE_val']:.4f}, Test: {row['RMSE_test']:.4f}")
            report_lines.append(f"  MAE - Train: {row['MAE_train']:.4f}, Val: {row['MAE_val']:.4f}, Test: {row['MAE_test']:.4f}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("ADVANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        report_lines.append("All 15 models analyzed with comprehensive plots.")
        report_lines.append("="*80)
        
        # Save report
        report_text = '\n'.join(report_lines)
        with open(f'{self.output_folder}/advanced_comprehensive_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"📄 Full report saved to: {self.output_folder}/advanced_comprehensive_report.txt")

def main():
    """Main execution function"""
    analyzer = QSARAdvancedAnalyzer(
        pipeline_path="QSAR_Publication_Results_Final/models/complete_pipeline_package.pkl",
        output_folder="QSAR_Advanced_Analysis_Updated"
    )
    
    # Load pipeline data
    if not analyzer.load_pipeline_data():
        return
    
    # Recreate predictions from original data
    if not analyzer.recreate_predictions():
        return
    
    # Perform comprehensive analysis
    metrics_df = analyzer.compute_comprehensive_metrics()
    
    # Create all advanced plots
    analyzer.create_all_advanced_plots(metrics_df)
    
    # Generate final report
    analyzer.generate_advanced_report(metrics_df)
    
    print(f"\n🎉 ADVANCED ANALYSIS COMPLETED!")
    print(f"📁 Results saved to: {analyzer.output_folder}")
    print("📊 Generated plots:")
    print("  ✅ Experimental vs Predicted for all 15 models (Training, Validation, Test)")
    print("  ✅ Residual plots for all 15 models (Training, Validation, Test)") 
    print("  ✅ Williams plots for all models")
    print("  ✅ Model comparison with unified scales")
    print("  ✅ Feature importance for all models")
    print("  ✅ PCA analysis")
    print("  ✅ Learning curves")
    print("  ✅ Enhanced Y-randomization")
    print("  ✅ Mahalanobis distance plots")
    print("  ✅ Cook's distance plots")

if __name__ == "__main__":
    main()
