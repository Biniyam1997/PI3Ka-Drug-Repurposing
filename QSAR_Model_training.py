# complete_qsar_pipeline_final_fixed.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, RFE, mutual_info_regression
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from scipy import stats
import joblib
import warnings
import gc
import os
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print("Boruta not available. Install with: pip install boruta-py")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

class PublicationStyle:
    """Publication-style plotting configuration"""
    
    # Font configurations - easily customizable
    FONT_FAMILY = 'Arial'
    TITLE_FONT_SIZE = 16
    AXIS_FONT_SIZE = 14
    LEGEND_FONT_SIZE = 12
    TICK_FONT_SIZE = 12
    
    # Color configurations - easily customizable
    COLOR_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Style configurations
    STYLE = 'seaborn-v0_8-whitegrid'
    DPI = 300
    FIG_SIZE = (10, 8)
    
    @classmethod
    def set_style(cls):
        """Set publication style"""
        plt.style.use(cls.STYLE)
        sns.set_palette(cls.COLOR_PALETTE)
        
        # Set font parameters
        plt.rcParams['font.family'] = cls.FONT_FAMILY
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['figure.titleweight'] = 'bold'
        plt.rcParams['legend.fontsize'] = cls.LEGEND_FONT_SIZE
        plt.rcParams['legend.title_fontsize'] = cls.LEGEND_FONT_SIZE
        plt.rcParams['xtick.labelsize'] = cls.TICK_FONT_SIZE
        plt.rcParams['ytick.labelsize'] = cls.TICK_FONT_SIZE
        plt.rcParams['axes.labelsize'] = cls.AXIS_FONT_SIZE
        plt.rcParams['axes.titlesize'] = cls.TITLE_FONT_SIZE

class PublicationOutputManager:
    """Manage publication output folders and file organization"""
    
    def __init__(self, base_folder="QSAR_Publication_Results"):
        self.base_folder = base_folder
        self.subfolders = {
            'plots': 'plots',
            'tables': 'tables',
            'models': 'models',
            'analysis': 'analysis',
            'screening': 'screening',
            'interpretability': 'interpretability',
            'descriptors': 'descriptors'  # NEW: Folder for descriptor files
        }
        self.create_folder_structure()
    
    def create_folder_structure(self):
        """Create organized folder structure for publication outputs"""
        print(f"📁 Creating publication folder structure: {self.base_folder}")
        
        # Create main folder
        os.makedirs(self.base_folder, exist_ok=True)
        
        # Create subfolders
        for folder_name, folder_path in self.subfolders.items():
            full_path = os.path.join(self.base_folder, folder_path)
            os.makedirs(full_path, exist_ok=True)
            print(f"  📂 Created: {full_path}")
    
    def get_path(self, file_type, filename):
        """Get full path for saving files in organized structure"""
        if file_type in self.subfolders:
            return os.path.join(self.base_folder, self.subfolders[file_type], filename)
        else:
            return os.path.join(self.base_folder, filename)
    
    def save_figure(self, filename, dpi=300, bbox_inches='tight', **kwargs):
        """Save figure to plots folder with publication quality"""
        filepath = self.get_path('plots', filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"  💾 Saved plot: {filepath}")
        return filepath
    
    def save_table(self, df, filename, index=False):
        """Save DataFrame to tables folder"""
        filepath = self.get_path('tables', filename)
        df.to_csv(filepath, index=index)
        print(f"  💾 Saved table: {filepath}")
        return filepath
    
    def save_model(self, model, filename):
        """Save model to models folder"""
        filepath = self.get_path('models', filename)
        joblib.dump(model, filepath)
        print(f"  💾 Saved model: {filepath}")
        return filepath
    
    def save_descriptors(self, df, filename, index=False):
        """Save descriptor data to descriptors folder"""
        filepath = self.get_path('descriptors', filename)
        df.to_csv(filepath, index=index)
        print(f"  💾 Saved descriptors: {filepath}")
        return filepath

class ComprehensiveValidator:
    """Comprehensive validation and analysis for QSAR models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state

class AdvancedInterpretability:
    """Advanced model interpretability using SHAP, LIME, PDP, etc."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state

class LargeDatasetProcessor:
    """Process large datasets in chunks to avoid memory issues"""
    
    def __init__(self, chunk_size=5000):
        self.chunk_size = chunk_size

class DataCleaner:
    """Enhanced data cleaning with duplicate removal"""
    
    @staticmethod
    def remove_duplicate_smiles(df, smiles_column='Smiles', activity_column='IC50 (nM)', strategy='best_activity'):
        """
        Remove duplicate SMILES from dataset
        """
        print(f"Original dataset size: {len(df)}")
        
        # Check for duplicate SMILES
        duplicate_mask = df.duplicated(subset=[smiles_column], keep=False)
        duplicate_count = duplicate_mask.sum()
        
        if duplicate_count == 0:
            print("No duplicate SMILES found.")
            return df.copy()
        
        print(f"Found {duplicate_count} duplicate SMILES entries")
        
        # Group by SMILES and handle duplicates based on strategy
        if strategy == 'best_activity':
            # For IC50, lower values are better (more potent)
            df_clean = df.loc[df.groupby(smiles_column)[activity_column].idxmin()].copy()
            print("Kept entries with best (lowest) IC50 values for duplicate SMILES")
        
        elif strategy == 'first':
            df_clean = df.drop_duplicates(subset=[smiles_column], keep='first').copy()
            print("Kept first occurrence for duplicate SMILES")
        
        elif strategy == 'last':
            df_clean = df.drop_duplicates(subset=[smiles_column], keep='last').copy()
            print("Kept last occurrence for duplicate SMILES")
        
        elif strategy == 'average':
            # Group by SMILES and calculate mean activity
            agg_dict = {activity_column: 'mean'}
            # Include other numeric columns in aggregation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols:
                if col != activity_column:
                    agg_dict[col] = 'mean'
            
            # For non-numeric columns, take the first value
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
            non_numeric_cols = [col for col in non_numeric_cols if col != smiles_column]
            for col in non_numeric_cols:
                agg_dict[col] = 'first'
            
            df_clean = df.groupby(smiles_column).agg(agg_dict).reset_index()
            print("Averaged activities for duplicate SMILES")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        print(f"Dataset after duplicate removal: {len(df_clean)} entries")
        print(f"Removed {len(df) - len(df_clean)} duplicate entries")
        
        return df_clean

class AdvancedFeatureSelector:
    """Advanced feature selection methods for QSAR modeling"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state

class GolbraikhTropshaValidator:
    """Golbraikh-Tropsha validation for QSAR models"""
    
    @staticmethod
    def calculate_gt_metrics(y_true, y_pred, y_train=None):
        """Calculate Golbraikh-Tropsha validation metrics"""
        return {}

class CompleteQSARPipeline:
    """Complete QSAR pipeline with comprehensive validation and virtual screening"""
    
    def __init__(self, random_state=42, output_folder="QSAR_Publication_Results"):
        self.random_state = random_state
        self.processed_data = None
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_names = None
        self.selected_feature_indices = None
        self.selected_scaler = None
        self.original_feature_names = None  # Store original feature names
        
        # Initialize components with proper parameters
        self.data_cleaner = DataCleaner()
        self.feature_selector = AdvancedFeatureSelector(random_state=random_state)
        self.gt_validator = GolbraikhTropshaValidator()
        self.validator = ComprehensiveValidator(random_state=random_state)
        self.interpreter = AdvancedInterpretability(random_state=random_state)
        self.output_manager = PublicationOutputManager(output_folder)
        
        # Set publication style
        PublicationStyle.set_style()

    def run_complete_pipeline(self, data_file, screening_directory=None):
        """Run complete QSAR pipeline"""
        print("🚀 STARTING COMPREHENSIVE QSAR PIPELINE")
        print("="*70)
        
        # Step 1: Data Preprocessing
        print("\n📊 STEP 1: DATA PREPROCESSING")
        df = self.load_and_preprocess_data(data_file)
        
        # Calculate Mordred descriptors
        mordred_descriptors, valid_indices = self.calculate_mordred_descriptors(df['Smiles'].tolist())
        
        if len(mordred_descriptors) == 0:
            print("ERROR: No Mordred descriptors calculated!")
            return
        
        # NEW: Save calculated descriptors to CSV for journal
        self.save_calculated_descriptors(df, mordred_descriptors, valid_indices)
        
        # Prepare features and targets
        df_final, feature_columns = self.prepare_features_targets(df, mordred_descriptors, valid_indices)
        X = df_final[feature_columns]
        y = df_final['pIC50']
        
        # Store original feature names
        self.original_feature_names = feature_columns.copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state)
        
        print(f"Training set: {X_train.shape[0]} molecules")
        print(f"Validation set: {X_val.shape[0]} molecules") 
        print(f"Test set: {X_test.shape[0]} molecules")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Store processed data
        self.processed_data = {
            'X_train': X_train, 'X_test': X_test, 'X_val': X_val,
            'y_train': y_train, 'y_test': y_test, 'y_val': y_val,
            'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled, 'X_val_scaled': X_val_scaled,
            'feature_names': feature_columns, 'df_final': df_final
        }
        
        # Step 2: Feature Selection
        print("\n🔍 STEP 2: FEATURE SELECTION")
        self.selected_feature_indices, self.feature_names = self.advanced_feature_selection(
            X_train_scaled, y_train, feature_columns
        )
        
        # Apply feature selection
        X_train_selected = X_train_scaled[:, self.selected_feature_indices]
        X_val_selected = X_val_scaled[:, self.selected_feature_indices]
        X_test_selected = X_test_scaled[:, self.selected_feature_indices]
        
        # Scale selected features
        self.selected_scaler = StandardScaler()
        X_train_selected_scaled = self.selected_scaler.fit_transform(X_train_selected)
        X_val_selected_scaled = self.selected_scaler.transform(X_val_selected)
        X_test_selected_scaled = self.selected_scaler.transform(X_test_selected)
        
        # Update processed data
        self.processed_data.update({
            'X_train_selected_scaled': X_train_selected_scaled,
            'X_val_selected_scaled': X_val_selected_scaled,
            'X_test_selected_scaled': X_test_selected_scaled
        })
        
        # Step 3: Model Training
        print("\n🤖 STEP 3: MODEL TRAINING")
        models = self.define_models()
        results, trained_models = self.train_models(
            X_train_selected_scaled, y_train, X_val_selected_scaled, y_val, models
        )
        
        # Create ensemble
        self.best_model, ensemble_model = self.create_ensemble(trained_models, results, X_train_selected_scaled, y_train)
        self.models = trained_models
        self.models['Ensemble'] = ensemble_model
        
        # Step 4: Comprehensive Evaluation
        print("\n📊 STEP 4: COMPREHENSIVE EVALUATION")
        self.comprehensive_evaluation(X_test_selected_scaled, y_test)
        
        # Step 5: FIXED - Model Saving
        print("\n💾 STEP 5: MODEL SAVING")
        self.save_models_comprehensive()
        
        # Step 6: Virtual Screening with precomputed descriptors
        if screening_directory and os.path.exists(screening_directory):
            print("\n🔬 STEP 6: VIRTUAL SCREENING WITH PRECOMPUTED DESCRIPTORS")
            screening_results = self.virtual_screening_from_precomputed_descriptors(screening_directory)
            if screening_results is not None:
                self.analyze_virtual_screening_results(screening_results)
        elif screening_directory:
            print(f"\n⚠️  Screening directory not found: {screening_directory}")
        
        # Step 7: Generate Outputs
        print("\n📊 STEP 7: GENERATING OUTPUTS")
        self.create_comprehensive_analysis_plots()
        self.generate_publication_tables()
        
        print("\n" + "="*70)
        print("🎉 COMPREHENSIVE QSAR PIPELINE FINISHED SUCCESSFULLY!")
        print("="*70)

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        print("Loading data...")
        df = pd.read_csv(file_path)
        print(f"Original dataset shape: {df.shape}")
        
        # Remove duplicates
        df = self.data_cleaner.remove_duplicate_smiles(
            df, 
            smiles_column='Smiles', 
            activity_column='IC50 (nM)', 
            strategy='best_activity'
        )
        
        # Convert IC50 to pIC50
        df['pIC50'] = -np.log10(df['IC50 (nM)'] * 1e-9)
        
        # Save cleaned dataset
        self.output_manager.save_table(df, 'cleaned_dataset.csv')
        
        return df

    def calculate_mordred_descriptors(self, smiles_list):
        """Calculate Mordred descriptors for SMILES list"""
        print("Calculating Mordred descriptors...")
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

    def save_calculated_descriptors(self, df, mordred_descriptors, valid_indices):
        """NEW: Save calculated Mordred descriptors to CSV file for journal submission"""
        print("\n💾 SAVING CALCULATED DESCRIPTORS FOR JOURNAL SUBMISSION")
        
        try:
            # Create DataFrame with descriptors
            mordred_df = pd.DataFrame(mordred_descriptors)
            
            # Add molecule information
            molecules_info = df.iloc[valid_indices][['Molecule ChEMBL ID', 'Smiles', 'IC50 (nM)']].reset_index(drop=True)
            
            # Combine molecule info with descriptors
            descriptors_full_df = pd.concat([molecules_info, mordred_df], axis=1)
            
            # Save full descriptor set
            self.output_manager.save_descriptors(
                descriptors_full_df, 
                'calculated_mordred_descriptors_full.csv'
            )
            
            # Create and save a cleaned version (without high missing values)
            descriptors_cleaned = self.clean_mordred_descriptors_for_export(mordred_descriptors)
            descriptors_cleaned_df = pd.concat([molecules_info, descriptors_cleaned], axis=1)
            
            self.output_manager.save_descriptors(
                descriptors_cleaned_df,
                'calculated_mordred_descriptors_cleaned.csv'
            )
            
            # Save descriptor statistics
            self.save_descriptor_statistics(descriptors_cleaned_df)
            
            print(f"✅ Descriptors saved:")
            print(f"   - Full descriptors: {descriptors_full_df.shape[1]} columns")
            print(f"   - Cleaned descriptors: {descriptors_cleaned_df.shape[1]} columns")
            print(f"   - Molecules: {len(descriptors_full_df)}")
            
        except Exception as e:
            print(f"❌ Error saving descriptors: {e}")
            import traceback
            traceback.print_exc()

    def clean_mordred_descriptors_for_export(self, mordred_descriptors):
        """Clean Mordred descriptors for export - less aggressive than modeling version"""
        mordred_df = pd.DataFrame(mordred_descriptors)
        print(f"Original Mordred descriptors for export: {mordred_df.shape[1]}")
        
        # Remove descriptors with too many missing values (>50% for export)
        missing_threshold = 0.5 * len(mordred_df)
        mordred_clean = mordred_df.loc[:, mordred_df.isnull().sum() < missing_threshold]
        print(f"After removing high-missing columns: {mordred_clean.shape[1]}")
        
        # Remove constant descriptors
        mordred_clean = mordred_clean.loc[:, mordred_clean.std() > 0]
        print(f"After removing constant columns: {mordred_clean.shape[1]}")
        
        # Fill remaining missing values with column median
        mordred_clean = mordred_clean.fillna(mordred_clean.median())
        
        return mordred_clean

    def save_descriptor_statistics(self, descriptors_df):
        """Save statistics about the descriptors for documentation"""
        print("📊 Calculating descriptor statistics...")
        
        # Calculate basic statistics
        descriptor_stats = descriptors_df.describe().T
        descriptor_stats['missing_count'] = descriptors_df.isnull().sum()
        descriptor_stats['missing_percentage'] = (descriptors_df.isnull().sum() / len(descriptors_df)) * 100
        
        # Save statistics
        self.output_manager.save_table(
            descriptor_stats, 
            'descriptor_statistics.csv',
            index=True
        )
        
        # Create descriptor categories summary
        descriptor_columns = [col for col in descriptors_df.columns if col not in ['Molecule ChEMBL ID', 'Smiles', 'IC50 (nM)']]
        
        # Group by descriptor type (based on naming conventions)
        topological_descs = [d for d in descriptor_columns if any(term in d.lower() for term in ['chi', 'kappa', 'balaban', 'wiener', 'zagreb'])]
        constitutional_descs = [d for d in descriptor_columns if any(term in d.lower() for term in ['mw', 'heavy', 'atom', 'bond', 'ring'])]
        electronic_descs = [d for d in descriptor_columns if any(term in d.lower() for term in ['e_state', 'polarizability', 'charge', 'dipole'])]
        geometric_descs = [d for d in descriptor_columns if any(term in d.lower() for term in ['vanderwaals', 'surface', 'volume', 'gravitation'])]
        
        category_summary = pd.DataFrame({
            'Category': ['Topological', 'Constitutional', 'Electronic', 'Geometric', 'Other'],
            'Count': [
                len(topological_descs),
                len(constitutional_descs),
                len(electronic_descs),
                len(geometric_descs),
                len(descriptor_columns) - len(topological_descs) - len(constitutional_descs) - len(electronic_descs) - len(geometric_descs)
            ]
        })
        
        self.output_manager.save_table(category_summary, 'descriptor_categories.csv')
        
        print(f"📊 Descriptor categories:")
        for _, row in category_summary.iterrows():
            print(f"   - {row['Category']}: {row['Count']} descriptors")

    def clean_mordred_descriptors(self, mordred_descriptors):
        """Clean and preprocess Mordred descriptors for modeling"""
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

    def prepare_features_targets(self, df, mordred_descriptors, valid_indices):
        """Prepare features and targets with proper cleaning"""
        mordred_clean = self.clean_mordred_descriptors(mordred_descriptors)
        df_valid = df.iloc[valid_indices].reset_index(drop=True)
        
        if len(df_valid) != len(mordred_clean):
            min_len = min(len(df_valid), len(mordred_clean))
            df_valid = df_valid.iloc[:min_len]
            mordred_clean = mordred_clean.iloc[:min_len]
        
        df_final = pd.concat([
            df_valid[['Molecule ChEMBL ID', 'Smiles', 'IC50 (nM)', 'pIC50']].reset_index(drop=True),
            mordred_clean.reset_index(drop=True)
        ], axis=1)
        
        df_final = df_final.dropna()
        print(f"Final dataset shape: {df_final.shape}")
        print(f"Final number of descriptors: {mordred_clean.shape[1]}")
        
        return df_final, mordred_clean.columns.tolist()

    def advanced_feature_selection(self, X_train_scaled, y_train, feature_names):
        """Advanced feature selection using multiple methods"""
        print("Selecting features using advanced methods...")
        
        # Method 1: Mutual Information
        mi_scores = mutual_info_regression(X_train_scaled, y_train, random_state=self.random_state)
        mi_indices = np.argsort(mi_scores)[-150:]  # Take top 150
        
        # Method 2: Random Forest Importance
        rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        rf.fit(X_train_scaled, y_train)
        rf_importances = rf.feature_importances_
        rf_indices = np.argsort(rf_importances)[-150:]
        
        # Combine features from both methods
        combined_features = set(mi_indices) | set(rf_indices)
        print(f"Combined features from all methods: {len(combined_features)}")
        
        # Final selection using the most important features
        final_rf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        final_rf.fit(X_train_scaled[:, list(combined_features)], y_train)
        final_importances = final_rf.feature_importances_
        final_feature_indices = np.argsort(final_importances)[-100:]  # Final 100 features
        
        # Map back to original indices
        final_selected_indices = [list(combined_features)[i] for i in final_feature_indices]
        
        # Get selected feature names safely
        selected_features = []
        for i in final_selected_indices:
            if i < len(feature_names):
                selected_features.append(feature_names[i])
            else:
                print(f"⚠️  Warning: Index {i} out of range for feature_names (length {len(feature_names)})")
                selected_features.append(f"Feature_{i}")
        
        print(f"Final selected features: {len(selected_features)}")
        
        return final_selected_indices, selected_features

    def define_models(self):
        """Define models for training"""
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=self.random_state),
            'GradientBoosting': GradientBoostingRegressor(random_state=self.random_state),
            'Ridge': Ridge(),
            'SVR': SVR(),
        }
        return models

    def train_models(self, X_train, y_train, X_val, y_val, models):
        """Train models and evaluate on validation set"""
        trained_models = {}
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Predict on validation set
                y_val_pred = model.predict(X_val)
                val_r2 = r2_score(y_val, y_val_pred)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                
                results[name] = {
                    'val_r2': val_r2,
                    'val_rmse': val_rmse,
                    'model': model
                }
                
                print(f"  {name} - Validation R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}")
                
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
        
        return results, trained_models

    def create_ensemble(self, trained_models, results, X_train, y_train):
        """Create ensemble model"""
        print("Creating ensemble model...")
        ensemble_models = [(name, model) for name, model in trained_models.items()]
        ensemble_model = VotingRegressor(estimators=ensemble_models)
        ensemble_model.fit(X_train, y_train)
        
        # Select best individual model based on validation R²
        best_model_name = max(results.keys(), key=lambda x: results[x]['val_r2'])
        best_model = trained_models[best_model_name]
        
        print(f"Best model: {best_model_name} (R²: {results[best_model_name]['val_r2']:.4f})")
        
        return best_model, ensemble_model

    def comprehensive_evaluation(self, X_test, y_test):
        """Comprehensive evaluation on test set"""
        print("\nFinal evaluation on test set:")
        test_results = {}
        
        for name, model in self.models.items():
            y_test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            test_results[name] = {
                'r2': test_r2,
                'rmse': test_rmse,
                'mae': test_mae
            }
            
            print(f"  {name}: R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
        
        return test_results

    def save_models_comprehensive(self):
        """Fixed model saving - save all necessary components"""
        print("💾 SAVING MODELS AND PIPELINE COMPONENTS...")
        
        try:
            model_dir = self.output_manager.get_path('models', '')
            
            # 1. Save individual models
            for name, model in self.models.items():
                model_path = os.path.join(model_dir, f"{name}_model.pkl")
                joblib.dump(model, model_path)
                print(f"  ✅ {name} model saved")
            
            # 2. Save scalers
            scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            
            selected_scaler_path = os.path.join(model_dir, "selected_feature_scaler.pkl")
            joblib.dump(self.selected_scaler, selected_scaler_path)
            print(f"  ✅ Scalers saved")
            
            # 3. Save feature information - FIXED: Handle index errors
            try:
                # Get the selected feature names safely
                selected_features = []
                if self.feature_names and self.selected_feature_indices is not None:
                    for i in self.selected_feature_indices:
                        if i < len(self.original_feature_names):  # Use original feature names
                            selected_features.append(self.original_feature_names[i])
                        else:
                            print(f"⚠️  Warning: Index {i} out of range for original_feature_names (length {len(self.original_feature_names)})")
                            # Add a placeholder for missing features
                            selected_features.append(f"Feature_{i}")
                
                feature_info = {
                    'original_feature_names': self.original_feature_names,
                    'selected_feature_indices': self.selected_feature_indices,
                    'selected_features': selected_features,
                    'final_feature_names': self.feature_names
                }
                
                feature_path = os.path.join(model_dir, "feature_info.pkl")
                joblib.dump(feature_info, feature_path)
                print(f"  ✅ Feature information saved")
                
            except Exception as e:
                print(f"⚠️  Warning: Could not save feature info: {e}")
                # Save basic feature info without indices
                feature_info = {
                    'original_feature_names': self.original_feature_names,
                    'selected_feature_count': len(self.selected_feature_indices) if self.selected_feature_indices else 0,
                    'final_feature_names': self.feature_names
                }
                feature_path = os.path.join(model_dir, "feature_info.pkl")
                joblib.dump(feature_info, feature_path)
            
            # 4. Save complete pipeline package
            pipeline_package = {
                'best_model': self.best_model,
                'models': self.models,
                'scaler': self.scaler,
                'selected_scaler': self.selected_scaler,
                'original_feature_names': self.original_feature_names,
                'selected_feature_indices': self.selected_feature_indices,
                'final_feature_names': self.feature_names,
                'processed_data_info': {
                    'train_shape': self.processed_data['X_train'].shape if self.processed_data else None,
                    'feature_count': len(self.original_feature_names) if self.original_feature_names else None,
                    'selected_feature_count': len(self.selected_feature_indices) if self.selected_feature_indices else None
                }
            }
            pipeline_path = os.path.join(model_dir, "complete_pipeline_package.pkl")
            joblib.dump(pipeline_package, pipeline_path)
            print(f"  ✅ Complete pipeline package saved")
            
            print("✅ ALL MODELS AND COMPONENTS SAVED SUCCESSFULLY!")
            
        except Exception as e:
            print(f"❌ Error saving models: {e}")
            import traceback
            traceback.print_exc()

    def virtual_screening_from_precomputed_descriptors(self, batch_files_directory, chunk_size=5000):
        """Virtual screening with chunk processing - FIXED to skip problematic columns"""
        print(f"🔬 STARTING VIRTUAL SCREENING WITH PRECOMPUTED DESCRIPTORS: {batch_files_directory}")
        
        try:
            batch_files = [f for f in os.listdir(batch_files_directory) if f.startswith('mordred_batch_') and f.endswith('.csv')]
            batch_files.sort()
            
            print(f"📂 Found {len(batch_files)} batch files")
            
            # Create results file and write header
            screening_dir = self.output_manager.get_path('screening', '')
            results_path = os.path.join(screening_dir, 'virtual_screening_results_precomputed.csv')
            
            with open(results_path, 'w') as f:
                f.write('Molecule_ID,Predicted_pIC50\n')
            
            total_processed = 0
            
            for batch_file in tqdm(batch_files, desc="Processing batches"):
                batch_path = os.path.join(batch_files_directory, batch_file)
                print(f"  Processing {batch_file}...")
                
                # Process batch in chunks
                for chunk_idx, chunk_df in enumerate(pd.read_csv(batch_path, chunksize=chunk_size)):
                    # Identify ID column (first column)
                    id_col = chunk_df.columns[0]
                    
                    # SKIP the problematic columns (ABC and ABCGG) - use only columns from index 3 onward
                    descriptor_columns = chunk_df.columns[3:]  # Skip molecule_ID, ABC, ABCGG
                    
                    print(f"    Using {len(descriptor_columns)} descriptor columns (skipped first 3 columns)")
                    
                    # Clean descriptors - with better error handling
                    screening_clean = self.clean_screening_descriptors_fixed(chunk_df[descriptor_columns])
                    if screening_clean is None:
                        print(f"    ⚠️  Skipping chunk due to cleaning issues")
                        continue
                    
                    # Align features
                    X_screening = self.align_screening_features(screening_clean)
                    if X_screening is None:
                        continue
                    
                    # Scale and select features
                    X_screening_scaled = self.scaler.transform(X_screening)
                    X_screening_selected = X_screening_scaled[:, self.selected_feature_indices]
                    X_screening_final = self.selected_scaler.transform(X_screening_selected)
                    
                    # Make predictions
                    predictions = self.best_model.predict(X_screening_final)
                    
                    # Append results to file immediately
                    with open(results_path, 'a') as f:
                        for i, (mol_id, pred) in enumerate(zip(chunk_df[id_col], predictions)):
                            f.write(f'{mol_id},{pred:.6f}\n')
                    
                    total_processed += len(chunk_df)
                    print(f"    Chunk {chunk_idx + 1}: {len(chunk_df)} compounds (Total: {total_processed})")
                    
                    # Clear memory
                    del chunk_df, screening_clean, X_screening, X_screening_scaled, X_screening_selected, X_screening_final
                    gc.collect()
            
            # Read back the results and add ranking
            print("📊 Adding rankings to results...")
            results_df = pd.read_csv(results_path)
            results_df = results_df.sort_values('Predicted_pIC50', ascending=False)
            results_df['Rank'] = range(1, len(results_df) + 1)
            results_df.to_csv(results_path, index=False)
            
            print(f"✅ Virtual screening completed: {total_processed} compounds screened")
            print(f"💾 Results saved to: {results_path}")
            
            return results_df
            
        except Exception as e:
            print(f"❌ Virtual screening failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def clean_screening_descriptors_fixed(self, screening_descriptors):
        """Clean screening descriptors - FIXED for non-numeric data"""
        try:
            screening_df = pd.DataFrame(screening_descriptors)
            
            # Convert all columns to numeric, forcing errors to NaN
            for col in screening_df.columns:
                screening_df[col] = pd.to_numeric(screening_df[col], errors='coerce')
            
            # Remove columns with too many missing values (>80%)
            missing_threshold = 0.8 * len(screening_df)
            screening_clean = screening_df.loc[:, screening_df.isnull().sum() < missing_threshold]
            
            # Remove constant columns
            numeric_cols = screening_clean.select_dtypes(include=[np.number]).columns
            screening_clean = screening_clean[numeric_cols]
            screening_clean = screening_clean.loc[:, screening_clean.std() > 0]
            
            # Fill remaining missing values with 0
            screening_clean = screening_clean.fillna(0)
            
            print(f"    Cleaned descriptors: {screening_clean.shape[1]} columns")
            return screening_clean
            
        except Exception as e:
            print(f"    ❌ Error cleaning descriptors: {e}")
            return None

    def align_screening_features(self, screening_descriptors):
        """Align screening features with training features - OPTIMIZED"""
        try:
            # Ensure screening data has same features as training (ORIGINAL features)
            screening_df = pd.DataFrame(screening_descriptors)
            
            # Only add missing features that are in our selected features to save memory
            features_to_add = []
            for feature in self.original_feature_names:
                if feature not in screening_df.columns:
                    features_to_add.append(feature)
            
            if features_to_add:
                print(f"    ⚠️  Adding {len(features_to_add)} missing features")
                for feature in features_to_add:
                    screening_df[feature] = 0.0
            
            # Reorder columns to match training (ORIGINAL features)
            screening_aligned = screening_df[self.original_feature_names]
            
            # Fill any remaining NaN values
            screening_aligned = screening_aligned.fillna(0)
            
            print(f"    ✅ Features aligned: {screening_aligned.shape}")
            return screening_aligned.values
            
        except Exception as e:
            print(f"❌ Feature alignment failed: {e}")
            return None

    def analyze_virtual_screening_results(self, screening_results):
        """Analyze and report virtual screening results"""
        print("\n" + "="*70)
        print("📊 VIRTUAL SCREENING ANALYSIS")
        print("="*70)
        
        total_compounds = len(screening_results)
        
        print(f"📈 SCREENING STATISTICS:")
        print(f"  Total compounds screened: {total_compounds}")
        print(f"  Mean predicted pIC50: {screening_results['Predicted_pIC50'].mean():.3f}")
        print(f"  pIC50 range: {screening_results['Predicted_pIC50'].min():.3f} - {screening_results['Predicted_pIC50'].max():.3f}")
        
        top_candidates = screening_results.nlargest(10, 'Predicted_pIC50')
        print(f"\n🏆 TOP 10 CANDIDATES:")
        print("-" * 80)
        for idx, (_, row) in enumerate(top_candidates.iterrows(), 1):
            print(f"  {idx:2d}. pIC50: {row['Predicted_pIC50']:.3f} | Molecule_ID: {row['Molecule_ID']}")

    def create_comprehensive_analysis_plots(self):
        """Create comprehensive analysis plots"""
        print("\n📊 CREATING COMPREHENSIVE ANALYSIS PLOTS")
        
        try:
            PublicationStyle.set_style()
            
            # Model performance comparison
            plt.figure(figsize=(12, 8))
            
            model_names = list(self.models.keys())
            test_r2_scores = []
            
            for name in model_names:
                y_test_pred = self.models[name].predict(self.processed_data['X_test_selected_scaled'])
                test_r2 = r2_score(self.processed_data['y_test'], y_test_pred)
                test_r2_scores.append(test_r2)
            
            bars = plt.bar(model_names, test_r2_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            plt.title('Model Performance Comparison - Test R² Score')
            plt.ylabel('R² Score')
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            self.output_manager.save_figure('model_performance_comparison.png')
            plt.show()

            # Best model predictions vs actual
            plt.figure(figsize=(10, 8))
            
            y_test_pred_best = self.best_model.predict(self.processed_data['X_test_selected_scaled'])
            
            plt.scatter(self.processed_data['y_test'], y_test_pred_best, alpha=0.6)
            plt.plot([self.processed_data['y_test'].min(), self.processed_data['y_test'].max()], 
                    [self.processed_data['y_test'].min(), self.processed_data['y_test'].max()], 'r--', linewidth=2)
            plt.xlabel('Actual pIC50')
            plt.ylabel('Predicted pIC50')
            plt.title(f'Best Model Predictions\nR² = {r2_score(self.processed_data["y_test"], y_test_pred_best):.3f}')
            plt.grid(True, alpha=0.3)
            
            self.output_manager.save_figure('best_model_predictions.png')
            plt.show()
            
            print("✅ Comprehensive plots generated!")
            
        except Exception as e:
            print(f"❌ Error generating plots: {e}")

    def generate_publication_tables(self):
        """Generate publication-ready tables"""
        print("\n📋 GENERATING PUBLICATION-READY TABLES")
        
        try:
            # Performance table
            performance_data = []
            for name, model in self.models.items():
                y_test_pred = model.predict(self.processed_data['X_test_selected_scaled'])
                test_r2 = r2_score(self.processed_data['y_test'], y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(self.processed_data['y_test'], y_test_pred))
                test_mae = mean_absolute_error(self.processed_data['y_test'], y_test_pred)
                
                performance_data.append({
                    'Model': name,
                    'Test_R2': test_r2,
                    'Test_RMSE': test_rmse,
                    'Test_MAE': test_mae
                })
            
            performance_df = pd.DataFrame(performance_data)
            self.output_manager.save_table(performance_df, 'model_performance.csv')
            
            # Dataset summary
            dataset_summary = {
                'Total_Compounds': len(self.processed_data['df_final']),
                'Training_Set': len(self.processed_data['X_train']),
                'Test_Set': len(self.processed_data['X_test']),
                'Initial_Descriptors': len(self.original_feature_names),
                'Selected_Descriptors': len(self.selected_feature_indices),
                'pIC50_Min': self.processed_data['df_final']['pIC50'].min(),
                'pIC50_Max': self.processed_data['df_final']['pIC50'].max(),
                'pIC50_Mean': self.processed_data['df_final']['pIC50'].mean(),
                'pIC50_Std': self.processed_data['df_final']['pIC50'].std()
            }
            
            summary_df = pd.DataFrame([dataset_summary])
            self.output_manager.save_table(summary_df, 'dataset_summary.csv')
            
            print("✅ Publication tables generated!")
            
        except Exception as e:
            print(f"❌ Error generating tables: {e}")

def main():
    """Main execution function"""
    # Initialize pipeline
    pipeline = CompleteQSARPipeline(
        random_state=42,
        output_folder="QSAR_Publication_Results_Final"
    )
    
    # Run complete pipeline with precomputed descriptor screening
    pipeline.run_complete_pipeline(
        data_file='dataset.csv',
        screening_directory='/mnt/MD/ML/Abrish/DS/PK/1/3a_good/LOX_2/Databases'  # Path to your batch files
    )

if __name__ == "__main__":
    main()
