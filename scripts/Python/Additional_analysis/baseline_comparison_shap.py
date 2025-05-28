# This script performs a comparative analysis of different feature sets (ART, TRT, combined)
# for classifying plant drought stress based on root system architecture data.
# It evaluates models using Random Forest, calculates feature importance using SHAP,
# performs Recursive Feature Elimination (RFE), and generates various visualizations
# and reports for biological interpretation and model performance comparison.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
from matplotlib.patches import Circle, ConnectionPatch, Patch

# --- Added for RFE ---
from sklearn.feature_selection import RFE
# --------------------

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_OUTPUT_DIR = r'C:\Users\ms\Desktop\data\output\rf\baseline' # Changed dir for new version
SHAP_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'shap_analysis')
RFE_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, 'rfe_analysis') # New dir for RFE results

try:
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)
    os.makedirs(RFE_OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directories created/ensured: {BASE_OUTPUT_DIR}, {SHAP_OUTPUT_DIR}, {RFE_OUTPUT_DIR}")
except Exception as e:
    logger.error(f"Failed to create output directories: {e}")
    raise

ART_PATH = r'C:\Users\ms\Desktop\data\ART_A_B.csv'
TRT_PATH = r'C:\Users\ms\Desktop\data\TRT_A_B.csv'
COMBINED_PATH = r'C:\Users\ms\Desktop\data\Combine_A_B.csv'

ART_PREFIXES = sorted(['DBSCAN', 'Density', 'FCM', 'GMM', 'HDBSCAN', 'K-mean', 'Mean-shift', 'Mean_shift', 'OPTICS', 'SLIC'], key=len, reverse=True)
BIO_INTERPRETATIONS = {
    'DBSCAN_density_points': 'Root cluster density representing resource allocation strategy',
    'DBSCAN_center': 'Spatial position of density-based clusters',
    'HDBSCAN_density_points': 'Hierarchical density clusters reflecting adaptive root organization',
    'HDBSCAN_centre_y': 'Vertical distribution depth, indicates deep water acquisition behavior',
    'HDBSCAN_centre_x': 'Horizontal distribution of root clusters',
    'FCM_center_x': 'Horizontal position of fuzzy root clusters, relates to lateral spread / soil exploration strategy',
    'FCM_density_points': 'Fuzzy cluster density showing gradual root density transitions',
    'K-mean_density_points': 'Primary root mass concentration, showing investment in main roots',
    'Mean-shift_centre_y': 'Vertical position of density peaks, reveals rooting depth preferences',
    'Mean-shift_density_points': 'Mode-seeking cluster size, indicates natural root grouping patterns',
    'OPTICS_density_points': 'Ordered clusters density, showing adaptive growth patterns',
    'GMM_centre_y': 'Vertical position of probabilistic root clusters, indicating depth targeting',
    'Density_centre_y': 'Root mass vertical center, reflecting overall rooting depth preference',
    'SLIC_density_points': 'Superpixel segmentation density, indicating compact root regions',
    'SLIC_num_superpixels': 'Root system segmentation complexity (SLIC related), potentially branching'
}
SPECIFIC_DROUGHT_MECHANISMS = {
    'HDBSCAN_centre_y': 'Deep rooting for lower soil water',
    'Mean-shift_centre_y': 'Vertical root distribution for drought',
    'K-mean_density_points': 'Root biomass allocation for water uptake',
    'DBSCAN_density_points': 'Adaptive clustering in moisture pockets',
    'FCM_center_x': 'Lateral exploration for soil volume coverage',
    'GMM_centre_y': 'Probabilistic targeting of deep moisture',
    'OPTICS_density_points': 'Sequential root organization for uptake',
    'Density_centre_y': 'Strategic positioning for water acquisition'
}
GENERIC_MECHANISM_VERTICAL = 'Generic: Vertical root distribution'
GENERIC_MECHANISM_LATERAL = 'Generic: Lateral root exploration'
GENERIC_MECHANISM_DENSITY = 'Generic: Root cluster density optimization'
GENERIC_MECHANISM_GENERAL_ADAPTATION = 'General root architecture adaptation'
MECHANISM_COLORS = {
    'Deep rooting for lower soil water': '#006400',
    'Vertical root distribution for drought': '#228B22',
    'Root biomass allocation for water uptake': '#3CB371',
    'Adaptive clustering in moisture pockets': '#48D1CC',
    'Lateral exploration for soil volume coverage': '#20B2AA',
    'Probabilistic targeting of deep moisture': '#87CEEB',
    'Sequential root organization for uptake': '#6A5ACD',
    'Strategic positioning for water acquisition': '#7B68EE',
    GENERIC_MECHANISM_VERTICAL: '#556B2F',
    GENERIC_MECHANISM_LATERAL: '#8FBC8F',
    GENERIC_MECHANISM_DENSITY: '#008080',
    GENERIC_MECHANISM_GENERAL_ADAPTATION: '#A9A9A9',
    'Other': '#D3D3D3',
    'Other Combined Mechanisms': '#BEBEBE' # Added for grouped mechanisms
}
FIG_SIZE_LARGE = (14, 8)
FIG_SIZE_MEDIUM = (10, 7)
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
# Set the seed for reproducibility at the module level for global effects
np.random.seed(42)
# Green-themed color configuration
CMAP_GREEN = 'Greens'
HIST_COLOR_GREEN = '#69c273'
PLOT_PALETTE = 'Greens_r' # Reversed green palette (darker is better)
FEATURE_TYPE_COLORS = {'ART': '#4CAF50', 'TRT': '#2196F3'}
KERNEL_EXPLAINER_SUBSET_SIZE = 50 # Can be adjusted based on dataset size and time
TOP_N_SHAP_FEATURES = 15
HIGH_CORRELATION_THRESHOLD = 0.8

# --- Helper Functions ---
def identify_feature_type(feature_name):
    for prefix in ART_PREFIXES:
        if feature_name.startswith(prefix) or feature_name.startswith(prefix.replace("-", "_")): # Handle Mean-shift/Mean_shift
            return 'ART'
    return 'TRT'

def get_biological_meaning(feature_name):
    if feature_name in BIO_INTERPRETATIONS: return BIO_INTERPRETATIONS[feature_name]
    for key in sorted(BIO_INTERPRETATIONS.keys(), key=len, reverse=True):
        if key in feature_name: return BIO_INTERPRETATIONS[key]
    if 'centre_y' in feature_name or 'center_y' in feature_name: return 'Vertical position of root clusters, related to rooting depth'
    if 'centre_x' in feature_name or 'center_x' in feature_name: return 'Horizontal position of root clusters, related to lateral spread'
    if 'density_points' in feature_name: return 'Size or density of root clusters, indicating root concentration'
    return 'Algorithmic representation of root architecture'

def get_drought_mechanism(feature_name):
    for key_pattern, mechanism_string in SPECIFIC_DROUGHT_MECHANISMS.items():
        if key_pattern in feature_name: return mechanism_string
    if 'centre_y' in feature_name or 'center_y' in feature_name: return GENERIC_MECHANISM_VERTICAL
    if 'centre_x' in feature_name or 'center_x' in feature_name: return GENERIC_MECHANISM_LATERAL
    if 'density_points' in feature_name: return GENERIC_MECHANISM_DENSITY
    return GENERIC_MECHANISM_GENERAL_ADAPTATION

def get_model_pipeline(classifier_component=None):
    if classifier_component is None:
        classifier_component = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier_component)
    ])

def generate_biological_interpretation_table(feature_importance_df, model_name):
    if feature_importance_df is None or feature_importance_df.empty:
        logger.warning(f"Cannot generate biological interpretation for {model_name}: missing importance data.")
        return None
    art_features_df = feature_importance_df[feature_importance_df['Feature_Type'] == 'ART'].copy()
    if art_features_df.empty:
        logger.info(f"No ART features found for biological interpretation in model {model_name}.")
        return None
    importance_col = 'Mean_Abs_SHAP' if 'Mean_Abs_SHAP' in art_features_df.columns else 'Importance'
    top_art_features = art_features_df.sort_values(by=importance_col, ascending=False).head(TOP_N_SHAP_FEATURES)
    bio_relevance_list = []
    for _, row in top_art_features.iterrows():
        feature = row['Feature']
        bio_relevance_list.append({
            'ART_Feature': feature,
            'Importance_Value': row[importance_col],
            'Potential_Biological_Interpretation': get_biological_meaning(feature),
            'Hypothesized_Drought_Adaptation_Mechanism': get_drought_mechanism(feature)
        })
    if not bio_relevance_list:
        logger.info(f"No biological relevance data generated for {model_name}.")
        return None
    bio_relevance_final_df = pd.DataFrame(bio_relevance_list)
    clean_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
    bio_interpretation_path = os.path.join(SHAP_OUTPUT_DIR, f'biological_interpretation_{clean_model_name}.csv')
    try:
        bio_relevance_final_df.to_csv(bio_interpretation_path, index=False)
        logger.info(f"Biological interpretation table for {model_name} saved to {bio_interpretation_path}")
    except Exception as e:
        logger.error(f"Failed to save biological interpretation for {model_name}: {e}")
    return bio_relevance_final_df

def evaluate_model_and_shap(X, y, model_name, label_encoder_classes, cv=5, is_rfe_model=False, rfe_selector_fitted=None):
    """Evaluates model, calculates SHAP values, and returns metrics & SHAP data."""
    if X.empty:
        logger.warning(f"Feature set for '{model_name}' is empty. Skipping evaluation.")
        metrics = {'Model': model_name, 'Features': 0}
        for m_key in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'CV_Accuracy_Mean', 'CV_Accuracy_Std']:
            metrics[m_key] = np.nan
        return metrics, None, (None, None), len(label_encoder_classes)

    if X.isna().any().any():
        logger.warning(f"Missing values detected in features for {model_name}. Filling with column median.")
        for col in X.columns[X.isna().any()]:
            if pd.api.types.is_numeric_dtype(X[col]): X[col] = X[col].fillna(X[col].median())
            else: X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "Unknown")
    
    numeric_cols = X.select_dtypes(include=np.number).columns
    if len(numeric_cols) != X.shape[1]:
        non_numeric_cols = X.select_dtypes(exclude=np.number).columns
        logger.error(f"Non-numeric columns detected in X for {model_name}: {list(non_numeric_cols)}. Evaluation aborted for this model.")
        metrics = {'Model': model_name, 'Features': X.shape[1]}
        for m_key in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'CV_Accuracy_Mean', 'CV_Accuracy_Std']:
            metrics[m_key] = np.nan
        return metrics, None, (None, None), len(label_encoder_classes)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build the pipeline
    # If it's an RFE model, the features X passed are already selected by an external RFE process.
    # The pipeline for RFE models will just be Scaler + RF.
    # For non-RFE models, it's Scaler + RF.
    model_pipeline = get_model_pipeline() # This always returns Scaler + RF
    
    try:
        model_pipeline.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Failed to fit model {model_name}: {e}")
        metrics = {'Model': model_name, 'Features': X.shape[1]}
        for m_key in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'CV_Accuracy_Mean', 'CV_Accuracy_Std']:
            metrics[m_key] = np.nan
        return metrics, None, (None, None), len(label_encoder_classes)

    y_pred = model_pipeline.predict(X_test)
    metrics = {
        'Model': model_name, 'Features': X.shape[1],
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'F1': f1_score(y_test, y_pred, average='macro', zero_division=0)
    }
    n_unique_classes = len(label_encoder_classes)
    if n_unique_classes == 2:
        try:
            y_proba = model_pipeline.predict_proba(X_test)[:, 1]
            metrics['ROC_AUC'] = roc_auc_score(y_test, y_proba)
        except Exception as e: metrics['ROC_AUC'] = np.nan
    else: metrics['ROC_AUC'] = np.nan

    try:
        cv_pipeline = get_model_pipeline() # Fresh pipeline for CV
        cv_scores = cross_val_score(cv_pipeline, X, y, cv=cv, scoring='accuracy')
        metrics['CV_Accuracy_Mean'] = cv_scores.mean()
        metrics['CV_Accuracy_Std'] = cv_scores.std()
    except Exception as e:
        metrics['CV_Accuracy_Mean'], metrics['CV_Accuracy_Std'] = np.nan, np.nan

    shap_values_output, X_df_for_shap_plot = None, None
    feature_importance_df = None
    classifier_component = model_pipeline.named_steps['classifier']
    scaler_component = model_pipeline.named_steps['scaler']
    
    X_test_scaled_np = scaler_component.transform(X_test)
    X_df_for_shap_plot = pd.DataFrame(X_test_scaled_np, columns=X.columns) # SHAP uses original feature names

    try:
        if X_df_for_shap_plot.shape[1] == 0: raise ValueError("No features to explain for TreeExplainer.")
        explainer = shap.TreeExplainer(classifier_component, X_df_for_shap_plot, model_output='probability')
        shap_values_output = explainer.shap_values(X_df_for_shap_plot)
        logger.info(f"SHAP TreeExplainer successful for {model_name}.")
        if isinstance(shap_values_output, list):
            mean_abs_shap = np.abs(shap_values_output[1]).mean(axis=0) if n_unique_classes == 2 else np.mean([np.abs(s).mean(axis=0) for s in shap_values_output], axis=0)
        else: mean_abs_shap = np.abs(shap_values_output).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns, 'Mean_Abs_SHAP': mean_abs_shap,
            'Feature_Type': [identify_feature_type(col) for col in X.columns]
        }).sort_values(by='Mean_Abs_SHAP', ascending=False)
    except Exception as e_tree:
        logger.warning(f"SHAP TreeExplainer failed for {model_name}: {e_tree}. Trying KernelExplainer.")
        try:
            if X_df_for_shap_plot.shape[1] == 0: raise ValueError("No features for KernelExplainer.")
            subset_indices_bg = np.random.choice(X_df_for_shap_plot.shape[0], min(KERNEL_EXPLAINER_SUBSET_SIZE, X_df_for_shap_plot.shape[0]), replace=False)
            X_background_kernel = X_df_for_shap_plot.iloc[subset_indices_bg]
            explainer = shap.KernelExplainer(classifier_component.predict_proba, X_background_kernel)
            X_test_subset_df_for_kernel = X_df_for_shap_plot.sample(min(KERNEL_EXPLAINER_SUBSET_SIZE, X_df_for_shap_plot.shape[0]), random_state=42)
            shap_values_output = explainer.shap_values(X_test_subset_df_for_kernel)
            X_df_for_shap_plot = X_test_subset_df_for_kernel # Update X_df to match SHAP values
            logger.info(f"SHAP KernelExplainer successful for {model_name} (on subset).")
            if isinstance(shap_values_output, list):
                mean_abs_shap = np.abs(shap_values_output[1]).mean(axis=0) if n_unique_classes == 2 else np.mean([np.abs(s).mean(axis=0) for s in shap_values_output], axis=0)
            else: mean_abs_shap = np.abs(shap_values_output).mean(axis=0)
            feature_importance_df = pd.DataFrame({
                'Feature': X.columns, 'Mean_Abs_SHAP': mean_abs_shap, # X.columns because kernel explainer gives importance for original features
                'Feature_Type': [identify_feature_type(col) for col in X.columns]
            }).sort_values(by='Mean_Abs_SHAP', ascending=False)
        except Exception as e_kernel:
            logger.warning(f"SHAP KernelExplainer failed for {model_name}: {e_kernel}. Using RF feature_importances_.")
            shap_values_output = None
            try:
                importances = classifier_component.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': X.columns, 'Importance': importances,
                    'Feature_Type': [identify_feature_type(col) for col in X.columns]
                }).sort_values(by='Importance', ascending=False)
            except Exception as e_rf:
                logger.error(f"Failed to get RF feature_importances_ for {model_name}: {e_rf}")
                feature_importance_df = None
    return metrics, feature_importance_df, (shap_values_output, X_df_for_shap_plot), n_unique_classes

def create_visualizations_and_reports(results_df, all_feature_importance_dfs, all_bio_interpretations, model_shap_data_map, label_encoder_classes):
    """Generates all plots and summary CSVs."""
    logger.info("Visualizing model performance metrics...")
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'CV_Accuracy_Mean']
    for metric in metrics_to_plot:
        if metric in results_df.columns and results_df[metric].notna().any():
            plt.figure(figsize=FIG_SIZE_LARGE)
            plot_df = results_df.sort_values(by=metric, ascending=False).dropna(subset=[metric])
            ax = sns.barplot(x='Model', y=metric, data=plot_df, palette=PLOT_PALETTE)
            for i, v_val in enumerate(plot_df[metric]):
                 ax.text(i, v_val + 0.01, f'{v_val:.3f}', ha='center', va='bottom', fontsize=TICK_FONTSIZE-2)
            plt.title(f'Model Comparison: {metric}', fontsize=TITLE_FONTSIZE)
            plt.ylabel(metric, fontsize=LABEL_FONTSIZE)
            plt.xlabel('Model Configuration', fontsize=LABEL_FONTSIZE)
            plt.ylim(0, max(1.0, plot_df[metric].max() * 1.1 if not plot_df.empty else 1.0))
            plt.xticks(rotation=45, ha='right', fontsize=TICK_FONTSIZE)
            plt.yticks(fontsize=TICK_FONTSIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_OUTPUT_DIR, f'comparison_{metric}.png'), dpi=300)
            plt.close()

    # --- START: Added code for Accuracy vs. Number of Features plot ---
    logger.info("Generating Accuracy vs. Number of Features plot...")
    if 'Features' in results_df.columns and 'Accuracy' in results_df.columns and \
       results_df['Features'].notna().any() and results_df['Accuracy'].notna().any():
        
        plt.figure(figsize=FIG_SIZE_MEDIUM) 
        
        # Prepare data: copy, ensure numeric, drop NaNs in essential columns, sort
        plot_df_acc_feat = results_df[['Model', 'Features', 'Accuracy']].copy()
        # Ensure 'Features' is numeric for sorting and plotting. Coerce errors to NaN.
        plot_df_acc_feat['Features'] = pd.to_numeric(plot_df_acc_feat['Features'], errors='coerce')
        # Drop rows where 'Features' or 'Accuracy' became NaN or were already NaN
        plot_df_acc_feat = plot_df_acc_feat.dropna(subset=['Features', 'Accuracy'])

        if not plot_df_acc_feat.empty:
            plot_df_acc_feat = plot_df_acc_feat.sort_values('Features')

            # Use the specified green color for scatter points
            ax_scatter = sns.scatterplot(
                x='Features', 
                y='Accuracy', 
                data=plot_df_acc_feat, 
                s=120,  # Size of the points
                color=HIST_COLOR_GREEN,
                edgecolor='black', # Edge color for points
                alpha=0.8 # Transparency - slightly higher for better visibility
            )

            # Add model labels to each point for identification
            for idx, row in plot_df_acc_feat.iterrows():
                ax_scatter.text(
                    row['Features'], 
                    row['Accuracy'] + 0.005, # Offset text slightly above the point
                    row['Model'], 
                    ha='center', 
                    va='bottom', 
                    fontsize=TICK_FONTSIZE - 3, # Slightly smaller font for annotations
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5, alpha=0.5) # Add a faint background to text
                )

            plt.title('Model Accuracy vs. Number of Features', fontsize=TITLE_FONTSIZE)
            plt.xlabel('Number of Features', fontsize=LABEL_FONTSIZE)
            plt.ylabel('Accuracy', fontsize=LABEL_FONTSIZE)
            
            # Customize x-axis ticks for better readability if there are many unique feature counts
            unique_feature_counts = sorted(plot_df_acc_feat['Features'].unique().astype(int))
            if unique_feature_counts:
                if len(unique_feature_counts) > 10 and (max(unique_feature_counts) - min(unique_feature_counts)) > 20:
                    # If many unique counts and wide range, let matplotlib decide or use a locator
                    pass 
                elif len(unique_feature_counts) > 0 :
                     plt.xticks(ticks=unique_feature_counts, rotation=30, ha='right', fontsize=TICK_FONTSIZE-1)
                # else: default ticks

            plt.yticks(fontsize=TICK_FONTSIZE)
            # Dynamic Y Lim based on data, ensuring it's within [0, 1.05]
            min_acc = plot_df_acc_feat['Accuracy'].min() if not plot_df_acc_feat.empty else 0
            max_acc = plot_df_acc_feat['Accuracy'].max() if not plot_df_acc_feat.empty else 1
            plt.ylim(max(0, min_acc - 0.05), min(1.05, max_acc + 0.05))
            
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(BASE_OUTPUT_DIR, 'accuracy_vs_features.png'), dpi=300)
            plt.close()
            logger.info(f"Accuracy vs. Number of Features plot saved to {os.path.join(BASE_OUTPUT_DIR, 'accuracy_vs_features.png')}")
        else:
            logger.warning("Not enough valid data (after NaN filtering) to generate Accuracy vs. Number of Features plot.")
    else:
        logger.warning("Skipping Accuracy vs. Number of Features plot: 'Features' or 'Accuracy' column missing, or all values are NaN in results_df.")
    # --- END: Added code for Accuracy vs. Number of Features plot ---

    logger.info("Generating SHAP summary plots...")
    for model_name, (shap_values, X_df_for_plot) in model_shap_data_map.items():
        if shap_values is None or X_df_for_plot is None or X_df_for_plot.empty:
            logger.info(f"No SHAP values or X_df_for_plot for {model_name}. Skipping SHAP plots.")
            continue
        clean_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
        try:
            plt.figure(figsize=FIG_SIZE_LARGE)
            shap.summary_plot(shap_values, X_df_for_plot, plot_type='bar', show=False, class_names=label_encoder_classes, max_display=TOP_N_SHAP_FEATURES)
            plt.title(f'SHAP Mean Abs Values: {model_name}', fontsize=TITLE_FONTSIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_OUTPUT_DIR, f'shap_summary_bar_{clean_model_name}.png'), dpi=300)
            plt.close()

            plt.figure(figsize=FIG_SIZE_LARGE)
            # Use green colormap for SHAP summary dot plot
            shap.summary_plot(shap_values, X_df_for_plot, show=False, class_names=label_encoder_classes, max_display=TOP_N_SHAP_FEATURES, cmap=CMAP_GREEN)
            plt.title(f'SHAP Value Distribution: {model_name}', fontsize=TITLE_FONTSIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_OUTPUT_DIR, f'shap_summary_dot_{clean_model_name}.png'), dpi=300)
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate SHAP summary plots for {model_name}: {e}")
    
    if all_feature_importance_dfs:
        combined_importance_df = pd.concat(all_feature_importance_dfs, ignore_index=True)
        importance_col_name = 'Mean_Abs_SHAP' if 'Mean_Abs_SHAP' in combined_importance_df.columns else 'Importance'
        if importance_col_name not in combined_importance_df.columns: # Fallback if even 'Importance' is missing somehow
            logger.warning(f"Neither 'Mean_Abs_SHAP' nor 'Importance' found in combined_importance_df. Importance report might be incomplete.")
        else:
            combined_importance_df = combined_importance_df.sort_values(by=['Model', importance_col_name], ascending=[True, False])
            combined_importance_df.to_csv(os.path.join(SHAP_OUTPUT_DIR, 'all_models_feature_importance.csv'), index=False)
            logger.info("Combined feature importance report saved.")

            top_overall_features = combined_importance_df.groupby('Feature')[importance_col_name].mean().nlargest(TOP_N_SHAP_FEATURES).reset_index()
            top_overall_features.to_csv(os.path.join(SHAP_OUTPUT_DIR, 'top_overall_shap_contributors.csv'), index=False)
            logger.info("Top N overall SHAP contributors report saved.")

            all_features_model_key = "All Features (ART+TRT)"
            all_features_importance_df = next((df for df in all_feature_importance_dfs if not df.empty and df['Model'].iloc[0] == all_features_model_key), None)
            if all_features_importance_df is not None and not all_features_importance_df.empty:
                type_contribution = all_features_importance_df.groupby('Feature_Type')[importance_col_name].sum().reset_index()
                if not type_contribution.empty and type_contribution[importance_col_name].sum() > 0:
                    plt.figure(figsize=FIG_SIZE_MEDIUM)
                    plt.pie(type_contribution[importance_col_name], labels=type_contribution['Feature_Type'],
                            colors=[FEATURE_TYPE_COLORS.get(ft, '#CCCCCC') for ft in type_contribution['Feature_Type']],
                            autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white'})
                    plt.title(f'Feature Type Contribution ({importance_col_name}) (Model: {all_features_model_key})', fontsize=TITLE_FONTSIZE)
                    plt.savefig(os.path.join(SHAP_OUTPUT_DIR, 'feature_type_contribution_pie.png'), dpi=300)
                    plt.close()
    
    if all_bio_interpretations:
        bio_df_for_plot = None
        target_models_for_drought_viz = ["All Features (ART+TRT)", "All ARTs"]
        for model_key_viz in target_models_for_drought_viz:
            bio_df_for_plot_candidate = next((df for df in all_bio_interpretations if not df.empty and df['Model_Name_Source'].iloc[0] == model_key_viz), None)
            if bio_df_for_plot_candidate is not None:
                bio_df_for_plot = bio_df_for_plot_candidate
                break
        if bio_df_for_plot is None and all_bio_interpretations:
             bio_df_for_plot = pd.concat([df for df in all_bio_interpretations if not df.empty])
        if bio_df_for_plot is not None and not bio_df_for_plot.empty:
            mechanism_counts = bio_df_for_plot['Hypothesized_Drought_Adaptation_Mechanism'].value_counts()
            if not mechanism_counts.empty:
                fig, ax = plt.subplots(figsize=(15,10))
                ax.set_facecolor('#FCFCFC')
                mechanisms_original = mechanism_counts.index
                sizes_original = mechanism_counts.values
                total_sum = sum(sizes_original)
                if total_sum > 0:
                    percentages_original = [(count / total_sum) * 100 for count in sizes_original]
                    sorted_mechanisms_data = sorted(zip(mechanisms_original, sizes_original, percentages_original), key=lambda x: x[2], reverse=True)
                    max_mechanisms_in_plot = 7
                    processed_mechanisms_data = []
                    if len(sorted_mechanisms_data) > max_mechanisms_in_plot:
                        processed_mechanisms_data = sorted_mechanisms_data[:max_mechanisms_in_plot]
                        other_mechanisms_group = sorted_mechanisms_data[max_mechanisms_in_plot:]
                        other_size = sum(item[1] for item in other_mechanisms_group)
                        other_percentage = sum(item[2] for item in other_mechanisms_group)
                        if other_percentage > 0.1:
                           processed_mechanisms_data.append(("Other Combined Mechanisms", other_size, other_percentage))
                    else: processed_mechanisms_data = sorted_mechanisms_data
                    plot_mechanisms = [item[0] for item in processed_mechanisms_data]
                    plot_sizes = np.array([item[1] for item in processed_mechanisms_data])
                    plot_percentages = [item[2] for item in processed_mechanisms_data]
                    colors_list = [MECHANISM_COLORS.get(name, MECHANISM_COLORS['Other']) for name in plot_mechanisms]
                    n_circles_to_plot = len(plot_mechanisms)
                    angles = np.linspace(0, 2 * np.pi, n_circles_to_plot + 1)[:-1]
                    radius_base = 0.32 if n_circles_to_plot <= 5 else (0.28 if n_circles_to_plot <= 7 else 0.24)
                    positions = [(0.5 + radius_base * np.cos(angle), 0.5 + radius_base * np.sin(angle)) for angle in angles]
                    legend_elements = []
                    current_max_size = plot_sizes.max() if len(plot_sizes) > 0 else 1
                    for i in range(n_circles_to_plot):
                        mechanism_name, size, percentage = plot_mechanisms[i], plot_sizes[i], plot_percentages[i]
                        color, pos = colors_list[i], positions[i]
                        radius_circle = 0.035 + 0.13 * (size / current_max_size)**0.65 if current_max_size > 0 else 0.035
                        circle = Circle(pos, radius_circle, color=color, alpha=0.80, ec='black', linewidth=1.2)
                        ax.add_patch(circle)
                        con = ConnectionPatch(xyA=(0.5, 0.5), xyB=pos, coordsA='data', coordsB='data', color='dimgrey', alpha=0.65, linestyle='--', linewidth=1.0)
                        ax.add_patch(con)
                        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=f"{mechanism_name} ({percentage:.1f}%)"))
                    root_center = Circle((0.5, 0.5), 0.065, color='saddlebrown', alpha=1.0, ec='#422d1a', linewidth=1.5)
                    ax.add_patch(root_center)
                    ax.text(0.5, 0.5, 'Root\nSystem', ha='center', va='center', fontsize=8.5, color='white', weight='bold')
                    legend_fontsize = 9 if n_circles_to_plot > 6 else 9.5
                    lgd = ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=legend_fontsize, title="Adaptation Mechanisms", title_fontsize=legend_fontsize, frameon=True, facecolor='white', edgecolor='silver', borderpad=0.8)
                    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
                    ax.set_aspect('equal', adjustable='box'); ax.axis('off')
                    plt.title('Hypothesized Drought Adaptation Mechanisms in Root Architecture', fontsize=TITLE_FONTSIZE-1, pad=15)
                    plt.savefig(os.path.join(SHAP_OUTPUT_DIR, 'drought_adaptation_mechanisms_diagram.png'), dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
                    plt.close(fig)

# --- New Function for RFE Analysis ---
def perform_rfe_analysis(X_all_features, y_target, label_encoder_classes, base_estimator_for_rfe):
    logger.info("--- Starting Feature Redundancy Analysis with RFE ---")
    rfe_results_list = []
    if X_all_features.empty:
        logger.warning("X_all_features is empty for RFE. Skipping.")
        return pd.DataFrame()

    corr_matrix = X_all_features.corr()
    plt.figure(figsize=(max(12, X_all_features.shape[1]*0.3), max(10, X_all_features.shape[1]*0.25))) # Dynamic size
    sns.heatmap(corr_matrix, annot=False, cmap=CMAP_GREEN, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix (All Features)', fontsize=TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(RFE_OUTPUT_DIR, 'all_features_correlation_heatmap.png'), dpi=150)
    plt.close()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated_pairs = [(column, index, upper[column][index]) for column in upper.columns for index in upper.index
                               if abs(upper[column][index]) > HIGH_CORRELATION_THRESHOLD]
    if highly_correlated_pairs:
        pd.DataFrame(highly_correlated_pairs, columns=['Feat1', 'Feat2', 'Corr']).to_csv(os.path.join(RFE_OUTPUT_DIR, 'highly_correlated_feature_pairs.csv'), index=False)
        logger.info(f"Found {len(highly_correlated_pairs)} pairs with |correlation| > {HIGH_CORRELATION_THRESHOLD}")

    num_total_features = X_all_features.shape[1]
    n_features_to_select_list = [10, 15, 20, 25, min(30, num_total_features-1 if num_total_features >1 else 1)] # Adjusted range
    # Add counts of ART and TRT features if they are meaningful selections
    num_art_features = sum(1 for col in X_all_features.columns if identify_feature_type(col) == 'ART')
    num_trt_features = sum(1 for col in X_all_features.columns if identify_feature_type(col) == 'TRT')
    if 0 < num_art_features < num_total_features: n_features_to_select_list.append(num_art_features)
    if 0 < num_trt_features < num_total_features: n_features_to_select_list.append(num_trt_features)
    
    n_features_to_select_list = sorted([n for n in list(set(n_features_to_select_list)) if 0 < n <= num_total_features])

    X_scaled_for_rfe_fit = StandardScaler().fit_transform(X_all_features)

    for n_select in n_features_to_select_list:
        logger.info(f"Running RFE to select top {n_select} features...")
        try:
            rfe_selector = RFE(estimator=base_estimator_for_rfe, n_features_to_select=n_select, step=0.1) # step as proportion
            rfe_selector.fit(X_scaled_for_rfe_fit, y_target)
            selected_features_rfe_names = X_all_features.columns[rfe_selector.support_]
            X_data_rfe_selected = X_all_features[selected_features_rfe_names]
            
            model_name_rfe = f"RFE (Top {n_select} Features)"
            metrics_rfe, importance_df_rfe, shap_data_rfe, _ = evaluate_model_and_shap(
                X_data_rfe_selected, y_target, model_name_rfe, label_encoder_classes,
                is_rfe_model=True, rfe_selector_fitted=rfe_selector # Pass fitted selector for context if needed
            )
            metrics_rfe['Features_Selected_By_RFE'] = n_select
            rfe_results_list.append(metrics_rfe)
            if importance_df_rfe is not None:
                importance_df_rfe['Model'] = model_name_rfe
                importance_df_rfe.to_csv(os.path.join(RFE_OUTPUT_DIR, f'rfe_{n_select}_feature_importance.csv'), index=False)
        except Exception as e:
            logger.error(f"Error during RFE for {n_select} features: {e}")

    if rfe_results_list:
        rfe_summary_df = pd.DataFrame(rfe_results_list)
        cols_order_rfe = ['Model', 'Features_Selected_By_RFE', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'CV_Accuracy_Mean', 'CV_Accuracy_Std']
        rfe_summary_df = rfe_summary_df[[col for col in cols_order_rfe if col in rfe_summary_df.columns]]
        rfe_summary_df.to_csv(os.path.join(RFE_OUTPUT_DIR, 'rfe_performance_summary.csv'), index=False)
        logger.info(f"RFE performance summary saved.")
        print("\nRFE Performance Summary:")
        print(rfe_summary_df)
        metrics_for_rfe_plot = ['Accuracy', 'F1', 'ROC_AUC']
        for metric in metrics_for_rfe_plot:
            if metric in rfe_summary_df.columns and rfe_summary_df[metric].notna().any():
                plt.figure(figsize=FIG_SIZE_MEDIUM)
                sns.lineplot(
                    x='Features_Selected_By_RFE', 
                    y=metric, 
                    data=rfe_summary_df.dropna(subset=[metric]), 
                    marker='o', 
                    color=HIST_COLOR_GREEN,
                    linewidth=2.5,
                    markersize=8
                ) 
                plt.title(f'RFE: {metric} vs. Number of Selected Features', fontsize=TITLE_FONTSIZE)
                plt.xlabel('Number of Features Selected by RFE', fontsize=LABEL_FONTSIZE)
                plt.ylabel(metric, fontsize=LABEL_FONTSIZE)
                unique_n_select = sorted(rfe_summary_df['Features_Selected_By_RFE'].unique())
                if unique_n_select: plt.xticks(unique_n_select, fontsize=TICK_FONTSIZE)
                plt.yticks(fontsize=TICK_FONTSIZE)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(RFE_OUTPUT_DIR, f'rfe_performance_vs_n_features_{metric}.png'), dpi=300)
                plt.close()
        return rfe_summary_df
    logger.info("No RFE results to summarize or plot.")
    return pd.DataFrame()

# --- Data Loading and Preprocessing Function ---
def load_and_preprocess_data(art_path, trt_path, combined_path):
    logger.info("Loading datasets...")
    try:
        art_data = pd.read_csv(art_path)
        trt_data = pd.read_csv(trt_path)
        combined_data = pd.read_csv(combined_path)
        logger.info("Datasets loaded successfully.")
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}. Exiting.")
        raise  # Re-raise the exception to be handled by the caller

    label_encoder = LabelEncoder()
    try:
        combined_data['Group_encoded'] = label_encoder.fit_transform(combined_data['Group'])
    except KeyError:
        logger.error("'Group' column missing in combined_data. Cannot encode target variable. Exiting.")
        raise
    
    y_target = combined_data['Group_encoded']
    meta_columns = ['Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication', 'Group', 'Group_encoded']
    
    # Ensure all_available_cols are numeric and exist after dropping meta_columns
    all_available_cols = combined_data.drop(columns=meta_columns, errors='ignore').select_dtypes(include=np.number).columns
    if all_available_cols.empty:
        logger.error("No numeric features found after removing metadata columns from combined_data. Exiting.")
        raise ValueError("No numeric features available for modeling.")
        
    X_all_features_df = combined_data[all_available_cols].copy()

    # Return all necessary components
    return art_data, trt_data, combined_data, X_all_features_df, y_target, label_encoder, meta_columns, all_available_cols

# --- Feature Set Definition Function ---
def define_feature_sets(combined_data, art_data, trt_data, meta_columns, all_available_cols):
    logger.info("Defining feature sets for evaluation...")
    baseline_feature_name = 'Total.Root.Length.mm' if 'Total.Root.Length.mm' in all_available_cols else all_available_cols[0]
    logger.info(f"Using '{baseline_feature_name}' as baseline feature.")

    selected_trt_features_list = [f for f in ['Total.Root.Length.mm', 'Network.Area.mm2', 'Volume.mm3', 'Surface.Area.mm2'] if f in all_available_cols]
    selected_art_features_list = [f for f in ['FCM_center_x', 'DBSCAN_density_points', 'HDBSCAN_density_points', 'OPTICS_density_points'] if f in all_available_cols]
    
    # Ensure art_feature_columns and trt_feature_columns are derived correctly based on *their respective* original dataframes (art_data, trt_data)
    # and also ensure they exist in all_available_cols (which are from combined_data)
    art_feature_columns = [col for col in art_data.columns if col not in meta_columns and col in all_available_cols]
    trt_feature_columns = [col for col in trt_data.columns if col not in meta_columns and col in all_available_cols]
    
    # X_all_features_df is already created in load_and_preprocess_data, using it directly is fine
    # For consistency, let's ensure we pass it if needed or re-derive from combined_data[all_available_cols]

    datasets = {
        f"Baseline ({baseline_feature_name})": combined_data[[baseline_feature_name]] if baseline_feature_name in combined_data else pd.DataFrame(),
        "Selected 4 TRTs": combined_data[selected_trt_features_list] if selected_trt_features_list else pd.DataFrame(),
        "Selected 4 ARTs": combined_data[selected_art_features_list] if selected_art_features_list else pd.DataFrame(),
        "Selected TRTs+ARTs": combined_data[list(set(selected_trt_features_list + selected_art_features_list))] if (selected_trt_features_list or selected_art_features_list) else pd.DataFrame(),
        "All TRTs": combined_data[trt_feature_columns] if trt_feature_columns else pd.DataFrame(),
        "All ARTs": combined_data[art_feature_columns] if art_feature_columns else pd.DataFrame(),
        "All Features (ART+TRT)": combined_data[all_available_cols].copy() # This uses all numeric cols from combined_data
    }
    
    # Filter out empty or no-feature dataframes
    datasets_to_evaluate = {name: df.select_dtypes(include=np.number) for name, df in datasets.items() if not df.empty and df.shape[1] > 0}
    logger.info(f"Defined {len(datasets_to_evaluate)} feature sets for evaluation.")
    return datasets_to_evaluate, baseline_feature_name # Return baseline_feature_name if needed elsewhere, though it's mostly for logging here

# --- Model Evaluation Function ---
def run_model_evaluations(datasets_to_evaluate, y_target, label_encoder_classes_):
    logger.info("Starting model evaluation for defined feature sets...")
    all_results_list = []
    all_feature_importance_dfs = []
    all_bio_interpretations_list = []
    model_shap_data_map = {}

    for model_name, X_df in datasets_to_evaluate.items():
        logger.info(f"--- Evaluating Model: {model_name} with {X_df.shape[1]} features ---")
        metrics, f_importance_df, s_data_tuple, n_classes_detected = evaluate_model_and_shap(
            X_df.copy(), y_target, model_name, label_encoder_classes_
        )
        all_results_list.append(metrics)
        model_shap_data_map[model_name] = s_data_tuple # Store (shap_values, X_df_for_plot)
        if f_importance_df is not None and not f_importance_df.empty:
            f_importance_df['Model'] = model_name
            all_feature_importance_dfs.append(f_importance_df)
            bio_df = generate_biological_interpretation_table(f_importance_df, model_name)
            if bio_df is not None and not bio_df.empty:
                bio_df['Model_Name_Source'] = model_name
                all_bio_interpretations_list.append(bio_df)
    
    results_summary_df = pd.DataFrame(all_results_list)
    cols_order = ['Model', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC', 'CV_Accuracy_Mean', 'CV_Accuracy_Std']
    results_summary_df = results_summary_df[[col for col in cols_order if col in results_summary_df.columns]]
    results_summary_df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'model_performance_summary.csv'), index=False)
    print("\nModel Performance Summary (Original Sets):")
    print(results_summary_df)

    logger.info("Model evaluation phase completed.")
    return results_summary_df, all_feature_importance_dfs, all_bio_interpretations_list, model_shap_data_map

# --- RFE and Final Reporting Function ---
def run_rfe_and_generate_outputs(datasets_to_evaluate, y_target, label_encoder_classes_,
                                 results_summary_df, all_feature_importance_dfs, 
                                 all_bio_interpretations_list, model_shap_data_map):
    logger.info("Starting RFE analysis and final report generation...")

    rfe_performed_successfully = False
    if "All Features (ART+TRT)" in datasets_to_evaluate and not datasets_to_evaluate["All Features (ART+TRT)"].empty:
        X_for_rfe_analysis = datasets_to_evaluate["All Features (ART+TRT)"]
        base_rf_for_rfe = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
        logger.info("Calling RFE analysis function...")
        rfe_performance_df = perform_rfe_analysis(X_for_rfe_analysis, y_target, label_encoder_classes_, base_rf_for_rfe)
        if rfe_performance_df is not None and not rfe_performance_df.empty:
            logger.info("RFE analysis completed and results obtained.")
            # Optional: Could merge rfe_performance_df with results_summary_df here if desired for unified reporting
            # For now, create_visualizations_and_reports expects the original summary, RFE plots separately.
            rfe_performed_successfully = True 
        else:
            logger.info("RFE analysis did not produce a performance DataFrame or it was empty.")
    else:
        logger.warning("Skipping RFE: 'All Features (ART+TRT)' dataset not available or empty.")

    if all_bio_interpretations_list:
        try:
            combined_bio_df = pd.concat(all_bio_interpretations_list, ignore_index=True)
            combined_bio_df.to_csv(os.path.join(SHAP_OUTPUT_DIR, 'all_models_biological_interpretations.csv'), index=False)
            logger.info("Combined biological interpretations saved.")
        except Exception as e:
            logger.error(f"Failed to save combined biological interpretations: {e}")

    logger.info("Calling final visualization and report generation function...")
    create_visualizations_and_reports(results_summary_df, all_feature_importance_dfs, 
                                      all_bio_interpretations_list, model_shap_data_map, 
                                      label_encoder_classes_)
    
    logger.info(f"Analysis complete. All results and visualizations saved to {BASE_OUTPUT_DIR} and its subdirectories.")
    if rfe_performed_successfully:
        logger.info("RFE analysis was also performed and its results are in the RFE output directory.")


# --- Main Execution --- 
def main():
    logger.info("Starting baseline comparison, SHAP, and RFE analysis script.")
    # Set seed at the beginning of main execution for operations within this scope
    np.random.seed(42)
    try:
        art_data, trt_data, combined_data, X_all_features_df, y_target, label_encoder, meta_columns, all_available_cols = load_and_preprocess_data(ART_PATH, TRT_PATH, COMBINED_PATH)
    except (FileNotFoundError, KeyError, ValueError) as e:
        logger.error(f"Failed during data loading and preprocessing: {e}")
        return
        
    datasets_to_evaluate, baseline_feature_name = define_feature_sets(combined_data, art_data, trt_data, meta_columns, all_available_cols)

    results_summary_df, all_feature_importance_dfs, all_bio_interpretations_list, model_shap_data_map = run_model_evaluations(datasets_to_evaluate, y_target, label_encoder.classes_)

    run_rfe_and_generate_outputs(datasets_to_evaluate, y_target, label_encoder.classes_,
                                 results_summary_df, all_feature_importance_dfs,
                                 all_bio_interpretations_list, model_shap_data_map)

if __name__ == "__main__":
    main()