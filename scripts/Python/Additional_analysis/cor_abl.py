# This script performs a comprehensive analysis of Algorithmic Root Trait (ART) data
# and Traditional Root Trait (TRT) data. It includes:
# 1. Correlation analysis between ART and TRT features with significance testing.
# 2. Feature importance analysis using Random Forest and permutation importance
#    to identify key discriminators between experimental groups.
# 3. Trait stability analysis (Coefficient of Variation) across different conditions.
# 4. Biological relevance and network analysis to link ART algorithms with TRT features.
# 5. Drought adaptation mechanism analysis based on important ART features and their
#    correlations with TRT features, including a conceptual diagram.
# 6. Simulated algorithm ablation study to estimate the impact of removing feature sets
#    from different ART algorithms.
# The script loads data from CSV files, performs calculations, generates various plots
# (heatmaps, bar plots, network diagrams), and saves results and plots to an output directory.
# It aims to provide insights into the relationships between different root measurement
# techniques and their biological significance, particularly in the context of drought adaptation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split # Keep if still needed elsewhere
# Remove unused imports like Pipeline, SelectFromModel if not used directly in cor.py
from scipy.stats import ttest_ind

# Set a global random seed for reproducibility where applicable (e.g., for numpy operations)
# Note: sklearn models often use their own random_state parameter, which is also set.
np.random.seed(42)


# --- Global Configuration ---
# Adjust these paths if your data and output directories are structured differently.
# It's recommended to have a 'data' subdirectory for input CSVs
# and an 'output/cor' subdirectory for saved results and plots, relative to the script's location.
OUTPUT_DIR = 'output/cor'
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEBUG = True
ART_FILE_PATH = 'data/ART_A_B.csv'
TRT_FILE_PATH = 'data/TRT_A_B.csv'
COMBINED_FILE_PATH = 'data/Combine_A_B.csv'
plt.style.use('seaborn-v0_8-whitegrid')
SMALL_FIGSIZE = (8, 6)
MEDIUM_FIGSIZE = (10, 8)
LARGE_FIGSIZE = (12, 10)
NETWORK_FIGSIZE = (14,12)
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
ANNOT_FONTSIZE = 8
LEGEND_FONTSIZE = 10
DIVERGING_CMAP = LinearSegmentedColormap.from_list('custom_div_cmap',
                                                 ['darkturquoise', 'white', 'forestgreen'],
                                                 N=256)
SEQUENTIAL_CMAP = 'BuGn'
SEQUENTIAL_PALETTE = 'BuGn'
CATEGORICAL_PALETTE_FEATURES = 'tab20'
CORRELATION_THRESHOLDS = [0.6, 0.7]
NETWORK_CORR_THRESHOLD_STRONG = 0.7
NETWORK_CORR_THRESHOLD_WEAK = 0.6
TOP_N_FEATURES_DISPLAY = 20
RF_N_ESTIMATORS = 100 # Or whatever your final model uses
RF_RANDOM_STATE = 42
KNOWN_ART_PREFIXES = sorted(['DBSCAN', 'Density', 'FCM', 'GMM', 'HDBSCAN', 'K-mean', 'Mean_shift', 'Mean-shift', 'OPTICS', 'SLIC'], key=len, reverse=True)


# --- Helper Functions (Keep as is) ---
def debug_print(message):
    if DEBUG:
        print(f"DEBUG: {message}")

def extract_feature_algorithm_type(feature_name, art_prefixes):
    if feature_name.startswith('Mean_shift') or feature_name.startswith('Mean-shift'):
        return 'Mean-shift'
    for prefix in art_prefixes:
        if feature_name.startswith(prefix):
            return prefix
    return 'TRT'

def save_plot(fig_or_plt, filename_base, output_dir=OUTPUT_DIR, dpi=300): # Ensure fig_or_plt is passed
    path = os.path.join(output_dir, f"{filename_base}.png")
    if isinstance(fig_or_plt, plt.Figure):
         fig_or_plt.savefig(path, dpi=dpi, bbox_inches='tight')
         plt.close(fig_or_plt)
    else: # Assuming it's plt
        plt.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close()
    debug_print(f"Saved plot: {path}")


def load_csv_data(file_path, identifier="data"):
    try:
        df = pd.read_csv(file_path)
        debug_print(f"Successfully loaded {identifier} from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        debug_print(f"Error: {identifier} file not found at {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        debug_print(f"Error loading {identifier} from {file_path}: {e}")
        return pd.DataFrame()

def draw_root(ax):
    """Draws a very simple stylized root system on the given axes."""
    # Main taproot
    ax.plot([5, 5], [8, 1.5], 'gray', linewidth=4, alpha=0.5, zorder=0) # Thicker, lighter
    # Some lateral roots
    ax.plot([5, 3.5], [7, 6], 'gray', linewidth=3, alpha=0.4, zorder=0)
    ax.plot([5, 6.5], [7, 6], 'gray', linewidth=3, alpha=0.4, zorder=0)
    ax.plot([5, 4], [5.5, 4.5], 'gray', linewidth=2.5, alpha=0.4, zorder=0)
    ax.plot([5, 6], [5.5, 4.5], 'gray', linewidth=2.5, alpha=0.4, zorder=0)
    ax.plot([5, 4.5], [3.5, 2.8], 'gray', linewidth=2, alpha=0.3, zorder=0)
    ax.plot([5, 5.5], [3.5, 2.8], 'gray', linewidth=2, alpha=0.3, zorder=0)
    debug_print("Drew stylized root for conceptual diagram.")

def _load_and_prepare_features(art_path, trt_path):
    """Loads ART and TRT data, aligns them, and prepares feature DataFrames."""
    art_data_full = load_csv_data(art_path, "ART data")
    trt_data_full = load_csv_data(trt_path, "TRT data")

    if art_data_full.empty or trt_data_full.empty:
        debug_print("ART or TRT data is empty in _load_and_prepare_features. Aborting.")
        return pd.DataFrame(), pd.DataFrame()

    art_data = art_data_full.set_index('Image_name')
    trt_data = trt_data_full.set_index('Image_name')
    common_indices = art_data.index.intersection(trt_data.index)

    if common_indices.empty:
        debug_print("No common Image_name indices found in _load_and_prepare_features. Aborting.")
        return pd.DataFrame(), pd.DataFrame()

    art_data_aligned = art_data.loc[common_indices].reset_index()
    trt_data_aligned = trt_data.loc[common_indices].reset_index()

    # Define meta_columns specifically for this feature preparation step
    meta_cols_for_prep = ['Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication', 'Group']

    art_features_df = art_data_aligned.drop(
        columns=[col for col in meta_cols_for_prep if col in art_data_aligned.columns and col != 'Image_name'],
        errors='ignore'
    ).set_index('Image_name')

    trt_features_df = trt_data_aligned.drop(
        columns=[col for col in meta_cols_for_prep if col in trt_data_aligned.columns and col != 'Image_name'],
        errors='ignore'
    ).set_index('Image_name')

    debug_print(f"Helper: ART features shape: {art_features_df.shape}, TRT features shape: {trt_features_df.shape}")

    if art_features_df.empty or trt_features_df.empty:
        debug_print("No features found after filtering metadata in _load_and_prepare_features. Aborting.")
        return pd.DataFrame(), pd.DataFrame()
        
    return art_features_df, trt_features_df

# --- art_trt_correlation_analysis_with_significance (Keep as is) ---
def art_trt_correlation_analysis_with_significance(art_path, trt_path):
    """Performs correlation analysis between ART and TRT features.

    Calculates Pearson correlations and p-values, generates heatmaps for
    specified thresholds, and computes inter-algorithm correlations for ART features.

    Args:
        art_path (str): Path to the ART data CSV file.
        trt_path (str): Path to the TRT data CSV file.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
            - correlations_df_r_threshold1: DataFrame of correlations >= first threshold.
            - correlations_df_r_threshold2: DataFrame of correlations >= second threshold.
    """
    debug_print("--- Starting ART-TRT Correlation Analysis with Significance Testing ---")

    art_features_df, trt_features_df = _load_and_prepare_features(art_path, trt_path)

    if art_features_df.empty: # Implies trt_features_df is also an empty DataFrame from helper
        debug_print("Feature dataframes are empty after preparation. Aborting correlation analysis.")
        return pd.DataFrame(), pd.DataFrame()

    # Initialize DataFrames for correlations and p-values
    correlation_matrix = pd.DataFrame(index=art_features_df.columns, columns=trt_features_df.columns, dtype=float)
    pvalue_matrix = pd.DataFrame(index=art_features_df.columns, columns=trt_features_df.columns, dtype=float)
    
    for art_col in art_features_df.columns:
        for trt_col in trt_features_df.columns:
            try:
                # Ensure columns are numeric before correlation
                art_series = pd.to_numeric(art_features_df[art_col], errors='coerce')
                trt_series = pd.to_numeric(trt_features_df[trt_col], errors='coerce')
                # Drop NaNs that might have been introduced by coerce, or that existed before
                valid_indices = art_series.notna() & trt_series.notna()
                if valid_indices.sum() < 2: # Need at least 2 pairs to correlate
                    correlation_matrix.loc[art_col, trt_col] = np.nan
                    pvalue_matrix.loc[art_col, trt_col] = np.nan
                    continue
                    
                # Calculate correlation and p-value using pearsonr
                corr, p_value = pearsonr(art_series[valid_indices], trt_series[valid_indices])
                correlation_matrix.loc[art_col, trt_col] = corr
                pvalue_matrix.loc[art_col, trt_col] = p_value
                
            except Exception as e:
                debug_print(f"Error calculating correlation between {art_col} and {trt_col}: {e}")
                correlation_matrix.loc[art_col, trt_col] = np.nan
                pvalue_matrix.loc[art_col, trt_col] = np.nan

    # Convert correlation matrix to long format with p-values
    correlation_long = correlation_matrix.stack().reset_index()
    correlation_long.columns = ['ART_Feature', 'TRT_Feature', 'Correlation']
    p_values_long = pvalue_matrix.stack().reset_index()
    p_values_long.columns = ['ART_Feature', 'TRT_Feature', 'P_Value']
    
    # Merge correlation and p-value data
    correlation_long = pd.merge(correlation_long, p_values_long, on=['ART_Feature', 'TRT_Feature'])
    correlation_long['Abs_Correlation'] = correlation_long['Correlation'].abs()
    correlation_long['Significant'] = correlation_long['P_Value'] < 0.05

    # Process for different correlation thresholds
    correlation_results = {}
    for threshold in CORRELATION_THRESHOLDS:
        high_corr_df = correlation_long[correlation_long['Abs_Correlation'] >= threshold].sort_values(by='Correlation', ascending=False)
        correlation_results[f'r{threshold}'] = high_corr_df
        high_corr_df.to_csv(os.path.join(OUTPUT_DIR, f'art_trt_correlations_r{threshold}.csv'), index=False)
        
        # Save a separate file with only significant correlations
        significant_corr_df = high_corr_df[high_corr_df['Significant']]
        significant_corr_df.to_csv(os.path.join(OUTPUT_DIR, f'art_trt_correlations_r{threshold}_significant.csv'), index=False)
        
        debug_print(f"Found {len(high_corr_df)} correlations with |r| >= {threshold}")
        debug_print(f"Of which {len(significant_corr_df)} are statistically significant (p < 0.05)")

        if not high_corr_df.empty:
            # Create a pivot table for visualization
            subset_pivot = high_corr_df.pivot_table(index='ART_Feature', columns='TRT_Feature', values='Correlation')
            
            if not subset_pivot.empty:
                fig_width = max(8, subset_pivot.shape[1] * 0.5)
                fig_height = max(6, subset_pivot.shape[0] * 0.4)
                fig, ax = plt.subplots(figsize=(fig_width, fig_height))
                
                # Create a mask for non-significant values
                significance_pivot = high_corr_df.pivot_table(index='ART_Feature', columns='TRT_Feature', values='Significant', fill_value=False)
                
                # Plot heatmap
                heatmap = sns.heatmap(subset_pivot, cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
                            linewidths=0.5, annot=True, fmt='.2f', annot_kws={"size": ANNOT_FONTSIZE}, ax=ax)
                
                # Add asterisks for significant correlations
                for i, art_feature in enumerate(subset_pivot.index):
                    for j, trt_feature in enumerate(subset_pivot.columns):
                        if significance_pivot.loc[art_feature, trt_feature]: # Check if the cell exists
                            ax.text(j + 0.85, i + 0.15, '*', fontsize=10, color='black', ha='center', va='center') # Adjusted position for asterisk
                
                ax.set_title(f'ART vs TRT Feature Correlations (|r| ≥ {threshold})\n* indicates p < 0.05', fontsize=TITLE_FONTSIZE)
                ax.set_xlabel("TRT Features", fontsize=LABEL_FONTSIZE)
                ax.set_ylabel("ART Features", fontsize=LABEL_FONTSIZE)
                plt.xticks(fontsize=TICK_FONTSIZE, rotation=45, ha='right')
                plt.yticks(fontsize=TICK_FONTSIZE)
                save_plot(fig, f'art_trt_correlation_heatmap_r{threshold}_with_significance') # Pass fig object
            else:
                debug_print(f"Skipping heatmap for |r| >= {threshold} due to empty pivot table.")

    # Inter-algorithm correlations (using ART features only)
    debug_print("Calculating inter-algorithm correlations for ART features")
    art_feature_columns = art_features_df.columns
    
    # Get unique algorithms present in the actual ART features
    algorithms_present = sorted(list(set(
        extract_feature_algorithm_type(col, KNOWN_ART_PREFIXES) for col in art_feature_columns
        if extract_feature_algorithm_type(col, KNOWN_ART_PREFIXES) != 'TRT'
    )))
    
    if not algorithms_present:
        debug_print("No ART algorithms found in ART features. Skipping inter-algorithm correlation.")
    else:
        debug_print(f"ART Algorithms for inter-correlation: {algorithms_present}")
        algo_features_map = {algo: [col for col in art_feature_columns if extract_feature_algorithm_type(col, KNOWN_ART_PREFIXES) == algo]
                             for algo in algorithms_present}

        num_algos = len(algorithms_present)
        algo_corr_matrix = np.ones((num_algos, num_algos))

        for i, algo1 in enumerate(algorithms_present):
            for j, algo2 in enumerate(algorithms_present):
                if i < j:
                    corrs = []
                    for col1 in algo_features_map[algo1]:
                        for col2 in algo_features_map[algo2]:
                            try:
                                s1 = pd.to_numeric(art_features_df[col1], errors='coerce')
                                s2 = pd.to_numeric(art_features_df[col2], errors='coerce')
                                valid_idx = s1.notna() & s2.notna()
                                if valid_idx.sum() < 2: continue
                                corr_val = abs(s1[valid_idx].corr(s2[valid_idx]))
                                if not np.isnan(corr_val):
                                    corrs.append(corr_val)
                            except Exception as e:
                                debug_print(f"Error in inter-algo corr between {col1} & {col2}: {e}")
                    
                    mean_corr = np.mean(corrs) if corrs else 0 # Use 0 if no valid correlations
                    algo_corr_matrix[i, j] = mean_corr
                    algo_corr_matrix[j, i] = mean_corr
        
        algo_corr_df = pd.DataFrame(algo_corr_matrix, index=algorithms_present, columns=algorithms_present)
        algo_corr_df.to_csv(os.path.join(OUTPUT_DIR, 'art_algorithm_correlation_matrix.csv'))

        if not algo_corr_df.empty:
            fig_algo_corr, ax_algo_corr = plt.subplots(figsize=MEDIUM_FIGSIZE) # Create figure and axes
            sns.heatmap(algo_corr_df, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1,
                        linewidths=0.5, annot=True, fmt='.2f', annot_kws={"size": ANNOT_FONTSIZE}, ax=ax_algo_corr)
            ax_algo_corr.set_title('Mean Absolute Correlation Between ART Algorithm Features', fontsize=TITLE_FONTSIZE)
            plt.xticks(fontsize=TICK_FONTSIZE, rotation=45, ha='right')
            plt.yticks(fontsize=TICK_FONTSIZE)
            save_plot(fig_algo_corr, 'art_algorithm_correlation_heatmap') # Pass fig_algo_corr object

    debug_print("--- ART-TRT Correlation Analysis with Significance Testing Complete ---")
    return correlation_results.get(f'r{CORRELATION_THRESHOLDS[0]}', pd.DataFrame()), \
           correlation_results.get(f'r{CORRELATION_THRESHOLDS[1]}', pd.DataFrame())


# --- feature_importance_analysis_with_permutation (Keep as is) ---
def feature_importance_analysis_with_permutation(combined_data_path):
    # ... (your existing implementation) ...
    debug_print("--- Starting Feature Importance Analysis with Permutation Importance ---")
    data = load_csv_data(combined_data_path, "combined data")
    if data.empty:
        debug_print("Combined data is empty. Aborting feature importance analysis.")
        return None, None

    if 'Group_encoded' not in data.columns:
        if 'Group' not in data.columns:
            debug_print("Error: 'Group' column missing for encoding. Aborting.")
            return None, None
        label_encoder = LabelEncoder()
        data['Group_encoded'] = label_encoder.fit_transform(data['Group'])
        debug_print(f"Encoded 'Group' column. Classes: {label_encoder.classes_}")
    
    meta_cols = ['Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication', 'Group', 'Group_encoded']
    X = data.drop(columns=[col for col in meta_cols if col in data.columns], errors='ignore')
    y = data['Group_encoded']

    if X.empty or X.shape[1] == 0:
        debug_print("Feature set X is empty. Skipping RandomForest training.")
        return None, None

    feature_algo_type_map = {col: extract_feature_algorithm_type(col, KNOWN_ART_PREFIXES) for col in X.columns}
    
    # Scale the data for more fair feature comparisons
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the model
    model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, class_weight='balanced')
    model.fit(X_scaled, y)

    # Calculate standard feature importances
    gini_importances = model.feature_importances_

    # Calculate permutation importance (more robust measure)
    # For permutation importance, it's common to use a held-out set.
    # If you want to evaluate on the training set itself (less common for final reporting but okay for exploration):
    # X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RF_RANDOM_STATE, stratify=y)
    # model.fit(X_train_scaled, y_train) # Refit model on the smaller training part
    # perm_importance = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=RF_RANDOM_STATE, n_jobs=-1)
    
    # Using full X_scaled and y for permutation importance (as in original context)
    perm_importance = permutation_importance(model, X_scaled, y, n_repeats=10, 
                                           random_state=RF_RANDOM_STATE, n_jobs=-1)
    perm_importances = perm_importance.importances_mean
    perm_importances_std = perm_importance.importances_std

    # Create a DataFrame with both importance measures
    importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Gini_Importance': gini_importances,
        'Permutation_Importance': perm_importances,
        'Permutation_Importance_Std': perm_importances_std
    })
    importances_df['Algorithm_Type'] = importances_df['Feature'].map(feature_algo_type_map)
    
    # Sort by permutation importance (more reliable)
    importances_df = importances_df.sort_values('Permutation_Importance', ascending=False)

    # Aggregate importance by algorithm type
    gini_algo_type_importance = importances_df.groupby('Algorithm_Type')['Gini_Importance'].sum().reset_index()
    perm_algo_type_importance = importances_df.groupby('Algorithm_Type')['Permutation_Importance'].sum().reset_index()
    
    # Merge the two importance measures
    algo_type_importance = pd.merge(gini_algo_type_importance, perm_algo_type_importance, 
                                  on='Algorithm_Type', suffixes=('_Gini', '_Perm'))
    algo_type_importance = algo_type_importance.sort_values('Permutation_Importance', ascending=False)

    # Visualizations
    # 1. Algorithm type importance (permutation)
    fig1, ax1 = plt.subplots(figsize=MEDIUM_FIGSIZE)
    bars = sns.barplot(x='Permutation_Importance', y='Algorithm_Type', data=algo_type_importance, 
              palette=SEQUENTIAL_PALETTE, orient='h', ax=ax1)
    
    # Add value annotations
    for bar in bars.patches:
        width = bar.get_width()
        ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    ax1.set_title('Aggregate Feature Importance by Algorithm Type (Permutation)', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Total Permutation Importance', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Algorithm Type', fontsize=LABEL_FONTSIZE)
    save_plot(fig1, 'algorithm_type_importance_permutation') # Pass fig1

    # 2. Top features with error bars (permutation)
    top_features_df = importances_df.head(TOP_N_FEATURES_DISPLAY)
    fig2, ax2 = plt.subplots(figsize=(MEDIUM_FIGSIZE[0], max(8, TOP_N_FEATURES_DISPLAY * 0.4)))
    bars = sns.barplot(x='Permutation_Importance', y='Feature', data=top_features_df, 
              hue='Algorithm_Type', dodge=False, palette=CATEGORICAL_PALETTE_FEATURES, ax=ax2)
    
    # Add error bars
    for i, (index, row) in enumerate(top_features_df.iterrows()): # Iterate with index for y position
        ax2.errorbar(row['Permutation_Importance'], i, xerr=row['Permutation_Importance_Std'], 
                    fmt='none', color='black', capsize=3)
    
    ax2.set_title(f'Top {TOP_N_FEATURES_DISPLAY} Features by Permutation Importance', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Permutation Importance', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Feature', fontsize=LABEL_FONTSIZE)
    plt.legend(title='Algorithm Type', fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_FONTSIZE, loc='best')
    save_plot(fig2, 'top_features_by_permutation_importance') # Pass fig2

    # 3. Comparison of Gini vs Permutation importance for top features
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    top_N = min(20, len(importances_df))
    imp_comparison = importances_df.head(top_N).melt(
        id_vars=['Feature', 'Algorithm_Type'], 
        value_vars=['Gini_Importance', 'Permutation_Importance'],
        var_name='Importance_Type', value_name='Importance_Value'
    )
    sns.barplot(x='Importance_Value', y='Feature', hue='Importance_Type', 
              data=imp_comparison, ax=ax3, palette=['lightblue', 'coral'])
    ax3.set_title('Comparison of Importance Metrics for Top Features', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Importance Score', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Feature', fontsize=LABEL_FONTSIZE)
    plt.legend(title='Importance Type', fontsize=LEGEND_FONTSIZE)
    save_plot(fig3, 'importance_metrics_comparison') # Pass fig3

    # Save results
    algo_type_importance.to_csv(os.path.join(OUTPUT_DIR, 'algorithm_type_importance_comparison.csv'), index=False)
    top_features_df.to_csv(os.path.join(OUTPUT_DIR, 'top_features_with_permutation.csv'), index=False)
    importances_df.to_csv(os.path.join(OUTPUT_DIR, 'all_feature_importances_with_permutation.csv'), index=False)

    debug_print("--- Feature Importance Analysis with Permutation Complete ---")
    return importances_df, top_features_df


# --- trait_stability_analysis (Keep as is) ---
def trait_stability_analysis(art_path, trt_path):
    # ... (your existing implementation) ...
    debug_print("--- Starting Trait Stability Analysis ---")

    art_data = load_csv_data(art_path, "ART data")
    trt_data = load_csv_data(trt_path, "TRT data")

    if art_data.empty or trt_data.empty:
        debug_print("ART or TRT data is empty. Aborting trait stability analysis.")
        return None

    # Check if data_type column exists
    if 'data_type' not in art_data.columns or 'data_type' not in trt_data.columns:
        debug_print("Error: 'data_type' column missing. Required for stability analysis.")
        return None

    # Function to calculate CV for a dataframe
    def calculate_cv(df, meta_columns_cv): # Renamed meta_columns to avoid conflict
        # Remove metadata columns
        features_df = df.drop(columns=meta_columns_cv, errors='ignore')
        
        # Initialize results
        results = []
        
        # Map data_type to consolidated conditions (glasshouse or field)
        df['consolidated_condition'] = df['data_type'].apply(
            lambda x: 'glasshouse' if 'glasshouse' in str(x).lower() else 
                      ('field' if 'field' in str(x).lower() else 'other')
        )
        
        # Calculate CV for each feature across all data
        all_cv = features_df.std() / features_df.mean()
        
        for feature, cv in all_cv.items():
            results.append({
                'Feature': feature,
                'Algorithm_Type': extract_feature_algorithm_type(feature, KNOWN_ART_PREFIXES),
                'Condition': 'All',
                'CV': cv
            })
        
        # Get consolidated conditions
        consolidated_conditions = df['consolidated_condition'].unique()
        debug_print(f"Consolidated conditions: {consolidated_conditions}")
        
        # Calculate CV for each consolidated condition
        for condition in consolidated_conditions:
            if condition in ['glasshouse', 'field']:  # Only process relevant conditions
                condition_df = df[df['consolidated_condition'] == condition]
                if len(condition_df) < 5:  # Skip if too few samples
                    debug_print(f"Skipping CV calculation for {condition} - too few samples")
                    continue
                
                condition_features = condition_df.drop(columns=meta_columns_cv + ['consolidated_condition'], errors='ignore')
                condition_cv = condition_features.std() / condition_features.mean()
                
                for feature, cv in condition_cv.items():
                    results.append({
                        'Feature': feature,
                        'Algorithm_Type': extract_feature_algorithm_type(feature, KNOWN_ART_PREFIXES),
                        'Condition': condition.capitalize(),
                        'CV': cv
                    })
        
        return pd.DataFrame(results)

    # Calculate CV for ART and TRT features
    meta_columns_for_cv = ['Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication', 'Group'] # Specific name
    
    art_cv_df = calculate_cv(art_data, meta_columns_for_cv)
    trt_cv_df = calculate_cv(trt_data, meta_columns_for_cv)
    
    # Add feature type column
    art_cv_df['Feature_Type'] = 'ART'
    trt_cv_df['Feature_Type'] = 'TRT'
    
    # Combine the results
    cv_df = pd.concat([art_cv_df, trt_cv_df])
    
    # Clean up data - remove NaN and infinite values
    cv_df = cv_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Save the complete CV results
    cv_df.to_csv(os.path.join(OUTPUT_DIR, 'trait_stability_cv_full.csv'), index=False)
    
    # Create summary statistics by algorithm type and condition
    cv_summary = cv_df.groupby(['Algorithm_Type', 'Condition'])['CV'].agg(['mean', 'std', 'median']).reset_index()
    cv_summary.to_csv(os.path.join(OUTPUT_DIR, 'trait_stability_cv_summary.csv'), index=False)
    
    # Create visualization - boxplot of CV by algorithm type and condition (original plot)
    fig_box, ax_box = plt.subplots(figsize=(14, 8)) # Create figure and axes
    sns.boxplot(x='Algorithm_Type', y='CV', hue='Condition', data=cv_df, palette='viridis', ax=ax_box)
    ax_box.set_title('Trait Stability (Coefficient of Variation) by Algorithm Type', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Algorithm Type', fontsize=LABEL_FONTSIZE)
    plt.ylabel('Coefficient of Variation (CV)', fontsize=LABEL_FONTSIZE)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Condition', fontsize=LEGEND_FONTSIZE)
    save_plot(fig_box, 'trait_stability_boxplot') # Pass fig_box
    
    # Create a summary table comparing ART vs TRT stability
    art_vs_trt = cv_df.groupby(['Algorithm_Type', 'Condition'])['CV'].mean().reset_index()
    art_vs_trt_pivot = art_vs_trt.pivot(index='Algorithm_Type', columns='Condition', values='CV')
    
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6)) # Create figure and axes
    sns.heatmap(art_vs_trt_pivot, cmap='YlGnBu_r', annot=True, fmt='.3f', ax=ax_heat)
    ax_heat.set_title('Mean Coefficient of Variation by Algorithm Type and Condition', fontsize=TITLE_FONTSIZE)
    save_plot(fig_heat, 'trait_stability_heatmap') # Pass fig_heat
    
    # NEW: Create bar plot comparing ART vs TRT stability
    type_comparison = cv_df.groupby(['Feature_Type', 'Condition'])['CV'].agg(['mean', 'std', 'count']).reset_index()
    overall_comparison = type_comparison[type_comparison['Condition'] == 'All']
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6)) # Create figure and axes
    bars = ax_bar.bar(overall_comparison['Feature_Type'], 
                  overall_comparison['mean'], 
                  yerr=overall_comparison['std']/np.sqrt(overall_comparison['count']),
                  color=['forestgreen', 'darkturquoise'])
    
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=12)
    
    art_cvs = cv_df[(cv_df['Feature_Type'] == 'ART') & (cv_df['Condition'] == 'All')]['CV'].dropna() # dropna before ttest
    trt_cvs = cv_df[(cv_df['Feature_Type'] == 'TRT') & (cv_df['Condition'] == 'All')]['CV'].dropna()
    
    if len(art_cvs) > 1 and len(trt_cvs) > 1: # ttest needs at least 2 samples per group
        t_stat, p_value = ttest_ind(art_cvs, trt_cvs, equal_var=False)
        sig_text = ""
        if p_value < 0.001: sig_text = "***"
        elif p_value < 0.01: sig_text = "**"
        elif p_value < 0.05: sig_text = "*"
        
        if sig_text:
            y_max = max(overall_comparison['mean']) + max(overall_comparison['std'])/np.sqrt(min(overall_comparison['count'])) + 0.02
            ax_bar.plot([0, 0, 1, 1], [y_max, y_max + 0.01, y_max + 0.01, y_max], 'k-')
            ax_bar.text(0.5, y_max + 0.015, sig_text, ha='center', va='bottom', fontsize=14)
        ax_bar.text(0.5, 0.9, f'p = {p_value:.4f}', ha='center', va='center', transform=ax_bar.transAxes, fontsize=12)
    
    ax_bar.set_title('Trait Stability Comparison: ART vs TRT Features', fontsize=TITLE_FONTSIZE)
    plt.ylabel('Mean Coefficient of Variation (Lower = More Stable)', fontsize=LABEL_FONTSIZE)
    ax_bar.set_ylim(0, max(overall_comparison['mean']) * 1.5 if not overall_comparison.empty else 0.1)
    plt.figtext(0.5, 0.01, 'Lower CV indicates higher trait stability', ha='center', fontsize=10, style='italic')
    save_plot(fig_bar, 'art_vs_trt_stability_comparison') # Pass fig_bar
    
    key_features = {
        'HDBSCAN_centre_y': 'ART', 'K-mean_density_points': 'ART',
        'Total.Root.Length.mm': 'TRT', 'Number.of.Root.Tips': 'TRT'
    }
    key_cv = cv_df[cv_df['Feature'].isin(key_features.keys())].copy() # Use .copy()
    key_cv['Type'] = key_cv['Feature'].map(key_features)
    supp_table = key_cv.pivot_table(index=['Feature', 'Type'], columns='Condition', values='CV', aggfunc='mean').reset_index()
    supp_table.to_csv(os.path.join(OUTPUT_DIR, 'trait_stability_supplementary_table.csv'), index=False)
    
    debug_print("--- Trait Stability Analysis Complete ---")
    return cv_df


# --- drought_adaptation_analysis (Keep as is, but ensure it uses correct importance column) ---
def drought_adaptation_analysis(top_features_df, correlations_df_06):
    debug_print("--- Starting Drought Adaptation Mechanism Analysis ---")

    if top_features_df is None or top_features_df.empty:
        debug_print("Top features data not available. Attempting to load.")
        top_features_df = load_csv_data(os.path.join(OUTPUT_DIR, 'top_features_with_permutation.csv'), "top features")
        if top_features_df.empty:
            debug_print("Cannot proceed with drought adaptation without top features data.")
            return

    if correlations_df_06 is None or correlations_df_06.empty:
        debug_print(f"Correlation r{CORRELATION_THRESHOLDS[0]} data not available. Attempting to load.")
        correlations_df_06 = load_csv_data(os.path.join(OUTPUT_DIR, f'art_trt_correlations_r{CORRELATION_THRESHOLDS[0]}.csv'), f"r{CORRELATION_THRESHOLDS[0]} correlations")
        # If still empty, the function will handle it by putting N/A for correlation info.

    # Define feature mechanisms (Consider loading from CSV as suggested previously)
    # This list maps patterns in ART feature names to general mechanisms.
    feature_mechanisms_data = [
        {"ART_Feature_Pattern": "HDBSCAN_centre_y", "Drought_Mechanism": "Deep Rooting Indicator", "Biological_Interpretation": "Vertical position of dense root clusters (HDBSCAN); higher values suggest deeper significant root mass."},
        {"ART_Feature_Pattern": "Mean-shift_centre_y", "Drought_Mechanism": "Deep Rooting Peak", "Biological_Interpretation": "Vertical position of natural root density peaks (Mean-shift); potential adaptive deeper rooting."},
        {"ART_Feature_Pattern": "K-mean_density_points", "Drought_Mechanism": "Root Mass Allocation", "Biological_Interpretation": "Pixel count in K-means clusters; reflects localized root biomass."},
        {"ART_Feature_Pattern": "DBSCAN_density_points", "Drought_Mechanism": "Irregular Dense Zones", "Biological_Interpretation": "Density within DBSCAN clusters; captures non-globular dense root zones."},
        {"ART_Feature_Pattern": "FCM_centre_x", "Drought_Mechanism": "Lateral Spread Indicator", "Biological_Interpretation": "Horizontal centroid of fuzzy clusters; indicates lateral root exploration."},
        {"ART_Feature_Pattern": "SLIC_num_superpixels", "Drought_Mechanism": "Root System Complexity", "Biological_Interpretation": "Number of superpixels; may relate to branching density or overall root system fragmentation."},
        {"ART_Feature_Pattern": "GMM_weights_0", "Drought_Mechanism": "Primary Root Zone", "Biological_Interpretation": "Weight of the first Gaussian component; could represent proportion of roots in a dominant cluster."},
         # Add more known mappings if necessary
    ]
    mechanisms_map_df = pd.DataFrame(feature_mechanisms_data)

    merged_mechanisms_list = []
    # Iterate through top_features_df to link important features to predefined mechanisms
    for _, top_f_row in top_features_df.iterrows():
        feature_name = top_f_row['Feature']
        # Find if this feature matches any ART_Feature_Pattern in mechanisms_map_df
        matched_mech_info = mechanisms_map_df[mechanisms_map_df['ART_Feature_Pattern'] == feature_name]

        if not matched_mech_info.empty:
            mech_details = matched_mech_info.iloc[0]
            entry = {
                'ART_Feature_Actual': feature_name,
                'Drought_Mechanism': mech_details['Drought_Mechanism'],
                'Biological_Interpretation': mech_details['Biological_Interpretation'],
                'Feature_Importance': top_f_row.get('Permutation_Importance', top_f_row.get('Gini_Importance', np.nan))
            }
            merged_mechanisms_list.append(entry)
        # Optionally, include top features even if no predefined mechanism:
        # else:
        #     entry = {
        #         'ART_Feature_Actual': feature_name,
        #         'Drought_Mechanism': 'N/A (Mechanism not predefined)',
        #         'Biological_Interpretation': 'N/A',
        #         'Feature_Importance': top_f_row.get('Permutation_Importance', top_f_row.get('Gini_Importance', np.nan))
        #     }
        #     merged_mechanisms_list.append(entry)


    if not merged_mechanisms_list:
         debug_print("No predefined mechanisms matched with top ART features. The mechanisms table will be based on all top features or be empty.")
         # Fallback: create a basic table from top_features if no mechanisms are defined for them
         if not top_features_df[top_features_df['Algorithm_Type'] != 'TRT'].empty :
             final_mechanisms_df = top_features_df[top_features_df['Algorithm_Type'] != 'TRT'][['Feature', 'Permutation_Importance']].copy()
             final_mechanisms_df.rename(columns={'Feature': 'ART_Feature_Actual', 'Permutation_Importance': 'Feature_Importance'}, inplace=True)
             final_mechanisms_df['Drought_Mechanism'] = 'N/A (Mechanism not predefined)'
             final_mechanisms_df['Biological_Interpretation'] = 'N/A'
         else:
            final_mechanisms_df = pd.DataFrame(columns=['ART_Feature_Actual', 'Drought_Mechanism', 'Biological_Interpretation', 'Feature_Importance'])
    else:
        final_mechanisms_df = pd.DataFrame(merged_mechanisms_list)
        final_mechanisms_df = final_mechanisms_df.drop_duplicates(subset=['ART_Feature_Actual', 'Drought_Mechanism'])

    # Add strongest TRT correlation from correlations_df_06
    if correlations_df_06 is not None and not correlations_df_06.empty:
        # Ensure necessary columns exist from Part 1 processing
        if 'Abs_Correlation' not in correlations_df_06.columns and 'Correlation' in correlations_df_06.columns:
             correlations_df_06['Abs_Correlation'] = correlations_df_06['Correlation'].abs()
        if 'Significant' not in correlations_df_06.columns and 'P_Value' in correlations_df_06.columns:
            correlations_df_06['Significant'] = correlations_df_06['P_Value'] < 0.05
        elif 'Significant' not in correlations_df_06.columns: # Fallback if P_Value also missing
            correlations_df_06['Significant'] = False


        strongest_corrs_for_mech_list = []
        for art_feature_actual in final_mechanisms_df['ART_Feature_Actual'].unique():
            # Filter correlations for the current ART feature
            feature_corrs = correlations_df_06[correlations_df_06['ART_Feature'] == art_feature_actual]
            if not feature_corrs.empty:
                # Find the correlation with the highest absolute value
                top_corr_for_feature = feature_corrs.loc[feature_corrs['Abs_Correlation'].idxmax()]
                strongest_corrs_for_mech_list.append({
                    'ART_Feature_Actual': art_feature_actual,
                    'Top_Correlated_TRT': top_corr_for_feature['TRT_Feature'],
                    'TRT_Correlation_Value': top_corr_for_feature['Correlation'],
                    'Correlation_Significant': top_corr_for_feature.get('Significant', False) # Use .get for safety
                })

        if strongest_corrs_for_mech_list:
            strongest_corrs_df = pd.DataFrame(strongest_corrs_for_mech_list)
            final_mechanisms_df = pd.merge(final_mechanisms_df, strongest_corrs_df, on='ART_Feature_Actual', how='left')
        else:
            final_mechanisms_df['Top_Correlated_TRT'] = "N/A"
            final_mechanisms_df['TRT_Correlation_Value'] = np.nan
            final_mechanisms_df['Correlation_Significant'] = pd.NA # Use pd.NA for boolean missing
    else:
        final_mechanisms_df['Top_Correlated_TRT'] = f"N/A (no r{CORRELATION_THRESHOLDS[0]} corr data)"
        final_mechanisms_df['TRT_Correlation_Value'] = np.nan
        final_mechanisms_df['Correlation_Significant'] = pd.NA

    final_mechanisms_df = final_mechanisms_df.sort_values(by='Feature_Importance', ascending=False).reset_index(drop=True)
    final_mechanisms_df.to_csv(os.path.join(OUTPUT_DIR, 'art_drought_adaptation_mechanisms_table.csv'), index=False)
    debug_print("Saved drought adaptation mechanisms table.")

    # --- Conceptual Diagram Plotting (Adopted from combine_final.py) ---
    fig_concept, ax_concept = plt.subplots(figsize=MEDIUM_FIGSIZE)
    ax_concept.set_xlim(0, 10)
    ax_concept.set_ylim(0, 10)

    # This list defines what's shown on the conceptual diagram.
    # It's hardcoded for consistency with the "nicer" plot style from combine_final.py.
    # You can modify this list or make it dynamic based on your key findings in final_mechanisms_df.
    display_mechanisms_for_plot = [
        {"name": "Deep Rooting", "art_features": ["HDBSCAN_centre_y", "Mean-shift_centre_y"], "pos": (2.5, 8), "color": "SaddleBrown"},
        {"name": "Root Distribution", "art_features": ["K-mean_density_points", "DBSCAN_density_points"], "pos": (7.5, 7), "color": "ForestGreen"},
        {"name": "Lateral Growth", "art_features": ["FCM_centre_x"], "pos": (2.5, 2.5), "color": "DarkGoldenRod"},
        {"name": "System Complexity", "art_features": ["SLIC_num_superpixels"], "pos": (7.5, 3), "color": "SteelBlue"},
        # Example of how to potentially make it dynamic (select top N from final_mechanisms_df):
        # Note: This requires careful selection of positions and colors.
        # dynamic_plot_items = []
        # if not final_mechanisms_df.empty:
        #     # Ensure ART_Feature_Actual exists for this selection
        #     top_n_for_plot = final_mechanisms_df[final_mechanisms_df['ART_Feature_Actual'].notna()].head(3)
        #     plot_positions = [(2.5, 8), (7.5, 7), (2.5, 2.5)] # Define enough positions
        #     plot_colors = ["SaddleBrown", "ForestGreen", "DarkGoldenRod"] # Define enough colors
        #     for i, row in enumerate(top_n_for_plot.iterrows()):
        #         idx, data = row
        #         if i < len(plot_positions): # Check bounds
        #             dynamic_plot_items.append({
        #                 "name": data['Drought_Mechanism'].split(" ")[0], # Just take first word for brevity if long
        #                 "art_features": [data['ART_Feature_Actual']],
        #                 "pos": plot_positions[i],
        #                 "color": plot_colors[i % len(plot_colors)]
        #             })
        # if dynamic_plot_items:
        #    display_mechanisms_for_plot = dynamic_plot_items #  Overwrite if dynamic items were generated
    ]


    # Central "Root System" concept
    ax_concept.text(5, 5, "Root System\nAdaptation", ha='center', va='center', fontsize=TITLE_FONTSIZE + 2,
                    bbox=dict(boxstyle="circle,pad=0.5", fc="lightblue", ec="royalblue", alpha=0.8))

    for mech_info in display_mechanisms_for_plot:
        # Ensure art_features is a list of strings
        if isinstance(mech_info['art_features'], list) and all(isinstance(f, str) for f in mech_info['art_features']):
            art_list_str = "\n".join(mech_info['art_features'])
        else: # Fallback if art_features is not as expected
            art_list_str = str(mech_info.get('art_features', 'N/A'))

        # Truncate if too long
        max_len_art_str = 28 # Adjust as needed
        if len(art_list_str) > max_len_art_str:
             art_list_str = art_list_str[:max_len_art_str-3] + "..."
        if "\n" in art_list_str: # if multiple features, take first one and indicate more
            first_feature = art_list_str.split("\n")[0]
            if len(art_list_str.split("\n")) > 1:
                art_list_str_display = f"{first_feature}\n& more..."
            else:
                art_list_str_display = first_feature
        else:
            art_list_str_display = art_list_str


        text_label = f"{mech_info['name']}\n(e.g., {art_list_str_display})"
        target_pos = mech_info['pos']

        # Line from center to mechanism
        ax_concept.plot([5, target_pos[0]], [5, target_pos[1]], linestyle="--", color='gray', alpha=0.7, linewidth=1.5)

        # Mechanism text box
        ax_concept.text(target_pos[0], target_pos[1], text_label, ha='center', va='center',
                        fontsize=LABEL_FONTSIZE - 3, # Slightly smaller text for compactness
                        bbox=dict(boxstyle="round,pad=0.4", fc=mech_info['color'], alpha=0.75, ec='black', linewidth=0.5))

    ax_concept.set_title("Conceptual Links: ART Features & Drought Adaptation", fontsize=TITLE_FONTSIZE + 1, y=1.03)
    ax_concept.axis('off')
    # save_plot in cor_abl.py handles tight_layout via bbox_inches='tight' and closes the figure.
    save_plot(fig_concept, 'art_drought_adaptation_conceptual_diagram')
    debug_print("Saved drought adaptation conceptual diagram (combine_final.py style).")
    # --- End of Conceptual Diagram Plotting ---

    debug_print("--- Drought Adaptation Mechanism Analysis Complete ---")


# --- NEW: Part 5: Simulated Algorithm Ablation Study ---
def simulate_algorithm_ablation(all_importances_df):
    """
    Simulates an ablation study by summing permutation importances for features
    belonging to each ART algorithm.
    """
    debug_print("--- Starting Simulated Algorithm Ablation Study ---")
    if all_importances_df is None or all_importances_df.empty:
        debug_print("Feature importances data not available. Skipping ablation simulation.")
        return

    # Ensure 'Permutation_Importance' column exists
    if 'Permutation_Importance' not in all_importances_df.columns:
        debug_print("Error: 'Permutation_Importance' column missing in importances_df. Skipping ablation.")
        return

    # Filter for ART algorithms only
    art_importances = all_importances_df[all_importances_df['Algorithm_Type'] != 'TRT'].copy()
    if art_importances.empty:
        debug_print("No ART features found in importances_df for ablation simulation.")
        return

    # Sum permutation importances for each ART algorithm
    # This sum represents the estimated drop in performance if that algorithm's features were removed
    ablation_scores = art_importances.groupby('Algorithm_Type')['Permutation_Importance'].sum().reset_index()
    ablation_scores.rename(columns={'Algorithm_Type': 'Algorithm', 'Permutation_Importance': 'Simulated_Performance_Drop'}, inplace=True)
    ablation_scores = ablation_scores.sort_values('Simulated_Performance_Drop', ascending=False)

    ablation_scores.to_csv(os.path.join(OUTPUT_DIR, 'simulated_algorithm_ablation_scores.csv'), index=False)
    debug_print("Saved simulated algorithm ablation scores.")

    # Create bar plot
    fig_abl, ax_abl = plt.subplots(figsize=MEDIUM_FIGSIZE)
    sns.barplot(x='Simulated_Performance_Drop', y='Algorithm', data=ablation_scores,
                palette=SEQUENTIAL_PALETTE, ax=ax_abl) # Using orient='h' implicitly by x,y assignment
    ax_abl.set_title('Simulated Performance Drop by Removing ART Algorithm Features', fontsize=TITLE_FONTSIZE)
    plt.xlabel('Aggregated Permutation Importance (Simulated Drop)', fontsize=LABEL_FONTSIZE)
    plt.ylabel('ART Algorithm', fontsize=LABEL_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    
    # Add value annotations to bars
    for i, v in enumerate(ablation_scores['Simulated_Performance_Drop']):
        ax_abl.text(v + 0.001, i, f'{v:.4f}', color='black', va='center', fontsize=ANNOT_FONTSIZE -1)

    save_plot(fig_abl, 'simulated_algorithm_ablation_plot')
    debug_print("--- Simulated Algorithm Ablation Study Complete ---")


# --- NEW/MODIFIED: Part for Biological Relevance and Network Diagram ---
def biological_relevance_and_network_analysis(correlation_data_r_strong, correlation_data_r_weak):
    """
    Analyzes biological relevance by creating a heatmap of strongest Algorithm-TRT correlations
    and a network diagram of these relationships.
    Based on logic from combine_final.py's biological_relevance_analysis.
    """
    debug_print("--- Starting Biological Relevance and Network Analysis ---")

    # Determine which correlation dataset to use (prioritize stronger threshold)
    corr_data_to_use = None
    strong_threshold_str = f'r{NETWORK_CORR_THRESHOLD_STRONG}' # e.g. r0.7
    weak_threshold_str = f'r{NETWORK_CORR_THRESHOLD_WEAK}'     # e.g. r0.6

    if correlation_data_r_strong is not None and not correlation_data_r_strong.empty:
        corr_data_to_use = correlation_data_r_strong.copy() # Use a copy
        debug_print(f"Using correlation data with |r| >= {NETWORK_CORR_THRESHOLD_STRONG} (count: {len(corr_data_to_use)})")
    elif correlation_data_r_weak is not None and not correlation_data_r_weak.empty:
        corr_data_to_use = correlation_data_r_weak.copy() # Use a copy
        debug_print(f"Using correlation data with |r| >= {NETWORK_CORR_THRESHOLD_WEAK} (count: {len(corr_data_to_use)})")
    else:
        debug_print("No primary correlation data provided. Attempting to load from file.")
        try:
            # Try loading the stronger threshold file first
            corr_file_strong = os.path.join(OUTPUT_DIR, f'art_trt_correlations_{strong_threshold_str}.csv')
            corr_data_to_use = load_csv_data(corr_file_strong, f"correlations {strong_threshold_str}")
            if corr_data_to_use.empty: raise FileNotFoundError
            debug_print(f"Loaded {len(corr_data_to_use)} correlations ({strong_threshold_str}) from file.")
        except FileNotFoundError:
            try:
                corr_file_weak = os.path.join(OUTPUT_DIR, f'art_trt_correlations_{weak_threshold_str}.csv')
                corr_data_to_use = load_csv_data(corr_file_weak, f"correlations {weak_threshold_str}")
                if corr_data_to_use.empty: raise FileNotFoundError
                debug_print(f"Loaded {len(corr_data_to_use)} correlations ({weak_threshold_str}) from file.")
            except FileNotFoundError:
                debug_print(f"Correlation files not found. Skipping biological relevance and network analysis.")
                return

    if corr_data_to_use.empty:
        debug_print("Correlation data is empty after attempting load. Skipping analysis.")
        return

    # Ensure 'Abs_Correlation' column exists (it should from part 1, but good check)
    if 'Abs_Correlation' not in corr_data_to_use.columns and 'Correlation' in corr_data_to_use.columns:
        corr_data_to_use['Abs_Correlation'] = corr_data_to_use['Correlation'].abs()

    # Extract algorithm type for ART features
    # KNOWN_ART_PREFIXES is a global from cor_abl.py
    corr_data_to_use['Algorithm'] = corr_data_to_use['ART_Feature'].apply(
        lambda x: extract_feature_algorithm_type(x, KNOWN_ART_PREFIXES)
    )
    # Filter out any rows where the ART_Feature was miscategorized as TRT (shouldn't happen if data is clean)
    # or if extract_feature_algorithm_type returned a generic 'Other' for non-ART prefixes.
    # We are only interested in actual ART algorithms here.
    valid_art_algorithms = [prefix for prefix in KNOWN_ART_PREFIXES if 'Mean' not in prefix] + ['Mean-shift'] # Get unique algo names
    corr_data_to_use = corr_data_to_use[corr_data_to_use['Algorithm'].isin(valid_art_algorithms)]

    if corr_data_to_use.empty:
        debug_print("No valid ART algorithm correlations found after filtering. Skipping further analysis.")
        return

    debug_print(f"Identified algorithms in filtered corr_data: {corr_data_to_use['Algorithm'].unique()}")

    # --- 1. Heatmap of Strongest Algorithm-TRT Correlations ---
    # Find the max absolute correlation for each Algorithm-TRT_Feature pair
    # Use .loc with idxmax to avoid potential issues with groupby().first() if NaNs are present in Correlation
    try:
        # Ensure 'Abs_Correlation' is present for idxmax
        if 'Abs_Correlation' not in corr_data_to_use.columns:
             corr_data_to_use['Abs_Correlation'] = corr_data_to_use['Correlation'].abs()

        idx = corr_data_to_use.groupby(['Algorithm', 'TRT_Feature'])['Abs_Correlation'].idxmax()
        grouped_max_abs = corr_data_to_use.loc[idx]

        pivot_table_heatmap = grouped_max_abs.pivot_table(index='Algorithm', columns='TRT_Feature', values='Correlation')
        pivot_table_heatmap = pivot_table_heatmap.fillna(0) # Fill NaNs for non-existent max pairs with 0 for heatmap

        if not pivot_table_heatmap.empty:
            n_rows, n_cols = pivot_table_heatmap.shape
            fig_width = max(10, n_cols * 0.75)
            fig_height = max(8, n_rows * 0.65)

            fig_heatmap, ax_heatmap = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(pivot_table_heatmap, cmap=DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
                        annot=True, fmt='.2f', annot_kws={"size": ANNOT_FONTSIZE}, ax=ax_heatmap,
                        linewidths=.5)
            ax_heatmap.set_title('Strongest ART Algorithm - TRT Feature Correlations', fontsize=TITLE_FONTSIZE)
            ax_heatmap.set_xlabel("Traditional Root Traits (TRT Features)", fontsize=LABEL_FONTSIZE)
            ax_heatmap.set_ylabel("ART Algorithms", fontsize=LABEL_FONTSIZE)
            plt.xticks(fontsize=TICK_FONTSIZE, rotation=45, ha='right')
            plt.yticks(fontsize=TICK_FONTSIZE)
            save_plot(fig_heatmap, 'algorithm_trt_strongest_correlation_heatmap')
        else:
            debug_print("Pivot table for algorithm-TRT strongest correlation heatmap is empty.")
    except Exception as e:
        debug_print(f"Error creating Algorithm-TRT strongest correlation heatmap: {e}")


    # --- 2. Correlation Network Diagram ---
    # Prepare data for network: select top positive and negative correlations for each algorithm
    top_correlations_list_for_network = []
    unique_algorithms_in_data = corr_data_to_use['Algorithm'].unique()

    for algo in unique_algorithms_in_data:
        algo_data = corr_data_to_use[corr_data_to_use['Algorithm'] == algo]
        if algo_data.empty:
            continue

        # Select top 3 positive and top 3 negative based on 'Correlation'
        top_pos = algo_data.nlargest(min(3, len(algo_data[algo_data['Correlation'] > 0])), 'Correlation')
        top_neg = algo_data.nsmallest(min(3, len(algo_data[algo_data['Correlation'] < 0])), 'Correlation')

        # Combine and remove duplicates (in case a feature is in both, though unlikely with strict pos/neg)
        combined = pd.concat([top_pos, top_neg]).drop_duplicates(subset=['ART_Feature', 'TRT_Feature', 'Correlation'])
        top_correlations_list_for_network.append(combined)

    if top_correlations_list_for_network:
        all_top_correlations_for_network_df = pd.concat(top_correlations_list_for_network).reset_index(drop=True)
        # Save this intermediate for inspection if needed
        all_top_correlations_for_network_df.to_csv(os.path.join(OUTPUT_DIR, 'algorithm_top_trt_correlations_for_network.csv'), index=False)
    else:
        debug_print("No top correlations found to build the network diagram.")
        all_top_correlations_for_network_df = pd.DataFrame() # Empty df

    if all_top_correlations_for_network_df.empty:
        debug_print("No data for network diagram. Skipping network plot.")
    else:
        # Filter connections for the network based on thresholds
        # NETWORK_CORR_THRESHOLD_STRONG and NETWORK_CORR_THRESHOLD_WEAK are globals from cor_abl.py
        strong_connections = all_top_correlations_for_network_df[
            all_top_correlations_for_network_df['Abs_Correlation'] >= NETWORK_CORR_THRESHOLD_STRONG
        ]
        current_network_threshold = NETWORK_CORR_THRESHOLD_STRONG

        if strong_connections.empty:
            debug_print(f"No connections with |r| >= {NETWORK_CORR_THRESHOLD_STRONG}, trying with {NETWORK_CORR_THRESHOLD_WEAK}")
            strong_connections = all_top_correlations_for_network_df[
                all_top_correlations_for_network_df['Abs_Correlation'] >= NETWORK_CORR_THRESHOLD_WEAK
            ]
            current_network_threshold = NETWORK_CORR_THRESHOLD_WEAK

        if strong_connections.empty:
            debug_print(f"Still no connections with |r| >= {current_network_threshold} for network diagram. Skipping plot.")
        else:
            debug_print(f"Creating correlation network diagram with |r| >= {current_network_threshold}...")
            # NETWORK_FIGSIZE is a global from cor_abl.py (default (14,12))
            fig_network, ax_network = plt.subplots(figsize=NETWORK_FIGSIZE)

            algorithms_in_network = sorted(strong_connections['Algorithm'].unique())
            traits_in_network = sorted(strong_connections['TRT_Feature'].unique())

            if not algorithms_in_network or not traits_in_network:
                debug_print("Not enough unique algorithms or TRT features for network diagram after filtering.")
            else:
                # Node positioning
                algo_y_max = 10
                trait_y_max = 10
                algo_y_step = algo_y_max / max(1, len(algorithms_in_network) -1) if len(algorithms_in_network) > 1 else algo_y_max
                trait_y_step = trait_y_max / max(1, len(traits_in_network) -1) if len(traits_in_network) > 1 else trait_y_max

                algo_positions = {algo: (1, i * algo_y_step) for i, algo in enumerate(algorithms_in_network)} # X=1 for ART
                trait_positions = {trait: (9, i * trait_y_step) for i, trait in enumerate(traits_in_network)} # X=9 for TRT

                # Plot edges (lines)
                for _, row in strong_connections.iterrows():
                    algo, trt, corr_val = row['Algorithm'], row['TRT_Feature'], row['Correlation']
                    if algo not in algo_positions or trt not in trait_positions:
                        continue # Should not happen if lists are derived from strong_connections

                    x1, y1 = algo_positions[algo]
                    x2, y2 = trait_positions[trt]

                    linewidth = abs(corr_val) * 5 # Scale linewidth by correlation strength
                    # Colors from DIVERGING_CMAP: forestgreen for positive, darkturquoise for negative
                    line_color = 'forestgreen' if corr_val > 0 else 'darkturquoise'
                    alpha_val = min(1.0, abs(corr_val) * 0.6 + 0.4) # Stronger correlations more opaque

                    ax_network.plot([x1, x2], [y1, y2], '-', color=line_color, linewidth=linewidth, alpha=alpha_val, zorder=1)

                # Plot nodes
                node_size = 250 # Adjust as needed
                label_fontsize = LEGEND_FONTSIZE # Use LEGEND_FONTSIZE or a specific one

                for algo, (x, y) in algo_positions.items():
                    ax_network.scatter(x, y, s=node_size, color='darkcyan', zorder=5, edgecolors='black', linewidth=0.5)
                    ax_network.text(x - 0.3, y, algo, fontsize=label_fontsize, ha='right', va='center', weight='bold', zorder=6)

                for trait, (x, y) in trait_positions.items():
                    ax_network.scatter(x, y, s=node_size, color='goldenrod', zorder=5, edgecolors='black', linewidth=0.5)
                    ax_network.text(x + 0.3, y, trait, fontsize=label_fontsize, ha='left', va='center', weight='bold', zorder=6)

                # Legend
                legend_elements = [
                    Line2D([0], [0], color='forestgreen', lw=3, label=f'Positive Correlation (r ≥ {current_network_threshold:.1f})'),
                    Line2D([0], [0], color='darkturquoise', lw=3, label=f'Negative Correlation (r ≤ -{current_network_threshold:.1f})'),
                    mpatches.Patch(facecolor='darkcyan', edgecolor='black', label='ART Algorithms'),
                    mpatches.Patch(facecolor='goldenrod', edgecolor='black', label='TRT Features')
                ]
                ax_network.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.1),
                                  ncol=2, fontsize=LEGEND_FONTSIZE, fancybox=True, shadow=True)

                ax_network.set_title(f'ART Algorithm - TRT Feature Correlation Network (|r| ≥ {current_network_threshold})',
                                     fontsize=TITLE_FONTSIZE + 2, y=1.15)
                ax_network.set_xlim(0, 10)
                ax_network.set_ylim(-1, algo_y_max + 1 if algorithms_in_network else 11) # Adjust ylim based on content
                ax_network.axis('off')
                save_plot(fig_network, 'algorithm_trt_correlation_network')

    debug_print("--- Biological Relevance and Network Analysis Complete ---")


# --- Main Execution ---
if __name__ == '__main__':
    debug_print("Starting combined analysis with all improvements...")
    
    # Improvement 1
    debug_print("\nRunning Part 1: Correlation analysis with significance...")
    corrs_r06, corrs_r07 = art_trt_correlation_analysis_with_significance(ART_FILE_PATH, TRT_FILE_PATH)
    
    # Improvement 2
    debug_print("\nRunning Part 2: Feature importance analysis with permutation...")
    all_importances_df, top_N_importances_df = feature_importance_analysis_with_permutation(COMBINED_FILE_PATH)
    
    # Improvement 3
    debug_print("\nRunning Part 3: Trait stability analysis...")
    cv_df_results = trait_stability_analysis(ART_FILE_PATH, TRT_FILE_PATH) # Using original ART/TRT for CV
    
    # NEW Part 4: Biological Relevance and Network Analysis
    # This uses the correlation results from Part 1
    debug_print("\nRunning Part 4: Biological Relevance and Network Analysis...")
    biological_relevance_and_network_analysis(corrs_r07, corrs_r06) # Pass the stronger and weaker threshold data
    
    # Original Part 3 (now Part 5 - Drought Adaptation)
    debug_print("\nRunning Part 5: Drought adaptation mechanism analysis...")
    # This uses top_N_importances_df from permutation analysis (Part 2)
    # and corrs_r06 from correlation analysis (Part 1)
    drought_adaptation_analysis(top_N_importances_df, corrs_r06) 
    
    # NEW Part 6: Simulated Algorithm Ablation Study (was Part 5)
    debug_print("\nRunning Part 6: Simulated algorithm ablation study...")
    # Uses all_importances_df from feature_importance_analysis_with_permutation (Part 2)
    simulate_algorithm_ablation(all_importances_df)
    
    debug_print(f"\nAll analyses complete. Results saved to {OUTPUT_DIR}")
