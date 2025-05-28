import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from scipy.stats import ttest_ind  # For Panel D
from matplotlib.colors import LinearSegmentedColormap

# This script generates a 3x2 composite figure visualizing various aspects of root system analysis.
# It includes panels for:
# A: Drought adaptation mechanisms of root systems.
# B: Performance metrics of clustering algorithms (t-SNE/UMAP).
# C: Correlation heatmap between ART (Automated Root Tracing) and TRT (Traditional Root Trait) features.
# D: Trait stability (Coefficient of Variation) comparison between ART and TRT features.
# E: Simulated performance drop from ablating ART algorithm features.
# F: Model accuracy versus the number of features used.
# The script loads data from various CSV files, processes it, and uses Matplotlib/Seaborn to create and save the figure.


# --- Global Plotting Configuration ---
OUTPUT_BASE_DIR = r'C:\Users\ms\Desktop\data\output'
# --- Seed for reproducibility ---
np.random.seed(42)
# --- End Seed ---

COMPOSITE_PLOT_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, 'plot_independent')
os.makedirs(COMPOSITE_PLOT_OUTPUT_DIR, exist_ok=True)

# Paths to data CSVs
COR_ABL_DATA_DIR = os.path.join(OUTPUT_BASE_DIR, 'cor')
TSNE_UMAP_DATA_DIR = os.path.join(OUTPUT_BASE_DIR, 't-SNE_UMAP')
ACC_VS_FEAT_DATA_DIR = os.path.join(OUTPUT_BASE_DIR, 'rf', 'baseline')


# Font Sizes
PANEL_LABEL_FONTSIZE = 28
TITLE_FONTSIZE = 21
AXIS_LABEL_FONTSIZE = 26
TICK_LABEL_FONTSIZE = 22
LEGEND_TEXT_FONTSIZE = 20
LEGEND_TITLE_FONTSIZE = 22
ANNOTATION_FONTSIZE = 18
POINT_LABEL_FONTSIZE = 20


# Colors
DEFAULT_TEXT_COLOR = 'black'
PANEL_LABEL_COLOR = 'black'
# DIVERGING_CMAP_COOLWARM = sns.color_palette("coolwarm", as_cmap=True) # Keep if used elsewhere, or remove if only Panel C used it
SEQUENTIAL_CMAP = 'BuGn'
PERFORMANCE_METRICS_COLORS = ['#0072B2', '#A0522D', '#00B2B2']
SCATTER_POINT_COLOR = '#69c273'

# --- NEW: Define the DIVERGING_CMAP from cor_abl_2.py for Panel C ---
DIVERGING_CMAP_PANEL_C = LinearSegmentedColormap.from_list(
    'custom_div_cmap_panel_c',
    ['darkturquoise', 'white', 'forestgreen'],
    N=256
)
# --- END NEW ---


# Panel Label Appearance
PANEL_LABEL_FONTWEIGHT = 'normal'
PANEL_LABEL_X_POS = 0.02
PANEL_LABEL_Y_POS = 1.09

# --- Helper Function to Load Data ---
def load_csv_data(file_path, identifier="data"):
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {identifier} from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: {identifier} file not found at {file_path}.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {identifier} from {file_path}: {e}")
        return pd.DataFrame()

# --- Plotting Functions for Each Panel ---

def plot_panel_a_drought_adaptation(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(5, 5, "Root System\nAdaptation", ha='center', va='center', fontsize=POINT_LABEL_FONTSIZE + 2,
            bbox=dict(boxstyle="circle,pad=0.6", fc="lightblue", ec="royalblue", alpha=0.7))
    
    # MODIFIED positions for the mechanisms
    mechanisms = [
        {"name": "Deep Rooting", "pos": (2, 9), "color": "SaddleBrown", "example_feature": "HDBSCAN_centre_y"},
        {"name": "Root Distribution", "pos": (8, 8.8), "color": "ForestGreen", "example_feature": "K-mean_density_points"},
        {"name": "Lateral Growth", "pos": (2.1, 1.0), "color": "DarkGoldenRod", "example_feature": "FCM_centre_x"},
        {"name": "System Complexity", "pos": (8.0, 1.0), "color": "SteelBlue", "example_feature": "SLIC_num_superpixels"},
    ]
    # END MODIFICATION

    for mech in mechanisms:
        ax.plot([5, mech['pos'][0]], [5, mech['pos'][1]], linestyle="--", color='darkgray', alpha=0.9, linewidth=2)
        text_label = f"{mech['name']}\n(e.g., {mech['example_feature']})"
        ax.text(mech['pos'][0], mech['pos'][1], text_label, ha='center', va='center',
                fontsize=POINT_LABEL_FONTSIZE -1,
                bbox=dict(boxstyle="round,pad=0.4", fc=mech['color'], alpha=0.8, ec='black', linewidth=0.75))
    ax.axis('off')

def plot_panel_b_performance_metrics(ax):
    csv_path = os.path.join(TSNE_UMAP_DATA_DIR, 'separation_metrics.csv')
    data = load_csv_data(csv_path, "Separation Metrics")
    if data.empty:
        ax.text(0.5, 0.5, "Data not found:\nseparation_metrics.csv", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    data = data.set_index(data.columns[0])
    data_to_plot = data

    bar_width = 0.8
    metric_colors = {
        'Silhouette Score': PERFORMANCE_METRICS_COLORS[0],
        'Davies-Bouldin Score': PERFORMANCE_METRICS_COLORS[1],
        'Calinski-Harabasz Score': PERFORMANCE_METRICS_COLORS[2]
    }
    colors_for_plot = [metric_colors.get(col, '#808080') for col in data_to_plot.columns]

    data_to_plot.plot(kind='bar', ax=ax, width=bar_width, color=colors_for_plot)

    ax.set_ylabel('Score', fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)
    ax.set_xlabel('', color=DEFAULT_TEXT_COLOR)
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE, labelrotation=45, colors=DEFAULT_TEXT_COLOR, pad=7)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE, colors=DEFAULT_TEXT_COLOR)

    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')

    ax.legend(title='Metric', fontsize=LEGEND_TEXT_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, loc='upper right', bbox_to_anchor=(1, 1.05), frameon=True, edgecolor='gray', framealpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y', linewidth=0.7)

    max_bar_height = 0
    for container in ax.containers:
        for bar_obj in container:
            height = bar_obj.get_height()
            if height > max_bar_height:
                 max_bar_height = height
            if height == 0 and data_to_plot.min().min() >= 0 : continue

            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2.,
                    height + (0.01 * max_bar_height if max_bar_height > 0 else 0.01 * np.abs(height) + 0.1),
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    rotation=90,
                    fontsize=ANNOTATION_FONTSIZE,
                    color=DEFAULT_TEXT_COLOR,
                    fontweight='normal')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], current_ylim[1] * 1.30 if max_bar_height > 0 else current_ylim[1] * 1.1)


def plot_panel_c_correlation_heatmap(ax):
    csv_path = os.path.join(COR_ABL_DATA_DIR, 'art_trt_correlations_r0.7_significant.csv')
    data = load_csv_data(csv_path, "ART-TRT Correlations r0.7 sig")

    if data.empty or 'ART_Feature' not in data.columns or 'TRT_Feature' not in data.columns or 'Correlation' not in data.columns:
        ax.text(0.5, 0.5, "Data not found or incomplete:\nart_trt_correlations_r0.7_significant.csv",
                ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    try:
        pivot_data = data.pivot_table(index='ART_Feature', columns='TRT_Feature', values='Correlation')
    except Exception as e:
        ax.text(0.5, 0.5, f"Error creating pivot table:\n{e}", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    if pivot_data.empty:
        ax.text(0.5, 0.5, "No data for heatmap.", ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    heatmap_annot_fontsize = ANNOTATION_FONTSIZE - 4
    if pivot_data.shape[0] > 10 or pivot_data.shape[1] > 10:
        heatmap_annot_fontsize = ANNOTATION_FONTSIZE - 6

    # --- MODIFIED CMAP FOR PLOT C ---
    sns.heatmap(pivot_data, cmap=DIVERGING_CMAP_PANEL_C, vmin=-1, vmax=1, center=0, # <<< --- USE THE NEW CMAP
                linewidths=0.75, linecolor='white', annot=True, fmt='.2f',
                annot_kws={"size": heatmap_annot_fontsize, "color": "black", "fontweight":"normal"},
                ax=ax, cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
    # --- END MODIFICATION ---

    cbar_ax = ax.figure.axes[-1]
    cbar_ax.yaxis.label.set_size(AXIS_LABEL_FONTSIZE - 2)
    cbar_ax.tick_params(labelsize=TICK_LABEL_FONTSIZE - 2)

    ax.set_xlabel("TRT Features", fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)
    ax.set_ylabel("ART Features", fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)

    xtick_labelsize = TICK_LABEL_FONTSIZE - 4
    ytick_labelsize = TICK_LABEL_FONTSIZE - 4
    if pivot_data.shape[1] > 8:
        xtick_labelsize = TICK_LABEL_FONTSIZE - 6
    if pivot_data.shape[0] > 10:
        ytick_labelsize = TICK_LABEL_FONTSIZE - 6

    ax.tick_params(axis='x', labelsize=xtick_labelsize, labelrotation=45, colors=DEFAULT_TEXT_COLOR, pad=5)
    ax.tick_params(axis='y', labelsize=ytick_labelsize, labelrotation=0, colors=DEFAULT_TEXT_COLOR, pad=5)

    plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')


def plot_panel_d_trait_stability(ax):
    csv_path = os.path.join(COR_ABL_DATA_DIR, 'trait_stability_cv_full.csv')
    cv_df = load_csv_data(csv_path, "Trait Stability CV Full")

    if cv_df.empty or 'Feature_Type' not in cv_df.columns or 'Condition' not in cv_df.columns or 'CV' not in cv_df.columns:
        ax.text(0.5, 0.5, "Data not found or incomplete:\ntrait_stability_cv_full.csv",
                ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    type_comparison = cv_df.groupby(['Feature_Type', 'Condition'])['CV'].agg(['mean', 'std', 'count']).reset_index()
    overall_comparison = type_comparison[type_comparison['Condition'] == 'All']

    if overall_comparison.empty:
        ax.text(0.5, 0.5, "No 'All' condition data for stability comparison.",
                ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    bar_colors = ['forestgreen', 'darkturquoise']
    bar_width = 0.6

    overall_comparison['sem'] = overall_comparison['std'] / np.sqrt(overall_comparison['count'].replace(0,1))

    bars = ax.bar(overall_comparison['Feature_Type'],
                  overall_comparison['mean'],
                  yerr=overall_comparison['sem'],
                  color=bar_colors, capsize=8, width=bar_width, ecolor='black', error_kw={'elinewidth':1.5})

    for i, bar_obj in enumerate(bars):
        height = bar_obj.get_height()
        sem_val = overall_comparison.iloc[i]['sem']
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2., height + sem_val + 0.02,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=ANNOTATION_FONTSIZE, color=DEFAULT_TEXT_COLOR, fontweight='normal')

    art_cvs = cv_df[(cv_df['Feature_Type'] == 'ART') & (cv_df['Condition'] == 'All')]['CV'].dropna()
    trt_cvs = cv_df[(cv_df['Feature_Type'] == 'TRT') & (cv_df['Condition'] == 'All')]['CV'].dropna()

    y_max_val = (overall_comparison['mean'] + overall_comparison['sem']).max() if not overall_comparison.empty else 0

    if len(art_cvs) > 1 and len(trt_cvs) > 1:
        t_stat, p_value = ttest_ind(art_cvs, trt_cvs, equal_var=False, nan_policy='omit')
        sig_text = ""
        if p_value < 0.001: sig_text = "***"
        elif p_value < 0.01: sig_text = "**"
        elif p_value < 0.05: sig_text = "*"

        if sig_text:
            line_y_start = y_max_val + 0.05
            line_height_increment = 0.02
            ax.plot([0, 0, 1, 1], [line_y_start, line_y_start + line_height_increment, line_y_start + line_height_increment, line_y_start],
                    'k-', lw=1.5)
            ax.text(0.5, line_y_start + line_height_increment + 0.01, sig_text,
                    ha='center', va='bottom', fontsize=AXIS_LABEL_FONTSIZE + 2, color=DEFAULT_TEXT_COLOR)

    ax.set_ylabel('Mean Coefficient of Variation\n(Lower = More Stable)', fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)
    ax.set_xlabel('', color=DEFAULT_TEXT_COLOR)
    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE + 2, colors=DEFAULT_TEXT_COLOR, pad=7)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE, colors=DEFAULT_TEXT_COLOR)
    ax.set_ylim(0, (y_max_val * 1.35) if y_max_val > 0 else 0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.6, axis='y', linewidth=0.7)

def plot_panel_e_ablation(ax):
    csv_path = os.path.join(COR_ABL_DATA_DIR, 'simulated_algorithm_ablation_scores.csv')
    data = load_csv_data(csv_path, "Simulated Ablation Scores")

    if data.empty or 'Algorithm' not in data.columns or 'Simulated_Performance_Drop' not in data.columns:
        ax.text(0.5, 0.5, "Data not found or incomplete:\nsimulated_algorithm_ablation_scores.csv",
                ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    palette = sns.color_palette(SEQUENTIAL_CMAP, n_colors=len(data))
    bars = sns.barplot(x='Simulated_Performance_Drop', y='Algorithm', data=data, palette=palette, ax=ax)

    # ax.set_title('Simulated Performance Drop by Removing ART Algorithm Features', fontsize=TITLE_FONTSIZE, color=DEFAULT_TEXT_COLOR)
    ax.set_xlabel('Aggregated Permutation Importance (Simulated Drop)', fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)
    ax.set_ylabel('ART Algorithm', fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)

    ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE, colors=DEFAULT_TEXT_COLOR)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE, colors=DEFAULT_TEXT_COLOR)

    for i, v_val in enumerate(data['Simulated_Performance_Drop']):
        ax.text(v_val + (0.0005 * data['Simulated_Performance_Drop'].max()), i, f'{v_val:.3f}',
                color=DEFAULT_TEXT_COLOR, va='center', fontsize=ANNOTATION_FONTSIZE, fontweight='normal')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.6, axis='x', linewidth=0.7)
    current_xlim = ax.get_xlim()
    ax.set_xlim(current_xlim[0], current_xlim[1] * 1.15)


def plot_panel_f_accuracy_vs_features(ax):
    csv_path = os.path.join(ACC_VS_FEAT_DATA_DIR, 'model_performance_summary.csv')
    data = load_csv_data(csv_path, "Model Performance Summary")

    if data.empty or 'Features' not in data.columns or 'Accuracy' not in data.columns or 'Model' not in data.columns:
        ax.text(0.5, 0.5, "Data not found or incomplete:\nmodel_performance_summary.csv",
                ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    plot_df = data[['Model', 'Features', 'Accuracy']].copy()
    plot_df['Features'] = pd.to_numeric(plot_df['Features'], errors='coerce')
    plot_df = plot_df.dropna(subset=['Features', 'Accuracy'])

    if plot_df.empty:
        ax.text(0.5, 0.5, "No valid data for Accuracy vs Features plot.",
                ha='center', va='center', fontsize=AXIS_LABEL_FONTSIZE, color='red')
        ax.axis('off')
        return

    plot_df = plot_df.sort_values('Features')

    sns.scatterplot(
        x='Features', y='Accuracy', data=plot_df,
        s=150, color=SCATTER_POINT_COLOR, edgecolor='black', alpha=0.8, ax=ax
    )

    x_feature_range = plot_df['Features'].max() - plot_df['Features'].min() if not plot_df['Features'].empty else 1
    horizontal_text_offset = 0.02 * x_feature_range if x_feature_range > 0 else 0.2
    if horizontal_text_offset == 0 : horizontal_text_offset = 0.2

    for idx, row in plot_df.iterrows():
        ax.text(
            row['Features'] + horizontal_text_offset,
            row['Accuracy'] + 0.005,
            row['Model'],
            ha='left',
            va='bottom',
            fontsize=ANNOTATION_FONTSIZE -2,
            color=DEFAULT_TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", lw=0.5, alpha=0.7)
        )

    # ax.set_title('Model Accuracy vs. Number of Features', fontsize=TITLE_FONTSIZE, color=DEFAULT_TEXT_COLOR)
    ax.set_xlabel('Number of Features', fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)
    ax.set_ylabel('Accuracy', fontsize=AXIS_LABEL_FONTSIZE, color=DEFAULT_TEXT_COLOR)

    unique_feature_counts = sorted(plot_df['Features'].unique().astype(int))
    if unique_feature_counts:
        if len(unique_feature_counts) > 10 and (max(unique_feature_counts) - min(unique_feature_counts)) > 20:
             ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE - 2, colors=DEFAULT_TEXT_COLOR)
        elif len(unique_feature_counts) > 0:
            ax.set_xticks(unique_feature_counts)
            ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE - 2, labelrotation=30, colors=DEFAULT_TEXT_COLOR)
            plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
    else:
        ax.tick_params(axis='x', labelsize=TICK_LABEL_FONTSIZE - 2, colors=DEFAULT_TEXT_COLOR)

    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONTSIZE, colors=DEFAULT_TEXT_COLOR)

    min_acc = plot_df['Accuracy'].min() if not plot_df.empty else 0
    max_acc = plot_df['Accuracy'].max() if not plot_df.empty else 1
    ax.set_ylim(max(0, min_acc - 0.05), min(1.05, max_acc + 0.12))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.6, axis='both', linewidth=0.7)


# --- Main Script to Create Composite Figure (Using 3x2 layout) ---
def create_main_composite_figure():
    fig, axs = plt.subplots(3, 2, figsize=(26, 28))

    panel_label_settings = {
        "fontsize": PANEL_LABEL_FONTSIZE,
        "fontweight": PANEL_LABEL_FONTWEIGHT,
        "color": PANEL_LABEL_COLOR,
        "va": 'top', "ha": 'left',
        "bbox": dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.0)
    }

    plot_panel_a_drought_adaptation(axs[0, 0])
    axs[0, 0].text(PANEL_LABEL_X_POS, PANEL_LABEL_Y_POS, 'A', transform=axs[0, 0].transAxes, **panel_label_settings)
    plot_panel_b_performance_metrics(axs[0, 1])
    axs[0, 1].text(PANEL_LABEL_X_POS, PANEL_LABEL_Y_POS, 'B', transform=axs[0, 1].transAxes, **panel_label_settings)

    plot_panel_c_correlation_heatmap(axs[1, 0])
    axs[1, 0].text(PANEL_LABEL_X_POS, PANEL_LABEL_Y_POS, 'C', transform=axs[1, 0].transAxes, **panel_label_settings)
    plot_panel_d_trait_stability(axs[1, 1])
    axs[1, 1].text(PANEL_LABEL_X_POS, PANEL_LABEL_Y_POS, 'D', transform=axs[1, 1].transAxes, **panel_label_settings)

    plot_panel_f_accuracy_vs_features(axs[2, 0])
    axs[2, 0].text(PANEL_LABEL_X_POS, PANEL_LABEL_Y_POS, 'F', transform=axs[2, 0].transAxes, **panel_label_settings)
    plot_panel_e_ablation(axs[2, 1])
    axs[2, 1].text(PANEL_LABEL_X_POS, PANEL_LABEL_Y_POS, 'E', transform=axs[2, 1].transAxes, **panel_label_settings)

    plt.tight_layout(pad=3.5, h_pad=2, w_pad=3.5)

    output_filename = os.path.join(COMPOSITE_PLOT_OUTPUT_DIR, 'composite_figure_3x2_independent_data_final.png')
    plt.savefig(output_filename, dpi=300)
    print(f"Independent 3x2 composite figure saved to: {output_filename}")
    plt.close(fig)

if __name__ == '__main__':
    create_main_composite_figure()