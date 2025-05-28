'''
This script performs analysis and visualization of ART (Algorithmic Root Trait )
and TRT (Traditional Root Trait) data.

Key functionalities include:
- Loading and preprocessing ART, TRT, and combined datasets.
- Calculating correlation matrices and performing hierarchical clustering.
- Generating Principal Component Analysis (PCA) plots.
- Performing Multidimensional Scaling (MDS) analysis.
- Creating two main figures:
    - Figure 1: Large heatmaps of ART and TRT data with a shared colorbar.
    - Figure 2: A composite figure showing smaller heatmaps, PCA plots for ART and TRT,
                 and an MDS plot comparing ART and TRT variables.
- Saving the generated figures to a specified directory.

The script is structured to define global configurations (font sizes, file paths),
load and process data, define plotting functions, and then generate and save figures.
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.colorbar import Colorbar
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA as SKLearnPCA # Renamed to avoid conflict with PCA type hint
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import numpy as np
import os
from typing import List, Tuple # Added for type hinting

# --- Set random seed for reproducibility ---
np.random.seed(42)

# --- Global Font Size Configuration ---
FONT_SIZE_LARGE_HEATMAP_TICK_LABEL = 13
FONT_SIZE_LARGE_HEATMAP_LEGEND = 16

FONT_SIZE_SUPTITLE = 24
FONT_SIZE_PLOT_TITLE = 20
FONT_SIZE_AXIS_LABEL = 18
FONT_SIZE_TICK_LABEL = 16
FONT_SIZE_LEGEND = 16
FONT_SIZE_PANEL_LABEL = 28 # Font size for A, B panel labels
FONT_SIZE_SMALL_HEATMAP_AXIS_LABEL = 16
FONT_SIZE_SMALL_HEATMAP_TICK_LABEL = 14

# --- Plot Saving Configuration ---
SAVE_PLOT_DIR = "C:\\\\Users\\\\ms\\\\Desktop\\\\data\\\\analysis\\\\plot"
PLOT_FILENAME_FIG1 = "figure1_large_heatmaps_top_cbar_adjusted.png"
PLOT_FILENAME_FIG2 = "figure2_analysis_plots.png"

# --- Custom Colormap Definition ---
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'custom_div_cmap', ['darkturquoise', 'white', 'forestgreen'], N=256
)
HEATMAP_CMAP = DIVERGING_CMAP

# --- File Paths ---
ART_FILE_PATH = 'C:\\\\Users\\\\ms\\\\Desktop\\\\data\\\\ART.csv'
TRT_FILE_PATH = 'C:\\\\Users\\\\ms\\\\Desktop\\\\data\\\\TRT.csv'
COMBINED_FILE_PATH = 'C:\\\\Users\\\\ms\\\\Desktop\\\\data\\\\Combine_A_B.csv'

# --- Data Loading and Preprocessing ---
# Note: Global imputer and scaler are generally fine for scripts, but for larger applications,
# consider passing them as arguments or managing them within classes/functions.
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

art_data_orig = pd.read_csv(ART_FILE_PATH)
trt_data_orig = pd.read_csv(TRT_FILE_PATH)

art_data_numeric_cols = art_data_orig.select_dtypes(include=[float, int]).columns
art_data_imputed = imputer.fit_transform(art_data_orig.select_dtypes(include=[float, int]))
art_data_imputed_df = pd.DataFrame(art_data_imputed, columns=art_data_numeric_cols)
art_data_scaled = scaler.fit_transform(art_data_imputed_df)
art_corr = pd.DataFrame(art_data_scaled, columns=art_data_imputed_df.columns).corr()
art_linkage = sch.linkage(pdist(art_corr, metric='euclidean'), method='ward')
art_dendro = sch.dendrogram(art_linkage, no_plot=True)
art_order = art_dendro['leaves']
art_corr_reordered = art_corr.iloc[art_order, art_order]
art_pca_data = SKLearnPCA(n_components=2, random_state=42).fit_transform(art_data_scaled)

trt_data_numeric_cols = trt_data_orig.select_dtypes(include=[float, int]).columns
trt_data_imputed = imputer.fit_transform(trt_data_orig.select_dtypes(include=[float, int]))
trt_data_imputed_df = pd.DataFrame(trt_data_imputed, columns=trt_data_numeric_cols)
trt_data_scaled = scaler.fit_transform(trt_data_imputed_df)
trt_corr = pd.DataFrame(trt_data_scaled, columns=trt_data_imputed_df.columns).corr()
trt_linkage = sch.linkage(pdist(trt_corr, metric='euclidean'), method='ward')
trt_dendro = sch.dendrogram(trt_linkage, no_plot=True)
trt_order = trt_dendro['leaves']
trt_corr_reordered = trt_corr.iloc[trt_order, trt_order]
trt_pca_data = SKLearnPCA(n_components=2, random_state=42).fit_transform(trt_data_scaled)

combined_data = pd.read_csv(COMBINED_FILE_PATH)
ART_VARIABLES = ['DBSCAN_density_points', 'DBSCAN_centre_x', 'DBSCAN_centre_y',
                 'Density_points', 'Density_centre_x', 'Density_centre_y',
                 'FCM_density_points', 'FCM_centre_x', 'FCM_centre_y',
                 'GMM_density_points', 'GMM_centre_x', 'GMM_centre_y',
                 'HDBSCAN_density_points', 'HDBSCAN_centre_x', 'HDBSCAN_centre_y',
                 'K-mean_density_points', 'K-mean_centre_x', 'K-mean_centre_y',
                 'SLIC_density_points', 'SLIC_centre_x', 'SLIC_centre_y',
                 'Mean_shift_density_points', 'Mean-shift_centre_x', 'Mean-shift_centre_y',
                 'OPTICS_density_points', 'OPTICS_centre_x', 'OPTICS_centre_y']
TRT_VARIABLES = ['Number.of.Root.Tips', 'Number.of.Branch.Points', 'Total.Root.Length.mm',
                 'Branching.frequency.per.mm', 'Network.Area.mm2', 'Average.Diameter.mm',
                 'Median.Diameter.mm', 'Maximum.Diameter.mm', 'Perimeter.mm',
                 'Volume.mm3', 'Surface.Area.mm2', 'Root.Length.Diameter.Range.1.mm',
                 'Root.Length.Diameter.Range.2.mm', 'Root.Length.Diameter.Range.3.mm',
                 'Projected.Area.Diameter.Range.1.mm2', 'Projected.Area.Diameter.Range.2.mm2',
                 'Projected.Area.Diameter.Range.3.mm2', 'Surface.Area.Diameter.Range.1.mm2',
                 'Surface.Area.Diameter.Range.2.mm2', 'Surface.Area.Diameter.Range.3.mm2',
                 'Volume.Diameter.Range.1.mm3', 'Volume.Diameter.Range.2.mm3',
                 'Volume.Diameter.Range.3.mm3']

data_art_mds = combined_data[ART_VARIABLES].copy()
data_trt_mds = combined_data[TRT_VARIABLES].copy()

data_art_mds_imputed = imputer.fit_transform(data_art_mds)
data_trt_mds_imputed = imputer.fit_transform(data_trt_mds)
data_art_mds_scaled = scaler.fit_transform(data_art_mds_imputed)
data_trt_mds_scaled = scaler.fit_transform(data_trt_mds_imputed)

df_art_mds_scaled = pd.DataFrame(data_art_mds_scaled, columns=ART_VARIABLES)
df_trt_mds_scaled = pd.DataFrame(data_trt_mds_scaled, columns=TRT_VARIABLES)

data_for_mds_dist_calc = pd.concat([df_art_mds_scaled, df_trt_mds_scaled], axis=1)
distance_matrix = pdist(data_for_mds_dist_calc.T, metric='euclidean')

mds_model = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
mds_results = mds_model.fit_transform(squareform(distance_matrix))
mds_df = pd.DataFrame(mds_results, columns=['MDS1', 'MDS2'])
mds_df['Group'] = ['ART'] * len(ART_VARIABLES) + ['TRT'] * len(TRT_VARIABLES)

combined_vars_corr = data_for_mds_dist_calc.corr()
combined_vars_linkage = sch.linkage(pdist(combined_vars_corr, metric='euclidean'), method='ward')
combined_vars_dendro = sch.dendrogram(combined_vars_linkage, no_plot=True)
combined_vars_order = combined_vars_dendro['leaves']
combined_vars_corr_reordered = combined_vars_corr.iloc[combined_vars_order, combined_vars_order]


# --- Plotting Functions ---
def plot_heatmap_on_ax(
    ax: Axes,
    correlation_matrix: pd.DataFrame,
    title: str,
    vmin: float = -1,
    vmax: float = 1,
    cmap: LinearSegmentedColormap = HEATMAP_CMAP,
    is_large_heatmap: bool = True,
    show_cbar: bool = False,
    show_xlabel: bool = True,
    show_ylabel: bool = True
) -> AxesImage:
    """Plots a heatmap on a given Matplotlib Axes object."""
    im = ax.imshow(correlation_matrix.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    if is_large_heatmap:
        ax.set_xticks(np.arange(len(correlation_matrix.columns)))
        ax.set_yticks(np.arange(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=90, ha='right')
        ax.set_yticklabels(correlation_matrix.index)
        ax.tick_params(axis='x', labelsize=FONT_SIZE_LARGE_HEATMAP_TICK_LABEL)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_LARGE_HEATMAP_TICK_LABEL)
    else:
        # Small heatmap logic
        num_cols = len(correlation_matrix.columns)
        num_rows = len(correlation_matrix.index)
        x_ticks_pos = np.arange(0, num_cols, max(1, num_cols // 5))
        y_ticks_pos = np.arange(0, num_rows, max(1, num_rows // 5))
        ax.set_xticks(x_ticks_pos)
        ax.set_yticks(y_ticks_pos)
        ax.set_xticklabels([str(int(x)) for x in x_ticks_pos]) # Ensure labels are strings
        ax.set_yticklabels([str(int(y)) for y in y_ticks_pos]) # Ensure labels are strings
        ax.tick_params(axis='x', labelsize=FONT_SIZE_SMALL_HEATMAP_TICK_LABEL)
        ax.tick_params(axis='y', labelsize=FONT_SIZE_SMALL_HEATMAP_TICK_LABEL)
        if show_xlabel:
            ax.set_xlabel("Variables", fontsize=FONT_SIZE_SMALL_HEATMAP_AXIS_LABEL)
        if show_ylabel:
            ax.set_ylabel("Variables", fontsize=FONT_SIZE_SMALL_HEATMAP_AXIS_LABEL)
    if title:
        ax.set_title(title, fontsize=FONT_SIZE_PLOT_TITLE)
    return im

def plot_pca_on_ax(
    ax: Axes,
    data: np.ndarray,  # PCA data is a numpy array
    title: str,
    color: str = '#1f77b4'
) -> None:
    """Plots PCA results on a given Matplotlib Axes object."""
    ax.scatter(data[:, 0], data[:, 1], alpha=0.7, color=color, s=50)
    ax.set_title(title, fontsize=FONT_SIZE_PLOT_TITLE)
    ax.set_xlabel('PCA Comp. 1', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('PCA Comp. 2', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK_LABEL)

def plot_mds_on_ax(
    ax: Axes,
    mds_df: pd.DataFrame, # MDS data is a DataFrame
    title: str
) -> None:
    """Plots MDS results on a given Matplotlib Axes object."""
    for group, color_hex in zip(['ART', 'TRT'], ['#228B22', '#00CED1']):
        subset = mds_df[mds_df['Group'] == group]
        # Jittering can be helpful for visualization if points overlap significantly
        jitter_strength_x = 0.01 * (subset['MDS1'].max() - subset['MDS1'].min()) if not subset['MDS1'].empty else 0
        jitter_strength_y = 0.01 * (subset['MDS2'].max() - subset['MDS2'].min()) if not subset['MDS2'].empty else 0
        
        jittered_x = subset['MDS1'] + np.random.normal(0, jitter_strength_x if jitter_strength_x > 0 else 0.01, size=subset['MDS1'].shape)
        jittered_y = subset['MDS2'] + np.random.normal(0, jitter_strength_y if jitter_strength_y > 0 else 0.01, size=subset['MDS2'].shape)
        
        ax.scatter(jittered_x, jittered_y, label=group, color=color_hex, s=100, alpha=0.7)
    ax.set_title(title, fontsize=FONT_SIZE_PLOT_TITLE)
    ax.set_xlabel('MDS Dim. 1', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('MDS Dim. 2', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.legend(fontsize=FONT_SIZE_LEGEND)
    ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK_LABEL)

# ============================================================================
# FIGURE GENERATION LOGIC MOVED TO MAIN FUNCTION
# ============================================================================

# --- Main Execution ---
def main():
    """Main function to orchestrate data processing, plot generation, and saving."""
    # Ensure the save directory exists
    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)

    # --- Figure 1: Large Heatmaps (Panels A, B) ---
    fig1: Figure = plt.figure(figsize=(17, 7.5))
    gs1 = fig1.add_gridspec(2, 2,
                           height_ratios=[0.06, 1],
                           width_ratios=[1, 1],
                           hspace=0.5, # Increased vertical spacing for colorbar
                           wspace=0.3) # Adjusted horizontal spacing between plots

    ax_cbar_top: Axes = fig1.add_subplot(gs1[0, :])
    ax_A: Axes = fig1.add_subplot(gs1[1, 0])
    ax_B: Axes = fig1.add_subplot(gs1[1, 1])

    # Plot heatmaps for Figure 1
    im_A: AxesImage = plot_heatmap_on_ax(ax_A, art_corr_reordered, title='', is_large_heatmap=True, show_cbar=False)
    plot_heatmap_on_ax(ax_B, trt_corr_reordered, title='', is_large_heatmap=True, show_cbar=False)

    # Add shared colorbar for Figure 1
    cbar1: Colorbar = fig1.colorbar(im_A, cax=ax_cbar_top, orientation='horizontal')
    cbar1.set_label('Correlation', fontsize=FONT_SIZE_LARGE_HEATMAP_LEGEND)
    cbar1.ax.tick_params(labelsize=FONT_SIZE_LARGE_HEATMAP_TICK_LABEL)
    cbar1.ax.xaxis.set_ticks_position('top')
    cbar1.ax.xaxis.set_label_position('top')

    # Add panel labels for Figure 1
    ax_A.text(-0.12, 1.08, 'A', transform=ax_A.transAxes, # Adjusted x for better positioning
              fontsize=FONT_SIZE_PANEL_LABEL, fontweight='normal', va='bottom', ha='right')
    ax_B.text(-0.12, 1.08, 'B', transform=ax_B.transAxes, # Adjusted x for better positioning
              fontsize=FONT_SIZE_PANEL_LABEL, fontweight='normal', va='bottom', ha='right')
    
    # Using tight_layout after all elements are added
    fig1.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95]) # rect to reserve space for suptitle if added

    # --- Figure 2: Analysis Plots (Panels C-H) ---
    fig2: Figure = plt.figure(figsize=(20, 11)) # Adjusted figsize for better layout
    gs2 = fig2.add_gridspec(2, 4, 
                           width_ratios=[1, 1, 1, 0.05], # Last ratio for colorbar
                           hspace=0.45, wspace=0.35) # Adjusted spacing

    ax_C: Axes = fig2.add_subplot(gs2[0, 0])
    ax_D: Axes = fig2.add_subplot(gs2[0, 1])
    ax_E: Axes = fig2.add_subplot(gs2[0, 2])
    ax_cbar2: Axes = fig2.add_subplot(gs2[0, 3])
    ax_F: Axes = fig2.add_subplot(gs2[1, 0])
    ax_G: Axes = fig2.add_subplot(gs2[1, 1])
    ax_H: Axes = fig2.add_subplot(gs2[1, 2])

    # Plot heatmaps for Figure 2
    im_C: AxesImage = plot_heatmap_on_ax(ax_C, art_corr_reordered, title='', is_large_heatmap=False,
                                      show_cbar=False, show_xlabel=False, show_ylabel=True)
    plot_heatmap_on_ax(ax_D, trt_corr_reordered, title='', is_large_heatmap=False,
                       show_cbar=False, show_xlabel=True, show_ylabel=False)
    plot_heatmap_on_ax(ax_E, combined_vars_corr_reordered, title='', is_large_heatmap=False,
                       show_cbar=False, show_xlabel=False, show_ylabel=False)

    # Add colorbar for Figure 2 heatmaps
    cbar2: Colorbar = fig2.colorbar(im_C, cax=ax_cbar2)
    cbar2.set_label('Correlation', fontsize=FONT_SIZE_LEGEND)
    cbar2.ax.tick_params(labelsize=FONT_SIZE_TICK_LABEL)

    # Plot PCA and MDS for Figure 2
    plot_pca_on_ax(ax_F, art_pca_data, title='PCA - ART', color='#228B22')
    plot_pca_on_ax(ax_G, trt_pca_data, title='PCA - TRT', color='#00CED1')
    plot_mds_on_ax(ax_H, mds_df, title='MDS of Variables')

    # Add panel labels for Figure 2
    panel_axes_fig2: List[Axes] = [ax_C, ax_D, ax_E, ax_F, ax_G, ax_H]
    panel_labels_fig2: List[str] = ['C', 'D', 'E', 'F', 'G', 'H']
    for ax, label in zip(panel_axes_fig2, panel_labels_fig2):
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=FONT_SIZE_PANEL_LABEL, fontweight='normal', va='top', ha='right')
    
    fig2.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96]) # rect to reserve space for suptitle if added

    # --- Save Plots ---
    full_save_path_fig1 = os.path.join(SAVE_PLOT_DIR, PLOT_FILENAME_FIG1)
    full_save_path_fig2 = os.path.join(SAVE_PLOT_DIR, PLOT_FILENAME_FIG2)

    try:
        fig1.savefig(full_save_path_fig1, dpi=300, bbox_inches='tight')
        print(f"Figure 1 successfully saved to: {full_save_path_fig1}")
        fig2.savefig(full_save_path_fig2, dpi=300, bbox_inches='tight')
        print(f"Figure 2 successfully saved to: {full_save_path_fig2}")
    except Exception as e:
        print(f"Error saving plots: {e}")

    plt.show()

    # --- Summary Print ---
    print("\n" + "="*60)
    print("SCRIPT EXECUTION SUMMARY") # Changed title for clarity
    print("="*60)
    print(f"Figure 1 saved to: {full_save_path_fig1}")
    print(f"Figure 2 saved to: {full_save_path_fig2}")
    print(f"Random seed set to 42 for reproducibility.")
    print(f"Script refactored for clarity, modularity, and PEP8 adherence.")
    # Detailed printout of Figure 1 changes can be removed or kept if still desired.
    # For this refactoring, focusing on the overall script changes.
    # print("SUMMARY OF CHANGES FOR FIGURE 1 (Adjusted Layout):")
    # print(f"  - Colorbar thickness reduced (height_ratios=[0.06, 1]).")
    # print(f"  - Vertical spacing (hspace) between colorbar and plots increased to 0.5.")
    # print(f"  - Horizontal spacing (wspace) between plots A & B set to 0.3.") 
    # print(f"  - Panel labels 'A' and 'B' repositioned to (-0.12, 1.08) with va='bottom' to sit above plots.")
    # print(f"  - Overall figure padding adjusted with fig1.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95]).")
    print("="*60)

if __name__ == "__main__":
    main()