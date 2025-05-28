# This script performs a comparative analysis of original and augmented datasets (TRT and ART)
# using Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE).
# It loads data, applies dimensionality reduction techniques, and generates a multi-panel plot
# visualizing the distributions of original versus augmented data in the reduced dimensional space.
# The script is configured to save the resulting plot and ensures reproducibility through seeding.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os # For robust path joining and directory creation

# --- Configuration ---
BASE_PATH = "C:\\Users\\ms\\Desktop\\data\\" # PLEASE VERIFY THIS PATH

# --- Plot Saving Configuration ---
SAVE_PLOT_DIR = "C:\\Users\\ms\\Desktop\\data\\analysis\\plot"
PLOT_FILENAME = "pca_tsne_grouped_comparison_plot.png" # Updated filename

# --- Global Font Size Configuration ---
FONT_SIZE_SUPTITLE = 24
FONT_SIZE_PLOT_TITLE = 22
FONT_SIZE_AXIS_LABEL = 20
FONT_SIZE_TICK_LABEL = 16
FONT_SIZE_LEGEND = 16
FONT_SIZE_IN_PLOT_TEXT = 14

# Configuration for base dataset types (TRT, ART)
datasets_config = [
    {
        "base_name": "TRT",
        "original_file": "TRT.csv",
        "augmented_file": "TRT_A_B.csv",
        "color": "#00CED1",  # Dark Turquoise for TRT
        "legend_orig": "TRT",
        "legend_aug": "TRT_A_B"
    },
    {
        "base_name": "ART",
        "original_file": "ART.csv",
        "augmented_file": "ART_A_B.csv",
        "color": "#228B22",  # Forest Green for ART
        "legend_orig": "ART",
        "legend_aug": "ART_A_B"
    }
]


def load_and_preprocess_data(file_path, dataset_name):
    """Loads data from a CSV file and extracts numeric features."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {dataset_name}: {os.path.basename(file_path)}, Shape: {data.shape}")
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            print(f"Warning: No numeric features found in {dataset_name}.")
        return data, numeric_data
    except FileNotFoundError:
        print(f"ERROR: Data file not found: {file_path}")
        return None, pd.DataFrame()

def apply_pca(data_numeric, dataset_name_for_log):
    """Applies PCA to the numeric data."""
    if data_numeric.shape[1] >= 2:
        pca_model = PCA(n_components=2, random_state=42)
        pca_result = pca_model.fit_transform(data_numeric)
        return pca_result
    else:
        print(f"Warning: Not enough numeric features for PCA on {dataset_name_for_log}.")
        return None

def apply_tsne(data_numeric, dataset_name_for_log):
    """Applies t-SNE to the numeric data."""
    if data_numeric.shape[0] > 1:
        perplexity = min(30, data_numeric.shape[0] - 1)
        if perplexity > 0:
            tsne_model = TSNE(n_components=2, n_iter=300, random_state=42, perplexity=perplexity)
            tsne_result = tsne_model.fit_transform(data_numeric)
            return tsne_result
        else:
            print(f"Warning: Perplexity too low for t-SNE on {dataset_name_for_log}.")
            return None
    else:
        print(f"Warning: Not enough samples for t-SNE on {dataset_name_for_log}.")
        return None

def add_jitter(arr, noise_level=0.05):
    if arr is None or arr.shape[0] == 0:
        return arr
    # Apply seeding for reproducible jitter
    np.random.seed(42) # Seed before every random operation for consistency
    if arr.ndim == 1:
        std_dev = np.std(arr) if np.std(arr) > 0 else 1.0
        return arr + np.random.normal(0, noise_level * std_dev, arr.shape)
    std_devs = np.std(arr, axis=0)
    std_devs[std_devs == 0] = 1.0
    noise_scaled = noise_level * std_devs
    return arr + np.random.normal(0, noise_scaled, arr.shape)

def plot_subplot(ax, data_result, color, label, title, xlabel, ylabel, data_available_check):
    """Helper function to plot a single subplot for PCA or t-SNE results."""
    if data_result is not None:
        ax.scatter(add_jitter(data_result[:, 0]), add_jitter(data_result[:, 1]),
                   c=color, label=label, alpha=0.7, s=50)
        ax.legend(loc='best', fontsize=FONT_SIZE_LEGEND)
    else:
        status_message = "Data\nUnavailable"
        status_color = 'orange' # Default for data processed but no result (e.g. not enough features)
        if data_available_check is None: # Check if the original data itself was unavailable
            status_color = 'red'
        ax.text(0.5, 0.5, status_message, ha='center', va='center',
                fontsize=FONT_SIZE_IN_PLOT_TEXT, color=status_color)
    ax.set_title(title, fontsize=FONT_SIZE_PLOT_TITLE)
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_AXIS_LABEL)
    if ylabel is not None: # Only set ylabel if provided (for left column plots)
        ax.set_ylabel(ylabel, fontsize=FONT_SIZE_AXIS_LABEL)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK_LABEL)
    ax.grid(True, linestyle='--', alpha=0.6)

def main():
    """Main function to orchestrate data loading, processing, and plotting."""
    # --- Create Figure (4 rows, 2 columns) ---
    num_plot_rows = len(datasets_config) * 2
    fig, axes = plt.subplots(num_plot_rows, 2,
                             figsize=(15, 6 * num_plot_rows),
                             squeeze=False)
    fig.suptitle('PCA and t-SNE Comparison: Original vs. Augmented', fontsize=FONT_SIZE_SUPTITLE, y=0.985)

    # --- Processing and Plotting Loop ---
    for dataset_idx, config in enumerate(datasets_config):
        base_name = config['base_name']
        original_file_path = os.path.join(BASE_PATH, config['original_file'])
        augmented_file_path = os.path.join(BASE_PATH, config['augmented_file'])
        plot_color = config['color']
        legend_orig_label = config['legend_orig']
        legend_aug_label = config['legend_aug']

        print(f"--- Processing dataset type: {base_name} ---")

        # Determine row indices for this dataset type
        pca_plot_row = dataset_idx * 2
        tsne_plot_row = dataset_idx * 2 + 1

        # --- Load and Process Original Data ---
        original_data, original_data_numeric = load_and_preprocess_data(original_file_path, f"original {base_name}")
        pca_original_result, tsne_original_result = None, None
        if not original_data_numeric.empty:
            pca_original_result = apply_pca(original_data_numeric, f"original {base_name}")
            tsne_original_result = apply_tsne(original_data_numeric, f"original {base_name}")
        
        # --- Load and Process Augmented Data ---
        augmented_data, augmented_data_numeric = load_and_preprocess_data(augmented_file_path, f"augmented {base_name}")
        pca_augmented_result, tsne_augmented_result = None, None
        if not augmented_data_numeric.empty:
            pca_augmented_result = apply_pca(augmented_data_numeric, f"augmented {base_name}")
            tsne_augmented_result = apply_tsne(augmented_data_numeric, f"augmented {base_name}")

        # --- Plotting --- 
        # PCA Original
        plot_subplot(axes[pca_plot_row, 0], pca_original_result, plot_color, legend_orig_label, 
                     f'PCA - {base_name} - Original', 'Principal Component 1', 'Principal Component 2', original_data)
        # PCA Augmented
        plot_subplot(axes[pca_plot_row, 1], pca_augmented_result, plot_color, legend_aug_label, 
                     f'PCA - {base_name} - Augmented', 'Principal Component 1', None, augmented_data) # No Y-label
        # t-SNE Original
        plot_subplot(axes[tsne_plot_row, 0], tsne_original_result, plot_color, legend_orig_label, 
                     f't-SNE - {base_name} - Original', 't-SNE Component 1', 't-SNE Component 2', original_data)
        # t-SNE Augmented
        plot_subplot(axes[tsne_plot_row, 1], tsne_augmented_result, plot_color, legend_aug_label, 
                     f't-SNE - {base_name} - Augmented', 't-SNE Component 1', None, augmented_data) # No Y-label

    # --- Final Adjustments, Save, and Show Plot ---
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust 0.96 (top margin) if suptitle overlaps

    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)
    full_save_path = os.path.join(SAVE_PLOT_DIR, PLOT_FILENAME)

    try:
        plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
        print(f"Plot successfully saved to: {full_save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()

if __name__ == "__main__":
    main()