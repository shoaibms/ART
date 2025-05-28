# This script generates a two-panel figure (A and B) to visualize and compare
# machine learning model performance metrics across different datasets and methods.
# Panel A displays a comparative plot of various metrics (Accuracy, Precision, etc.)
# for different methods grouped by dataset type.
# Panel B shows a line plot comparing performance metrics for three distinct methods:
# TRT, ART, and a Combined approach.
# The script loads data from CSV files, processes it, and uses matplotlib to create
# and customize the plots, including font sizes, colors, and layout adjustments,
# finally saving the figure to a PNG file.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- Global Font Size Configuration ---
FONT_SIZE_SUPTITLE = 24
FONT_SIZE_PLOT_TITLE = 22  # For A, B, C, D titles
FONT_SIZE_AXIS_LABEL = 20  # For X and Y axis labels
FONT_SIZE_TICK_LABEL = 16  # For X and Y tick labels
FONT_SIZE_LEGEND = 16
FONT_SIZE_IN_PLOT_TEXT = 14

# --- Reproducibility ---
np.random.seed(42)

# Define colors - different shades of green and blue
COLORS_METRICS = ['#2E8B57', '#b2f74a', '#3CB371', '#025f6e', '#32CD32', '#87CEEB']  # Green and blue shades for metrics
COLORS_METHODS = ['#00CED1', '#228B22', '#1a4f24']  # User defined colors


def add_jitter(values, jitter_amount=0.01):
    """Adds a small amount of random noise to data for better visualization."""
    return values + np.random.normal(0, jitter_amount, len(values))


def create_subplot_a(ax, df, metrics):
    """Creates Subplot A: Performance Comparison Across Datasets."""
    first_group_methods = df['Method'][:3]
    second_group_methods = df['Method'][3:6]
    third_group_methods = df['Method'][6:]

    # Plotting the first group
    for idx, metric in enumerate(metrics):
        ax.plot(first_group_methods, add_jitter(df[metric][:3]),
                marker='o', label=metric if first_group_methods.iloc[0] == 'TRT_baseline' else "",
                color=COLORS_METRICS[idx])

    # Plotting the second group
    for idx, metric in enumerate(metrics):
        ax.plot(second_group_methods, add_jitter(df[metric][3:6]),
                marker='o', color=COLORS_METRICS[idx])

    # Plotting the third group
    for idx, metric in enumerate(metrics):
        ax.plot(third_group_methods, add_jitter(df[metric][6:]),
                marker='o', color=COLORS_METRICS[idx])

    ax.axvline(x=2.5, color='grey', linestyle='--')
    ax.axvline(x=5.5, color='grey', linestyle='--')

    ax.set_xlabel('Datasets', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Score', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title('A', fontsize=FONT_SIZE_PLOT_TITLE, fontweight='normal', loc='left')

    all_methods = df['Method'].tolist()
    ax.set_xticks(range(len(all_methods)))
    ax.set_xticklabels(all_methods, rotation=45, ha='right', fontsize=FONT_SIZE_TICK_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK_LABEL)
    ax.legend(fontsize=FONT_SIZE_LEGEND, loc='upper left', framealpha=0.4)
    ax.grid(False)


def create_subplot_b(ax, df):
    """Creates Subplot B: Performance Metrics Comparison."""
    methods = df['Method'].unique()
    method_labels = ['TRT', 'ART', 'Combined']  # Simplified names for legend

    for i, method in enumerate(methods):
        subset = df[df['Method'] == method]
        ax.plot(subset.columns[1:], subset.iloc[0, 1:],
                label=method_labels[i],
                color=COLORS_METHODS[i],
                linestyle='-',
                marker='o',
                markersize=8,
                linewidth=3)

    ax.set_xlabel('Metrics', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel('Score', fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_title('B', fontsize=FONT_SIZE_PLOT_TITLE, fontweight='normal', loc='left')

    metric_names = df.columns[1:].tolist()
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right', fontsize=FONT_SIZE_TICK_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK_LABEL)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_b = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                         ncol=3, fontsize=FONT_SIZE_LEGEND, framealpha=0.4)
    plt.setp(legend_b.get_texts(), color='black')
    for leg_line in legend_b.get_lines():
        leg_line.set_linewidth(3)

    ax.set_ylim([0.88, 1])
    ax.grid(False)


def main():
    """Main function to load data, create plots, and save the figure."""
    # Load the data from both files
    df_a = pd.read_csv(r'C:\Users\ms\Desktop\p\rf\ART_TRT_Combine_result_compare.csv')
    df_b = pd.read_csv(r'C:\Users\ms\Desktop\p\rf\Result_compare_all_A_B _ TRT_ART_Combine3.csv')

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # --- SUBPLOT A: Performance Comparison Across Datasets ---
    metrics_subplot_a = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Specificity']
    create_subplot_a(ax1, df_a, metrics_subplot_a)

    # --- SUBPLOT B: Performance Metrics Comparison ---
    create_subplot_b(ax2, df_b)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(r'C:\Users\ms\Desktop\data\analysis\plot\fig_6_A-B.png',
                dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    main()