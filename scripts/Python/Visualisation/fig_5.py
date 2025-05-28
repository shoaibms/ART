"""
This script generates a composite 2x3 figure (Figure 5) for data visualization.
It combines multiple individual plot types:
- Plots A, B, C: Line plots comparing model performance across different metrics
                 for TRT, ART, and Combined datasets.
- Plots D, E: Cumulative Distribution Function (CDF) plots for ROC AUC and Precision
              metrics from 'cdf.csv'.
- Plot F: Bar chart comparing Model and Validation scores across various metrics.

The script handles data loading from CSV files, plot customization
(colors, line styles, fonts), legend creation, and subplot arrangement.
The final composite figure is saved as 'fig_5.png'.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D # For CDF plot legend

# --- Plot A, B, C: Adapted from fig_5a-c.py ---
def plot_abc_on_axes(fig, ax_a, ax_b, ax_c):
    """
    Plots data from TRT, ART, and Combined datasets onto three provided axes (ax_a, ax_b, ax_c).

    Each axis displays a line plot showing model performance across various metrics
    for one of the datasets. Handles missing data and customizes plot aesthetics.

    Args:
        fig: The main matplotlib Figure object.
        ax_a: Matplotlib Axes object for the TRT dataset plot.
        ax_b: Matplotlib Axes object for the ART dataset plot.
        ax_c: Matplotlib Axes object for the Combined dataset plot.

    Returns:
        tuple: (legend_handles, legend_labels, legend_fontsize)
               - legend_handles: List of Line2D objects for the legend.
               - legend_labels: List of model names for the legend.
               - legend_fontsize: Font size for the legend.
    """
    datasets_paths = {
        'TRT': r'C:\\Users\\ms\\Desktop\\data\\data\\plot_data\\Output_TRT.csv',
        'ART': r'C:\\Users\\ms\\Desktop\\data\\data\\plot_data\\Output_ART.csv',
        'Combined': r'C:\\Users\\ms\\Desktop\\data\\data\\plot_data\\Output_Combine.csv'
    }
    datasets = {}
    for name, path in datasets_paths.items():
        try:
            datasets[name] = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Warning: File not found for dataset {name} at {path}. Plot section for this will be empty.")
            datasets[name] = pd.DataFrame(columns=['Model'] + [f'Metric{i}' for i in range(4)])
        except Exception as e:
            print(f"Warning: Error reading CSV for dataset {name}: {e}. Plot section for this will be empty.")
            datasets[name] = pd.DataFrame(columns=['Model'] + [f'Metric{i}' for i in range(4)])

    # Plot parameters
    axis_label_fontsize = 16
    tick_label_fontsize = 14
    subplot_title_fontsize = 16
    legend_fontsize = 14

    # Consistent color and line style definitions
    colors = [
        '#044f35', '#2d8659', '#4a90b8', '#5cb85c',
        '#7bb3d9', '#8fbc8f', '#65a19f', '#98fb98'
    ]
    line_styles = [
        '-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1)),
        (0, (1, 1)), (0, (5, 1, 1, 1, 1, 1))
    ]

    axes_list = [ax_a, ax_b, ax_c]
    legend_handles = []
    legend_labels = []
    dataset_names_ordered = ['TRT', 'ART', 'Combined']

    for idx, dataset_name in enumerate(dataset_names_ordered):
        df = datasets.get(dataset_name)
        ax = axes_list[idx]

        # Check for 'Model' column and numeric columns
        if 'Model' not in df.columns:
            ax.text(0.5, 0.5, f"'Model' column missing in {dataset_name} data",
                    ha='center', va='center', fontsize=10, color='grey')
            numeric_cols = [] # Ensure numeric_cols is defined
        else:
            numeric_cols = [col for col in df.columns if col != 'Model' and pd.api.types.is_numeric_dtype(df[col])]

        if df.empty or not numeric_cols or df[numeric_cols].isnull().all().all():
            ax.text(0.5, 0.5, "No data available" if df.empty else "No numeric data to plot",
                    ha='center', va='center', fontsize=10, color='grey')
            y_min_plot, y_max_plot = 0, 1 # Default y-limits for empty plots
            ax.set_title(f'{dataset_name} Dataset', fontsize=subplot_title_fontsize, pad=10)
            if idx == 0:
                ax.set_ylabel('Performance Score', fontsize=axis_label_fontsize)
            ax.set_xlabel('Metrics', fontsize=axis_label_fontsize)
            ax.set_xticks([])
            ax.set_xticklabels([])
        else:
            numeric_df = df[numeric_cols].copy() # Use .copy() to avoid SettingWithCopyWarning
            data_min_val = numeric_df.min().min()
            data_max_val = numeric_df.max().max()

            data_range = data_max_val - data_min_val
            padding = 0.05 if pd.isna(data_range) or data_range == 0 else data_range * 0.03

            y_min_candidate = data_min_val - padding
            y_max_candidate = data_max_val + padding

            # Ensure y-limits are within [0, 1] and sensible
            y_min_plot = max(0, y_min_candidate if pd.notna(y_min_candidate) else 0)
            y_max_plot = min(1, y_max_candidate if pd.notna(y_max_candidate) else 1)

            if y_min_plot >= y_max_plot: # Fallback for tricky data
                y_min_plot = max(0, (data_min_val if pd.notna(data_min_val) else 0.5) - 0.1)
                y_max_plot = min(1, (data_max_val if pd.notna(data_max_val) else 0.5) + 0.1)
                if y_min_plot >= y_max_plot: # Final fallback
                    y_min_plot, y_max_plot = 0, 1

            ax.set_ylim(y_min_plot, y_max_plot)

            models = df['Model'].unique()

            for i, model_name in enumerate(models):
                model_data = df[df['Model'] == model_name]
                if model_data.empty or model_data[numeric_cols].isnull().all().all():
                    continue

                values = model_data[numeric_cols].values[0]
                x_pos = range(len(numeric_cols))

                line = ax.plot(x_pos, values,
                               color=colors[i % len(colors)],
                               linestyle=line_styles[i % len(line_styles)],
                               linewidth=2, marker='o', markersize=4, label=model_name, alpha=0.9)

                if idx == 0 and model_name not in legend_labels: # Collect legend items only from the first plot
                    legend_handles.append(line[0])
                    legend_labels.append(model_name)

            ax.set_title(f'{dataset_name} Dataset', fontsize=subplot_title_fontsize, pad=10)
            ax.set_xlabel('Metrics', fontsize=axis_label_fontsize)
            if idx == 0:
                ax.set_ylabel('Performance Score', fontsize=axis_label_fontsize)

            ax.set_xticks(range(len(numeric_cols)))
            ax.set_xticklabels(numeric_cols, rotation=45, ha='right')

        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, color='lightgrey', alpha=0.7)
        ax.grid(True, axis='y', linestyle=':', linewidth=0.3, color='lightgrey', alpha=0.5)
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_edgecolor('darkgrey')
            spine.set_linewidth(1)

    return legend_handles, legend_labels, legend_fontsize

# --- Plot D, E: Adapted from cdf_plot_2.py ---
def plot_cdf_on_axis(ax, dataframe, metric_name, color_mapping, title_fontsize, axis_label_fontsize, tick_label_fontsize, legend_fontsize, legend_line_thickness):
    """
    Plots a Cumulative Distribution Function (CDF) for a given metric on a specified axis.

    Args:
        ax: Matplotlib Axes object to plot on.
        dataframe: Pandas DataFrame containing the data. Must include 'Data_type' and metric_name columns.
        metric_name: String, the name of the column in dataframe to plot the CDF for.
        color_mapping: Dict mapping 'Data_type' values to plot properties (e.g., color, linewidth).
        title_fontsize: Font size for the plot title.
        axis_label_fontsize: Font size for axis labels.
        tick_label_fontsize: Font size for tick labels.
        legend_fontsize: Font size for the legend.
        legend_line_thickness: Thickness of lines in the legend.
    """
    lines = []
    ax.set_title(f'CDF of {metric_name}', fontsize=title_fontsize, pad=10)
    ax.set_xlabel(metric_name, fontsize=axis_label_fontsize)
    ax.set_ylabel('CDF', fontsize=axis_label_fontsize)
    ax.tick_params(axis='x', labelsize=tick_label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)
    ax.grid(True)

    if metric_name not in dataframe.columns:
        ax.text(0.5, 0.5, f"'{metric_name}' column not found", ha='center', va='center', fontsize=10, color='grey')
        return

    unique_data_types = dataframe['Data_type'].unique()
    if not any(data_type in color_mapping for data_type in unique_data_types):
         # Check for 'Combine' vs 'Combined' variations as a special case
         has_mappable_type = False
         for data_type in unique_data_types:
             alternative_name = 'Combined' if data_type == 'Combine' else 'Combine' if data_type == 'Combined' else None
             if alternative_name and alternative_name in color_mapping:
                 has_mappable_type = True
                 break
         if not has_mappable_type:
            ax.text(0.5, 0.5, "Data types in CSV\nnot in color_map", ha='center', va='center', fontsize=10, color='grey')
            return

    plotted_anything = False
    for data_type in unique_data_types:
        actual_label = data_type
        color_map_entry = None

        if data_type in color_mapping:
            color_map_entry = color_mapping[data_type]
        else:
            # Try to match 'Combine' vs 'Combined' variations
            alternative_name = None
            if data_type == 'Combine' and 'Combined' in color_mapping:
                alternative_name = 'Combined'
            elif data_type == 'Combined' and 'Combine' in color_mapping:
                alternative_name = 'Combine' # Should use the key that IS in color_mapping

            if alternative_name and alternative_name in color_mapping:
                 color_map_entry = color_mapping[alternative_name]
                 actual_label = alternative_name # Use the key from color_map for legend
            else:
                print(f"Warning: Data_type '{data_type}' in CDF data not found in color_map. Skipping.")
                continue

        subset = dataframe[dataframe['Data_type'] == data_type][metric_name]
        if subset.empty or subset.isnull().all():
            continue
        
        x_sorted = np.sort(subset.dropna()) # Use a more descriptive name
        if len(x_sorted) == 0:
            continue
        
        y_cdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted) # Use a more descriptive name
        line = ax.plot(x_sorted, y_cdf, label=actual_label,
                       color=color_map_entry['color'],
                       linewidth=color_map_entry['linewidth'])[0]
        lines.append(line)
        plotted_anything = True

    if not plotted_anything and metric_name in dataframe.columns:
        ax.text(0.5, 0.5, f"No valid data for {metric_name}", ha='center', va='center', fontsize=10, color='grey')

    if lines:
        unique_legend_items = {}
        for line_obj in lines:
            # Ensure legend handles are unique based on their labels
            # This handles cases where 'Combine' and 'Combined' might map to the same style
            unique_legend_items[line_obj.get_label()] = line_obj
        
        legend_handles_cdf = [Line2D([0], [0], color=item.get_color(),
                                     linewidth=legend_line_thickness, label=label)
                              for label, item in unique_legend_items.items()]
        ax.legend(handles=legend_handles_cdf, loc='best', fontsize=legend_fontsize)


def plot_d_e_on_axes(ax_d, ax_e):
    """
    Sets up and plots CDF data for 'ROC AUC' and 'Precision' on two provided axes.

    Args:
        ax_d: Matplotlib Axes object for the 'ROC AUC' CDF plot.
        ax_e: Matplotlib Axes object for the 'Precision' CDF plot.
    """
    title_fontsize = 18
    axis_label_fontsize = 16
    tick_label_fontsize = 14
    legend_fontsize = 14
    legend_line_thickness = 2.5

    file_path_cdf = r'C:\\Users\\ms\\Desktop\\data\\data\\plot_data\\cdf.csv'
    # Ensure 'Combined' is used consistently if that's the key in your color map
    color_map_cdf = {
        'TRT': {'color': '#00CED1', 'linewidth': 2},
        'ART': {'color': '#228B22', 'linewidth': 2},
        'Combined': {'color': '#1a4f24', 'linewidth': 2}
    }

    try:
        df_cdf = pd.read_csv(file_path_cdf)
        # Standardize 'Data_type' column if 'Combine' exists and 'Combined' is the target
        if 'Data_type' in df_cdf.columns and 'Combine' in df_cdf['Data_type'].unique() and 'Combined' in color_map_cdf:
            df_cdf['Data_type'] = df_cdf['Data_type'].replace({'Combine': 'Combined'})

    except FileNotFoundError:
        print(f"Error: CDF file '{file_path_cdf}' not found.")
        for ax_item, metric_title in [(ax_d, 'ROC AUC'), (ax_e, 'Precision')]:
             ax_item.text(0.5, 0.5, "CDF Data not found", ha='center', va='center', fontsize=10, color='grey')
             ax_item.set_title(f'CDF of {metric_title}', fontsize=title_fontsize, pad=10)
             ax_item.set_xlabel(metric_title, fontsize=axis_label_fontsize)
             ax_item.set_ylabel('CDF', fontsize=axis_label_fontsize)
             ax_item.tick_params(axis='both', labelsize=tick_label_fontsize) # Apply to both axes
             ax_item.grid(True)
        return
    except Exception as e:
        print(f"Error reading CDF CSV file: {e}")
        for ax_item, metric_title in [(ax_d, 'ROC AUC'), (ax_e, 'Precision')]:
            ax_item.text(0.5, 0.5, "Error loading CDF data", ha='center', va='center', fontsize=10, color='grey')
            ax_item.set_title(f'CDF of {metric_title}', fontsize=title_fontsize, pad=10)
            ax_item.set_xlabel(metric_title, fontsize=axis_label_fontsize)
            ax_item.set_ylabel('CDF', fontsize=axis_label_fontsize)
            ax_item.tick_params(axis='both', labelsize=tick_label_fontsize) # Apply to both axes
            ax_item.grid(True)
        return

    # Check if 'Data_type' column exists after attempting to load
    if 'Data_type' not in df_cdf.columns:
        print(f"Error: 'Data_type' column not found in {file_path_cdf}.")
        for ax_item, metric_title in [(ax_d, 'ROC AUC'), (ax_e, 'Precision')]:
            ax_item.text(0.5, 0.5, "'Data_type' missing\nin CDF data", ha='center', va='center', fontsize=10, color='grey')
            ax_item.set_title(f'CDF of {metric_title}', fontsize=title_fontsize, pad=10)
            # Basic axis setup even if data is bad
            ax_item.set_xlabel(metric_title, fontsize=axis_label_fontsize)
            ax_item.set_ylabel('CDF', fontsize=axis_label_fontsize)
            ax_item.tick_params(axis='both', labelsize=tick_label_fontsize)
            ax_item.grid(True)
        return


    plot_cdf_on_axis(ax_d, df_cdf, 'ROC AUC', color_map_cdf, title_fontsize, axis_label_fontsize, tick_label_fontsize, legend_fontsize, legend_line_thickness)
    plot_cdf_on_axis(ax_e, df_cdf, 'Precision', color_map_cdf, title_fontsize, axis_label_fontsize, tick_label_fontsize, legend_fontsize, legend_line_thickness)

# --- Plot F: Adapted from Fig_5F.py ---
def plot_f_on_axis(ax_f):
    """
    Plots a grouped bar chart for Model and Validation scores on the provided axis.

    Args:
        ax_f: Matplotlib Axes object to plot on.
    """
    # Hardcoded data for Plot F
    data_dict = {
      'ModelPerformance': ["Model score"] * 6 + ["Validation score"] * 6,
      'Metric': ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Specificity"] * 2,
      'Score': [0.96, 0.96, 0.96, 0.96, 0.99, 0.96, 0.5, 0.25, 0.5, 0.3333, 0.5, 0.55]
    }
    df_f = pd.DataFrame(data_dict) # Renamed to df_f for clarity

    metric_order = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Specificity"]
    model_performance_order = ["Model score", "Validation score"]

    # Ensure categorical ordering for consistent plotting
    df_f['Metric'] = pd.Categorical(df_f['Metric'], categories=metric_order, ordered=True)
    df_f['ModelPerformance'] = pd.Categorical(df_f['ModelPerformance'], categories=model_performance_order, ordered=True)
    df_f = df_f.sort_values(by=['Metric', 'ModelPerformance'])

    # Plot parameters
    title_fontsize = 16
    axis_label_fontsize = 15 # Harmonized variable name
    tick_label_fontsize = 14 # Harmonized variable name
    legend_fontsize = 14

    metrics_unique = df_f['Metric'].unique() # From df_f
    model_scores = df_f[df_f['ModelPerformance'] == 'Model score']['Score'].values
    validation_scores = df_f[df_f['ModelPerformance'] == 'Validation score']['Score'].values

    bar_width = 0.35 # Renamed for clarity
    positions = np.arange(len(metrics_unique))

    ax_f.bar(positions - bar_width / 2, model_scores, bar_width,
             label='Model score', color="#2dfab2")
    ax_f.bar(positions + bar_width / 2, validation_scores, bar_width,
             label='Validation score', color="#7dc7b6")

    ax_f.set_xlabel("Metric", fontsize=axis_label_fontsize, labelpad=10)
    ax_f.set_ylabel("Score", fontsize=axis_label_fontsize, labelpad=10)
    ax_f.set_title("Model and Validation Scores", fontsize=title_fontsize, loc='center', pad=10)

    ax_f.set_xticks(positions)
    ax_f.set_xticklabels(metrics_unique, rotation=45, ha="right", fontsize=tick_label_fontsize) # Use tick_label_fontsize
    ax_f.tick_params(axis='y', labelsize=tick_label_fontsize) # Use tick_label_fontsize

    ax_f.legend(loc='upper right', ncol=1, title=None, fontsize=legend_fontsize,
                frameon=True, facecolor='white', edgecolor='gray', framealpha=0.5)

    # Spine and grid styling
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.spines['bottom'].set_color('black')
    ax_f.spines['left'].set_color('black')
    ax_f.spines['bottom'].set_linewidth(1)
    ax_f.spines['left'].set_linewidth(1)
    ax_f.grid(False) # Explicitly turn off grid if not desired

    # Set y-axis limit based on data
    max_score = 0
    if len(model_scores) > 0: max_score = max(max_score, np.max(model_scores))
    if len(validation_scores) > 0: max_score = max(max_score, np.max(validation_scores))
    ax_f.set_ylim(0, max_score * 1.05 if max_score > 0 else 1.05) # Ensure ylim starts at 0

# --- Main script to create combined plot ---
def main():
    """
    Main function to generate and save the combined 2x3 plot.
    """
    np.random.seed(42) # for reproducibility
    fig, axes = plt.subplots(2, 3, figsize=(14, 11))

    # Plot A, B, C (Row 0)
    # Unpack returned values correctly
    legend_handles_abc, legend_labels_abc, legend_fs_abc = plot_abc_on_axes(fig, axes[0, 0], axes[0, 1], axes[0, 2])

    # Plot D, E (Row 1, Col 0 and 1)
    plot_d_e_on_axes(axes[1, 0], axes[1, 1])

    # Plot F (Row 1, Col 2)
    plot_f_on_axis(axes[1, 2])

    # Add subplot labels (A, B, C, D, E, F)
    subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F'] # Renamed for clarity
    for i, ax in enumerate(axes.flatten()):
        ax.text(-0.08, 1.07, subplot_labels[i], transform=ax.transAxes,
                 fontsize=20, fontweight='normal', va='bottom', ha='right')

    # Add overall legend for A,B,C models, placed at the TOP of the figure
    if legend_handles_abc and legend_labels_abc: # Check both
        fig_legend = fig.legend(legend_handles_abc, legend_labels_abc,
                                loc='upper center',
                                bbox_to_anchor=(0.5, 1.0), # Adjusted y for better placement
                                ncol=min(len(legend_labels_abc), 4),
                                fontsize=legend_fs_abc, # Use the returned fontsize
                                frameon=True, fancybox=True, shadow=False, borderpad=0.5, handlelength=2.0)
        if fig_legend:
            for leg_line in fig_legend.get_lines():
                leg_line.set_linewidth(2.5) # Match line weight in plot_abc

    # Adjust layout AFTER placing the legend
    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # Adjusted rect for better fit

    # Save the plot
    save_directory = r'C:\\Users\\ms\\Desktop\\data\\analysis\\plot' # Renamed for clarity
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, 'fig_5.png')

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Combined plot saved successfully to: {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")

    plt.show()

if __name__ == '__main__':
    main()