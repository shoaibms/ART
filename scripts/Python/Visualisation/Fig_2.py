# This script generates a 2x2 facet plot comparing different algorithm performances
# and variable effect sizes. It includes:
# A: Bump chart showing rank changes.
# B: Grouped bar chart for effect sizes.
# C: Line plot from a primary dataset.
# D: Line plot from a secondary dataset.
# The script handles data loading, plot creation using matplotlib, styling,
# and saving the final composite figure.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# --- Reproducibility ---
np.random.seed(42)

# --- Configuration ---
OUTPUT_DIR = r'C:\\Users\\ms\\Desktop\\data\\analysis\\plot'
# Data file for plot type originally like fig_2_c.py (now for Facet C)
DATA_FILE_FACET_C_SOURCE = r'C:\\Users\\ms\\Desktop\\p\\cluster_comparison_results_T0&T1.csv'
# Data file for plot type originally like fig_2_d.py (now for Facet D)
DATA_FILE_FACET_D_SOURCE = r'C:\\Users\\ms\\Desktop\\p\\cluster_comparison_results_T0&T1_2.csv'

# --- Global Font Size Configuration ---
FONT_SIZE_SUPTITLE = 24
FONT_SIZE_PLOT_TITLE = 22  # For A, B, C, D titles
FONT_SIZE_AXIS_LABEL = 20  # For X and Y axis labels
FONT_SIZE_TICK_LABEL = 16  # For X and Y tick labels
FONT_SIZE_LEGEND = 16
FONT_SIZE_IN_PLOT_TEXT = 14 # For text annotations within plots (e.g., on bars, bump chart labels)

# --- Global Style Definitions for Line Plots ---
LINE_PLOT_LINECOLORS = {
    'ARI-T0': '#c7e9c0',  # light green
    'ARI-T1': '#74c476',  # medium green
    'NMI-T0': '#a1d99b',  # another light green
    'NMI-T1': '#41ab5d',  # another medium-dark green
    'FMS-T0': '#cdf549',  # very light green
    'FMS-T1': '#238b45',  # dark green
}

LINE_PLOT_LINEWIDTHS = {
    'ARI-T0': 3.0, 'ARI-T1': 5.5,
    'NMI-T0': 3.0, 'NMI-T1': 5.5,
    'FMS-T0': 3.0, 'FMS-T1': 5.5,
}

# --- Helper function to ensure output directory exists ---
def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

# --- Plot 1 & 2: Line plot (common function) ---
def create_line_comparison_plot(ax, data_file_path, plot_title="", show_legend=True):
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}")
        ax.text(0.5, 0.5, f"Data file not found:\n{os.path.basename(data_file_path)}", 
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(plot_title, loc='left', fontsize=FONT_SIZE_PLOT_TITLE, 
                     fontweight='normal', color='black')
        return {} # Return empty dict on error

    df_melted = df.melt(id_vars=['Algorithm', 'Treatment'], var_name='Metric', 
                        value_name='Value')

    plotted_labels = set() # To track labels already added to plots, preventing duplicate legend entries from ax.plot()

    # Plot each metric for each treatment
    for metric in df_melted['Metric'].unique():
        key_t0 = f'{metric}-T0'
        key_t1 = f'{metric}-T1'

        df_metric_t0 = df_melted[(df_melted['Metric'] == metric) & (df_melted['Treatment'] == 'T0')]
        df_metric_t1 = df_melted[(df_melted['Metric'] == metric) & (df_melted['Treatment'] == 'T1')]
        
        if not df_metric_t0.empty and key_t0 in LINE_PLOT_LINECOLORS:
            ax.plot(df_metric_t0['Algorithm'], df_metric_t0['Value'], marker='', 
                    color=LINE_PLOT_LINECOLORS[key_t0], 
                    linewidth=LINE_PLOT_LINEWIDTHS[key_t0], 
                    alpha=0.9, label=key_t0 if key_t0 not in plotted_labels else "")
            plotted_labels.add(key_t0)
        if not df_metric_t1.empty and key_t1 in LINE_PLOT_LINECOLORS:
            ax.plot(df_metric_t1['Algorithm'], df_metric_t1['Value'], marker='', 
                    color=LINE_PLOT_LINECOLORS[key_t1], 
                    linewidth=LINE_PLOT_LINEWIDTHS[key_t1], 
                    alpha=0.9, label=key_t1 if key_t1 not in plotted_labels else "")
            plotted_labels.add(key_t1)

    ax.set_facecolor('none')
    ax.grid(False)
    ax.tick_params(axis='x', rotation=45, labelsize=FONT_SIZE_TICK_LABEL)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    ax.set_ylim(bottom=-0.05, top=1.05) # Set bottom slightly below 0 to avoid overlap

    handles, labels = ax.get_legend_handles_labels()
    
    unique_legend_items = {} # Dictionary to store unique handles and labels for the legend
    for handle, label_text in zip(handles, labels):
        if label_text and label_text not in unique_legend_items : # Ensure label is not empty and unique
             unique_legend_items[label_text] = handle
    
    if show_legend and unique_legend_items: 
        legend = ax.legend(handles=unique_legend_items.values(), 
                           labels=unique_legend_items.keys(), 
                           loc='upper left', bbox_to_anchor=(1.02, 1), 
                           ncol=1, fontsize=FONT_SIZE_LEGEND)
        for handle, label_text in zip(legend.legend_handles, unique_legend_items.keys()):
            if label_text in LINE_PLOT_LINEWIDTHS:
                 handle.set_linewidth(LINE_PLOT_LINEWIDTHS[label_text])

    ax.set_title(plot_title, loc='left', fontsize=FONT_SIZE_PLOT_TITLE, 
                 fontweight='normal', color='black')
    ax.set_xlabel("Algorithm", fontsize=FONT_SIZE_AXIS_LABEL, fontweight='normal')
    ax.set_ylabel("Score", fontsize=FONT_SIZE_AXIS_LABEL, fontweight='normal')
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK_LABEL)
    return unique_legend_items # Return the collected items for potential external legend

# --- Plot 3: Grouped Bar Chart (for Facet B) ---
def create_grouped_bar_chart(ax, plot_title=""):
    categories = ['g_s_1', 'g_s_2', 'RWC', 'Tiller_no']
    T_values = [0.736111111, 0.430555556, 0.486111111, 0.722222222]
    S_values = [0.958333333, 0.986111111, 0.569444444, 1]

    colors_T = ['#edf8e9', '#c7e9c0', '#a1d99b', '#74c476']
    colors_S = ['#41ab5d', '#238b45', '#006d2c', '#005a32'] 

    bar_width = 0.35
    x_coords_group1 = np.arange(len(T_values))
    x_coords_group2 = [x + bar_width for x in x_coords_group1]

    for i in range(len(categories)):
        ax.bar(x_coords_group1[i], T_values[i], color=colors_T[i], 
               width=bar_width, edgecolor='grey', label='T' if i == 0 else "")
        ax.bar(x_coords_group2[i], S_values[i], color=colors_S[i], 
               width=bar_width, edgecolor='grey', label='S' if i == 0 else "")
        
        ax.text(x_coords_group1[i], T_values[i] + 0.02, 'T', ha='center', 
                va='bottom', fontsize=FONT_SIZE_IN_PLOT_TEXT, fontweight='bold')
        ax.text(x_coords_group2[i], S_values[i] + 0.02, 'S', ha='center', 
                va='bottom', fontsize=FONT_SIZE_IN_PLOT_TEXT, fontweight='bold')

    ax.set_xlabel('Variable', fontsize=FONT_SIZE_AXIS_LABEL, fontweight='normal')
    ax.set_xticks([r + bar_width / 2 for r in range(len(T_values))])
    ax.set_xticklabels(categories, fontsize=FONT_SIZE_TICK_LABEL)
    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK_LABEL)
    
    ax.set_title(plot_title, loc='left', fontsize=FONT_SIZE_PLOT_TITLE, 
                 fontweight='normal', color='black')
    ax.set_ylabel('Effect Size (Cliff\'s Delta)', fontsize=FONT_SIZE_AXIS_LABEL, 
                  fontweight='normal')
    ax.set_ylim(bottom=0, top=1.15) # Set bottom to 0


# --- Plot 4: Bump Chart (for Facet A) ---
def create_bump_chart(ax, plot_title=""):
    data = {
        'Genotype': ["DT_3", "DT_2", "DT_1", "DS_1", "DS_3", "DS_2"],
        'Rank_1': [1, 2, 3, 4, 5, 6], 'Rank_2': [1, 2, 3, 4, 6, 5], 'Rank_3': [1, 2, 3, 4, 5, 6]
    }
    df = pd.DataFrame(data)
    custom_colors = {
        "DT_1": "#78ccb5", "DT_2": "#66dec9", "DT_3": "#10b397",
        "DS_1": "#f7ba36", "DS_2": "#f5c47a", "DS_3": "#ada358"
    }
    rank_types = ["Rank_1", "Rank_2", "Rank_3"] # x-axis categories
    x_coords = np.arange(len(rank_types))    # numerical x-coordinates for plotting

    # Text label nudging (visual effect, sign depends on axis inversion)
    y_nudge_visual_up = 0.3  # Nudge text UP visually (towards smaller rank values)
    x_nudge_horizontal = 0.08 # Small horizontal nudge to separate text from points if needed

    for _, row in df.iterrows():
        genotype = row['Genotype']
        ranks = [row[rt] for rt in rank_types]
        ax.plot(x_coords, ranks, marker='o', markersize=10, linewidth=3, 
                color=custom_colors[genotype])
        
        # Add text labels for Genotype at Rank_1 and Rank_3, mimicking R's nudge_y
        # For Rank_1 labels:
        if ranks[0] is not None: 
             ax.text(x_coords[0] + x_nudge_horizontal, ranks[0] - y_nudge_visual_up, 
                    genotype, ha='left', va='center', 
                    fontsize=FONT_SIZE_IN_PLOT_TEXT, color=custom_colors[genotype])
        # For Rank_3 labels:
        if ranks[2] is not None: # ranks[2] corresponds to Rank_3
             ax.text(x_coords[2] - x_nudge_horizontal, ranks[2] - y_nudge_visual_up, 
                    genotype, ha='right', va='center', 
                    fontsize=FONT_SIZE_IN_PLOT_TEXT, color=custom_colors[genotype])

    ax.set_xticks(x_coords)
    ax.set_xticklabels(rank_types, fontsize=FONT_SIZE_TICK_LABEL, rotation=45, ha='right')
    ax.invert_yaxis() 
    
    # Determine the maximum rank to set y-axis ticks appropriately
    max_rank = 0
    for rt in rank_types:
        if rt in df.columns and df[rt].notna().any(): # Check if rank type column exists and has data
            current_max_for_rank_type = df[rt].max()
            if pd.notna(current_max_for_rank_type) and current_max_for_rank_type > max_rank:
                max_rank = current_max_for_rank_type
    
    if max_rank > 0:
        ax.set_yticks(np.arange(1, int(max_rank) + 1)) # Set integer ticks from 1 to max_rank

    ax.tick_params(axis='y', labelsize=FONT_SIZE_TICK_LABEL)
    ax.set_title(plot_title, loc='left', fontsize=FONT_SIZE_PLOT_TITLE, 
                 fontweight='normal', color='black')
    ax.set_xlabel("Rank Type", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.set_ylabel("Rank", fontsize=FONT_SIZE_AXIS_LABEL)
    ax.grid(True, linestyle=':', linewidth=0.5, color='lightgrey')
    ax.set_facecolor('white') 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- Main script execution ---
if __name__ == '__main__':
    ensure_dir(OUTPUT_DIR)

    # --- Generate and save 2x2 facet plot with NEW ARRANGEMENT ---
    print("\nGenerating 2x2 facet plot with rearranged layout...")
    fig_facet, axs = plt.subplots(2, 2, figsize=(17, 15))

    # NEW ARRANGEMENT:
    # A (top-left): Bump Chart (was D)
    # B (top-right): Bar Chart (was C) 
    # C (bottom-left): Line plot with DATA_FILE_FACET_C_SOURCE (was B)
    # D (bottom-right): Line plot with DATA_FILE_FACET_D_SOURCE (was A)
    
    create_bump_chart(axs[0, 0], plot_title="A")
    create_grouped_bar_chart(axs[0, 1], plot_title="B")
    legend_items_C = create_line_comparison_plot(axs[1, 0], 
                                               DATA_FILE_FACET_C_SOURCE, 
                                               plot_title="C", 
                                               show_legend=False)
    legend_items_D = create_line_comparison_plot(axs[1, 1], 
                                               DATA_FILE_FACET_D_SOURCE, 
                                               plot_title="D", 
                                               show_legend=False)
    
    fig_facet.suptitle("Comparison Plots", fontsize=FONT_SIZE_SUPTITLE, 
                       fontweight='bold', y=0.98)

    # Combine legend items from C and D for a shared legend
    combined_legend_items = {**legend_items_C, **legend_items_D} # Merge dicts, D's items will overwrite C's if keys clash (should be consistent)
    
    if combined_legend_items: # Only add legend if there are items
        # Extract handles and labels in a consistent order
        sorted_labels = sorted(combined_legend_items.keys()) # Sort labels for consistent order
        sorted_handles = [combined_legend_items[label] for label in sorted_labels]

        # Create the shared legend
        shared_legend = fig_facet.legend(handles=sorted_handles, 
                                         labels=sorted_labels,
                                         loc='center right', # Position to the right, centered vertically
                                         bbox_to_anchor=(0.97, 0.3), # Adjust to place outside plots C & D (0.3 is approx vertical center of bottom row)
                                         ncol=1, fontsize=FONT_SIZE_LEGEND)
        # Apply line widths to the legend handles
        for handle, label_text in zip(shared_legend.legend_handles, sorted_labels):
            if label_text in LINE_PLOT_LINEWIDTHS:
                handle.set_linewidth(LINE_PLOT_LINEWIDTHS[label_text])
    
    # Adjust layout to prevent overlap and make space for the shared legend
    fig_facet.tight_layout(pad=3.0, h_pad=4.0, w_pad=4.0, 
                           rect=[0, 0, 0.90, 0.95]) # Reserved 10% on right for legend
                                                        
    facet_plot_path = os.path.join(OUTPUT_DIR, 'facet_plot_2x2_final.png')
    fig_facet.savefig(facet_plot_path, dpi=300)
    plt.close(fig_facet)
    print(f"Saved facet_plot_2x2_final.png to {OUTPUT_DIR}")

    print("\nAll plots generated and saved with new arrangement:")
    print("A: Bump Chart (top-left)")
    print("B: Bar Chart (top-right)")
    print("C: Line Plot with first dataset (bottom-left)")
    print("D: Line Plot with second dataset (bottom-right)")
