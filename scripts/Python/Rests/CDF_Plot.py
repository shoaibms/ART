import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  # Import Line2D for custom legend

# Customize plot parameters
title_fontsize = 18
axis_label_fontsize = 18
tick_label_fontsize = 18  # Font size for tick labels

# Replace this file path with the path to your CSV file
file_path = r'path/to/your/merged.csv'  # Update with your actual file path

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Define colors and line thicknesses using hex colors
color_map = {
    'TRT': {'color': '#EED2A8', 'linewidth': 3},  
    'ART': {'color': '#C3DAEC', 'linewidth': 3},  
    'Combine': {'color': '#f2a7a7', 'linewidth': 3}  
}

# Plot CDF for ROC AUC
plt.figure(figsize=(6, 5))

# Track lines for the legend
lines = [] 

# Calculate CDF for each data type
for data_type in df['Data_type'].unique():
    subset = df[df['Data_type'] == data_type]['ROC AUC']
    x = np.sort(subset)
    y = np.arange(1, len(x) + 1) / len(x)
    line = plt.plot(x, y, label=f'{data_type}',
                    color=color_map[data_type]['color'],
                    linewidth=color_map[data_type]['linewidth'])[0]
    lines.append(line)

# Label axes
plt.xlabel('ROC AUC', fontsize=axis_label_fontsize)
plt.ylabel('CDF', fontsize=axis_label_fontsize)

# Set font size for tick labels
plt.xticks(fontsize=tick_label_fontsize)
plt.yticks(fontsize=tick_label_fontsize)

# Adjust subplot parameters
plt.subplots_adjust(bottom=0.15)

# Create custom legend handles with thicker lines
legend_handles = [Line2D([0], [0], color=line.get_color(), linewidth=6, label=line.get_label()) for line in lines]

# Use custom legend handles and adjust font size
plt.legend(handles=legend_handles, loc='best', fontsize=18)
plt.grid(True)

# Show plot with tight layout
plt.tight_layout()  
plt.show()
