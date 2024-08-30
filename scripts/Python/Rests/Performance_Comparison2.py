import pandas as pd
import matplotlib.pyplot as plt

# Load the data from your CSV file
df = pd.read_csv('path_to_your_file/Result_compare_all_A_B_TRT_ART_Combine.csv')

# Customize plot parameters
title_fontsize = 20
axis_label_fontsize = 18
tick_label_fontsize = 19
marker_size = 10  # Size of the plot markers
line_width = 3  # Width of the lines
legend_fontsize = 14  # Font size for the legend text

# Define colors for each line
colors = ['#C3DAEC', '#EED2A8', '#f2a7a7']  # Distinct colors for better visibility

methods = df['Method'].unique()  # Assuming 'Method' is the column with model names

# Create the plot
plt.figure(figsize=(7, 5))
for i, method in enumerate(methods):
    subset = df[df['Method'] == method]
    plt.plot(subset.columns[1:], subset.iloc[0, 1:], 
             label=method, 
             color=colors[i], 
             linestyle='-',  # Use solid lines
             marker='o',  # Use the same marker for all lines
             markersize=marker_size, 
             linewidth=line_width)

# Customize plot title and axes labels
plt.xlabel('Metrics', fontsize=axis_label_fontsize)
plt.ylabel('Score', fontsize=axis_label_fontsize)

# Customize the tick labels for both axes
plt.xticks(rotation=45, ha='right', fontsize=tick_label_fontsize)
plt.yticks(fontsize=tick_label_fontsize)

# Remove top and right spines
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust legend placement and appearance
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=legend_fontsize)
plt.setp(legend.get_texts(), color='black')
for leg_line in legend.get_lines():
    leg_line.set_linewidth(3)

# Remove grid for better readability
plt.grid(False)

# Adjust y-axis limits to show all data points clearly
plt.ylim([0.88, 1])

# Ensure layout is tight to avoid clipping
plt.tight_layout()

# Show the plot
plt.show()
