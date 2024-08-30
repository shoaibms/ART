import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Load the data from CSV file
df = pd.read_csv('Output_ART.csv')

# Customize plot parameters
title_fontsize = 20
axis_label_fontsize = 18
tick_label_fontsize = 18  # Font size for axis ticks

# Set the color and line styles for each model
colors = ['#acfcd1', '#f2cedc', '#6ec5cc', '#ffc882', '#58ed9e', '#d6a3b5', '#4d6c94', '#7f7f7f']
line_styles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]

# Create the parallel coordinates plot
plt.figure(figsize=(8, 5))
for i, (color, line_style) in enumerate(zip(colors, line_styles)):
    parallel_coordinates(df[df['Model'] == df['Model'].unique()[i]], 'Model', color=[color], linestyle=line_style, linewidth=3)

# Customize plot title and axes labels
plt.xlabel('Metrics', fontsize=axis_label_fontsize)
plt.ylabel('', fontsize=axis_label_fontsize)

# Customize tick labels for both axes
plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)

# Customize legend and place it outside the plot
legend = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=axis_label_fontsize)
plt.setp(legend.get_texts(), color='black')

# Adjust the legend's line width
for leg_line in legend.get_lines():
    leg_line.set_linewidth(10)

# Customize grid lines
plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5, color='lightgrey')
plt.grid(False, which='both', axis='y')

plt.xticks(rotation=45, ha='right')

# Set the y-axis to finish at 1
plt.ylim([plt.ylim()[0], 1])

plt.tight_layout()
plt.show()
