import matplotlib.pyplot as plt
import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('cluster_comparison_results_T0&T1_2.csv')

# Reshape the dataframe to have one row per algorithm and treatment combination
df_melted = df.melt(id_vars=['Algorithm', 'Treatment'], var_name='Metric', value_name='Value')

# Define the hex color for each metric and treatment
colors = {
    'ARI-T0': '#89fab6',  # light green
    'ARI-T1': '#54c481',  # dark green
    'NMI-T0': '#add8e6',  # light blue
    'NMI-T1': '#48a4f0',  # midnight blue
    'FMS-T0': '#ffe4b5',  # moccasin
    'FMS-T1': '#ffc154',  # dark orange
}

# Define the linewidth for each metric and treatment
line_widths = {
    'ARI-T0': 2.0,
    'ARI-T1': 3.5,
    'NMI-T0': 2.0,
    'NMI-T1': 3.5,
    'FMS-T0': 2.0,
    'FMS-T1': 3.5,
}

# Create the plot
plt.figure(figsize=(6, 6))  # Adjust the width as necessary

# Plot each metric for each treatment
for metric in df_melted['Metric'].unique():
    df_metric_t0 = df_melted[(df_melted['Metric'] == metric) & (df_melted['Treatment'] == 'T0')]
    df_metric_t1 = df_melted[(df_melted['Metric'] == metric) & (df_melted['Treatment'] == 'T1')]
    plt.plot(df_metric_t0['Algorithm'], df_metric_t0['Value'], marker='', color=colors[f'{metric}-T0'], linewidth=line_widths[f'{metric}-T0'], alpha=0.9, label=f'{metric}-T0')
    plt.plot(df_metric_t1['Algorithm'], df_metric_t1['Value'], marker='', color=colors[f'{metric}-T1'], linewidth=line_widths[f'{metric}-T1'], alpha=0.9, label=f'{metric}-T1')

# Remove background color
plt.gca().set_facecolor('none')
plt.grid(False)

# Rotate x-axis labels and align to the right
plt.xticks(rotation=45, ha='right', fontsize=14)  # Increased fontsize

# Set y-axis maximum value to 1
plt.ylim(top=1.05)

# Add legend with appropriate line thickness and fontsize
handles, labels = plt.gca().get_legend_handles_labels()
legend = plt.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=14)  # Increased fontsize

# Adjust the linewidth for each legend handle
for handle, label in zip(legend.legend_handles, labels):  # Corrected attribute name
    handle.set_linewidth(line_widths[label])

# Add titles and labels with increased fontsize
plt.title("", loc='left', fontsize=18, fontweight=0, color='black')
plt.xlabel("Algorithm", fontsize=16, fontweight=0)
plt.ylabel("Score", fontsize=18, fontweight=0)
plt.yticks(fontsize=18)  # Increased fontsize for y-axis ticks

# Adjust the plot layout to make sure the x-axis labels and legend are not cut off
plt.tight_layout()

# Show the plot
plt.show()
