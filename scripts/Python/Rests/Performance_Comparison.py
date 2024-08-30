import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the data from the specified file path
file_path = 'data/ART_TRT_Combine_result_compare.csv'
df = pd.read_csv(file_path)

# Define the metrics and colors
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'Specificity']
colors = ['#ADD8E6', '#F0E68C', '#FFC0CB', '#90EE90', '#FFB6C1', '#B0C4DE']

# Add jitter to the data
def add_jitter(values, jitter_amount=0.01):
    return values + np.random.normal(0, jitter_amount, len(values))

# Split the data into the three groups
first_group_methods = df['Method'][:3]
second_group_methods = df['Method'][3:6]
third_group_methods = df['Method'][6:]

# Parameters for text size
xlabel_size = 16
ylabel_size = 16
ticks_size = 14
legend_size = 14
title_size = 16

plt.figure(figsize=(10, 6))

# Plotting the first group
for idx, metric in enumerate(metrics):
    plt.plot(first_group_methods, add_jitter(df[metric][:3]), marker='o', label=metric if first_group_methods[0] == 'TRT_baseline' else "", color=colors[idx])

# Plotting the second group
for idx, metric in enumerate(metrics):
    plt.plot(second_group_methods, add_jitter(df[metric][3:6]), marker='o', color=colors[idx])

# Plotting the third group
for idx, metric in enumerate(metrics):
    plt.plot(third_group_methods, add_jitter(df[metric][6:]), marker='o', color=colors[idx])

# Adding vertical lines to indicate splits
plt.axvline(x=2.5, color='grey', linestyle='--')
plt.axvline(x=5.5, color='grey', linestyle='--')

# Setting text sizes
plt.xlabel('Datasets', fontsize=xlabel_size)
plt.ylabel('Score', fontsize=ylabel_size)
plt.title('Performance Comparison Across Datasets', fontsize=title_size)
plt.xticks(rotation=45, ha='right', fontsize=ticks_size)
plt.yticks(fontsize=ticks_size)
plt.legend(fontsize=legend_size, loc='upper left')
plt.tight_layout()

plt.show()
