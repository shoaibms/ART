import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Function to clean data
def clean_data(group):
    group = group[group > 0]  # Remove negative values
    lower_quantile = group.quantile(0.10)
    upper_quantile = group.quantile(0.90)
    return group[(group > lower_quantile) & (group < upper_quantile)]

# Load the dataset
df = pd.read_csv('Combine_A_B_T_S.csv')  # Make sure to update the path as needed

# Preparing data
variables = ['DBSCAN_density_points', 'FCM_centre_x', 'HDBSCAN_centre_y'] 

groups = ['T', 'S']
data = {}
colors = {'T': '#b654f7', 'S': '#f5b70c'}  # Hex colors for each group

for var in variables:
    data[var] = {}
    for group in groups:
        group_data = df[df['Group'] == group][var]
        data[var][group] = clean_data(group_data)

# Creating subplots
fig, axs = plt.subplots(1, 3, figsize=(9, 5))

for i, var in enumerate(variables):
    for group in groups:
        sns.kdeplot(data[var][group], ax=axs[i], label=f'Group {group}', bw_adjust=0.5, fill=True, color=colors[group])
    axs[i].set_title(var)
    axs[i].set_xlabel(var, fontsize=16)  # Adjust font size for x-axis label
    axs[i].set_ylabel('Density', fontsize=16)  # Adjust font size for y-axis label
    axs[i].legend(fontsize=16)  # Adjust font size for legend

plt.tight_layout()
plt.show()


