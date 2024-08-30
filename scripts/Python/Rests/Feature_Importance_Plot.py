import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data_file_path = "Combine_A_B_Feature_Importances.csv"
data = pd.read_csv(data_file_path)

# Sort the data by importance
data = data.sort_values(by='Importance', ascending=False)

# Define color thresholds
high_threshold = 0.08
medium_threshold = 0.05

# Assign colors based on importance values
colors = []
for importance in data['Importance']:
    if importance >= high_threshold:
        colors.append('salmon')
    elif importance >= medium_threshold:
        colors.append('gold')
    else:
        colors.append('skyblue')

# Plot the feature importances
plt.figure(figsize=(9, 6))
bars = plt.barh(data['Feature'], data['Importance'], color=colors)

# Customize text size
x_label_size = 18
y_label_size = 18
tick_label_size = 18

plt.xlabel('Importance', fontsize=x_label_size)
plt.ylabel('Feature', fontsize=y_label_size)
plt.xticks(fontsize=tick_label_size)
plt.yticks(fontsize=tick_label_size)
plt.title('', fontsize=18)
plt.gca().invert_yaxis()

# Display the plot
plt.tight_layout()
plt.show()
