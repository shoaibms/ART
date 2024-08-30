import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['g_s_1', 'g_s_2', 'RWC', 'Tiller_no']
T_values = [0.736111111, 0.430555556, 0.486111111, 0.722222222]
S_values = [0.958333333, 0.986111111, 0.569444444, 1]

# Colors for each group (lighter for T, darker for S)
colors_T = ['#afe9ed', '#fce0bb', '#acfcd1', '#f2cedc']  # Lighter colors for T
colors_S = ['#6ec5cc', '#ffc882', '#58ed9e', '#d6a3b5']  # Darker colors for S

# Set position of bar on X axis
barWidth = 0.35
r1 = np.arange(len(T_values))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.figure(figsize=(5, 5))

# Create bars for T and S and add text labels
for i in range(len(categories)):
    bar1 = plt.bar(r1[i], T_values[i], color=colors_T[i], width=barWidth, edgecolor='grey')
    bar2 = plt.bar(r2[i], S_values[i], color=colors_S[i], width=barWidth, edgecolor='grey')
    
    # Add text labels on the bars
    plt.text(r1[i], T_values[i] + 0.02, 'T', ha='center', va='bottom')
    plt.text(r2[i], S_values[i] + 0.02, 'S', ha='center', va='bottom')

# Add xticks on the middle of the group bars
plt.xlabel('Variable', fontsize=16, fontweight=0)
plt.xticks([r + barWidth / 2 for r in range(len(T_values))], categories, fontsize=14)

plt.ylabel('Effect Size (Cliff\'s Delta)', fontsize=16, fontweight=0)
plt.yticks(fontsize=16)

# Set y-axis maximum value to 1
plt.ylim(top=1.1)

# Show the plot
plt.tight_layout()  # Adjust the layout to make room for the text
plt.show()


