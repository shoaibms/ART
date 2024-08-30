import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator, FuncFormatter

# Load the data
art_file_path = 'path_to_your_data/ART.csv'
trt_file_path = 'path_to_your_data/TRT.csv'

art_data = pd.read_csv(art_file_path)
trt_data = pd.read_csv(trt_file_path)

# Variables belonging to ART and TRT groups
art_variables = ['DBSCAN_density_points', 'DBSCAN_center_x', 'DBSCAN_center_y',
                 'Density_points', 'Density_center_x', 'Density_center_y',
                 'FCM_density_points', 'FCM_center_x', 'FCM_center_y',
                 'GMM_density_points', 'GMM_center_x', 'GMM_center_y',
                 'HDBSCAN_density_points', 'HDBSCAN_center_x', 'HDBSCAN_center_y',
                 'K-mean_density_points', 'K-mean_center_x', 'K-mean_center_y',
                 'SLIC_density_points', 'SLIC_center_x', 'SLIC_center_y',
                 'Mean_shift_density_points', 'Mean-shift_center_x', 'Mean-shift_center_y',
                 'OPTICS_density_points', 'OPTICS_center_x', 'OPTICS_center_y']

trt_variables = ['Number.of.Root.Tips', 'Number.of.Branch.Points', 'Total.Root.Length.mm',
                 'Branching.frequency.per.mm', 'Network.Area.mm2', 'Average.Diameter.mm',
                 'Median.Diameter.mm', 'Maximum.Diameter.mm', 'Perimeter.mm',
                 'Volume.mm3', 'Surface.Area.mm2', 'Root.Length.Diameter.Range.1.mm',
                 'Root.Length.Diameter.Range.2.mm', 'Root.Length.Diameter.Range.3.mm',
                 'Projected.Area.Diameter.Range.1.mm2', 'Projected.Area.Diameter.Range.2.mm2',
                 'Projected.Area.Diameter.Range.3.mm2', 'Surface.Area.Diameter.Range.1.mm2',
                 'Surface.Area.Diameter.Range.2.mm2', 'Surface.Area.Diameter.Range.3.mm2',
                 'Volume.Diameter.Range.1.mm3', 'Volume.Diameter.Range.2.mm3',
                 'Volume.Diameter.Range.3.mm3']

# Extract ART and TRT data
art_data = art_data[art_variables]
trt_data = trt_data[trt_variables]

# Standardize the data
scaler = StandardScaler()
art_scaled_data = scaler.fit_transform(art_data)
trt_scaled_data = scaler.fit_transform(trt_data)

# Calculate Euclidean distances for ART and TRT variables
art_euclidean_dist = pdist(art_scaled_data.T, metric='euclidean')
trt_euclidean_dist = pdist(trt_scaled_data.T, metric='euclidean')

# Convert to distance matrices
art_euclidean_dist_matrix = squareform(art_euclidean_dist)
trt_euclidean_dist_matrix = squareform(trt_euclidean_dist)

# Quantify dissimilarity for ART and TRT
art_mean_dist = np.mean(art_euclidean_dist)
art_std_dist = np.std(art_euclidean_dist)
trt_mean_dist = np.mean(trt_euclidean_dist)
trt_std_dist = np.std(trt_euclidean_dist)

# Save the individual distance matrices to CSV files
art_distances_df = pd.DataFrame(art_euclidean_dist_matrix)
art_distances_df.to_csv('path_to_your_data/art_distances.csv', index=False)

trt_distances_df = pd.DataFrame(trt_euclidean_dist_matrix)
trt_distances_df.to_csv('path_to_your_data/trt_distances.csv', index=False)

# Function to plot heatmap with adjustable font sizes
def plot_heatmap(matrix, title, xlabel, ylabel, font_size=12, tick_size=10, legend_size=10):
    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(matrix, cmap='vlag')  # Use 'vlag' colormap
    plt.title(title, fontsize=font_size)
    plt.xlabel(xlabel, fontsize=tick_size)
    plt.ylabel(ylabel, fontsize=tick_size)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x)}'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{int(y)}'))
    plt.xticks(fontsize=tick_size, rotation=90)
    plt.yticks(fontsize=tick_size, rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_size)
    plt.show()

# Custom tick locator to manage tick positions
max_n_ticks = 20

# Visualize ART distance matrix
plot_heatmap(art_euclidean_dist_matrix, '', 
             'Variables', 'Variables', font_size=20, tick_size=20, legend_size=20)

# Visualize TRT distance matrix
plot_heatmap(trt_euclidean_dist_matrix, '', 
             'Variables', 'Variables', font_size=20, tick_size=20, legend_size=20)

# Combine the ART and TRT data before scaling
combined_data = pd.concat([art_data, trt_data], axis=1)

# Standardize the combined data
combined_scaled_data = scaler.fit_transform(combined_data)

# Calculate Euclidean distances for the combined data between variables
combined_euclidean_dist = pdist(combined_scaled_data.T, metric='euclidean')
combined_euclidean_dist_matrix = squareform(combined_euclidean_dist)

# Quantify combined dissimilarity
combined_mean_dist = np.mean(combined_euclidean_dist)
combined_std_dist = np.std(combined_euclidean_dist)

# Save the combined distance matrix to a CSV file
combined_distances_df = pd.DataFrame(combined_euclidean_dist_matrix)
combined_distances_df.to_csv('path_to_your_data/combined_distances.csv', index=False)

# Visualize the combined distance matrix comparison
plot_heatmap(combined_euclidean_dist_matrix, '', 
             'Variables', 'Variables', font_size=20, tick_size=20, legend_size=20)

# Print summary statistics
print(f"ART Mean Euclidean Distance (Variables): {art_mean_dist:.2f}")
print(f"ART Standard Deviation of Euclidean Distances (Variables): {art_std_dist:.2f}")
print(f"TRT Mean Euclidean Distance (Variables): {trt_mean_dist:.2f}")
print(f"TRT Standard Deviation of Euclidean Distances (Variables): {trt_std_dist:.2f}")
print(f"Combined Mean Euclidean Distance (Variables): {combined_mean_dist:.2f}")
print(f"Combined Standard Deviation of Euclidean Distances (Variables): {combined_std_dist:.2f}")
