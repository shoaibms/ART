import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova

# Load the data
file_path = 'path_to_your_csv_file/Combine_A_B.csv'
data = pd.read_csv(file_path)

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
art_data = data[art_variables]
trt_data = data[trt_variables]

# Combine the ART and TRT data before scaling
combined_data = pd.concat([art_data, trt_data], axis=1)

# Standardize the combined data
scaler = StandardScaler()
combined_scaled_data = scaler.fit_transform(combined_data)

# Calculate Euclidean distance for the combined data
combined_euclidean_dist = pdist(combined_scaled_data.T, metric='euclidean')
combined_euclidean_dist_matrix = squareform(combined_euclidean_dist)

# Create a group vector (0 for ART, 1 for TRT) for the combined data
group_vector = ['ART'] * len(art_variables) + ['TRT'] * len(trt_variables)

# Convert distance matrix to skbio DistanceMatrix object
distance_matrix = DistanceMatrix(combined_euclidean_dist_matrix)

# Perform PERMANOVA
result = permanova(distance_matrix, group_vector, permutations=999)

# Save the result to a CSV file
result_df = pd.DataFrame([result])
result_df.to_csv('path_to_save_results/permanova_result.csv', index=False)
