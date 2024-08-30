import os
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd

# Specify the image directory and the directory to save plots
image_dir = r"path_to_your_image_folder" # select images to plot
save_dir = r"path_to_save_plots"  # Directory to save plots

# Initialize an empty list to store the results
results = []

# Create the save directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates
        coordinates = np.array([(i, j) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])
        
        # Estimate bandwidth for Mean Shift
        bandwidth = estimate_bandwidth(coordinates, quantile=0.1, n_samples=500)
        
        # Mean Shift cluster
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        clusters = ms.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters, return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        mean_shift_density_points = coordinates[clusters == densest_cluster_label]
        
        # Calculate the center point of the densest cluster
        mean_shift_center_y = np.mean(mean_shift_density_points[:, 0])
        mean_shift_center_x = np.mean(mean_shift_density_points[:, 1])
        
        # Append the results to the list
        results.append([filename, len(mean_shift_density_points), mean_shift_center_x, mean_shift_center_y])

# Convert the results to a DataFrame and save as a CSV file
df = pd.DataFrame(results, columns=['Image Name', 'Mean-shift_density_points', 'Mean-shift_center_x', 'Mean-shift_center_y'])
df.to_csv('seg_image_1_Mean-shift.csv', index=False)
