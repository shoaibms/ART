import cv2
import numpy as np
import hdbscan
import pandas as pd
import os 

# Specify the image directory
image_dir = r"path_to_your_image_folder"# all images

# Initialize an empty list to store the results
results = []

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates
        coordinates = np.array([(j, i) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])  # Changed to 0 for black pixels
        
        # HDBSCAN cluster
        hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=10)
        clusters = hdbscan_cluster.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        HDBSCAN_density_points = coordinates[clusters == densest_cluster_label]
        
        # Calculate the center point of the densest cluster
        HDBSCAN_center_x = np.mean(HDBSCAN_density_points[:, 0])
        HDBSCAN_center_y = np.mean(HDBSCAN_density_points[:, 1])
        
        # Append the results to the list
        results.append([filename, len(HDBSCAN_density_points), HDBSCAN_center_x, HDBSCAN_center_y])

# Convert the results to a DataFrame and save as a CSV file
df = pd.DataFrame(results, columns=['Image Name', 'HDBSCAN_density_points', 'HDBSCAN_center_x', 'HDBSCAN_center_y'])
df.to_csv('seg_image_1_hdbscan.csv', index=False)
