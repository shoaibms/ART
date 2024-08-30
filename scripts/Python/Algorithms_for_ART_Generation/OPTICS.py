import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS

# Specify the image directory
image_dir = r"path_to_your_image_folder" # all images

# Prepare an empty list to store the results
results = []

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates
        coordinates = np.array([(i, j) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])  # Consider black pixels
        
        # OPTICS cluster
        optics = OPTICS(min_samples=10)
        clusters = optics.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        OPTICS_density_points = coordinates[clusters == densest_cluster_label]
        
        # Calculate the center point of the densest cluster
        OPTICS_center_y = np.mean(OPTICS_density_points[:, 0])
        OPTICS_center_x = np.mean(OPTICS_density_points[:, 1])
        
        # Append the results
        results.append([filename, len(OPTICS_density_points), OPTICS_center_x, OPTICS_center_y])

# Convert the results to a DataFrame and save as .csv
df = pd.DataFrame(results, columns=['Image File Name', 'OPTICS_density_points', 'OPTICS_center_x', 'OPTICS_center_y'])
df.to_csv('seg_image_1_optics.csv', index=False)
