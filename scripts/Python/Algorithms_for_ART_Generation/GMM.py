import os
import cv2
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance

# Specify the image directory
image_dir = r"path_to_your_image_folder"# all images

# Prepare an empty list to store the data
data = []

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates
        coordinates = np.array([(j, i) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])
        
        # GMM cluster
        gmm = GaussianMixture(n_components=2)
        clusters = gmm.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters, return_counts=True)
        
        if len(labels) > 0:  # Check if labels is not empty
            densest_cluster_label = labels[np.argmax(counts)]
            GMM_density_points = coordinates[clusters == densest_cluster_label]
            
            # Calculate the center point of the densest cluster
            GMM_center_x = np.mean(GMM_density_points[:, 0])
            GMM_center_y = np.mean(GMM_density_points[:, 1])
            
            # Find the black pixel closest to the center point
            closest_point = GMM_density_points[distance.cdist([(GMM_center_x, GMM_center_y)], GMM_density_points).argmin()]
            closest_x, closest_y = closest_point
            
            # Append the data to the list
            data.append([filename, len(GMM_density_points), closest_x, closest_y])
        else:
            # Append the data to the list with 'NA' values for images without clusters
            data.append([filename, 'NA', 'NA', 'NA'])

# Convert the list to a DataFrame and write it to a .csv file
df = pd.DataFrame(data, columns=['image_name', 'GMM_density_points', 'GMM_center_x', 'GMM_center_y'])
df.to_csv('image_gmm.csv', index=False)
