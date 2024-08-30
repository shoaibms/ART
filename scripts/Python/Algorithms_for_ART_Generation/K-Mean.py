import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

# Specify the image directory
image_dir = r"path_to_your_image_folder"# all images 

data = []  # List to store the data for each image

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates of black pixels
        coordinates = np.array([(i, j) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])

        # KMeans cluster
        kmeans = KMeans(n_clusters=5, random_state=0)
        clusters = kmeans.fit_predict(coordinates)

        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        KMean_density_points = coordinates[clusters == densest_cluster_label]

        if len(KMean_density_points) > 0:
            # Calculate the center point of the densest cluster
            KMean_center_x = np.mean(KMean_density_points[:, 1])
            KMean_center_y = np.mean(KMean_density_points[:, 0])
            
            # Append the data to the list
            data.append([filename, len(KMean_density_points), KMean_center_x, KMean_center_y])
        else:
            # Append the data to the list with 'NA' values for images without clusters
            data.append([filename, 'NA', 'NA', 'NA'])

# Convert the list to a DataFrame and write it to a .csv file
df = pd.DataFrame(data, columns=['image_name', 'KMean_density_points', 'KMean_center_x', 'KMean_center_y'])
csv_filename = r"path_to_save_csv\KMean.csv"
df.to_csv(csv_filename, index=False)

print(f"Data saved to {csv_filename}")
