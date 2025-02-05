
import os
import cv2
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from scipy.spatial import distance

# Specify image directory
image_dir = r"C:\Users\Mirza Shoiab\Desktop\data\seg_images"

# Prepare an empty list to store the data
data = []

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates
        coordinates = np.array([(j, i) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])  # Changed to 0 for black pixels
        
        # FCM cluster
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            coordinates.T, 2, 2, error=0.005, maxiter=1000)
        
        # Find the cluster with the maximum number of points (densest cluster)
        cluster_membership = np.argmax(u, axis=0)
        labels, counts = np.unique(cluster_membership, return_counts=True)
        
        if len(labels) > 0:  # Check if labels is not empty
            densest_cluster_label = labels[np.argmax(counts)]
            densest_cluster_points = coordinates[cluster_membership == densest_cluster_label]
            
            # Calculate the center point of the densest cluster
            center_x = np.mean(densest_cluster_points[:, 0])
            center_y = np.mean(densest_cluster_points[:, 1])
            
            # Find the black pixel closest to the center point
            closest_point = densest_cluster_points[distance.cdist([(center_x, center_y)], densest_cluster_points).argmin()]
            closest_x, closest_y = closest_point
            
            # Append the data to the list
            data.append([filename, len(densest_cluster_points), closest_x, closest_y])
        else:
            # Append the data to the list with 'NA' values for images without clusters
            data.append([filename, 'NA', 'NA', 'NA'])

# Convert the list to a DataFrame and write it to a .csv file
df = pd.DataFrame(data, columns=['image_name', 'number_of_density_points', 'center_x', 'center_y'])
df.to_csv('all_fcm.csv', index=False)




"""
#CODE1: IT PRODUCE PLOT HIGHLIGHTS HOTTEST REGION AND GIVE OUTPUT IN TERMINAL

import os
import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Specify image directory
image_dir = r"C:/Users/shoai/Dropbox/analysis/misc/test/image"

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates
        coordinates = np.array([(i, j) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])  # Changed to 0 for black pixels
        
        # FCM cluster
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            coordinates.T, 2, 2, error=0.005, maxiter=1000)
        
        # Find the cluster with the maximum number of points (densest cluster)
        cluster_membership = np.argmax(u, axis=0)
        labels, counts = np.unique(cluster_membership, return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        densest_cluster_points = coordinates[cluster_membership == densest_cluster_label]
        
        # Calculate the center point of the densest cluster
        center_y = np.mean(densest_cluster_points[:, 0])  # Swapped here
        center_x = np.mean(densest_cluster_points[:, 1])  # Swapped here
      
        # Create annotated image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in densest_cluster_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)  # Corrected here
        
        # Add a blue dot at the center of the densest cluster
        cv2.circle(viz, (int(center_x), int(center_y)), 100, (255, 0, 0), -1)
               
        # Display result    
        plt.figure(figsize=(4,5))  # Set the figure size to 4 inches wide and 5 inches tall
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        plt.title(filename) 
        plt.show()
        
        print(f"Number of density points in the densest cluster: {len(densest_cluster_points)}")
        print(f"Center point of the densest cluster: ({center_x}, {center_y})")  # Swapped here
"""