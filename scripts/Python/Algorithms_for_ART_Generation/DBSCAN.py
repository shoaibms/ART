import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

# Specify image directory
image_dir = r"C:\Users\Mirza Shoiab\Desktop\data2\seg_image_1"

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
        
        # DBSCAN cluster
        dbscan = DBSCAN(eps=5, min_samples=10)
        clusters = dbscan.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        
        if len(labels) > 0:  # Check if labels is not empty
            densest_cluster_label = labels[np.argmax(counts)]
            DBSCAN_density_points = coordinates[clusters == densest_cluster_label]
            
            # Calculate the DBSCAN center point of the densest cluster
           DBSCAN_center_x = np.mean(DBSCAN_density_points[:, 0])
           DBSCAN_center_y = np.mean(DBSCAN_density_points[:, 1])
            
            # Find the black pixel closest to the DBSCAN center point
            closest_point = DBSCAN_density_points[distance.cdist([(DBSCAN_center_x, DBSCAN_center_y)], DBSCAN_density_points).argmin()]
            closest_x, closest_y = closest_point
            
            # Append the data to the list
            data.append([filename, len(DBSCAN_density_points), closest_x, closest_y])
        else:
            # Append the data to the list with 'NA' values for images without clusters
            data.append([filename, 'NA', 'NA', 'NA'])

# Convert the list to a DataFrame and write it to a .csv file
df = pd.DataFrame(data, columns=['image_name', 'DBSCAN_density_points', 'DBSCAN_center_x', 'DBSCAN_center_y'])
df.to_csv('seg_image_1_dbscan.csv', index=False)







#Old_CODE2: IT PRODUCE .CSV WITH DENSITY CLUSTER POINT AND X AN Y VALUE OF THE CENTER POINT OF THE HOTTEST REGION 

#This code calculates the Euclidean distance from the center point to each black pixel in the densest cluster and finds the black pixel 
#that is closest to the center point. It then writes the coordinates of this black pixel to the .csv file
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

# Specify image directory
image_dir = r"C:\Users\Mirza Shoiab\Desktop\data2\seg_image_1"

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
        
        # DBSCAN cluster
        dbscan = DBSCAN(eps=5, min_samples=10)
        clusters = dbscan.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        
        if len(labels) > 0:  # Check if labels is not empty
            densest_cluster_label = labels[np.argmax(counts)]
            densest_cluster_points = coordinates[clusters == densest_cluster_label]
            
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
df.to_csv('seg_image_1_dbscan.csv', index=False)


# CODE1
#IT PRODUCE PLOT densest_cluster REGION AND SAVE PLOT IN FOLDER

import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Specify image directory
image_dir = r"C:\Users\Mirza Shoiab\Desktop\data\image2"

# Specify save directory for plots
save_dir = r"C:/Users/Mirza Shoiab/Desktop/data/plot"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates
        coordinates = np.array([(i, j) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])  # Changed to 0 for black pixels
        
        # DBSCAN cluster
        dbscan_cluster = DBSCAN(eps=3, min_samples=2)
        clusters = dbscan_cluster.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        DBSCAN_density_points = coordinates[clusters == densest_cluster_label]
        
        # Calculate the center point of the densest cluster
        DBSCAN_center_y = np.mean(DBSCAN_density_points[:, 0])  # Swapped here
        DBSCAN_center_x = np.mean(DBSCAN_density_points[:, 1])  # Swapped here
        
        # Create annotated image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in DBSCAN_density_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)  # Corrected here
        
        # Add a blue dot at the center of the densest cluster
        cv2.circle(viz, (int(DBSCAN_center_x), int(DBSCAN_center_y)), 100, (255, 0, 0), -1)
               
        # Create plot
        plt.figure(figsize=(4,5))  # Set the figure size
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        plt.title(filename)

        # Save plot to file instead of displaying
        plot_filename = os.path.join(save_dir, f'DBSCAN_{filename}.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        
        print(f"Plot saved to {plot_filename}")
        print(f"Number of density points in the densest cluster: {len(DBSCAN_density_points)}")
        print(f"Center point of the densest cluster: ({DBSCAN_center_x}, {DBSCAN_center_y})")




"""
#Old_CODE1: IT PRODUCE PLOT HIGHLIGHTS HOTTEST REGION AND GIVE OUTPUT IN TERMINAL

import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
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
        
        # DBSCAN cluster
        dbscan_cluster = DBSCAN(eps=3, min_samples=2)
        clusters = dbscan_cluster.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (densest cluster)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        densest_cluster_points = coordinates[clusters == densest_cluster_label]
        
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
