import os
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Specify image directory
image_dir = r"C:\Users\ms\Desktop\data\minirhizotron_test"

# Specify save directory for plots
save_dir = r"C:\Users\ms\Desktop\data\plot"
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
        #plt.title(filename)

        # Save plot to file instead of displaying
        plot_filename = os.path.join(save_dir, f'DBSCAN_{filename}.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        
        print(f"Plot saved to {plot_filename}")
        print(f"Number of density points in the densest cluster: {len(DBSCAN_density_points)}")
        print(f"Center point of the densest cluster: ({DBSCAN_center_x}, {DBSCAN_center_y})")


