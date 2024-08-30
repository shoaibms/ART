import os
import cv2
import numpy as np
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt

# Specify the image directory and the directory to save plots
image_dir = r"path_to_your_image_folder" # select images to plot
save_dir = r"path_to_save_plots"

# Create the save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Flatten coordinates of black pixels
        coordinates = np.array([(i, j) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])
        
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
        
        # Create annotated image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in OPTICS_density_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)
        
        # Add a blue dot at the center of the densest cluster
        cv2.circle(viz, (int(OPTICS_center_x), int(OPTICS_center_y)), 100, (255, 0, 0), -1)

        # Prepare plot
        plt.figure(figsize=(4,5))  # Set the figure size
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))

        # Save plot to file
        plot_filename = os.path.join(save_dir, f"OPTICS_{filename}.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        
        print(f"Plot saved to {plot_filename}")
        print(f"Number of density points in the densest cluster: {len(OPTICS_density_points)}")
        print(f"Center point of the densest cluster: ({OPTICS_center_x}, {OPTICS_center_y})")
