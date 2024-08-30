import os
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Specify the image directory and the directory to save plots
image_dir = r"path_to_your_image_folder"# select images to plot
save_dir = r"path_to_save_plots"  # Directory to save plots

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
        
        # Create annotated image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in mean_shift_density_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)
        
        # Add a blue dot at the center of the densest cluster
        cv2.circle(viz, (int(mean_shift_center_x), int(mean_shift_center_y)), 100, (255, 0, 0), -1)
               
        # Prepare the plot    
        plt.figure(figsize=(4,5))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))

        # Save the plot
        save_path = os.path.join(save_dir, f"Mean_shift_{filename}.png")
        plt.savefig(save_path)
        plt.close()  # Close the plot to free up memory
        
        print(f"Plot saved to: {save_path}")
        print(f"Number of density points in the densest cluster: {len(mean_shift_density_points)}")
        print(f"Center point of the densest cluster: ({mean_shift_center_x}, {mean_shift_center_y})")
