import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Specify the image directory and the directory to save outputs
image_dir = r"path_to_your_image_folder"# select images to plot
save_dir = r"path_to_save_outputs"

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
        K_Mean_density_points = coordinates[clusters == densest_cluster_label]

        # Calculate the center point of the densest cluster
        K_Mean_center_x = np.mean(K_Mean_density_points[:, 1])
        K_Mean_center_y = np.mean(K_Mean_density_points[:, 0])
        
        # Create annotated image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in K_Mean_density_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)
        
        # Add a blue dot at the center of the densest cluster
        cv2.circle(viz, (int(K_Mean_center_x), int(K_Mean_center_y)), 100, (255, 0, 0), -1)
               
        # Save the plot to the specified directory
        plot_filename = os.path.join(save_dir, f'KMean_{filename}.png')
        plt.figure(figsize=(4,5))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        plt.savefig(plot_filename)
        plt.close()

        print(f"Annotated image saved: {plot_filename}")
        print(f"Number of density points in the densest cluster: {len(K_Mean_density_points)}")
        print(f"Center point of the densest cluster: ({K_Mean_center_x}, {K_Mean_center_y})")
