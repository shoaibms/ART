import os
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

def detect_orientation(img):
    height, width = img.shape[:2]
    return 'landscape' if width > height else 'portrait'

# Specify image and save directories
image_dir = r"path_to_your_image_folder"# select image to plot
save_dir = r"plot"  # Define directory to save plots

# Create the save directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check orientation and rotate if necessary
        orientation = detect_orientation(image)
        if orientation == 'landscape':
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, 90, 1.0)
            image = cv2.warpAffine(image, M, (h, w))
        
        # Flatten coordinates of black pixels
        coordinates = np.array([(i, j) for i in range(image.shape[0]) 
                                for j in range(image.shape[1]) if image[i, j] == 0])

        # GMM cluster
        gmm_cluster = GaussianMixture(n_components=2)
        clusters = gmm_cluster.fit_predict(coordinates)
        
        # Find the cluster with the maximum number of points (most dense)
        labels, counts = np.unique(clusters[clusters != -1], return_counts=True)
        GMM_densest_cluster_label = labels[np.argmax(counts)]
        GMM_density_points = coordinates[clusters == GMM_densest_cluster_label]
        
        # Calculate the center point of the most dense GMM cluster
        GMM_center_y = np.mean(GMM_density_points[:, 0])
        GMM_center_x = np.mean(GMM_density_points[:, 1])
        
        # Create annotated image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in GMM_density_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)
        
        # Add a blue dot at the center of the most dense GMM cluster
        cv2.circle(viz, (int(GMM_center_x), int(GMM_center_y)), 100, (255, 0, 0), -1)
                
        # Create plot
        plt.figure(figsize=(4,5))  # Set the figure size
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))

        # Save plot to file
        plot_filename = os.path.join(save_dir, f'GMM_{filename}.png')
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        
        print(f"Plot saved to {plot_filename}")
        print(f"Number of density points in the densest cluster: {len(GMM_density_points)}")
        print(f"Center point of the densest cluster: ({GMM_center_x}, {GMM_center_y})")
