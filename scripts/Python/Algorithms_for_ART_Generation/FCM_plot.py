import os
import cv2
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Specify image and save directories
image_dir = r"C:\Users\ms\Desktop\data\minirhizotron_test"
save_dir = r"C:\Users\ms\Desktop\data\plot"  # Ensure this directory exists

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
        
        # FCM cluster
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            coordinates.T, 2, 2, error=0.005, maxiter=1000)
        
        # Find the densest cluster
        cluster_membership = np.argmax(u, axis=0)
        labels, counts = np.unique(cluster_membership, return_counts=True)
        densest_cluster_label = labels[np.argmax(counts)]
        FCM_density_points = coordinates[cluster_membership == densest_cluster_label]
        
        # Calculate center point of the densest cluster
        FCM_center_y = np.mean(FCM_density_points[:, 0])
        FCM_center_x = np.mean(FCM_density_points[:, 1])
      
        # Annotate image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in FCM_density_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)
        
        cv2.circle(viz, (int(FCM_center_x), int(FCM_center_y)), 100, (255, 0, 0), -1)

        # Create and save plot
        plt.figure(figsize=(4,5))
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))
        #plt.title(filename)
        plot_filename = os.path.join(save_dir, f'FCM_{filename}.png')
        plt.savefig(plot_filename)
        plt.close()
        
        print(f"Plot saved to {plot_filename}")
        print(f"Number of density points in the densest cluster: {len(FCM_density_points)}")
        print(f"Center point of the densest cluster: ({FCM_center_x}, {FCM_center_y})")



