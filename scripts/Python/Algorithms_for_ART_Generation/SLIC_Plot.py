import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Specify image directory and save directory
image_dir = r"path_to_your_image_folder" # select images to plot
save_dir = r"path_to_save_plots"  # Directory to save plots

for filename in os.listdir(image_dir):
    if filename.endswith(".png"):
        image_path = os.path.join(image_dir, filename)
        
        # Load image as grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert grayscale image to 3-channel image
        image_3channel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Apply SLIC and extract (approximately) the number of superpixels
        superpixels = slic(image_3channel, n_segments=50, compactness=10)
        
        # Find the superpixel with the maximum number of black pixels
        labels, counts = np.unique(superpixels[image == 0], return_counts=True)
        densest_superpixel_label = labels[np.argmax(counts)]
        SLIC_density_points = np.array([(i, j) for i in range(image.shape[0]) 
                                        for j in range(image.shape[1]) 
                                        if superpixels[i, j] == densest_superpixel_label])
        
        # Calculate the center point of the densest superpixel
        SLIC_center_y = np.mean(SLIC_density_points[:, 0])
        SLIC_center_x = np.mean(SLIC_density_points[:, 1])
        
        # Create annotated image
        viz = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for x, y in SLIC_density_points:
            cv2.circle(viz, (int(y), int(x)), 10, (0, 0, 255), -1)
        
        # Add a blue dot at the center of the densest cluster
        cv2.circle(viz, (int(SLIC_center_x), int(SLIC_center_y)), 100, (255, 0, 0), -1)
               
        # Prepare plot
        plt.figure(figsize=(4, 5))  # Set the figure size
        plt.imshow(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB))

        # Save plot to file
        plot_filename = os.path.join(save_dir, f"SLIC_{filename}.png")
        plt.savefig(plot_filename)
        plt.close()  # Close the plot to free memory
        
        print(f"Number of black pixels in the densest superpixel: {len(SLIC_density_points)}")
        print(f"SLIC center point of the densest superpixel: ({SLIC_center_x}, {SLIC_center_y})")
