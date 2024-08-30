import os
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import pandas as pd

# Specify the image directory and the directory to save plots
image_dir = r"path_to_your_image_folder" # all images 
save_dir = r"path_to_save_plots"  # Directory to save plots

results = []

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
        
        # Append the results to the list
        results.append([filename, len(SLIC_density_points), SLIC_center_x, SLIC_center_y])

# Convert the results to a DataFrame and save as a CSV file
df = pd.DataFrame(results, columns=['Image Name', 'SLIC_density_points', 'SLIC_center_x', 'SLIC_center_y'])
df.to_csv('seg_image_1_hdbscan.csv', index=False)
