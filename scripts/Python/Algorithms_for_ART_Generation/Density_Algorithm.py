import os
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import pandas as pd

# Define the paths for image folder and output file
IMAGE_FOLDER = r'path_to_your_image_folder'# all images 
OUTPUT_FILE = r'path_to_your_output_file\all_density.csv'

# Initialize an empty list to store the data
data = []

for filename in os.listdir(IMAGE_FOLDER):
    if not filename.endswith('.png'):
        continue
    image_path = os.path.join(IMAGE_FOLDER, filename)
    img = Image.open(image_path).convert('1')
    width, height = img.size
    image = np.zeros((height, width))
    black_pixels = []
    for y in range(height):
        for x in range(width):
            if img.getpixel((x, y)) == 0:
                image[y, x] += 1
                black_pixels.append((y, x))

    densest_points = np.argpartition(image.flatten(), -200)[-200:]
    densest_counts = image.flatten()[densest_points]
    sorted_indices = np.argsort(densest_counts)[::-1]
    densest_points = densest_points[sorted_indices]
    y_coords, x_coords = np.unravel_index(densest_points, image.shape)

    # Get the x and y values
    xy = np.vstack([x_coords, y_coords])

    # Calculate the density
    density = gaussian_kde(xy)(xy)

    # Use KMeans clustering to find the largest densest region
    clustering = KMeans(n_clusters=min(5, len(xy.T))).fit(xy.T)

    # Check for multiple clusters 
    cluster_centers = clustering.cluster_centers_
    num_clusters = len(cluster_centers)
    if num_clusters > 1:
        # Get cluster sizes
        cluster_sizes = [len(clustering.labels_[clustering.labels_==i]) for i in range(num_clusters)]

        # Find index of largest cluster
        largest_cluster = np.argmax(cluster_sizes)

        # Find indices of largest cluster
        largest_cluster_indices = clustering.labels_ == largest_cluster

        # Take points of largest cluster
        Density_points = xy.T[largest_cluster_indices]

        # Calculate center of largest cluster
        Density_center_x, Density_center_y = np.mean(Density_points, axis=0)    

        # Use the size of the largest cluster as 'Density_points'
        Density_points_size = cluster_sizes[largest_cluster]

    else:
        # Original case of single cluster
        Density_center_x, Density_center_y = cluster_centers[0]
        largest_cluster = 0  # There is only one cluster
        Density_points_size = len(xy.T)  # Size of the cluster is the total number of points

    # Append the data to the list
    data.append([filename, Density_points_size, Density_center_x, Density_center_y])

# Convert the list to a DataFrame and write it to a .csv file
df = pd.DataFrame(data, columns=['filename', 'Density_points', 'Density_center_x', 'Density_center_y'])
df.to_csv(OUTPUT_FILE, index=False)

print('Done processing images')
