import os
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Define the paths for the image folder and save directory
IMAGE_FOLDER = r'path_to_your_image_folder'# select images to plot
save_dir = r'path_to_your_save_directory'

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

    xy = np.vstack([x_coords, y_coords])
    density = gaussian_kde(xy)(xy)
    clustering = KMeans(n_clusters=min(5, len(xy.T))).fit(xy.T)
    cluster_centers = clustering.cluster_centers_
    num_clusters = len(cluster_centers)
    if num_clusters > 1:
        cluster_sizes = [len(clustering.labels_[clustering.labels_ == i]) for i in range(num_clusters)]
        largest_cluster = np.argmax(cluster_sizes)
        largest_cluster_indices = clustering.labels_ == largest_cluster
        Density_points = xy.T[largest_cluster_indices]
        Density_center_x, Density_center_y = np.mean(Density_points, axis=0)
    else:
        Density_center_x, Density_center_y = cluster_centers[0]

    plt.imshow(img, cmap='gray')
    plt.scatter(x_coords, y_coords, c='r', s=1)
    plt.scatter(Density_center_x, Density_center_y, c='b', s=100)
    plt.xticks([0, 1000, 2000])  # Set x-axis ticks

    plot_filename = os.path.join(save_dir, f'Density_{filename}')
    plt.savefig(plot_filename)
    plt.clf()  # Clear the figure

print('Done processing images')
