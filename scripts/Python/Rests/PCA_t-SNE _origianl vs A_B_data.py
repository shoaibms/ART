import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# File paths
original_data_path = "path_to_original_data/TRT.csv"
augmented_data_path = "path_to_augmented_data/TRT_A_B.csv"

# Load datasets
original_data = pd.read_csv(original_data_path)
augmented_data = pd.read_csv(augmented_data_path)

# Assuming the datasets have the same structure, drop any non-numeric columns
original_data_numeric = original_data.select_dtypes(include=['float64', 'int64'])
augmented_data_numeric = augmented_data.select_dtypes(include=['float64', 'int64'])

# PCA
pca = PCA(n_components=2)
pca_original_result = pca.fit_transform(original_data_numeric)
pca_augmented_result = pca.fit_transform(augmented_data_numeric)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
tsne_original_result = tsne.fit_transform(original_data_numeric)
tsne_augmented_result = tsne.fit_transform(augmented_data_numeric)

# Add jitter function
def add_jitter(arr, noise_level=0.05):
    return arr + np.random.normal(0, noise_level, arr.shape)

# Font size settings
title_fontsize = 16
label_fontsize = 15
tick_fontsize = 14

# Plot PCA results
plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)
plt.scatter(add_jitter(pca_original_result[:, 0]), add_jitter(pca_original_result[:, 1]), c='#79a4c7', label='Original Data')
plt.title('PCA - TRT Data', fontsize=title_fontsize)
plt.xlabel('Principal Component 1', fontsize=label_fontsize)
plt.ylabel('Principal Component 2', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.subplot(1, 2, 2)
plt.scatter(add_jitter(pca_augmented_result[:, 0]), add_jitter(pca_augmented_result[:, 1]), c='#de8781', label='Augmented Data')
plt.title('PCA - TRT_A_B Data', fontsize=title_fontsize)
plt.xlabel('Principal Component 1', fontsize=label_fontsize)
plt.ylabel('Principal Component 2', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.tight_layout()
plt.show()

# Plot t-SNE results
plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)
plt.scatter(add_jitter(tsne_original_result[:, 0]), add_jitter(tsne_original_result[:, 1]), c='#79a4c7', label='Original Data')
plt.title('t-SNE - TRT Data', fontsize=title_fontsize)
plt.xlabel('t-SNE Component 1', fontsize=label_fontsize)
plt.ylabel('t-SNE Component 2', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.subplot(1, 2, 2)
plt.scatter(add_jitter(tsne_augmented_result[:, 0]), add_jitter(tsne_augmented_result[:, 1]), c='#de8781', label='Augmented Data')
plt.title('t-SNE - TRT_A_B Data', fontsize=title_fontsize)
plt.xlabel('t-SNE Component 1', fontsize=label_fontsize)
plt.ylabel('t-SNE Component 2', fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

plt.tight_layout()
plt.show()
