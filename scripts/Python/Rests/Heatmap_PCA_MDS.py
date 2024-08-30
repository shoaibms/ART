import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
import numpy as np

# Load the datasets
art_file_path = 'C:\\Users\\ms\\Desktop\\data\\ART.csv'
trt_file_path = 'C:\\Users\\ms\\Desktop\\data\\TRT.csv'

art_data = pd.read_csv(art_file_path)
trt_data = pd.read_csv(trt_file_path)

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
art_data_imputed = imputer.fit_transform(art_data.select_dtypes(include=[float, int]))
trt_data_imputed = imputer.fit_transform(trt_data.select_dtypes(include=[float, int]))

# Convert imputed data back to DataFrame
art_data_imputed_df = pd.DataFrame(art_data_imputed, columns=art_data.select_dtypes(include=[float, int]).columns)
trt_data_imputed_df = pd.DataFrame(trt_data_imputed, columns=trt_data.select_dtypes(include=[float, int]).columns)

# Standardize the datasets
scaler = StandardScaler()
art_data_scaled = scaler.fit_transform(art_data_imputed_df)
trt_data_scaled = scaler.fit_transform(trt_data_imputed_df)

# Create correlation matrices
art_corr = pd.DataFrame(art_data_scaled, columns=art_data_imputed_df.columns).corr()
trt_corr = pd.DataFrame(trt_data_scaled, columns=trt_data_imputed_df.columns).corr()

# Generate linkage matrices for ordering
art_linkage = sch.linkage(art_corr, method='ward')
trt_linkage = sch.linkage(trt_corr, method='ward')

# Reorder the correlation matrix based on the clustering
art_dendro = sch.dendrogram(art_linkage, no_plot=True)
trt_dendro = sch.dendrogram(trt_linkage, no_plot=True)

art_order = art_dendro['leaves']
trt_order = trt_dendro['leaves']

art_corr_reordered = art_corr.iloc[art_order, art_order]
trt_corr_reordered = trt_corr.iloc[trt_order, trt_order]

# Define the common color map limits
vmin = -1
vmax = 1

# Plot Heatmap without Dendrogram for ART and TRT
def plot_heatmap(correlation_matrix, title, figsize=(11, 9), fontsize=12, legend_fontsize=16, vmin=-1, vmax=1, cmap='vlag'):
    plt.figure(figsize=figsize)
    ax = sns.heatmap(correlation_matrix, annot=False, cmap=cmap, vmin=vmin, vmax=vmax, cbar_kws={'label': 'Correlation'})
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=legend_fontsize)
    cbar.set_label('Correlation', size=legend_fontsize)  
    plt.title(title, fontsize=fontsize + 2)
    plt.xticks(fontsize=fontsize, rotation=90)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout(pad=2)
    plt.show()

# Customize the plot size and text size
plot_size = (11, 9)
text_size = 14
legend_size = 16

plot_heatmap(art_corr_reordered, '', figsize=plot_size, fontsize=text_size, legend_fontsize=legend_size, vmin=vmin, vmax=vmax)
plot_heatmap(trt_corr_reordered, '', figsize=plot_size, fontsize=text_size, legend_fontsize=legend_size, vmin=vmin, vmax=vmax)

# Perform PCA for initial dimensionality reduction
pca_art = PCA(n_components=2)
pca_trt = PCA(n_components=2)

art_pca = pca_art.fit_transform(art_data_scaled)
trt_pca = pca_trt.fit_transform(trt_data_scaled)

# Plot PCA results with color option
def plot_pca(data, title, color='#1f77b4', figsize=(6, 6), fontsize=20):  # Default color is blue
    plt.figure(figsize=figsize)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.7, color=color)
    plt.title(title, fontsize=fontsize + 2)
    plt.xlabel('PCA Component 1', fontsize=fontsize)
    plt.ylabel('PCA Component 2', fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout(pad=2)
    plt.show()

# Customize the PCA plot size and text size
pca_plot_size = (6, 6)
pca_text_size = 20

plot_pca(art_pca, 'PCA Plot - ART', color='#C3DAEC', figsize=pca_plot_size, fontsize=pca_text_size)
plot_pca(trt_pca, 'PCA Plot - TRT', color='#EED2A8', figsize=pca_plot_size, fontsize=pca_text_size)

# Load combined dataset for MDS
combined_file_path = 'C:\\Users\\ms\\Desktop\\data\\Combine_A_B.csv'
combined_data = pd.read_csv(combined_file_path)

# Variables belonging to ART and TRT groups in the combined dataset
art_variables = ['DBSCAN_density_points', 'DBSCAN_center_x', 'DBSCAN_center_y',
                 'Density_points', 'Density_center_x', 'Density_center_y',
                 'FCM_density_points', 'FCM_center_x', 'FCM_center_y',
                 'GMM_density_points', 'GMM_center_x', 'GMM_center_y',
                 'HDBSCAN_density_points', 'HDBSCAN_center_x', 'HDBSCAN_center_y',
                 'K-mean_density_points', 'K-mean_center_x', 'K-mean_center_y',
                 'SLIC_density_points', 'SLIC_center_x', 'SLIC_center_y',
                 'Mean_shift_density_points', 'Mean-shift_center_x', 'Mean-shift_center_y',
                 'OPTICS_density_points', 'OPTICS_center_x', 'OPTICS_center_y']

trt_variables = ['Number.of.Root.Tips', 'Number.of.Branch.Points', 'Total.Root.Length.mm',
                 'Branching.frequency.per.mm', 'Network.Area.mm2', 'Average.Diameter.mm',
                 'Median.Diameter.mm', 'Maximum.Diameter.mm', 'Perimeter.mm',
                 'Volume.mm3', 'Surface.Area.mm2', 'Root.Length.Diameter.Range.1.mm',
                 'Root.Length.Diameter.Range.2.mm', 'Root.Length.Diameter.Range.3.mm',
                 'Projected.Area.Diameter.Range.1.mm2', 'Projected.Area.Diameter.Range.2.mm2',
                 'Projected.Area.Diameter.Range.3.mm2', 'Surface.Area.Diameter.Range.1.mm2',
                 'Surface.Area.Diameter.Range.2.mm2', 'Surface.Area.Diameter.Range.3.mm2',
                 'Volume.Diameter.Range.1.mm3', 'Volume.Diameter.Range.2.mm3',
                 'Volume.Diameter.Range.3.mm3']

# Check the initial counts of variables
print(f'Number of ART variables: {len(art_variables)}')
print(f'Number of TRT variables: {len(trt_variables)}')

# Select the columns for ART and TRT variables
data_art = combined_data[art_variables]
data_trt = combined_data[trt_variables]

# Check for missing values
print(f'Missing values in ART data: {data_art.isnull().sum().sum()}')
print(f'Missing values in TRT data: {data_trt.isnull().sum().sum()}')

# Handle missing values by imputing with the mean
data_art_imputed = imputer.fit_transform(data_art)
data_trt_imputed = imputer.fit_transform(data_trt)

# Check the counts after imputation
print(f'ART data shape after imputation: {data_art_imputed.shape}')
print(f'TRT data shape after imputation: {data_trt_imputed.shape}')

# Standardize the datasets
data_art_scaled = scaler.fit_transform(data_art_imputed)
data_trt_scaled = scaler.fit_transform(data_trt_imputed)

# Combine data
data_combined_scaled = pd.concat([pd.DataFrame(data_art_scaled, columns=art_variables),
                                  pd.DataFrame(data_trt_scaled, columns=trt_variables)], axis=1)

# Check the shape of combined data
print(f'Combined data shape: {data_combined_scaled.shape}')

# Compute the distance matrix
distance_matrix = pdist(data_combined_scaled.T, metric='euclidean')

# Apply MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds_results = mds.fit_transform(squareform(distance_matrix))

# Create a DataFrame with the MDS results
mds_df = pd.DataFrame(mds_results, columns=['MDS1', 'MDS2'])
mds_df['Group'] = ['ART'] * len(art_variables) + ['TRT'] * len(trt_variables)

# Verify the count of variables in the MDS DataFrame
print(f'Number of ART points in MDS: {mds_df[mds_df["Group"] == "ART"].shape[0]}')
print(f'Number of TRT points in MDS: {mds_df[mds_df["Group"] == "TRT"].shape[0]}')

# Plot the Multidimensional Scaling (MDS) results with larger dot sizes, hex colors, and jitter
def plot_mds(mds_df, title, figsize=(6, 6), fontsize=20):
    plt.figure(figsize=figsize)
    for group, color in zip(['ART', 'TRT'], ['#C3DAEC', '#EED2A8']):  # Using hex colors for blue and green
        subset = mds_df[mds_df['Group'] == group]
        plt.scatter(subset['MDS1'] + np.random.normal(0, 0.5, size=subset['MDS1'].shape), 
                    subset['MDS2'] + np.random.normal(0, 0.5, size=subset['MDS2'].shape), 
                    label=group, color=color, s=100)  # Adjust dot size with `s` parameter
    plt.title(title, fontsize=fontsize + 2)
    plt.xlabel('MDS Dimension 1', fontsize=fontsize)
    plt.ylabel('MDS Dimension 2', fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout(pad=2)
    plt.show()

# Customize the MDS plot size and text size
mds_plot_size = (6, 6)
mds_text_size = 20

plot_mds(mds_df, 'MDS Plot of ART and TRT Variables', figsize=mds_plot_size, fontsize=mds_text_size)
