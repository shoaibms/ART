# This script performs dimensionality reduction (t-SNE, UMAP, PCA) on ART and TRT datasets.
# It visualizes the reduced data, coloring points by genotype and treatment,
# and by drought tolerance group. It also calculates and compares separation metrics
# for these different feature spaces and methods. The script aims to identify
# which combination of features (ART/TRT) and dimensionality reduction technique
# best separates the defined groups.
#
# The script expects two CSV files, ART.csv and TRT.csv, in a specified input directory.
# Outputs include:
#   - CSV files of the reduced data for each method and dataset.
#   - PNG images of the visualizations:
#     - Genotype and Treatment separation plots for each method and dataset.
#     - Drought Tolerance Group separation plots for each method and dataset.
#   - A CSV file and a bar chart comparing separation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz).
#   - Side-by-side comparison plots for t-SNE and UMAP (if available) showing ART vs. TRT feature performance.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.patches as mpatches

# Set a global random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Define base directories for input and output
# Modify these paths if your data is located elsewhere or you want to save outputs to a different place.
BASE_INPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "data")
BASE_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "data", "output", "t-SNE_UMAP")

# Create output directory
# output_dir = r'C:\\Users\\ms\\Desktop\\data\\output\\t-SNE_UMAP'
output_dir = BASE_OUTPUT_DIR # Use the defined base output directory
os.makedirs(output_dir, exist_ok=True)

# Check if UMAP is available, otherwise skip UMAP visualizations
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
    print("UMAP package found and loaded successfully.")
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP package not found. Please install with 'pip install umap-learn'.")
    print("UMAP visualizations will be skipped.")

# Load datasets
print("Loading datasets...")
# art_path = r'C:\\Users\\ms\\Desktop\\data\\ART.csv'
# trt_path = r'C:\\Users\\ms\\Desktop\\data\\TRT.csv'
art_path = os.path.join(BASE_INPUT_DIR, 'ART.csv')
trt_path = os.path.join(BASE_INPUT_DIR, 'TRT.csv')

art_data = pd.read_csv(art_path)
trt_data = pd.read_csv(trt_path)

print(f"ART dataset shape: {art_data.shape}")
print(f"TRT dataset shape: {trt_data.shape}")

# Define metadata columns
meta_columns = ['Image_name', 'data_type', 'Genotype', 'Treatment', 'Replication', 'Group']

# Check for missing or invalid values in Treatment column
print("Unique Treatment values:", art_data['Treatment'].unique())

# Replace 'na' or None values with 'Unknown'
art_data['Treatment'] = art_data['Treatment'].fillna('Unknown')
art_data['Treatment'] = art_data['Treatment'].replace('na', 'Unknown')
trt_data['Treatment'] = trt_data['Treatment'].fillna('Unknown')
trt_data['Treatment'] = trt_data['Treatment'].replace('na', 'Unknown')

# Verify common indices between datasets
art_indices = set(art_data['Image_name'])
trt_indices = set(trt_data['Image_name'])
common_indices = list(art_indices.intersection(trt_indices))  # Convert set to list
print(f"Common samples between datasets: {len(common_indices)}")

# Filter to common indices and align datasets
art_data = art_data.set_index('Image_name').loc[common_indices].reset_index()
trt_data = trt_data.set_index('Image_name').loc[common_indices].reset_index()

# Extract features and metadata
art_features = art_data.drop(meta_columns, axis=1)
trt_features = trt_data.drop(meta_columns, axis=1)
metadata = art_data[meta_columns]

# Create color and marker mappings for visualization
# Generate distinct colors for genotypes
genotypes = metadata['Genotype'].unique()
treatments = metadata['Treatment'].unique()
groups = metadata['Group'].unique()

print("Unique genotypes:", genotypes)
print("Unique treatments:", treatments)
print("Unique groups:", groups)

# Define palette for genotypes (using a colorblind-friendly palette)
colors = sns.color_palette("colorblind", len(genotypes))
genotype_color_map = dict(zip(genotypes, colors))

# Define markers for treatments (including 'Unknown')
treatment_marker_map = {
    'T0': 'o',     # Control
    'T1': 's',     # Drought
    'Unknown': 'x'  # Unknown/NA treatment
}

# Add any other treatments found in the data
for treatment in treatments:
    if treatment not in treatment_marker_map:
        treatment_marker_map[treatment] = 'd'  # Default marker for any other treatments

# Define style for Group
group_style_map = {
    'T': True,   # Solid line for Tolerant
    'S': False   # Dashed line for Susceptible
}

def create_custom_legend_elements():
    """Create custom legend elements for genotypes, treatments, and groups"""
    genotype_elements = [mpatches.Patch(color=genotype_color_map[g], label=f'Genotype: {g}') 
                        for g in genotypes]
    
    # Treatment markers
    treatment_elements = [plt.Line2D([0], [0], marker=treatment_marker_map[t], color='black', 
                         linestyle='', markersize=8, label=f'Treatment: {t}')
                         for t in treatments]
    
    # Group style (solid vs dashed)
    group_elements = [plt.Line2D([0], [0], color='black', linestyle='-' if group_style_map[g] else '--',
                     markersize=8, label=f'Group: {g}')
                     for g in groups]
    
    return genotype_elements + treatment_elements + group_elements

# Function to perform dimensionality reduction and visualize
def dimensionality_reduction_plot(features, metadata, method='tsne', perplexity=30, 
                                 n_components=2, n_neighbors=15, min_dist=0.1, title_prefix=''):
    """
    Performs dimensionality reduction and creates plots
    
    Parameters:
    -----------
    features : pandas DataFrame
        Feature matrix
    metadata : pandas DataFrame
        Contains Genotype, Treatment, Group information
    method : str
        'tsne', 'umap', or 'pca'
    perplexity : int
        Perplexity parameter for t-SNE
    n_components : int
        Number of components to reduce to
    n_neighbors : int
        Number of neighbors for UMAP
    min_dist : float
        Minimum distance for UMAP
    title_prefix : str
        Prefix for plot titles
    """
    # Skip UMAP if not available
    if method == 'umap' and not UMAP_AVAILABLE:
        print("Skipping UMAP visualization as package is not available.")
        return None
        
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    # Perform dimensionality reduction
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=RANDOM_SEED, perplexity=perplexity)
        reducer_name = f"t-SNE (perplexity={perplexity})"
    elif method == 'umap' and UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_SEED, 
                          n_neighbors=n_neighbors, min_dist=min_dist)
        reducer_name = f"UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})"
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
        reducer_name = "PCA"
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply dimensionality reduction
    X_reduced = reducer.fit_transform(X_scaled)
    
    # Create dataframe with reduced dimensions and metadata
    result_df = pd.DataFrame(X_reduced, columns=[f"Component_{i+1}" for i in range(n_components)])
    result_df = pd.concat([result_df, metadata.reset_index(drop=True)], axis=1)
    
    # Save reduced data
    result_df.to_csv(os.path.join(output_dir, f"{title_prefix}_{method}_reduced_data.csv"), index=False)
    
    # Create visualization - By Genotype and Treatment
    plt.figure(figsize=(12, 10))
    
    # Plot points
    for genotype in genotypes:
        for treatment in treatments:
            mask = (result_df['Genotype'] == genotype) & (result_df['Treatment'] == treatment)
            if sum(mask) > 0:  # Only plot if there are points with this combination
                plt.scatter(
                    result_df.loc[mask, 'Component_1'],
                    result_df.loc[mask, 'Component_2'],
                    color=genotype_color_map[genotype],
                    marker=treatment_marker_map[treatment],
                    s=100,
                    alpha=0.7,
                    label=f"{genotype}-{treatment}"
                )
    
    # Create convex hulls around genotypes
    for genotype in genotypes:
        genotype_mask = result_df['Genotype'] == genotype
        if sum(genotype_mask) >= 3:  # Need at least 3 points for convex hull
            points = result_df.loc[genotype_mask, ['Component_1', 'Component_2']].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], 
                    color=genotype_color_map[genotype], alpha=0.1)
    
    plt.title(f"{title_prefix} {reducer_name} - Genotype and Treatment Separation", fontsize=14)
    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add custom legend with better organization
    legend_elements = create_custom_legend_elements()
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_{method}_genotype_treatment.png"), dpi=300)
    plt.close()
    
    # Create visualization - By Drought Tolerance Group
    plt.figure(figsize=(12, 10))
    
    # Define color map for Groups
    group_color_map = {'T': 'darkgreen', 'S': 'firebrick'}
    
    # Plot points
    for group in groups:
        mask = result_df['Group'] == group
        plt.scatter(
            result_df.loc[mask, 'Component_1'],
            result_df.loc[mask, 'Component_2'],
            color=group_color_map[group],
            s=80,
            alpha=0.7,
            label=f"Group {group}"
        )
    
    # Create convex hulls around groups
    for group in groups:
        group_mask = result_df['Group'] == group
        if sum(group_mask) >= 3:  # Need at least 3 points for convex hull
            points = result_df.loc[group_mask, ['Component_1', 'Component_2']].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], 
                    color=group_color_map[group], alpha=0.2)
    
    plt.title(f"{title_prefix} {reducer_name} - Drought Tolerance Group Separation", fontsize=14)
    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title_prefix}_{method}_group.png"), dpi=300)
    plt.close()
    
    return result_df

# Calculate separation metrics
def calculate_separation_metrics(reduced_df, group_col='Group'):
    """Calculate metrics for group separation in reduced space"""
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    if reduced_df is None:
        return {
            'Silhouette Score': np.nan,
            'Davies-Bouldin Score': np.nan,
            'Calinski-Harabasz Score': np.nan
        }
    
    X = reduced_df[['Component_1', 'Component_2']].values
    y = reduced_df[group_col].values
    
    try:
        # Silhouette score (higher is better)
        sil_score = silhouette_score(X, y)
        
        # Davies-Bouldin score (lower is better)
        db_score = davies_bouldin_score(X, y)
        
        # Calinski-Harabasz score (higher is better)
        ch_score = calinski_harabasz_score(X, y)
        
        return {
            'Silhouette Score': sil_score,
            'Davies-Bouldin Score': db_score,
            'Calinski-Harabasz Score': ch_score
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            'Silhouette Score': np.nan,
            'Davies-Bouldin Score': np.nan,
            'Calinski-Harabasz Score': np.nan
        }

# Run t-SNE for ART features
print("Generating t-SNE visualization for ART features...")
art_tsne_df = dimensionality_reduction_plot(
    art_features, metadata, method='tsne', perplexity=30, title_prefix='ART'
)

# Run t-SNE for TRT features
print("Generating t-SNE visualization for TRT features...")
trt_tsne_df = dimensionality_reduction_plot(
    trt_features, metadata, method='tsne', perplexity=30, title_prefix='TRT'
)

# Run PCA as a baseline comparison 
print("Generating PCA visualization for ART features...")
art_pca_df = dimensionality_reduction_plot(
    art_features, metadata, method='pca', title_prefix='ART'
)

print("Generating PCA visualization for TRT features...")
trt_pca_df = dimensionality_reduction_plot(
    trt_features, metadata, method='pca', title_prefix='TRT'
)

# Only run UMAP if available
art_umap_df = None
trt_umap_df = None
if UMAP_AVAILABLE:
    # Run UMAP for ART features
    print("Generating UMAP visualization for ART features...")
    art_umap_df = dimensionality_reduction_plot(
        art_features, metadata, method='umap', n_neighbors=15, min_dist=0.1, title_prefix='ART'
    )

    # Run UMAP for TRT features
    print("Generating UMAP visualization for TRT features...")
    trt_umap_df = dimensionality_reduction_plot(
        trt_features, metadata, method='umap', n_neighbors=15, min_dist=0.1, title_prefix='TRT'
    )

# Calculate separation metrics for all visualizations
separation_results = {
    'ART t-SNE': calculate_separation_metrics(art_tsne_df),
    'TRT t-SNE': calculate_separation_metrics(trt_tsne_df),
    'ART PCA': calculate_separation_metrics(art_pca_df),
    'TRT PCA': calculate_separation_metrics(trt_pca_df)
}

# Add UMAP metrics if available
if UMAP_AVAILABLE:
    separation_results['ART UMAP'] = calculate_separation_metrics(art_umap_df)
    separation_results['TRT UMAP'] = calculate_separation_metrics(trt_umap_df)

# Convert to DataFrame
separation_df = pd.DataFrame(separation_results).T
separation_df.to_csv(os.path.join(output_dir, 'separation_metrics.csv'))

# Create a bar chart comparing separation metrics
plt.figure(figsize=(14, 8))
ax = separation_df.plot(kind='bar', figsize=(14, 8))
plt.title('Comparison of Group Separation in Different Feature Spaces', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.xlabel('Feature Space and Method', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7, axis='y')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Metric')

# Add value labels on top of bars
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'separation_metrics_comparison.png'), dpi=300)

# Create a side-by-side comparison visualization for t-SNE 
plt.figure(figsize=(18, 8))

# ART t-SNE
plt.subplot(1, 2, 1)
for group in groups:
    mask = art_tsne_df['Group'] == group
    plt.scatter(
        art_tsne_df.loc[mask, 'Component_1'],
        art_tsne_df.loc[mask, 'Component_2'],
        color={'T': 'darkgreen', 'S': 'firebrick'}[group],
        s=80,
        alpha=0.7,
        label=f"Group {group}"
    )
    
    # Create convex hulls around groups
    group_mask = art_tsne_df['Group'] == group
    if sum(group_mask) >= 3:
        points = art_tsne_df.loc[group_mask, ['Component_1', 'Component_2']].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        plt.fill(hull_points[:, 0], hull_points[:, 1], 
                color={'T': 'darkgreen', 'S': 'firebrick'}[group], alpha=0.2)

plt.title(f"ART Features - t-SNE\nSilhouette Score: {separation_results['ART t-SNE']['Silhouette Score']:.3f}", fontsize=14)
plt.xlabel(f"Component 1", fontsize=12)
plt.ylabel(f"Component 2", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')

# TRT t-SNE
plt.subplot(1, 2, 2)
for group in groups:
    mask = trt_tsne_df['Group'] == group
    plt.scatter(
        trt_tsne_df.loc[mask, 'Component_1'],
        trt_tsne_df.loc[mask, 'Component_2'],
        color={'T': 'darkgreen', 'S': 'firebrick'}[group],
        s=80,
        alpha=0.7,
        label=f"Group {group}"
    )
    
    # Create convex hulls around groups
    group_mask = trt_tsne_df['Group'] == group
    if sum(group_mask) >= 3:
        points = trt_tsne_df.loc[group_mask, ['Component_1', 'Component_2']].values
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        plt.fill(hull_points[:, 0], hull_points[:, 1], 
                color={'T': 'darkgreen', 'S': 'firebrick'}[group], alpha=0.2)

plt.title(f"TRT Features - t-SNE\nSilhouette Score: {separation_results['TRT t-SNE']['Silhouette Score']:.3f}", fontsize=14)
plt.xlabel(f"Component 1", fontsize=12)
plt.ylabel(f"Component 2", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='best')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'art_vs_trt_tsne_comparison.png'), dpi=300)

# Only create UMAP comparison if available
if UMAP_AVAILABLE:
    # Create a side-by-side comparison visualization for UMAP
    plt.figure(figsize=(18, 8))

    # ART UMAP
    plt.subplot(1, 2, 1)
    for group in groups:
        mask = art_umap_df['Group'] == group
        plt.scatter(
            art_umap_df.loc[mask, 'Component_1'],
            art_umap_df.loc[mask, 'Component_2'],
            color={'T': 'darkgreen', 'S': 'firebrick'}[group],
            s=80,
            alpha=0.7,
            label=f"Group {group}"
        )
        
        # Create convex hulls around groups
        group_mask = art_umap_df['Group'] == group
        if sum(group_mask) >= 3:
            points = art_umap_df.loc[group_mask, ['Component_1', 'Component_2']].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], 
                    color={'T': 'darkgreen', 'S': 'firebrick'}[group], alpha=0.2)

    plt.title(f"ART Features - UMAP\nSilhouette Score: {separation_results['ART UMAP']['Silhouette Score']:.3f}", fontsize=14)
    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    # TRT UMAP
    plt.subplot(1, 2, 2)
    for group in groups:
        mask = trt_umap_df['Group'] == group
        plt.scatter(
            trt_umap_df.loc[mask, 'Component_1'],
            trt_umap_df.loc[mask, 'Component_2'],
            color={'T': 'darkgreen', 'S': 'firebrick'}[group],
            s=80,
            alpha=0.7,
            label=f"Group {group}"
        )
        
        # Create convex hulls around groups
        group_mask = trt_umap_df['Group'] == group
        if sum(group_mask) >= 3:
            points = trt_umap_df.loc[group_mask, ['Component_1', 'Component_2']].values
            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], 
                    color={'T': 'darkgreen', 'S': 'firebrick'}[group], alpha=0.2)

    plt.title(f"TRT Features - UMAP\nSilhouette Score: {separation_results['TRT UMAP']['Silhouette Score']:.3f}", fontsize=14)
    plt.xlabel(f"Component 1", fontsize=12)
    plt.ylabel(f"Component 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'art_vs_trt_umap_comparison.png'), dpi=300)

print(f"All visualizations saved to {output_dir}")
print("Analysis complete.")