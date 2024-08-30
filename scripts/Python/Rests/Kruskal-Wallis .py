

# Kruskal-Wallis, POST_HOC (Dunn's test) and EFFECT size (Cliff's Delta) calulcaion for g_s_1”, “g_s_2” and 'Tiller_no'
#g_s_1”, “g_s_2” and 'Tiller_no' DID NOT meet the normality assumption 

import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp
import numpy as np

# Load the dataset
data = pd.read_csv("path_to_your_data_file.csv")

# List of variables to analyze
variables = ["g_s_1", "g_s_2", "Tiller_no"]

# Function to calculate Cliff's Delta
def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    mat_x = np.tile(x, (n_y, 1)).T
    mat_y = np.tile(y, (n_x, 1))
    delta = np.mean(np.sign(mat_x - mat_y))
    return delta

# Perform Kruskal-Wallis and Dunn's test, and calculate Cliff's Delta
for var in variables:
    print(f"\nAnalyzing {var}:")

    # Kruskal-Wallis test
    kruskal_stat, kruskal_p = kruskal(*[group[var].values for name, group in data.groupby('Treatment')])
    print("Kruskal-Wallis Test Result: Statistic = {:.3f}, p-value = {:.3f}".format(kruskal_stat, kruskal_p))

    # Dunn's test for post-hoc analysis
    dunn_result = sp.posthoc_dunn(data, val_col=var, group_col='Treatment', p_adjust='bonferroni')
    print("Dunn's Test Result:\n", dunn_result)

    # Calculate Cliff's Delta for drought-tolerant and susceptible genotypes
    tolerant_T0 = data[(data['Group'] == 'T') & (data['Treatment'] == 'T0')][var]
    tolerant_T1 = data[(data['Group'] == 'T') & (data['Treatment'] == 'T1')][var]
    susceptible_T0 = data[(data['Group'] == 'S') & (data['Treatment'] == 'T0')][var]
    susceptible_T1 = data[(data['Group'] == 'S') & (data['Treatment'] == 'T1')][var]

    tolerant_delta = cliffs_delta(tolerant_T0, tolerant_T1)
    susceptible_delta = cliffs_delta(susceptible_T0, susceptible_T1)

    print(f"Cliff's Delta for Drought Tolerant Genotypes ({var}): {tolerant_delta}")
    print(f"Cliff's Delta for Drought Susceptible Genotypes ({var}): {susceptible_delta}")
