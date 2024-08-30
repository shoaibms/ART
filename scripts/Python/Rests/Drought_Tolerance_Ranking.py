
import pandas as pd
import scipy.stats as stats
from scipy.stats import trim_mean
from tqdm.auto import tqdm

def load_data(file_path):
    """Load the CSV data file."""
    return pd.read_csv(file_path)

def calculate_robust_difference(group, numerical_cols):
    """Calculate the robust difference (trimmed mean) between control and drought treatments."""
    control = group[group['Treatment'] == 'T0']
    drought = group[group['Treatment'] == 'T1']
    differences = {}
    for col in numerical_cols:
        control_mean = trim_mean(control[col], 0.1)
        drought_mean = trim_mean(drought[col], 0.1)
        differences[col] = control_mean - drought_mean
    return pd.Series(differences)

def calculate_ssi(data, variable):
    """Calculate the Stress Susceptibility Index (SSI) for a given variable."""
    grouped = data.groupby(['Genotype', 'Treatment'])
    means = grouped[variable].mean().unstack()
    ssi = 1 - (means['T1'] / means['T0'])
    return ssi

def calculate_weighted_ranks(df, weights):
    """Calculate weighted ranks for variables."""
    for col in df.columns[1:]:
        df[col + '_Rank'] = df[col].rank(method='min').astype(int)
        df[col + '_WeightedRank'] = df[col + '_Rank'] * weights[col]
    return df

def main_ranking(file_path, output_file):
    """Main function to perform ranking and output the result to a CSV file."""
    data = load_data(file_path)
    numerical_cols = ['g_s_1', 'g_s_2', 'RWC', 'Tiller_no']
    
    filtered_data = data[data['Treatment'].isin(['T0', 'T1'])]
    genotype_differences = filtered_data.groupby('Genotype').apply(calculate_robust_difference, numerical_cols).reset_index()
    genotype_differences.columns = ['Genotype'] + [f'{col}_Diff' for col in numerical_cols]

    weights = {'g_s_1_Diff': 2, 'g_s_2_Diff': 2, 'RWC_Diff': 2, 'Tiller_no_Diff': 1}
    genotype_differences = calculate_weighted_ranks(genotype_differences, weights)
    genotype_differences['Combined_Score'] = genotype_differences[[col + '_WeightedRank' for col in weights]].sum(axis=1)

    ssi_values = pd.DataFrame()
    for variable in tqdm(numerical_cols, desc='Calculating SSI'):
        ssi_values[variable + '_SSI'] = calculate_ssi(data, variable)
    ssi_values.reset_index(inplace=True)
    
    combined_data = pd.merge(genotype_differences, ssi_values, on='Genotype')
    
    for col in [x + '_SSI' for x in numerical_cols]:
        combined_data[col + '_Rank'] = combined_data[col].rank(method='min').astype(int)
        combined_data['Combined_Score'] += combined_data[col + '_Rank']
    
    combined_data['Final_Rank'] = combined_data['Combined_Score'].rank(method='min').astype(int)
    final_rankings = combined_data[['Genotype', 'Combined_Score', 'Final_Rank']].sort_values(by='Final_Rank')
    final_rankings.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Specify your input file path and output file path
    input_file_path = 'data/data_irga_rwc.csv'  # Adjust this path as needed
    output_file_path = 'data/RANK03.csv'        # Adjust this path as needed
    main_ranking(input_file_path, output_file_path)
    print(f"The final rankings have been saved to '{output_file_path}'.")
