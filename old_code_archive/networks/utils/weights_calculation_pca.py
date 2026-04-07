import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

def calculate_pca_weights(monthly_nodes, months):
    """
    Calculates the weights for each climate variable using PCA.

    Args:
        monthly_nodes (dict): A dictionary where keys are month strings and values are dicts
                              of climate data for that month.
        months (list): A list of month strings to ensure the order of the data.

    Returns:
        dict: A dictionary of weights for each climate variable.
    """

    # --- 1. Prepare the data ---
    data = []
    for month in months:
        node = monthly_nodes[month]
        data.append([
            node['mean_tg'],
            node['mean_tn'],
            node['mean_tx'],
            node['rr_sum'],
            node['mean_qq'],
            node['mean_hu'],
            node['mean_fg']
        ])

    df = pd.DataFrame(data, columns=['tg', 'tn', 'tx', 'rr_sum', 'qq', 'hu', 'fg'])

    # --- 2. Scale the data ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # --- 3. Apply PCA ---
    pca = PCA(n_components=1)
    pca.fit(scaled_data)

    # --- 4. Get component loadings and normalize ---
    # Loadings are the correlations between the variables and the component.
    # We take the absolute value as the direction doesn't matter for importance.
    loadings = np.abs(pca.components_[0])

    # --- 5. Create the weights dictionary ---
    # The 'deriv' weight is not part of the PCA, so we'll have to decide how to set it.
    # Let's start by distributing the PCA-based weights and consider 'deriv' separately.
    
    # Let's assign a fixed proportion of the total weight to 'deriv'
    # and distribute the rest according to PCA loadings.
    deriv_weight = 0.1  # Heuristic: assign 10% importance to the derivative trend
    
    # Normalize the PCA loadings to sum to (1 - deriv_weight)
    pca_loadings_sum = np.sum(loadings)
    normalized_loadings = loadings / pca_loadings_sum
    
    # The PCA-derived variables get the remaining 90% of the weight
    remaining_weight = 1.0 - deriv_weight
    
    weights = {
        'deriv': deriv_weight,
        'tg': normalized_loadings[0] * remaining_weight,
        'tn': normalized_loadings[1] * remaining_weight,
        'tx': normalized_loadings[2] * remaining_weight,
        'rr_sum': normalized_loadings[3] * remaining_weight,
        'qq': normalized_loadings[4] * remaining_weight,
        'hu': normalized_loadings[5] * remaining_weight,
        'fg': normalized_loadings[6] * remaining_weight,
    }

    return weights
