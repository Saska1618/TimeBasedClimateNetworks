import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def calculate_pca_weights(monthly_data, *args, variance_threshold=0.85):
    """
    Calculates weights for different climate variables using PCA and integrates a pre-defined derivative weight.

    The method calculates weights for scalar variables based on PCA. It then scales these
    weights to accommodate a specified weight for the 'deriv' (time-series derivative) feature.

    Args:
        monthly_data (dict): A dictionary of monthly climate data, like the one
                             returned by get_rich_monthly_nodes.
        variance_threshold (float): The cumulative variance threshold to decide 
                                    how many principal components to keep for scalar variables.

    Returns:
        dict: A dictionary with all variable names and their final, normalized weights, summing to 1.0.
    """
    
    # Drop tg_derivatives and convert to DataFrame to handle scalar variables
    df_data = {
        month: {k: v for k, v in data.items() if k != 'tg_derivatives'}
        for month, data in monthly_data.items()
    }
    df = pd.DataFrame.from_dict(df_data, orient='index')

    if df.empty:
        return {}

    # --- Data Preparation for PCA ---
    # Per user request, 'fg' weight is always 0, so remove it from PCA.
    if 'mean_fg' in df.columns:
        df = df.drop(columns=['mean_fg'])

    # Fill any other potential NaNs with 0 for robustness.
    df.fillna(0, inplace=True)
    
    # If all other scalar variables were unusable, handle gracefully.
    if df.empty:
        final_weights = {'deriv': 0.0, 'fg': 0.0}
        # Normalize weights to sum to 1.0
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v / total_weight for k, v in final_weights.items()}
        return final_weights, np.array([])

    # --- PCA on Scalar Variables ---
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(scaled_data)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components_to_keep = np.argmax(cumulative_variance >= variance_threshold) + 1

    loadings = pca.components_
    
    raw_scalar_weights = {}
    for j, col_name in enumerate(df.columns):
        weight_j = 0
        for i in range(n_components_to_keep):
            explained_variance_pc_i = explained_variance_ratio[i]
            loading_j_pc_i = loadings[i, j]
            weight_j += explained_variance_pc_i * abs(loading_j_pc_i)
        
        # Map column name to shorter key, e.g., 'mean_tg' -> 'tg'
        short_key = col_name.replace('mean_', '')
        raw_scalar_weights[short_key] = weight_j

    # --- Combine with deriv_weight ---
    total_scalar_weight = sum(raw_scalar_weights.values())
    
    final_weights = {}

    if total_scalar_weight > 0:
        scalar_weight_factor = 1 / total_scalar_weight
        for key, value in raw_scalar_weights.items():
            final_weights[key] = value * scalar_weight_factor

    final_weights['deriv'] = 0.0
    final_weights['fg'] = 0.0

    return final_weights, explained_variance_ratio
