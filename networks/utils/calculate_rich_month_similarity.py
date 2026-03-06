import numpy as np
from dtaidistance import dtw
from .weight_calculation_pca import calculate_pca_weights

def calculate_rich_month_similarity(
    month1_data,
    month2_data,
    stats,
    weights
):
    """
    Calculates the similarity between two "rich" nodes, which include more climate variables.

    Args:
        month1_data (dict): Data for the first month.
        month2_data (dict): Data for the second month.
        stats (dict): A dictionary with global means and standard deviations for z-scoring.
                      e.g., {'dtw_mean': ..., 'dtw_std': ..., 'tg_mean': ..., 'tg_std': ...}
        weights (dict): A dictionary of weights for each variable. It is recommended to
                        generate this using the `calculate_pca_weights` function to get
                        objective, data-driven weights.

    Returns:
        float: The calculated similarity score between 0 and 1.
    """

    # --- 1. DTW on derivatives (z-normalized DTW) ---
    d1 = np.array(month1_data['tg_derivatives'], dtype=np.double)
    d2 = np.array(month2_data['tg_derivatives'], dtype=np.double)

    # optional but recommended: z-normalize the sequences themselves
    if d1.std() > 0 and d2.std() > 0:
        d1 = (d1 - d1.mean()) / d1.std()
        d2 = (d2 - d2.mean()) / d2.std()

    dtw_distance = dtw.distance(d1, d2)

    # z-score the DTW distance
    dtw_z = (dtw_distance - stats['dtw_mean']) / stats['dtw_std']

    # --- 2. Monthly statistics (already scalar → z-score differences) ---
    tg_diff = abs(month1_data['mean_tg'] - month2_data['mean_tg'])
    tn_diff = abs(month1_data['mean_tn'] - month2_data['mean_tn'])
    tx_diff = abs(month1_data['mean_tx'] - month2_data['mean_tx'])
    rr_sum_diff = abs(month1_data['rr_sum'] - month2_data['rr_sum'])
    qq_diff = abs(month1_data['mean_qq'] - month2_data['mean_qq'])
    hu_diff = abs(month1_data['mean_hu'] - month2_data['mean_hu'])
    fg_diff = abs(month1_data['mean_fg'] - month2_data['mean_fg'])

    tg_z = (tg_diff - stats['tg_mean']) / stats['tg_std']
    tn_z = (tn_diff - stats['tn_mean']) / stats['tn_std']
    tx_z = (tx_diff - stats['tx_mean']) / stats['tx_std']
    rr_sum_z = (rr_sum_diff - stats['rr_sum_mean']) / stats['rr_sum_std']
    qq_z = (qq_diff - stats['qq_mean']) / stats['qq_std']
    hu_z = (hu_diff - stats['hu_mean']) / stats['hu_std']
    fg_z = (fg_diff - stats['fg_mean']) / stats['fg_std']


    # --- 3. Weighted combined distance ---
    combined_distance = (
        weights['deriv'] * dtw_z +
        weights['tg'] * tg_z +
        weights['tn'] * tn_z +
        weights['tx'] * tx_z +
        weights['rr_sum'] * rr_sum_z +
        weights['qq'] * qq_z +
        weights['hu'] * hu_z +
        weights['fg'] * fg_z
    )

    # --- 4. Convert distance → similarity ---
    similarity = 1 / (1 + np.exp(combined_distance))  # sigmoid, stable

    return similarity
