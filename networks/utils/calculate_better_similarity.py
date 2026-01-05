import numpy as np
from dtaidistance import dtw

def calculate_similarity(
    month1_data,
    month2_data,
    stats,  # <-- NEW: global means & stds
    weights={'deriv': 0.5, 'tg': 0.3, 'tn': 0.1, 'tx': 0.1}
):
    """
    stats = {
        'dtw_mean': ..., 'dtw_std': ...,
        'tg_mean': ...,  'tg_std': ...,
        'tn_mean': ...,  'tn_std': ...,
        'tx_mean': ...,  'tx_std': ...
    }
    """

    # --- 1. DTW on derivatives (z-normalized DTW) ---
    d1 = np.array(month1_data['tg_derivatives'], dtype=np.double)
    d2 = np.array(month2_data['tg_derivatives'], dtype=np.double)

    # optional but recommended: z-normalize the sequences themselves
    d1 = (d1 - d1.mean()) / d1.std()
    d2 = (d2 - d2.mean()) / d2.std()

    dtw_distance = dtw.distance(d1, d2)

    # z-score the DTW distance
    dtw_z = (dtw_distance - stats['dtw_mean']) / stats['dtw_std']

    # --- 2. Monthly statistics (already scalar → z-score differences) ---
    tg_diff = abs(month1_data['mean_tg'] - month2_data['mean_tg'])
    tn_diff = abs(month1_data['mean_tn'] - month2_data['mean_tn'])
    tx_diff = abs(month1_data['mean_tx'] - month2_data['mean_tx'])

    tg_z = (tg_diff - stats['tg_mean']) / stats['tg_std']
    tn_z = (tn_diff - stats['tn_mean']) / stats['tn_std']
    tx_z = (tx_diff - stats['tx_mean']) / stats['tx_std']

    # --- 3. Weighted combined distance ---
    combined_distance = (
        weights['deriv'] * dtw_z +
        weights['tg'] * tg_z +
        weights['tn'] * tn_z +
        weights['tx'] * tx_z
    )

    # --- 4. Convert distance → similarity ---
    similarity = 1 / (1 + np.exp(combined_distance))  # sigmoid, stable

    return similarity