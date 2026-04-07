import os
import pandas as pd
import numpy as np
from dtaidistance import dtw
import random
import json
from itertools import combinations

from get_monthly_nodes import get_monthly_nodes

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
EARLY_PERIOD = ('1961-01-01', '1990-12-31')
LATE_PERIOD = ('1991-01-01', '2024-12-31')
ALL_PERIOD = ('1960-01-01', '2024-12-31')
WEIGHTS = {'deriv': 0.1, 'tg': 0.5, 'tn': 0.2, 'tx': 0.2}
N_STATS_SAMPLES = 10000  # Number of pairs to sample for global statistics

# --- Load City Names ---
with open(os.path.join(SCRIPT_DIR, 'cities.txt'), 'r') as f:
    CITIES = [line.strip() for line in f.readlines()]

# --- Adapted Distance Function ---
def calculate_distance(month1_data, month2_data, stats):
    """
    Calculates a composite distance between two months based on climate data,
    using z-scored components. This function is adapted from
    'networks/utils/calculate_better_similarity.py'.

    Returns the raw combined distance, not the final similarity score.
    """
    # --- 1. DTW on derivatives (z-normalized DTW) ---
    d1 = np.array(month1_data['tg_derivatives'], dtype=np.double)
    d2 = np.array(month2_data['tg_derivatives'], dtype=np.double)

    if len(d1) == 0 or len(d2) == 0:
        return np.inf # Cannot compute distance if one series is empty

    # Z-normalize the sequences themselves
    d1_norm = (d1 - d1.mean()) / d1.std() if d1.std() > 0 else d1
    d2_norm = (d2 - d2.mean()) / d2.std() if d2.std() > 0 else d2

    dtw_distance = dtw.distance(d1_norm, d2_norm)

    # z-score the DTW distance
    dtw_z = (dtw_distance - stats['dtw_mean']) / stats['dtw_std']

    # --- 2. Monthly statistics (z-score differences) ---
    tg_diff = abs(month1_data['mean_tg'] - month2_data['mean_tg'])
    tn_diff = abs(month1_data['mean_tn'] - month2_data['mean_tn'])
    tx_diff = abs(month1_data['mean_tx'] - month2_data['mean_tx'])

    tg_z = (tg_diff - stats['tg_mean']) / stats['tg_std']
    tn_z = (tn_diff - stats['tn_mean']) / stats['tn_std']
    tx_z = (tx_diff - stats['tx_mean']) / stats['tx_std']

    # --- 3. Weighted combined distance ---
    combined_distance = (
        WEIGHTS['deriv'] * dtw_z +
        WEIGHTS['tg'] * tg_z +
        WEIGHTS['tn'] * tn_z +
        WEIGHTS['tx'] * tx_z
    )

    return combined_distance

def get_all_nodes(cities):
    """Loads all monthly nodes for all given cities over the entire period."""
    all_nodes = []
    print("Loading all monthly nodes for all cities...")
    for city in cities:
        # Correctly call get_monthly_nodes which is in the global scope
        nodes = get_monthly_nodes(city, ALL_PERIOD[0], ALL_PERIOD[1])
        for month_str, data in nodes.items():
            all_nodes.append({'city': city, 'month_str': month_str, 'data': data})
    print(f"Loaded {len(all_nodes)} total nodes.")
    return all_nodes

def calculate_global_stats(nodes):
    """
    Calculates the mean and standard deviation of distance components
    by sampling a large number of pairs from all nodes.
    """
    print(f"Calculating global statistics from {len(nodes)} nodes...")
    dtw_distances, tg_diffs, tn_diffs, tx_diffs = [], [], [], []

    sample_pairs = random.sample(list(combinations(nodes, 2)), min(N_STATS_SAMPLES, len(nodes) * (len(nodes) - 1) // 2))

    for node1, node2 in sample_pairs:
        # Raw DTW distance
        d1 = np.array(node1['data']['tg_derivatives'], dtype=np.double)
        d2 = np.array(node2['data']['tg_derivatives'], dtype=np.double)
        if len(d1) == 0 or len(d2) == 0: continue
        d1_norm = (d1 - d1.mean()) / d1.std() if d1.std() > 0 else d1
        d2_norm = (d2 - d2.mean()) / d2.std() if d2.std() > 0 else d2
        dtw_distances.append(dtw.distance(d1_norm, d2_norm))

        # Raw diffs
        tg_diffs.append(abs(node1['data']['mean_tg'] - node2['data']['mean_tg']))
        tn_diffs.append(abs(node1['data']['mean_tn'] - node2['data']['mean_tn']))
        tx_diffs.append(abs(node1['data']['mean_tx'] - node2['data']['mean_tx']))

    stats = {
        'dtw_mean': np.mean(dtw_distances), 'dtw_std': np.std(dtw_distances),
        'tg_mean': np.mean(tg_diffs), 'tg_std': np.std(tg_diffs),
        'tn_mean': np.mean(tn_diffs), 'tn_std': np.std(tn_diffs),
        'tx_mean': np.mean(tx_diffs), 'tx_std': np.std(tx_diffs),
    }
    print("Global statistics calculated.")
    return stats

def find_medoid(nodes, stats):
    """
    Finds the medoid of a set of monthly nodes. The medoid is the node
    with the minimum total distance to all other nodes.
    """
    if not nodes:
        return None
    if len(nodes) == 1:
        return list(nodes.keys())[0]

    node_keys = list(nodes.keys())
    dist_matrix = pd.DataFrame(np.zeros((len(node_keys), len(node_keys))), index=node_keys, columns=node_keys)

    for i in range(len(node_keys)):
        for j in range(i + 1, len(node_keys)):
            dist = calculate_distance(nodes[node_keys[i]], nodes[node_keys[j]], stats)
            dist_matrix.iloc[i, j] = dist
            dist_matrix.iloc[j, i] = dist
    
    # The medoid is the node with the minimum sum of distances to all other nodes
    total_distances = dist_matrix.sum(axis=1)
    medoid_key = total_distances.idxmin()
    return medoid_key

if __name__ == '__main__':
    # --- 1. Load all data and calculate global statistics ---
    all_nodes = get_all_nodes(CITIES)
    global_stats = calculate_global_stats(all_nodes)
    
    # --- 2. Find Medoids for each city, month, and period ---
    prototypes = {}
    for city in CITIES:
        prototypes[city] = {}
        print(f"Processing city: {city}...")
        for month in range(1, 13):
            # Early period
            early_nodes = get_monthly_nodes(city, EARLY_PERIOD[0], EARLY_PERIOD[1], target_month=month)
            early_medoid = find_medoid(early_nodes, global_stats)
            prototypes[city][f'early_{month}'] = early_medoid
            
            # Late period
            late_nodes = get_monthly_nodes(city, LATE_PERIOD[0], LATE_PERIOD[1], target_month=month)
            late_medoid = find_medoid(late_nodes, global_stats)
            prototypes[city][f'late_{month}'] = late_medoid
            
            print(f"  Month {month}: Early={early_medoid}, Late={late_medoid}")

    # --- 3. Save Prototypes ---
    output_path = os.path.join(PROJECT_ROOT, 'prototypes')
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, 'prototypes.json'), 'w') as f:
        json.dump(prototypes, f, indent=4)
        
    print(f"\nPrototypes saved to {os.path.join(output_path, 'prototypes.json')}")
