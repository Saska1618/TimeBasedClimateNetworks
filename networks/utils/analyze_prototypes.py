
import os
import sys
import pandas as pd
import numpy as np
import json
from itertools import combinations, product
import random

# Temporarily add the utils directory to the path to import from the other script
# This is not ideal, but helps reuse code without modifying existing files.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from get_monthly_nodes import get_monthly_nodes
# We will copy the necessary functions from compute_prototypes to avoid circular dependencies
# and to adhere to the "no modification" rule.

# --- Constants ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ALL_PERIOD = ('1960-01-01', '2024-12-31')
WEIGHTS = {'deriv': 0.1, 'tg': 0.5, 'tn': 0.2, 'tx': 0.2}
N_STATS_SAMPLES = 10000

# --- Functions copied from compute_prototypes.py ---

# This is a temporary solution to reuse the dtw function from dtaidistance
# without modifying the original file.
try:
    from dtaidistance import dtw
except ImportError:
    print("dtaidistance is not installed. Please install it using 'pip install dtaidistance'")
    sys.exit(1)


def calculate_distance(month1_data, month2_data, stats):
    """
    Calculates a composite distance between two months based on climate data,
    using z-scored components.
    """
    d1 = np.array(month1_data['tg_derivatives'], dtype=np.double)
    d2 = np.array(month2_data['tg_derivatives'], dtype=np.double)
    if len(d1) == 0 or len(d2) == 0: return np.inf
    d1_norm = (d1 - d1.mean()) / d1.std() if d1.std() > 0 else d1
    d2_norm = (d2 - d2.mean()) / d2.std() if d2.std() > 0 else d2
    dtw_distance = dtw.distance(d1_norm, d2_norm)
    dtw_z = (dtw_distance - stats['dtw_mean']) / stats['dtw_std']

    tg_diff = abs(month1_data['mean_tg'] - month2_data['mean_tg'])
    tn_diff = abs(month1_data['mean_tn'] - month2_data['mean_tn'])
    tx_diff = abs(month1_data['mean_tx'] - month2_data['mean_tx'])
    tg_z = (tg_diff - stats['tg_mean']) / stats['tg_std']
    tn_z = (tn_diff - stats['tn_mean']) / stats['tn_std']
    tx_z = (tx_diff - stats['tx_mean']) / stats['tx_std']

    combined_distance = (
        WEIGHTS['deriv'] * dtw_z +
        WEIGHTS['tg'] * tg_z +
        WEIGHTS['tn'] * tn_z +
        WEIGHTS['tx'] * tx_z
    )
    return combined_distance

def get_all_monthly_nodes_data(cities):
    """
    Loads all monthly node data for all cities and stores it in a nested dictionary
    for quick lookup.
    """
    all_data = {}
    print("Loading all monthly time series data...")
    for city in cities:
        all_data[city] = get_monthly_nodes(city, ALL_PERIOD[0], ALL_PERIOD[1])
    print("Finished loading all data.")
    return all_data

def get_all_nodes_list(all_monthly_data):
    """Converts the nested dictionary of all nodes into a flat list."""
    all_nodes_list = []
    for city, nodes in all_monthly_data.items():
        for month_str, data in nodes.items():
            all_nodes_list.append({'city': city, 'month_str': month_str, 'data': data})
    return all_nodes_list


def calculate_global_stats(nodes_list):
    """
    Calculates the mean and standard deviation of distance components
    by sampling a large number of pairs from all nodes.
    """
    print(f"Calculating global statistics from {len(nodes_list)} nodes...")
    dtw_distances, tg_diffs, tn_diffs, tx_diffs = [], [], [], []
    
    # Ensure we don't try to sample more pairs than exist
    num_possible_pairs = len(nodes_list) * (len(nodes_list) - 1) // 2
    num_samples = min(N_STATS_SAMPLES, num_possible_pairs)

    sample_pairs = random.sample(list(combinations(nodes_list, 2)), num_samples)

    for node1, node2 in sample_pairs:
        d1 = np.array(node1['data']['tg_derivatives'], dtype=np.double)
        d2 = np.array(node2['data']['tg_derivatives'], dtype=np.double)
        if len(d1) == 0 or len(d2) == 0: continue
        d1_norm = (d1 - d1.mean()) / d1.std() if d1.std() > 0 else d1
        d2_norm = (d2 - d2.mean()) / d2.std() if d2.std() > 0 else d2
        dtw_distances.append(dtw.distance(d1_norm, d2_norm))

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


if __name__ == '__main__':
    # --- 1. Load Prototypes and All Node Data ---
    prototypes_path = os.path.join(PROJECT_ROOT, 'prototypes', 'prototypes.json')
    with open(prototypes_path, 'r') as f:
        prototypes = json.load(f)

    cities = list(prototypes.keys())
    all_monthly_data = get_all_monthly_nodes_data(cities)
    all_nodes_list = get_all_nodes_list(all_monthly_data)

    # --- 2. Calculate Global Stats ---
    global_stats = calculate_global_stats(all_nodes_list)

    # --- 3. Build Similarity Matrices ---
    similarity_matrices = {}
    print("Building similarity matrices for each city...")
    for city in cities:
        similarity_matrices[city] = np.zeros((12, 12))
        for i, j in product(range(1, 13), range(1, 13)):
            early_month_id = f'early_{i}'
            late_month_id = f'late_{j}'
            
            early_proto_key = prototypes[city].get(early_month_id)
            late_proto_key = prototypes[city].get(late_month_id)

            if not early_proto_key or not late_proto_key:
                similarity = np.nan
            else:
                early_data = all_monthly_data[city][early_proto_key]
                late_data = all_monthly_data[city][late_proto_key]
                
                distance = calculate_distance(early_data, late_data, global_stats)
                similarity = 1 / (1 + distance) if distance != np.inf else 0
            
            similarity_matrices[city][i-1, j-1] = similarity
        print(f"  - Matrix for {city} created.")

    # --- 4. Save Matrices ---
    # Convert numpy arrays to lists for JSON serialization
    matrices_as_lists = {city: mat.tolist() for city, mat in similarity_matrices.items()}
    
    output_path = os.path.join(PROJECT_ROOT, 'prototypes', 'similarity_matrices.json')
    with open(output_path, 'w') as f:
        json.dump(matrices_as_lists, f, indent=4)

    print(f"\nSimilarity matrices saved to {output_path}")
