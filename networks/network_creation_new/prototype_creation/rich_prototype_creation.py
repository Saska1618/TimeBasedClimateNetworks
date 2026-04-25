
import os
import sys
import networkx as nx
import numpy as np
import json
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
PROTOTYPE_DIR = os.path.join(PROJECT_ROOT, 'prototypes')
EARLY_PERIOD_YEARS = range(1961, 1991)
LATE_PERIOD_YEARS = range(1995, 2025)
FEATURES = ['mean_tn', 'mean_tx', 'mean_tg', 'rr_sum', 'mean_qq', 'mean_hu']

sys.path.append(PROJECT_ROOT)

# --- Load City Names ---
with open(os.path.join(SCRIPT_DIR, '..', '..', 'utils', 'cities.txt'), 'r') as f:
    CITIES = [line.strip() for line in f.readlines()]

def get_pca_scores(graph):
    """
    Extracts features, scales them, and computes PCA scores for all nodes in the graph.
    """
    nodes = list(graph.nodes())
    
    # Create a data matrix X with nodes as rows and features as columns
    X = np.array([[graph.nodes[node].get(feature, np.nan) for feature in FEATURES] for node in nodes])

    # Handle missing values, e.g., by filling with the mean of the column
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    
    return scores, nodes

def find_medoid(node_ids, all_nodes, distance_matrix):
    """
    Finds the medoid for a specific subset of nodes using the global distance matrix.
    The medoid is the node with the minimum total distance to all other nodes in the subset.
    """
    if not node_ids:
        return None
        
    # Get the indices of the nodes in our subset
    subset_indices = [all_nodes.index(nid) for nid in node_ids if nid in all_nodes]
    if not subset_indices:
        return None

    # Sum distances for each node in the subset to all other nodes in the subset
    total_distances = {
        node_idx: sum(distance_matrix[node_idx, other_idx] for other_idx in subset_indices)
        for node_idx in subset_indices
    }
    
    # Find the index of the node with the minimum total distance
    medoid_idx = min(total_distances, key=total_distances.get)
    
    # Return the ID of the medoid node
    return all_nodes[medoid_idx]

def process_city(city):
    """
    Computes and saves prototypes for a single city.
    """
    print(f"Processing city: {city}...")
    city_prototypes = {}

    # 1. Load the full period graph to create a single, consistent PCA space.
    full_graph_path = os.path.join(GRAPH_DIR, f'{city}_1961-01_2024-12.graphml')
    if not os.path.exists(full_graph_path):
        print(f"  - WARNING: Full period graph file not found for {city}. Skipping city.")
        return {}

    G_full = nx.read_graphml(full_graph_path)

    # 2. Get PCA scores and node list from the full graph. This defines the common space.
    print("  - Creating a single PCA space from the full 1961-2024 graph...")
    scores, all_nodes = get_pca_scores(G_full)

    # 3. Compute the full distance matrix from the PCA scores of the full graph.
    distance_matrix = squareform(pdist(scores, metric='euclidean'))

    # 4. Now find medoids for subsets (early/late months) within this global space.
    period_definitions = {
        'early': EARLY_PERIOD_YEARS,
        'late': LATE_PERIOD_YEARS
    }

    for period_name, years in period_definitions.items():
        # We no longer need to load individual graph files, as all nodes are in G_full.
        # We just need the list of nodes for each period.

        for month in range(1, 13):
            month_str = f'{month:02d}'

            # Get the node IDs for the current month and period
            node_ids_for_month = [f'{year}-{month_str}' for year in years]

            # Find the medoid for this group of nodes using the global distance matrix
            medoid = find_medoid(node_ids_for_month, all_nodes, distance_matrix)
            city_prototypes[f'{period_name}_{month}'] = medoid
            print(f"  - {period_name.capitalize()} Month {month}: {medoid}")
            
    return city_prototypes

if __name__ == '__main__':
    all_prototypes = {}
    
    for city in CITIES:
        all_prototypes[city] = process_city(city)

    # --- Save Prototypes ---
    os.makedirs(PROTOTYPE_DIR, exist_ok=True)
    output_path = os.path.join(PROTOTYPE_DIR, 'rich_prototypes.json')
    with open(output_path, 'w') as f:
        json.dump(all_prototypes, f, indent=4)
        
    print(f"Rich prototypes saved to {output_path}")
