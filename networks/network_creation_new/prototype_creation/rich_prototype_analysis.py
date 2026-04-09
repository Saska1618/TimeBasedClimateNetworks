
import os
import sys
import networkx as nx
import numpy as np
import json
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
PROTOTYPES_PATH = os.path.join(PROJECT_ROOT, 'prototypes', 'rich_prototypes.json')
THRESHOLDS_PATH = os.path.join(PROJECT_ROOT, 'networks', 'network_creation_new', 'network_rich_thresholds.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'prototypes')
FEATURES = ['mean_tn', 'mean_tx', 'mean_tg', 'rr_sum', 'mean_qq', 'mean_hu']
USE_THRESHOLD = True

sys.path.append(PROJECT_ROOT)

def get_pca_scores(graph):
    """
    Extracts features, scales them, and computes PCA scores for all nodes in the graph.
    """
    nodes = list(graph.nodes())
    
    X = np.array([[graph.nodes[node].get(feature, np.nan) for feature in FEATURES] for node in nodes])

    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    scores = pca.fit_transform(X_scaled)
    
    return scores, nodes

def process_city(city, prototypes, thresholds):
    """
    Computes and returns the similarity matrix for a single city, applying thresholds.
    """
    print(f"Processing city: {city}...")
    
    full_period_graph_file = f'{city}_1961-01_2024-12.graphml'
    graph_path = os.path.join(GRAPH_DIR, full_period_graph_file)

    if not os.path.exists(graph_path):
        print(f"  - WARNING: Full period graph not found for {city}. Skipping.")
        return None
        
    G = nx.read_graphml(graph_path)
    
    scores, all_nodes = get_pca_scores(G)
    
    distances = pdist(scores, metric='euclidean')
    scaled_distances = (distances - distances.min()) / (distances.max() - distances.min())
    similarity_matrix_full = 1 - squareform(scaled_distances)

    similarity_matrix_12x12 = np.zeros((12, 12))

    city_prototypes = prototypes.get(city)
    if not city_prototypes:
        print(f"  - WARNING: No prototypes found for {city}. Skipping.")
        return None

    # Get the thresholds for the early and late periods for the current city
    early_thresh_key = f'{city}_1961-01_1990-12'
    late_thresh_key = f'{city}_1995-01_2024-12'
    
    early_threshold = thresholds.get(early_thresh_key, 0)
    late_threshold = thresholds.get(late_thresh_key, 0)
    
    # Use the minimum of the two thresholds as the cutoff
    threshold = min(early_threshold, late_threshold)

    if not USE_THRESHOLD:
        threshold = 0.0

    print(f"  - Using threshold: {threshold:.4f}")

    for i, j in product(range(1, 13), range(1, 13)):
        early_proto_id = city_prototypes.get(f'early_{i}')
        late_proto_id = city_prototypes.get(f'late_{j}')

        if not early_proto_id or not late_proto_id:
            continue

        try:
            idx1 = all_nodes.index(early_proto_id)
            idx2 = all_nodes.index(late_proto_id)
            similarity = similarity_matrix_full[idx1, idx2]
            
            # Apply the threshold
            if similarity < threshold:
                similarity = 0.0
            
            similarity_matrix_12x12[i-1, j-1] = similarity
        except ValueError as e:
            print(f"  - WARNING: Prototype node not found in full graph: {e}. Skipping pair.")
            continue
            
    print(f"  - Matrix for {city} created.")
    return similarity_matrix_12x12

if __name__ == '__main__':
    # --- 1. Load Rich Prototypes and Thresholds ---
    try:
        with open(PROTOTYPES_PATH, 'r') as f:
            prototypes = json.load(f)
        with open(THRESHOLDS_PATH, 'r') as f:
            thresholds = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: A required file was not found: {e.filename}")
        print("Please run 'rich_prototype_creation.py' and 'network_creation_rich_months.ipynb' first.")
        exit(1)

    cities = list(prototypes.keys())
    final_similarity_matrices = {}

    for city in cities:
        matrix = process_city(city, prototypes, thresholds)
        if matrix is not None:
            final_similarity_matrices[city] = matrix.tolist()

    # --- 3. Save Matrices ---
    output_path = os.path.join(OUTPUT_DIR, 'rich_similarity_matrices.json')

    if not USE_THRESHOLD:
        output_path = os.path.join(OUTPUT_DIR, 'rich_similarity_matrices_no_threshold.json')

    with open(output_path, 'w') as f:
        json.dump(final_similarity_matrices, f, indent=4)

    print(f"\nRich similarity matrices saved to {output_path}")
