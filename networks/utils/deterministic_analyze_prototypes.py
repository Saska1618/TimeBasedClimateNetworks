
import os
import networkx as nx
import numpy as np
import json
from itertools import product

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'correct_avg_degree')
PROTOTYPES_PATH = os.path.join(PROJECT_ROOT, 'prototypes', 'deterministic_prototypes.json')

if __name__ == '__main__':
    # --- 1. Load Deterministic Prototypes ---
    try:
        with open(PROTOTYPES_PATH, 'r') as f:
            prototypes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prototype file not found at {PROTOTYPES_PATH}")
        print("Please run 'deterministic_compute_prototypes.py' first.")
        exit(1)

    cities = list(prototypes.keys())
    similarity_matrices = {}
    print("Starting deterministic similarity matrix computation...")

    for city in cities:
        similarity_matrices[city] = np.zeros((12, 12))
        print(f"Processing city: {city}...")

        # Load the full-period graph for the city
        graph_file = os.path.join(GRAPH_DIR, f'{city}_global_pruned_1960-01_2024-12.graphml')
        if not os.path.exists(graph_file):
            print(f"  - WARNING: Graph file not found for {city}. Skipping.")
            continue
        
        G = nx.read_graphml(graph_file)

        # --- 2. Build Similarity Matrix from Graph Weights ---
        for i, j in product(range(1, 13), range(1, 13)):
            early_month_id = f'early_{i}'
            late_month_id = f'late_{j}'
            
            early_proto_key = prototypes[city].get(early_month_id)
            late_proto_key = prototypes[city].get(late_month_id)

            similarity = 0.0 # Default to 0 if edge doesn't exist
            if early_proto_key and late_proto_key and G.has_edge(early_proto_key, late_proto_key):
                similarity = G.get_edge_data(early_proto_key, late_proto_key).get('weight', 0.0)
            
            similarity_matrices[city][i-1, j-1] = similarity
        
        print(f"  - Matrix for {city} created.")

    # --- 3. Save Matrices ---
    # Convert numpy arrays to lists for JSON serialization
    matrices_as_lists = {city: mat.tolist() for city, mat in similarity_matrices.items()}
    
    output_dir = os.path.join(PROJECT_ROOT, 'prototypes')
    output_path = os.path.join(output_dir, 'deterministic_similarity_matrices.json')
    with open(output_path, 'w') as f:
        json.dump(matrices_as_lists, f, indent=4)

    print(f"\nDeterministic similarity matrices saved to {output_path}")
