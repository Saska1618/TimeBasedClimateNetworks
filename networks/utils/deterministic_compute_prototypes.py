
import os
import networkx as nx
import pandas as pd
import numpy as np
import json
from itertools import combinations

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'correct_avg_degree')
EARLY_PERIOD_YEARS = range(1961, 1991)
LATE_PERIOD_YEARS = range(1991, 2025)
# Penalty for missing edges in the graph
MISSING_EDGE_PENALTY = 1_000_000

# --- Load City Names ---
with open(os.path.join(SCRIPT_DIR, 'cities.txt'), 'r') as f:
    CITIES = [line.strip() for line in f.readlines()]

def find_medoid_from_graph(node_ids, G):
    """
    Finds the medoid of a set of nodes using pre-calculated weights from a graph.
    The medoid is the node with the minimum total distance to all other nodes in the set.
    """
    if not node_ids:
        return None
    # Filter out nodes that are not actually in the graph
    node_ids = [nid for nid in node_ids if G.has_node(nid)]
    if len(node_ids) <= 1:
        return node_ids[0] if node_ids else None

    total_distances = {node: 0.0 for node in node_ids}

    for u, v in combinations(node_ids, 2):
        distance = MISSING_EDGE_PENALTY
        if G.has_edge(u, v):
            weight = G.get_edge_data(u, v).get('weight', 0)
            if weight > 0:
                # similarity to distance conversion
                distance = (1 / weight) - 1
        
        total_distances[u] += distance
        total_distances[v] += distance

    # Find the node with the minimum total distance
    medoid = min(total_distances, key=total_distances.get)
    return medoid

if __name__ == '__main__':
    prototypes = {}
    print("Starting deterministic prototype computation...")

    for city in CITIES:
        prototypes[city] = {}
        print(f"Processing city: {city}...")
        
        # Load the full-period graph for the city
        graph_file = os.path.join(GRAPH_DIR, f'{city}_global_pruned_1960-01_2024-12.graphml')
        if not os.path.exists(graph_file):
            print(f"  - WARNING: Graph file not found for {city}. Skipping.")
            continue
        
        G = nx.read_graphml(graph_file)

        for month in range(1, 13):
            month_str = f'{month:02d}' # e.g., "01", "02", ...

            # --- Early Period ---
            early_node_ids = [f'{year}-{month_str}' for year in EARLY_PERIOD_YEARS]
            early_medoid = find_medoid_from_graph(early_node_ids, G)
            prototypes[city][f'early_{month}'] = early_medoid

            # --- Late Period ---
            late_node_ids = [f'{year}-{month_str}' for year in LATE_PERIOD_YEARS]
            late_medoid = find_medoid_from_graph(late_node_ids, G)
            prototypes[city][f'late_{month}'] = late_medoid
            
            print(f"  Month {month}: Early={early_medoid}, Late={late_medoid}")

    # --- Save Prototypes ---
    output_dir = os.path.join(PROJECT_ROOT, 'prototypes')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'deterministic_prototypes.json')
    with open(output_path, 'w') as f:
        json.dump(prototypes, f, indent=4)
        
    print(f"\nDeterministic prototypes saved to {output_path}")
