import os
import sys
import networkx as nx
import numpy as np
import json

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
PROTOTYPE_DIR = os.path.join(PROJECT_ROOT, 'prototypes', 'mean_prototypes')
EARLY_PERIOD_YEARS = range(1961, 1991)
LATE_PERIOD_YEARS = range(1995, 2025)
FEATURES = ['mean_tn', 'mean_tx', 'mean_tg', 'rr_sum', 'mean_qq', 'mean_hu']

sys.path.append(PROJECT_ROOT)

# --- Load City Names ---
with open(os.path.join(SCRIPT_DIR, '..', '..', 'utils', 'cities.txt'), 'r') as f:
    CITIES = [line.strip() for line in f.readlines()]

def get_mean_prototype(graph, node_ids):
    """
    Computes the mean value for each feature across the provided node_ids.
    """
    valid_nodes = [node for node in node_ids if node in graph.nodes]
    if not valid_nodes:
        return {feature: None for feature in FEATURES}
    
    # Extract features for valid nodes
    X = np.array([[graph.nodes[node].get(feature, np.nan) for feature in FEATURES] for node in valid_nodes])
    
    mean_prototype = {}
    for i, feature in enumerate(FEATURES):
        col_data = X[:, i]
        
        # Check if all values for a feature are NaN to avoid warnings with np.nanmean
        if np.isnan(col_data).all():
            mean_prototype[feature] = None
        else:
            mean_prototype[feature] = float(np.nanmean(col_data))
            
    return mean_prototype

def process_city(city):
    """
    Computes and saves mean prototypes for a single city.
    """
    print(f"Processing city: {city}...")
    city_prototypes = {}
    
    # Define time periods
    periods = {
        'early': (f'{city}_1961-01_1990-12.graphml', EARLY_PERIOD_YEARS),
        'late': (f'{city}_1995-01_2024-12.graphml', LATE_PERIOD_YEARS)
    }

    for period_name, (graph_file, years) in periods.items():
        graph_path = os.path.join(GRAPH_DIR, graph_file)
        if not os.path.exists(graph_path):
            print(f"  - WARNING: Graph file not found for {city} {period_name}. Skipping.")
            continue
            
        G = nx.read_graphml(graph_path)
        
        for month in range(1, 13):
            month_str = f'{month:02d}'
            
            # Get the node IDs for the current month and period
            node_ids_for_month = [f'{year}-{month_str}' for year in years]
            
            # Compute the mean prototype for this group of nodes
            mean_prototype = get_mean_prototype(G, node_ids_for_month)
            city_prototypes[f'{period_name}_{month}'] = mean_prototype
            print(f"  - {period_name.capitalize()} Month {month} computed.")
            
    return city_prototypes

if __name__ == '__main__':
    all_prototypes = {}
    
    for city in CITIES:
        all_prototypes[city] = process_city(city)

    # --- Save Prototypes ---
    os.makedirs(PROTOTYPE_DIR, exist_ok=True)
    output_path = os.path.join(PROTOTYPE_DIR, 'rich_mean_prototypes.json')
    with open(output_path, 'w') as f:
        json.dump(all_prototypes, f, indent=4)
        
    print(f"Rich mean prototypes saved to {output_path}")