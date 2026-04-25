import networkx as nx
import numpy as np
import json
from scipy.spatial.distance import pdist, squareform
import os

def load_graph(file_path):
    """Loads a graph from a GraphML file."""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    return nx.read_graphml(file_path)

def get_monthly_vectors(graph, month):
    """Extracts the climate variable vectors for a specific month from the graph."""
    vectors = []
    for node, data in graph.nodes(data=True):
        if data.get('month') == month:
            vector = [
                data.get('mean_tg', 0),
                data.get('mean_tn', 0),
                data.get('mean_tx', 0),
                data.get('mean_qq', 0),
                data.get('mean_hu', 0),
                data.get('rr_sum', 0)
            ]
            vectors.append(vector)
    return np.array(vectors)

def create_centroid_prototype(vectors):
    """Creates a prototype by calculating the centroid of the vectors."""
    if vectors.shape[0] == 0:
        return None
    return np.mean(vectors, axis=0)

def calculate_network_metrics(graph):
    """Calculates basic network metrics."""
    if graph is None:
        return {}
    return {
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'average_degree': np.mean([d for n, d in graph.degree()]),
        'average_clustering': nx.average_clustering(graph)
    }

def main():
    # --- Configuration ---
    cities = ['Cluj']
    periods = {
        'period1': ('1961-01', '1990-12'),
        'period2': ('1991-01', '2024-12')
    }
    base_path = 'networks/global_networks/rich_global/'

    # --- Analysis ---
    all_results = {}

    for city in cities:
        all_results[city] = {}
        
        # 1. Compare Network Metrics
        print(f"--- Comparing network metrics for {city} ---")
        for period_name, (start, end) in periods.items():
            graph_path = os.path.join(base_path, f"{city}_{start}_{end}.graphml")
            graph = load_graph(graph_path)
            metrics = calculate_network_metrics(graph)
            all_results[city][f"{period_name}_metrics"] = metrics
            print(f"  {period_name}: {metrics}")

        # 2. Create and Compare Prototypes
        print(f"
--- Creating and comparing prototypes for {city} ---")
        all_results[city]['prototype_comparison'] = {}
        
        prototypes = {}
        for period_name, (start, end) in periods.items():
            prototypes[period_name] = {}
            graph_path = os.path.join(base_path, f"{city}_{start}_{end}.graphml")
            graph = load_graph(graph_path)
            if graph is None:
                continue
            
            for month in range(1, 13):
                vectors = get_monthly_vectors(graph, month)
                prototype = create_centroid_prototype(vectors)
                prototypes[period_name][month] = prototype

        # Compare prototypes between the two periods
        for month in range(1, 13):
            p1 = prototypes.get('period1', {}).get(month)
            p2 = prototypes.get('period2', {}).get(month)
            
            if p1 is not None and p2 is not None:
                # Using Euclidean distance to compare prototypes
                distance = np.linalg.norm(p1 - p2)
                all_results[city]['prototype_comparison'][f'month_{month}_distance'] = distance
                print(f"  Distance between prototypes for month {month}: {distance:.4f}")

    # --- Save results ---
    with open('prototype_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print("
Analysis complete. Results saved to prototype_analysis_results.json")

if __name__ == '__main__':
    main()
