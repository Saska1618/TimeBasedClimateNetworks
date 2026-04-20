import os
import sys
import json
import math
from collections import defaultdict
import networkx as nx

try:
    import igraph as ig
    import leidenalg
except ImportError:
    print("Error: Required packages are missing.")
    print("Please run: pip install igraph leidenalg")
    sys.exit(1)

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

CITIES_FILE = os.path.join(PROJECT_ROOT, 'networks', 'utils', 'cities.txt')
EARLY_PERIOD = '1961-01_1990-12'
LATE_PERIOD = '1995-01_2024-12'

def load_cities():
    try:
        with open(CITIES_FILE, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Cities file not found at {CITIES_FILE}")
        sys.exit(1)

def convert_nx_to_igraph(G):
    """
    Safely converts a NetworkX graph to an igraph Graph,
    preserving node names and edge weights.
    """
    node_names = list(G.nodes())
    g_ig = ig.Graph(directed=G.is_directed())
    g_ig.add_vertices(len(node_names))
    g_ig.vs['name'] = node_names
    
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    edges = list(G.edges(data=True))
    ig_edges = [(name_to_idx[u], name_to_idx[v]) for u, v, d in edges]
    g_ig.add_edges(ig_edges)
    
    # Preserve weights if they exist
    if edges and 'weight' in edges[0][2]:
        weights = [d.get('weight', 1.0) for u, v, d in edges]
        g_ig.es['weight'] = weights
        
    return g_ig

def detect_communities(g_ig, resolution=1.0):
    """
    Runs the Leiden algorithm on an igraph object.
    Returns a dictionary mapping node names to their community ID.
    """
    weights = g_ig.es['weight'] if 'weight' in g_ig.edge_attributes() else None
    
    partition = leidenalg.find_partition(
        g_ig, 
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=42  # Adding a fixed seed for completely deterministic results
    )
    
    node_to_comm = {}
    for idx, cluster in enumerate(partition):
        for node_index in cluster:
            node_name = g_ig.vs[node_index]['name']
            node_to_comm[node_name] = idx
            
    return node_to_comm

def calculate_shannon_entropy(counts):
    """
    Calculates Shannon Entropy for the community distribution.
    Returns both the raw entropy and normalized entropy [0, 1].
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0, 0.0
    
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p)
            
    max_entropy = math.log(total) if total > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return entropy, normalized_entropy

def get_graph_fragmentation(G):
    if G.number_of_nodes() == 0:
        return {}
        
    g_ig = convert_nx_to_igraph(G)
    node_to_comm = detect_communities(g_ig, resolution=1.0)
    
    total_graph_comms = len(set(node_to_comm.values()))
    print(f"    - Detected {total_graph_comms} total communities in the graph.")
    
    month_community_counts = defaultdict(lambda: defaultdict(int))
    month_total_nodes = defaultdict(int)
    
    for node_name, comm_id in node_to_comm.items():
        try:
            # Node names are in the format "YYYY-MM"
            year, month = node_name.split('-')
            month_community_counts[month][comm_id] += 1
            month_total_nodes[month] += 1
        except ValueError:
            continue
            
    fragmentation = {}
    for month in sorted(month_community_counts.keys()):
        counts = month_community_counts[month]
        
        entropy, normalized_entropy = calculate_shannon_entropy(counts)
        
        total = month_total_nodes[month]
        simpson = 1.0 - sum((count / total) ** 2 for count in counts.values()) if total > 0 else 0.0
        
        fragmentation[month] = {
            "num_unique_communities": len(counts),
            "shannon_entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "simpson_index": simpson,
            "community_distribution": counts
        }
        
    return fragmentation

def safe_ratio(late_val, early_val, eps=1e-3):
    if early_val == 0 and late_val == 0:
        return 1.0
    if early_val == 0:
        return late_val / eps
    return late_val / early_val

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    cities = load_cities()
    all_fragmentation_results = {}
    
    for city in cities:
        early_path = os.path.join(GRAPH_DIR, f'{city}_{EARLY_PERIOD}.graphml')
        late_path = os.path.join(GRAPH_DIR, f'{city}_{LATE_PERIOD}.graphml')
        
        if not os.path.exists(early_path) or not os.path.exists(late_path):
            print(f"  - WARNING: Missing graph data for {city}. Skipping.")
            continue
            
        print(f"Processing {city}...")
        G_early = nx.read_graphml(early_path)
        G_late = nx.read_graphml(late_path)
        
        early_frag = get_graph_fragmentation(G_early)
        late_frag = get_graph_fragmentation(G_late)
        
        city_results = {}
        for m in range(1, 13):
            month = f"{m:02d}"
            e_data = early_frag.get(month)
            l_data = late_frag.get(month)
            
            if not e_data and not l_data:
                continue
                
            ratio = {}
            if e_data and l_data:
                ratio = {
                    "num_unique_communities": safe_ratio(l_data["num_unique_communities"], e_data["num_unique_communities"]),
                    "shannon_entropy": safe_ratio(l_data["shannon_entropy"], e_data["shannon_entropy"]),
                    "normalized_entropy": safe_ratio(l_data["normalized_entropy"], e_data["normalized_entropy"]),
                    "simpson_index": safe_ratio(l_data["simpson_index"], e_data["simpson_index"])
                }
                
            city_results[month] = {
                "early": e_data,
                "late": l_data,
                "ratio_late_to_early": ratio
            }
            
        all_fragmentation_results[city] = city_results
        
    # Save the results to JSON
    output_file = os.path.join(RESULTS_DIR, "fragmentation_indices.json")
    with open(output_file, "w") as f:
        json.dump(all_fragmentation_results, f, indent=4)
        
    print(f"\nFragmentation indices saved to {output_file}")

if __name__ == "__main__":
    main()