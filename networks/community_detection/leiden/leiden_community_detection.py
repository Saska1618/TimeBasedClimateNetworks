import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    import igraph as ig
    import leidenalg
except ImportError:
    print("Error: Required packages are missing.")
    print("Please run: pip install igraph leidenalg")
    exit(1)

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

def convert_nx_to_igraph(G):
    """
    Safely converts a NetworkX graph to an igraph Graph,
    preserving node names and edge weights.
    """
    node_names = list(G.nodes())
    g_ig = ig.Graph(directed=G.is_directed())
    g_ig.add_vertices(len(node_names))
    g_ig.vs['name'] = node_names
    
    edges = list(G.edges(data=True))
    ig_edges = [(node_names.index(u), node_names.index(v)) for u, v, d in edges]
    g_ig.add_edges(ig_edges)
    
    # Preserve weights if they exist
    if edges and 'weight' in edges[0][2]:
        weights = [d.get('weight', 1.0) for u, v, d in edges]
        g_ig.es['weight'] = weights
        
    return g_ig

def get_num_communities(g_ig, resolution):
    """Helper to quickly get the number of communities for a given resolution."""
    weights = g_ig.es['weight'] if 'weight' in g_ig.edge_attributes() else None
    partition = leidenalg.find_partition(
        g_ig, 
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution
    )
    return len(partition)

def find_resolution_for_target(g_ig, target_num, max_iters=50):
    """
    Binary search to find the optimal resolution parameter 
    to achieve a target number of communities.
    """
    low = 0.001
    high = 5.0
    
    # Expand upper bound if necessary
    while get_num_communities(g_ig, high) < target_num and high < 1000.0:
        high *= 2.0
        
    best_res = 1.0
    best_diff = float('inf')

    for _ in range(max_iters):
        mid = (low + high) / 2.0
        num_comms = get_num_communities(g_ig, mid)

        diff = abs(num_comms - target_num)
        if diff < best_diff:
            best_diff = diff
            best_res = mid

        if num_comms == target_num:
            break
        elif num_comms < target_num:
            low = mid  # Need more communities -> higher resolution
        else:
            high = mid # Need fewer communities -> lower resolution

    return best_res

def detect_communities(g_ig, resolution=1.0):
    """
    Runs the Leiden algorithm on an igraph object.
    """
    # Determine if we have edge weights
    weights = g_ig.es['weight'] if 'weight' in g_ig.edge_attributes() else None
    
    # Run Leiden Algorithm with a customizable resolution parameter
    print(f"  - Running Leiden algorithm (resolution={resolution:.4f})...")
    partition = leidenalg.find_partition(
        g_ig, 
        leidenalg.RBConfigurationVertexPartition,
        weights=weights,
        resolution_parameter=resolution
    )
    
    # Format the results
    communities = {}
    for idx, cluster in enumerate(partition):
        # Get the original node names (e.g., "1960-01")
        node_names = [g_ig.vs[node_index]['name'] for node_index in cluster]
        communities[f"Community_{idx + 1}"] = node_names
        
    result = {
        "modularity_score": partition.quality(),
        "num_communities": len(communities),
        "communities": communities
    }
    
    print(f"  - Found {len(communities)} communities (Modularity: {partition.quality():.4f})")
    return result

def plot_communities(graph_name, results, output_path):
    """
    Creates a temporal scatter plot of the detected communities.
    X-axis: Year, Y-axis: Month, Color: Community.
    """
    plt.figure(figsize=(14, 7))
    
    # Get a good colormap
    colors = list(mcolors.TABLEAU_COLORS.values())
    if results['num_communities'] > len(colors):
        # Fallback to a larger colormap if many communities are found
        cmap = plt.cm.get_cmap('tab20', results['num_communities'])
        colors = [cmap(i) for i in range(results['num_communities'])]
        
    all_years = []
    
    for idx, (comm_name, nodes) in enumerate(results['communities'].items()):
        years = []
        months = []
        for node in nodes:
            try:
                y, m = node.split('-')
                years.append(int(y))
                months.append(int(m))
                all_years.append(int(y))
            except ValueError:
                continue # Skip nodes that don't match the "YYYY-MM" format
                
        color = colors[idx % len(colors)]
        plt.scatter(years, months, c=[color], label=comm_name, alpha=0.8, edgecolors='w', s=80)
        
    plt.title(f"Temporal Community Structure: {graph_name} ({results['num_communities']} Clusters)\nModularity: {results['modularity_score']:.3f}", fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Month", fontsize=14)
    plt.yticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    if all_years:
        plt.xlim(min(all_years) - 1, max(all_years) + 1)
        
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  - Saved plot to {output_path}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    optimal_resolutions = {}
    all_results_base = {}
    all_results_4 = {}
    all_results_12 = {}
    
    for filename in os.listdir(GRAPH_DIR):
        if filename.endswith(".graphml"):
            graph_name = filename.replace(".graphml", "")
            file_path = os.path.join(GRAPH_DIR, filename)
            
            print(f"\nProcessing {graph_name}...")
            G = nx.read_graphml(file_path)
            
            if G.number_of_nodes() == 0:
                print("  - Graph is empty, skipping.")
                continue
                
            g_ig = convert_nx_to_igraph(G)
            
            print("  - Finding optimal resolution for 4 communities...")
            res_4 = find_resolution_for_target(g_ig, 4)
            print("  - Finding optimal resolution for 12 communities...")
            res_12 = find_resolution_for_target(g_ig, 12)
            
            optimal_resolutions[graph_name] = {
                "base_1.0": 1.0,
                "target_4": res_4,
                "target_12": res_12
            }
            
            # Run and plot for Base (1.0)
            print("  => Running for Base Resolution (1.0)")
            res_base = detect_communities(g_ig, resolution=1.0)
            all_results_base[graph_name] = res_base
            plot_communities(graph_name, res_base, os.path.join(PLOTS_DIR, f"{graph_name}_leiden_base.png"))
            
            # Run and plot for Target 4
            print(f"  => Running for Target 4 Communities")
            res_tgt4 = detect_communities(g_ig, resolution=res_4)
            all_results_4[graph_name] = res_tgt4
            plot_communities(graph_name, res_tgt4, os.path.join(PLOTS_DIR, f"{graph_name}_leiden_target_4.png"))
            
            # Run and plot for Target 12
            print(f"  => Running for Target 12 Communities")
            res_tgt12 = detect_communities(g_ig, resolution=res_12)
            all_results_12[graph_name] = res_tgt12
            plot_communities(graph_name, res_tgt12, os.path.join(PLOTS_DIR, f"{graph_name}_leiden_target_12.png"))
            
    # Save resolutions
    res_file = os.path.join(RESULTS_DIR, "leiden_optimal_resolutions.json")
    with open(res_file, "w") as f:
        json.dump(optimal_resolutions, f, indent=4)
    print(f"\nOptimal resolutions saved to {res_file}")
            
    # Save all raw community data to JSON
    with open(os.path.join(RESULTS_DIR, "leiden_communities_base.json"), "w") as f:
        json.dump(all_results_base, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "leiden_communities_target_4.json"), "w") as f:
        json.dump(all_results_4, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "leiden_communities_target_12.json"), "w") as f:
        json.dump(all_results_12, f, indent=4)
    print("All community data saved to JSON files in results directory.")

if __name__ == "__main__":
    main()