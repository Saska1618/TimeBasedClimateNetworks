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

def detect_communities(file_path, resolution=1.0):
    """
    Loads a graph and runs the Leiden algorithm.
    """
    print(f"Loading {os.path.basename(file_path)}...")
    G = nx.read_graphml(file_path)
    
    if G.number_of_nodes() == 0:
        print("  - Graph is empty, skipping.")
        return None
        
    g_ig = convert_nx_to_igraph(G)
    
    # Determine if we have edge weights
    weights = g_ig.es['weight'] if 'weight' in g_ig.edge_attributes() else None
    
    # Run Leiden Algorithm with a customizable resolution parameter
    print("  - Running Leiden algorithm...")
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
        
    plt.title(f"Temporal Community Structure: {graph_name}\nModularity: {results['modularity_score']:.3f}", fontsize=16)
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
    
    all_results = {}
    
    # Set your desired resolution parameter here!
    # > 1.0 favors more/smaller communities, < 1.0 favors fewer/larger communities
    TARGET_RESOLUTION = 2.25
    
    for filename in os.listdir(GRAPH_DIR):
        if filename.endswith(".graphml"):
            graph_name = filename.replace(".graphml", "")
            file_path = os.path.join(GRAPH_DIR, filename)
            
            result = detect_communities(file_path, resolution=TARGET_RESOLUTION)
            if result:
                all_results[graph_name] = result
                plot_path = os.path.join(PLOTS_DIR, f"{graph_name}_leiden_res_{TARGET_RESOLUTION}.png")
                plot_communities(graph_name, result, plot_path)
                
    # Save all raw community data to JSON
    json_output_path = os.path.join(RESULTS_DIR, "leiden_communities.json")
    with open(json_output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll community data saved to {json_output_path}")

if __name__ == "__main__":
    main()