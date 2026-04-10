import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    from sklearn.cluster import SpectralClustering
except ImportError:
    print("Error: Required package 'scikit-learn' is missing.")
    print("Please run: pip install scikit-learn")
    exit(1)

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

def detect_communities(file_path, n_clusters=4):
    """
    Loads a graph and runs Spectral Clustering to find a specific number of communities.
    """
    print(f"Loading {os.path.basename(file_path)}...")
    G = nx.read_graphml(file_path)
    
    if G.number_of_nodes() == 0:
        print("  - Graph is empty, skipping.")
        return None
        
    # For Spectral Clustering, we need the adjacency matrix (acts as our affinity matrix)
    # We use 'weight' to ensure the similarity scores between months are considered
    adj_matrix = nx.to_numpy_array(G, weight='weight')
    
    print(f"  - Running Spectral Clustering (n_clusters={n_clusters})...")
    
    # Initialize and fit the Spectral Clustering model
    # We use 'precomputed' because we are passing our own graph adjacency matrix
    sc = SpectralClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        assign_labels='kmeans',
        random_state=42 # Set for reproducibility
    )
    
    labels = sc.fit_predict(adj_matrix)
    
    # Format the results
    node_names = list(G.nodes())
    communities = {f"Community_{i + 1}": [] for i in range(n_clusters)}
    
    for node, label in zip(node_names, labels):
        communities[f"Community_{label + 1}"].append(node)
        
    # Calculate Modularity to compare quality with Leiden
    # NetworkX expects a list of sets for the communities parameter
    comm_list = [set(nodes) for nodes in communities.values()]
    try:
        mod_score = nx.community.modularity(G, comm_list, weight='weight')
    except Exception:
        mod_score = 0.0 # Fallback if graph is disconnected or modularity fails
        
    result = {
        "modularity_score": mod_score,
        "num_communities": n_clusters,
        "communities": communities
    }
    
    print(f"  - Grouped into {n_clusters} communities (Modularity: {mod_score:.4f})")
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
        
    plt.title(f"Spectral Clustering Structure: {graph_name} ({results['num_communities']} Clusters)\nModularity: {results['modularity_score']:.3f}", fontsize=16)
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
    
    # Set your desired number of clusters here!
    TARGET_CLUSTERS = 12
    
    for filename in os.listdir(GRAPH_DIR):
        if filename.endswith(".graphml"):
            graph_name = filename.replace(".graphml", "")
            file_path = os.path.join(GRAPH_DIR, filename)
            
            result = detect_communities(file_path, n_clusters=TARGET_CLUSTERS)
            if result:
                all_results[graph_name] = result
                plot_path = os.path.join(PLOTS_DIR, f"{graph_name}_spectral_{TARGET_CLUSTERS}.png")
                plot_communities(graph_name, result, plot_path)
                
    # Save all raw community data to JSON
    json_output_path = os.path.join(RESULTS_DIR, f"spectral_communities_{TARGET_CLUSTERS}.json")
    with open(json_output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nAll community data saved to {json_output_path}")

if __name__ == "__main__":
    main()