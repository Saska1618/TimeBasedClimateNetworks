import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
PLOTS_DIR = os.path.join(SCRIPT_DIR, 'plots')

def run_percolation(G, thresholds):
    """
    Runs percolation by incrementally removing edges below the given thresholds.
    Returns the Largest Connected Component (LCC) fractions and the critical 
    threshold (the split point where the graph fragments the most).
    """
    total_nodes = G.number_of_nodes()
    if total_nodes == 0:
        return None
        
    lcc_fractions = []
    
    for threshold in thresholds:
        filtered_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight', 1.0) >= threshold]
        
        H = nx.Graph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(filtered_edges)
        
        if H.number_of_edges() == 0:
            lcc_fractions.append(1.0 / total_nodes)
        else:
            largest_cc = max(nx.connected_components(H), key=len)
            lcc_fractions.append(len(largest_cc) / total_nodes)
            
    # Find critical threshold (steepest drop in LCC fraction implies graph splitting)
    drops = -np.diff(lcc_fractions)
    if len(drops) > 0 and np.max(drops) > 0:
        max_drop_idx = np.argmax(drops)
        # The threshold where the massive drop begins
        critical_threshold = float(thresholds[max_drop_idx])
    else:
        critical_threshold = None
        
    return {
        "lcc_fractions": lcc_fractions,
        "critical_threshold": critical_threshold
    }

def plot_comparison(city, thresholds, early_data, late_data, output_path):
    """
    Creates a line plot comparing the Percolation transition for Early vs Late periods.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Early Period
    plt.plot(thresholds, early_data['lcc_fractions'], marker='o', linestyle='-', color='blue', markersize=4, alpha=0.8, label='Early (1961-1990)')
    if early_data.get('critical_threshold') is not None:
        plt.axvline(x=early_data['critical_threshold'], color='blue', linestyle='--', alpha=0.7, label=f"Early Split Point ({early_data['critical_threshold']:.3f})")
        
    # Plot Late Period
    plt.plot(thresholds, late_data['lcc_fractions'], marker='s', linestyle='-', color='red', markersize=4, alpha=0.8, label='Late (1995-2024)')
    if late_data.get('critical_threshold') is not None:
        plt.axvline(x=late_data['critical_threshold'], color='red', linestyle='--', alpha=0.7, label=f"Late Split Point ({late_data['critical_threshold']:.3f})")
    
    plt.title(f"Percolation Analysis Comparison: {city}", fontsize=16)
    plt.xlabel("Similarity Threshold (Edge Weight)", fontsize=14)
    plt.ylabel("Fraction of Nodes in Largest Component", fontsize=14)
    
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  - Saved comparison plot to {output_path}")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    all_results = {}
    NUM_STEPS = 50
    
    # Identify unique cities based on the filenames of the early period
    cities = set()
    for filename in os.listdir(GRAPH_DIR):
        if filename.endswith("_1961-01_1990-12.graphml"):
            cities.add(filename.replace("_1961-01_1990-12.graphml", ""))
            
    for city in sorted(cities):
        print(f"Processing percolation for {city}...")
        
        early_path = os.path.join(GRAPH_DIR, f"{city}_1961-01_1990-12.graphml")
        late_path = os.path.join(GRAPH_DIR, f"{city}_1995-01_2024-12.graphml")
        
        if not os.path.exists(early_path) or not os.path.exists(late_path):
            print(f"  - WARNING: Missing early or late graph for {city}. Skipping.")
            continue
            
        G_early = nx.read_graphml(early_path)
        G_late = nx.read_graphml(late_path)
        
        # Combine edge weights from both graphs to get a common range for the threshold X-axis
        early_weights = [d.get('weight', 1.0) for u, v, d in G_early.edges(data=True)]
        late_weights = [d.get('weight', 1.0) for u, v, d in G_late.edges(data=True)]
        all_weights = early_weights + late_weights
        
        if not all_weights:
            print(f"  - WARNING: No edges found for {city}. Skipping.")
            continue
            
        min_w, max_w = min(all_weights), max(all_weights)
        thresholds = np.linspace(min_w, max_w, NUM_STEPS)
        
        print("  - Simulating early period...")
        early_res = run_percolation(G_early, thresholds)
        
        print("  - Simulating late period...")
        late_res = run_percolation(G_late, thresholds)
        
        if early_res and late_res:
            all_results[city] = {
                "thresholds": thresholds.tolist(),
                "early": early_res,
                "late": late_res,
                "min_weight": float(min_w),
                "max_weight": float(max_w)
            }
            
            plot_path = os.path.join(PLOTS_DIR, f"{city}_percolation_comparison.png")
            plot_comparison(city, thresholds, early_res, late_res, plot_path)
                
    json_output_path = os.path.join(RESULTS_DIR, "percolation_results.json")
    with open(json_output_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nAll percolation comparison data saved to {json_output_path}")

if __name__ == "__main__":
    main()