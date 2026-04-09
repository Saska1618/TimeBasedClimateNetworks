import os
import sys
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'degree_distributions', 'unweighted', 'all_months')
STATS_DIR = os.path.join(SCRIPT_DIR, 'monthly_degrees')
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

EARLY_PERIOD = '1961-01_1990-12'
LATE_PERIOD = '1995-01_2024-12'

# --- Load City Names ---
CITIES_FILE = os.path.join(PROJECT_ROOT, 'networks', 'utils', 'cities.txt')

def load_cities():
    try:
        with open(CITIES_FILE, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Error: Cities file not found at {CITIES_FILE}")
        sys.exit(1)

def process_city_degree_distributions(city):
    print(f"Processing degree distributions for: {city}...")
    
    early_graph_path = os.path.join(GRAPH_DIR, f'{city}_{EARLY_PERIOD}.graphml')
    late_graph_path = os.path.join(GRAPH_DIR, f'{city}_{LATE_PERIOD}.graphml')
    
    if not os.path.exists(early_graph_path) or not os.path.exists(late_graph_path):
        print(f"  - WARNING: Missing graph data for {city}. Skipping.")
        return
        
    # Load the graphs
    G_early = nx.read_graphml(early_graph_path)
    G_late = nx.read_graphml(late_graph_path)
    
    # Dictionaries to hold degrees per month (1-12)
    early_degrees = {m: [] for m in range(1, 13)}
    late_degrees = {m: [] for m in range(1, 13)}
    
    # Extract degrees for the early period
    for node in G_early.nodes():
        # Node IDs are typically formatted as 'YYYY-MM'
        month = int(node.split('-')[1])
        early_degrees[month].append(G_early.degree(node))
        
    # Extract degrees for the late period
    for node in G_late.nodes():
        month = int(node.split('-')[1])
        late_degrees[month].append(G_late.degree(node))
        
    # --- Statistics ---
    city_stats = {
        "early_period": EARLY_PERIOD,
        "late_period": LATE_PERIOD,
        "months": {}
    }

    # --- Plotting ---
    fig, axes = plt.subplots(4, 3, figsize=(18, 16), sharex=False, sharey=False)
    axes = axes.flatten()
    
    sns.set_theme(style="whitegrid")
    
    for month in range(1, 13):
        e_deg = early_degrees[month]
        l_deg = late_degrees[month]
        
        city_stats["months"][MONTH_LABELS[month - 1]] = {
            "early": {
                "mean": float(np.mean(e_deg)) if e_deg else 0.0,
                "std": float(np.std(e_deg)) if e_deg else 0.0,
                "median": float(np.median(e_deg)) if e_deg else 0.0,
                "min": int(np.min(e_deg)) if e_deg else 0,
                "max": int(np.max(e_deg)) if e_deg else 0,
            },
            "late": {
                "mean": float(np.mean(l_deg)) if l_deg else 0.0,
                "std": float(np.std(l_deg)) if l_deg else 0.0,
                "median": float(np.median(l_deg)) if l_deg else 0.0,
                "min": int(np.min(l_deg)) if l_deg else 0,
                "max": int(np.max(l_deg)) if l_deg else 0,
            }
        }

        ax = axes[month - 1]
        
        # Plot density histograms with KDE lines
        sns.histplot(early_degrees[month], color='blue', label='Early (1961-1990)', 
                     alpha=0.4, ax=ax, kde=True, stat="density", discrete=True)
        sns.histplot(late_degrees[month], color='darkorange', label='Late (1995-2024)', 
                     alpha=0.4, ax=ax, kde=True, stat="density", discrete=True)
        
        ax.set_title(MONTH_LABELS[month - 1], fontsize=14)
        ax.set_xlabel('Node Degree', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        
        # Only show legend on the first subplot to keep it clean
        if month == 1:
            ax.legend(loc='upper right', fontsize=11)
            
    plt.suptitle(f'Monthly Unweighted Degree Distributions: {city}', fontsize=22, y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save the plot
    output_path = os.path.join(OUTPUT_DIR, f'{city}_monthly_degree_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"  - Saved plot to {output_path}")
    
    # Save the stats
    stats_output_path = os.path.join(STATS_DIR, f'{city}_monthly_degree_stats.json')
    with open(stats_output_path, 'w') as f:
        json.dump(city_stats, f, indent=4)
    print(f"  - Saved stats to {stats_output_path}")

if __name__ == '__main__':
    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    
    cities = load_cities()
    for city in cities:
        process_city_degree_distributions(city)