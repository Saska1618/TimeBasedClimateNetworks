import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
GRAPH_DIR = os.path.join(PROJECT_ROOT, 'networks', 'global_networks', 'rich_global')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'degree_distributions', 'per_month')
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

def get_close_months(month):
    """Returns the set of months considered 'close' (previous, current, next)."""
    prev_m = 12 if month == 1 else month - 1
    next_m = 1 if month == 12 else month + 1
    return {prev_m, month, next_m}

def process_city(city):
    print(f"Processing monthly degree distributions for: {city}...")
    
    early_graph_path = os.path.join(GRAPH_DIR, f'{city}_{EARLY_PERIOD}.graphml')
    late_graph_path = os.path.join(GRAPH_DIR, f'{city}_{LATE_PERIOD}.graphml')
    
    if not os.path.exists(early_graph_path) or not os.path.exists(late_graph_path):
        print(f"  - WARNING: Missing graph data for {city}. Skipping.")
        return
        
    # Load the graphs
    G_early = nx.read_graphml(early_graph_path)
    G_late = nx.read_graphml(late_graph_path)
    
    # Initialize dictionary to hold all 4 variants per month
    data = {m: {
        'unweighted_all': {'early': [], 'late': []},
        'weighted_all': {'early': [], 'late': []},
        'unweighted_close': {'early': [], 'late': []},
        'weighted_close': {'early': [], 'late': []},
    } for m in range(1, 13)}
    
    # Extract degrees for the early period
    for node in G_early.nodes():
        month = int(node.split('-')[1])
        close_months = get_close_months(month)
        
        # All months
        data[month]['unweighted_all']['early'].append(G_early.degree(node))
        data[month]['weighted_all']['early'].append(G_early.degree(node, weight='weight'))
        
        # Close months
        unw_close = sum(1 for neighbor in G_early.neighbors(node) if int(neighbor.split('-')[1]) in close_months)
        w_close = sum(G_early[node][neighbor].get('weight', 1.0) for neighbor in G_early.neighbors(node) if int(neighbor.split('-')[1]) in close_months)
        
        data[month]['unweighted_close']['early'].append(unw_close)
        data[month]['weighted_close']['early'].append(w_close)
        
    # Extract degrees for the late period
    for node in G_late.nodes():
        month = int(node.split('-')[1])
        close_months = get_close_months(month)
        
        # All months
        data[month]['unweighted_all']['late'].append(G_late.degree(node))
        data[month]['weighted_all']['late'].append(G_late.degree(node, weight='weight'))
        
        # Close months
        unw_close = sum(1 for neighbor in G_late.neighbors(node) if int(neighbor.split('-')[1]) in close_months)
        w_close = sum(G_late[node][neighbor].get('weight', 1.0) for neighbor in G_late.neighbors(node) if int(neighbor.split('-')[1]) in close_months)
        
        data[month]['unweighted_close']['late'].append(unw_close)
        data[month]['weighted_close']['late'].append(w_close)

    # --- Plotting per month ---
    sns.set_theme(style="whitegrid")
    
    for month in range(1, 13):
        month_label = MONTH_LABELS[month - 1]
        
        # Combine all values for this month to calculate common bins so they match precisely across the 4 axes
        all_vals = (
            data[month]['unweighted_all']['early'] + data[month]['unweighted_all']['late'] +
            data[month]['weighted_all']['early'] + data[month]['weighted_all']['late'] +
            data[month]['unweighted_close']['early'] + data[month]['unweighted_close']['late'] +
            data[month]['weighted_close']['early'] + data[month]['weighted_close']['late']
        )
        global_min = min(all_vals) if all_vals else 0
        global_max = max(all_vals) if all_vals else 1
        common_bins = np.linspace(global_min, global_max, 31) # 30 bins
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        
        configs = [
            ('unweighted_all', axes[0, 0], 'Unweighted - All Months'),
            ('weighted_all', axes[0, 1], 'Weighted - All Months'),
            ('unweighted_close', axes[1, 0], 'Unweighted - Close Months'),
            ('weighted_close', axes[1, 1], 'Weighted - Close Months')
        ]
        
        for key, ax, title in configs:
            sns.histplot(data[month][key]['early'], color='blue', label='Early (1961-1990)', 
                         alpha=0.4, ax=ax, kde=True, stat="density", bins=common_bins)
            sns.histplot(data[month][key]['late'], color='darkorange', label='Late (1995-2024)', 
                         alpha=0.4, ax=ax, kde=True, stat="density", bins=common_bins)
            
            ax.set_title(title, fontsize=14)
            if ax in [axes[1, 0], axes[1, 1]]:
                ax.set_xlabel('Degree', fontsize=12)
            if ax in [axes[0, 0], axes[1, 0]]:
                ax.set_ylabel('Density', fontsize=12)
                
        axes[0, 1].legend(loc='upper right', fontsize=11)
        plt.suptitle(f'{city} - {month_label} Degree Distributions', fontsize=20, y=0.98)
        plt.tight_layout()
        
        out_file = os.path.join(OUTPUT_DIR, f'{city}_{month_label}_degree_distributions.png')
        plt.savefig(out_file, dpi=300)
        plt.close()
        
    print(f"  - Generated 12 monthly plots for {city}")

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for city in load_cities():
        process_city(city)