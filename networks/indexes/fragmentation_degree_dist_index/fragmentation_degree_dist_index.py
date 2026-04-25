import os
import sys
import json
import networkx as nx

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

def get_close_months(month):
    """Returns the set of months considered 'close' (previous, current, next)."""
    prev_m = 12 if month == 1 else month - 1
    next_m = 1 if month == 12 else month + 1
    return {prev_m, month, next_m}

def process_graph(G):
    """
    Calculates the accumulated weighted and unweighted degrees for 'all months' 
    and 'close months' for each month of the year (1-12).
    """
    stats = {f"{m:02d}": {'unw_all': 0, 'unw_close': 0, 'w_all': 0.0, 'w_close': 0.0} for m in range(1, 13)}
    
    for node in G.nodes():
        try:
            month = int(node.split('-')[1])
        except (IndexError, ValueError):
            continue
            
        month_str = f"{month:02d}"
        close_months = get_close_months(month)
        
        # All neighbors
        unw_all = G.degree(node)
        w_all = G.degree(node, weight='weight')
        
        # Close neighbors
        unw_close = 0
        w_close = 0.0
        
        for neighbor in G.neighbors(node):
            try:
                n_month = int(neighbor.split('-')[1])
            except (IndexError, ValueError):
                continue
                
            if n_month in close_months:
                unw_close += 1
                w_close += G[node][neighbor].get('weight', 1.0)
                
        stats[month_str]['unw_all'] += unw_all
        stats[month_str]['unw_close'] += unw_close
        stats[month_str]['w_all'] += w_all
        stats[month_str]['w_close'] += w_close
        
    return stats

def safe_ratio(numerator, denominator, eps=1e-5):
    """Safely calculates the ratio to handle division by zero."""
    if denominator == 0 and numerator == 0:
        return 1.0
    if denominator == 0:
        return numerator / eps
    return numerator / denominator

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    cities = load_cities()
    all_results = {}
    
    for city in cities:
        print(f"Processing {city}...")
        early_path = os.path.join(GRAPH_DIR, f'{city}_{EARLY_PERIOD}.graphml')
        late_path = os.path.join(GRAPH_DIR, f'{city}_{LATE_PERIOD}.graphml')
        
        if not os.path.exists(early_path) or not os.path.exists(late_path):
            print(f"  - WARNING: Missing graph data for {city}. Skipping.")
            continue
            
        G_early = nx.read_graphml(early_path)
        G_late = nx.read_graphml(late_path)
        
        early_stats = process_graph(G_early)
        late_stats = process_graph(G_late)
        
        city_results = {}
        
        for m in range(1, 13):
            month_str = f"{m:02d}"
            e_stat = early_stats[month_str]
            l_stat = late_stats[month_str]
            
            # Calculate Early indices (close / all)
            e_unw_index = safe_ratio(e_stat['unw_close'], e_stat['unw_all'])
            e_w_index = safe_ratio(e_stat['w_close'], e_stat['w_all'])
            
            # Calculate Late indices (close / all)
            l_unw_index = safe_ratio(l_stat['unw_close'], l_stat['unw_all'])
            l_w_index = safe_ratio(l_stat['w_close'], l_stat['w_all'])
            
            city_results[month_str] = {
                "unweighted": {
                    "early_index": e_unw_index,
                    "late_index": l_unw_index,
                    "ratio_late_to_early": safe_ratio(l_unw_index, e_unw_index)
                },
                "weighted": {
                    "early_index": e_w_index,
                    "late_index": l_w_index,
                    "ratio_late_to_early": safe_ratio(l_w_index, e_w_index)
                }
            }
            
        all_results[city] = city_results
        
    # Save the results to JSON
    output_file = os.path.join(RESULTS_DIR, "fragmentation_degree_dist_index.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\nFragmentation degree distribution indices saved to {output_file}")

if __name__ == "__main__":
    main()