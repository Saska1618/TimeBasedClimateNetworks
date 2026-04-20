import os
import json

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
INPUT_FILE = os.path.join(PROJECT_ROOT, 'networks', 'percolation', 'results', 'percolation_results.json')
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'percolation_index.json')

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Percolation results not found at {INPUT_FILE}")
        print("Please run 'networks/percolation/percolation_analysis.py' first.")
        return

    with open(INPUT_FILE, 'r') as f:
        percolation_data = json.load(f)

    percolation_indices = {}

    for city, data in percolation_data.items():
        early_threshold = data.get('early', {}).get('critical_threshold')
        late_threshold = data.get('late', {}).get('critical_threshold')

        # Calculate the index if both thresholds exist
        if early_threshold is not None and late_threshold is not None:
            # A positive index means the graph breaks apart at a higher threshold in the late period
            perc_index = late_threshold - early_threshold
            
            percolation_indices[city] = {
                "early_split_point": early_threshold,
                "late_split_point": late_threshold,
                "percolation_index": perc_index
            }
            print(f"[{city}] Early: {early_threshold:.4f} | Late: {late_threshold:.4f} | Index: {perc_index:+.4f}")
        else:
            print(f"  - WARNING: Missing critical threshold for {city}. Skipping.")

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(percolation_indices, f, indent=4)

    print(f"\nCalculated percolation index for {len(percolation_indices)} cities.")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()