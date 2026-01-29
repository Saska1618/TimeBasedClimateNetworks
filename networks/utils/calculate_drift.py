
import os
import json
import numpy as np

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MATRICES_PATH = os.path.join(PROJECT_ROOT, 'prototypes', 'deterministic_similarity_matrices.json')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'prototypes', 'drift_analysis.json')

# Month numbers for each season
SEASONS = {
    'Winter (DJF)': [12, 1, 2],
    'Spring (MAM)': [3, 4, 5],
    'Summer (JJA)': [6, 7, 8],
    'Autumn (SON)': [9, 10, 11]
}

def calculate_circular_diff(best_early_month_idx, late_month_idx):
    """
    Calculates the circular difference between two month indices (0-11).
    Result is in the range [-6, 6].
    """
    diff = best_early_month_idx - late_month_idx
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12
    return diff

if __name__ == '__main__':
    # --- 1. Load the similarity matrices ---
    try:
        with open(MATRICES_PATH, 'r') as f:
            matrices = json.load(f)
    except FileNotFoundError:
        print(f"Error: Matrices file not found at {MATRICES_PATH}")
        print("Please run 'deterministic_analyze_prototypes.py' first.")
        exit(1)

    drift_results = {}
    print("Calculating drift for each city...")

    # --- 2. Process each city's matrix ---
    for city, matrix_list in matrices.items():
        print(f"  - Processing {city}...")
        matrix = np.array(matrix_list)
        city_results = {
            'monthly_drift': {},
            'mean_drift': 0,
            'seasonal_drift': {}
        }
        
        monthly_drifts = []
        # For each late month j (column)
        for j_idx in range(12): # 0-11 for columns
            late_month_similarities = matrix[:, j_idx]
            
            # Find the best matching early month i (row)
            # np.argmax gives the index of the max value
            i_star_idx = np.argmax(late_month_similarities)
            
            # Calculate the circular difference
            drift = calculate_circular_diff(i_star_idx, j_idx)
            
            # Store drift for the month (using 1-based month numbers for keys)
            city_results['monthly_drift'][j_idx + 1] = drift
            monthly_drifts.append(drift)
            
        # --- 3. Summarize the drift ---
        
        # Mean drift over all months
        city_results['mean_drift'] = np.mean(monthly_drifts)
        
        # Drift by season
        for season_name, season_months in SEASONS.items():
            seasonal_drifts = [city_results['monthly_drift'][m] for m in season_months]
            city_results['seasonal_drift'][season_name] = np.mean(seasonal_drifts)
            
        drift_results[city] = city_results

    # --- 4. Save the results ---
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(drift_results, f, indent=4)

    print(f"\nDrift analysis complete. Results saved to {OUTPUT_PATH}")
