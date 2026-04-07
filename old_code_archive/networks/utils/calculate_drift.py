import json
import os
import numpy as np

def circular_diff(i_star, j):
    """Computes the circular month shift."""
    # This is the difference in months between the best matching early month (i_star) and the late month (j)
    diff = i_star - j
    if diff > 6:
        diff -= 12
    elif diff < -6:
        diff += 12
    return diff

def calculate_drift(similarity_matrix):
    """
    Calculates drift measures from a similarity matrix.

    The similarity matrix has early months as rows and late months as columns.
    A high value at M[i, j] means early month `i` is very similar to late month `j`.
    """
    matrix = np.array(similarity_matrix)
    
    # For each late month `j` (each column), find the early month `i` (row index) that has the maximum similarity.
    # `i_star_indices[j]` will hold the index `i` of the most similar early month for the late month `j`.
    i_star_indices = np.argmax(matrix, axis=0)
    
    # For each late month `j`, compute the circular difference between `j` and its best match `i_star_indices[j]`.
    # A positive drift means the season is appearing earlier (e.g., late April looks like early May).
    delta = np.array([circular_diff(i_star, j) for j, i_star in enumerate(i_star_indices)])
    
    # --- Summarize Drift ---
    
    # Overall mean drift across all months
    mean_drift = np.mean(delta)
    
    # Define seasons for late months (0=Jan, 11=Dec)
    djf_indices = [11, 0, 1]  # December, January, February
    mam_indices = [2, 3, 4]   # March, April, May
    jja_indices = [5, 6, 7]   # June, July, August
    son_indices = [8, 9, 10]  # September, October, November

    seasonal_drift = {
        "DJF": np.mean(delta[djf_indices]),
        "MAM": np.mean(delta[mam_indices]),
        "JJA": np.mean(delta[jja_indices]),
        "SON": np.mean(delta[son_indices]),
    }
    
    # Find the month that experienced the largest absolute drift
    max_drift_month_index = np.argmax(np.abs(delta))
    max_drift_value = delta[max_drift_month_index]
    
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Store the drift value for each individual month
    drift_per_month = {months[i]: delta[i] for i in range(12)}

    summary = {
        "mean_drift": float(mean_drift),
        "seasonal_drift": {k: float(v) for k, v in seasonal_drift.items()},
        "most_drifting_month": {
            "month": months[max_drift_month_index],
            "drift": int(max_drift_value),
        },
        "drift_per_month": {k: int(v) for k, v in drift_per_month.items()}
    }
    
    return summary

def main():
    """
    Main function to load data, calculate drift for each city, and print the results.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    similarity_matrices_path = os.path.join(script_dir, "../../prototypes/deterministic_similarity_matrices.json")
    
    try:
        with open(similarity_matrices_path, 'r') as f:
            similarity_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Similarity data not found at '{similarity_matrices_path}'")
        return
        
    all_drifts = {}
    
    # Calculate drift for each city using its similarity matrix
    for city, similarity_matrix in similarity_data.items():
        drift_summary = calculate_drift(similarity_matrix)
        all_drifts[city] = drift_summary
        
    # --- Output Results ---
    
    # Pretty print the summary for each city to the console
    for city, drift in all_drifts.items():
        print(f"--- Drift Analysis for {city.replace('_', ' ')} ---")
        print(f"Mean annual drift: {drift['mean_drift']:.2f} months")
        print("Seasonal drift (mean months):")
        for season, value in drift['seasonal_drift'].items():
            print(f"  {season}: {value:.2f}")
        print("Most drifting month:")
        most_drifting = drift['most_drifting_month']
        print(f"  {most_drifting['month']} with a drift of {most_drifting['drift']:.2f} months")
        print("-" * (27 + len(city)))
        print()

    # Save the detailed drift calculations to a file for later use
    output_path = os.path.join(script_dir, "../../prototypes/drift_analysis.json")
    with open(output_path, "w") as f:
        json.dump(all_drifts, f, indent=4)
    
    print(f"Complete drift analysis saved to '{output_path}'")


if __name__ == "__main__":
    main()