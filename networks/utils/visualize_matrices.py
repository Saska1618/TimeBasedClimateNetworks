
import os
import json
import numpy as np

# Attempt to import plotting libraries, provide helpful error if they are not installed.
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Error: Matplotlib or Seaborn is not installed.")
    print("Please install them using: pip install matplotlib seaborn")
    exit(1)

# --- Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
MATRICES_PATH = os.path.join(PROJECT_ROOT, 'prototypes', 'deterministic_similarity_matrices.json')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'plots', 'similarity_heatmaps')
MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def create_heatmap(matrix, city_name, output_path):
    """Generates and saves a heatmap for a given similarity matrix."""
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.1)
    
    # Create the heatmap
    ax = sns.heatmap(
        matrix,
        xticklabels=MONTH_LABELS,
        yticklabels=MONTH_LABELS,
        annot=True,          # Show the values in the cells
        fmt=".2f",           # Format values to two decimal places
        cmap="viridis",      # Color map
        linewidths=.5
    )
    
    # Set titles and labels
    ax.set_title(f'Monthly Similarity Matrix for {city_name}', fontsize=18, pad=20)
    ax.set_xlabel('Late Period (1991-2024)', fontsize=14, labelpad=15)
    ax.set_ylabel('Early Period (1960-1990)', fontsize=14, labelpad=15)
    
    # Ensure labels are not rotated
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent labels/titles from being cut off
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory
    print(f"  - Saved heatmap to {output_path}")

if __name__ == '__main__':
    # --- 1. Load the similarity matrices ---
    try:
        with open(MATRICES_PATH, 'r') as f:
            matrices = json.load(f)
    except FileNotFoundError:
        print(f"Error: Matrices file not found at {MATRICES_PATH}")
        print("Please run 'deterministic_analyze_prototypes.py' first.")
        exit(1)

    # --- 2. Create output directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # --- 3. Generate a plot for each city ---
    for city, matrix_list in matrices.items():
        print(f"Processing city: {city}...")
        matrix = np.array(matrix_list)
        output_file_path = os.path.join(OUTPUT_DIR, f'{city}_heatmap.png')
        create_heatmap(matrix, city, output_file_path)

    print("\nAll heatmap visualizations have been generated.")
